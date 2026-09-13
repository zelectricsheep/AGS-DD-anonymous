#!/usr/bin/env python3
"""
Duration sweep: constant guidance × {low_noise, high_noise} × {10, 15, 25, 35} steps.

Isolates guidance window position from cumulative guidance duration.
ImageNette, IPC=10, 10 classes, 5 datasets per config, ConvNet7 20-epoch eval.

Usage (dual GPU):
  CUDA_VISIBLE_DEVICES=0 python duration_sweep.py --window low_noise  &
  CUDA_VISIBLE_DEVICES=1 python duration_sweep.py --window high_noise &
"""

import os
import sys
import time
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from collections import defaultdict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from tsne_plots import get_loader, get_features_per_class
from ags import (
    ClassComplexityAnalyzer,
    AdaptiveStopTiming,
    TimestepAdaptiveSchedule,
    AGSSampler,
)


def load_model_and_vae(device, image_size=256):
    latent_size = image_size // 8
    model = DiT_models["DiT-XL/2"](
        input_size=latent_size, num_classes=1000
    ).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    diffusion = create_diffusion("50")
    return model, vae, diffusion, latent_size


def setup_classes(spec="nette", nclass=10):
    with open("./misc/class_indices.txt", "r") as fp:
        all_classes = [c.strip() for c in fp.readlines()]
    file_map = {
        "woof": "./misc/class_woof.txt",
        "nette": "./misc/class_nette.txt",
        "imagenet100": "./misc/class100.txt",
        "imagenet1k": "./misc/class_indices.txt",
        "animals": "./misc/class_animals_subset.txt",
        "objects": "./misc/class_objects_subset.txt",
        "imagenet200": "./misc/class200.txt",
    }
    with open(file_map[spec], "r") as fp:
        sel_classes = [c.strip() for c in fp.readlines()]
    sel_classes = sel_classes[:nclass]
    class_labels = [all_classes.index(c) for c in sel_classes]
    return class_labels, sel_classes


def compute_clusters(spec, imagenet_dir, vae, device, nclass=10,
                        sigmoid_slope=3.0, sigmoid_center=0.6, complexity_k=None,
                        alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    import argparse as ap
    args_eval = ap.Namespace(
        dataset="imagenet", net_type="resnet_ap", depth=10, width=1.0,
        norm_type="instance", nch=3,
        data_path=os.path.join(imagenet_dir, "train"),
        size=224, aug_type="color_crop_cutout", augment=True,
        dseed=0, global_batch_size=1, use_vae=True, finetune_ipc=-1,
        nclass=nclass, spec=spec, phase=0, image_size=256,
    )
    loader = get_loader(args_eval, return_path=True)
    features_per_class, paths_per_class = get_features_per_class(args_eval, loader, vae)

    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(2, 20), alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        use_pca=True, sigmoid_slope=sigmoid_slope, sigmoid_center=sigmoid_center,
    )
    analyzer.analyze_all_classes(features_per_class, paths_per_class, fixed_k=complexity_k)
    return analyzer


def generate_config(
    model, vae, diffusion, latent_size, device,
    analyzer, class_labels, sel_classes, ipc,
    window, guidance_steps, num_datasets, save_dir,
    schedule="constant",
    no_cags=False, fixed_scale=0.1, fixed_stop_t=None,
    cags_min_scale=0.05, cags_max_scale=0.3,
    sigmoid_slope=3.0, sigmoid_center=0.6,
    use_iast=False,
):
    t_max = 49
    if window == "high_noise":
        default_stop_t = t_max - guidance_steps
    else:
        default_stop_t = guidance_steps

    if fixed_stop_t is not None:
        default_stop_t = fixed_stop_t

    use_tags = schedule != "constant"
    sched_type = "cosine" if schedule == "constant" else schedule

    adaptive_stop = AdaptiveStopTiming(t_max=50, lam=0.316, window=window)
    guidance_schedule = TimestepAdaptiveSchedule(
        w_max=0.3, schedule_type=sched_type,
    )
    use_cags = not no_cags
    sampler = AGSSampler(
        model=model, vae=vae, diffusion=diffusion,
        complexity_analyzer=analyzer,
        adaptive_stop=adaptive_stop,
        guidance_schedule=guidance_schedule,
        device=device, num_sampling_steps=50,
        cfg_scale=4.0, latent_size=latent_size,
        use_cags=use_cags, use_iast=use_iast, use_tags=use_tags,
        guidance_scale_range=(cags_min_scale, cags_max_scale),
        guidance_window=window,
    )
    sampler.default_guidance_scale = fixed_scale
    sampler.default_stop_t = default_stop_t

    clusters_centers = analyzer.compute_clusters_for_ipc(
        ipc, use_pca=False, closest_point=False
    )

    args = argparse.Namespace(num_samples=ipc, num_datasets=num_datasets)
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()
    sampler.generate_dataset(
        args=args,
        class_labels=class_labels,
        sel_classes=sel_classes,
        clusters_centers=clusters_centers,
        save_dir=save_dir,
        num_datasets=num_datasets,
        use_same_noise=False,
        total_shift=0,
    )
    elapsed = time.time() - start
    print(f"  Generated {num_datasets} datasets in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return elapsed


class GeneratedDataset(Dataset):
    def __init__(self, root, class_names, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        for idx, cls_name in enumerate(class_names):
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class ConvNet(nn.Module):
    def __init__(self, channel=128, num_classes=10, depth=7, norm="instance"):
        super().__init__()
        self.features = nn.ModuleList()
        in_ch = 3
        for i in range(depth):
            out_ch = channel
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            if norm == "instance":
                norm_layer = nn.InstanceNorm2d(out_ch)
            elif norm == "batch":
                norm_layer = nn.BatchNorm2d(out_ch)
            else:
                norm_layer = nn.Identity()
            block = nn.Sequential(
                conv, norm_layer, nn.ReLU(inplace=True), nn.AvgPool2d(2),
            )
            self.features.append(block)
            in_ch = out_ch
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def evaluate_dataset(train_dir, test_dir, class_names, num_classes, device,
                     img_size=224, epochs=20, batch_size=64, seed=0,
                     depth=6, use_cutmix=True, use_rrc=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ConvNet(num_classes=num_classes, depth=depth).to(device)
    aug_list = []
    if use_rrc:
        aug_list.append(transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)))
    else:
        aug_list.append(transforms.Resize((img_size, img_size)))
    aug_list.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose(aug_list)
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = GeneratedDataset(train_dir, class_names, train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=False,
                             persistent_workers=True, pin_memory=True)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            if use_cutmix and np.random.rand() < 1.0:
                lam = np.random.beta(1.0, 1.0)
                idx = torch.randperm(images.size(0), device=device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[idx, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
                outputs = model(images)
                loss = criterion(outputs, targets) * ratio + criterion(outputs, targets[idx]) * (1. - ratio)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = 0
    total = 0
    top5_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if outputs.size(1) >= 5:
                _, pred5 = outputs.topk(5, 1, True, True)
                top5_correct += pred5.eq(targets.view(-1, 1)).any(dim=1).sum().item()

    top1 = 100.0 * correct / total
    top5 = 100.0 * top5_correct / total if total > 0 else 0
    del model
    torch.cuda.empty_cache()
    return top1, top5


def main():
    parser = argparse.ArgumentParser(description="Duration sweep experiment")
    parser.add_argument("--window", type=str, default="low_noise",
                        choices=["low_noise", "high_noise"])
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenette2/")
    parser.add_argument("--save-base", type=str, default="./results/sweep")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--num-datasets", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--durations", type=int, nargs="+", default=[10, 15, 25, 35])
    parser.add_argument("--schedule", type=str, default="constant",
                        choices=["constant", "cosine", "linear", "exponential"],
                        help="Weight profile inside the guidance window")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Classifier training seeds to average over")
    parser.add_argument("--tag", type=str, default=None,
                        help="Override config-name suffix (default derived from schedule)")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip generation entirely; only evaluate existing images")
    parser.add_argument("--spec", type=str, default="nette",
                        choices=["nette", "woof", "imagenet100", "imagenet1k"],
                        help="Dataset spec for class selection")
    parser.add_argument("--nclass", type=int, default=10,
                        help="Number of classes (use 100 for imagenet100)")
    parser.add_argument("--depth", type=int, default=6,
                        help="ConvNet depth (MGD3 default=6)")
    parser.add_argument("--no-cags", action="store_true", default=False,
                        help="Disable CAGS; use fixed guidance scale (MGD3 mode)")
    parser.add_argument("--fixed-scale", type=float, default=0.1,
                        help="Fixed guidance scale when CAGS disabled (MGD3 default=0.1)")
    parser.add_argument("--use-iast", action="store_true", default=False,
                        help="Enable IAST adaptive stop timing (ablation)")
    parser.add_argument("--fixed-stop-t", type=int, default=None,
                        help="Override IAST/duration with a fixed stop_t (MGD3 default=25)")
    parser.add_argument("--cags-min-scale", type=float, default=0.05,
                        help="Minimum CAGS guidance scale (default 0.05)")
    parser.add_argument("--cags-max-scale", type=float, default=0.3,
                        help="Maximum CAGS guidance scale (default 0.3, lower to 0.12-0.15 for tuning)")
    parser.add_argument("--sigmoid-slope", type=float, default=3.0,
                        help="Sigmoid slope for CAGS complexity mapping (default 3.0)")
    parser.add_argument("--sigmoid-center", type=float, default=0.6,
                        help="Sigmoid center for CAGS complexity mapping (default 0.6)")
    parser.add_argument("--regen-cache", action="store_true", default=False,
                        help="Delete existing cluster_cache.pkl and regenerate")
    parser.add_argument("--complexity-k", type=int, default=None,
                        help="Fixed K for complexity computation (default: silhouette selection)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Weight for mode_count in complexity (default 0.3)")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Weight for entropy in complexity (default 0.3)")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Weight for intra-class variance in complexity (default 0.2)")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="Weight for 1-separability in complexity (default 0.2)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sched_tag = args.tag or ("const" if args.schedule == "constant" else args.schedule)
    print(f"Device: {device}, Window: {args.window}")
    print(f"Durations: {args.durations}, Schedule: {args.schedule} (tag={sched_tag})")
    print(f"Seeds: {args.seeds}, Eval-only: {args.eval_only}")
    print(f"Data dir: {args.imagenet_dir}, Spec: {args.spec}, Nclass: {args.nclass}")

    class_labels, sel_classes = setup_classes(spec=args.spec, nclass=args.nclass)
    print(f"Classes: {sel_classes}")

    os.makedirs(args.save_base, exist_ok=True)
    cluster_cache = os.path.join(args.save_base, "cluster_cache.pkl")

    if args.eval_only:
        print("\n[eval-only] Skipping DiT/VAE load and CAGS clustering.")
        model = vae = diffusion = analyzer = None
        latent_size = 32
    else:
        print("\nLoading DiT model and VAE...")
        model, vae, diffusion, latent_size = load_model_and_vae(device)
        print("Model loaded.")

        if args.regen_cache and os.path.isfile(cluster_cache):
            print(f"\n[regen-cache] Deleting {cluster_cache}")
            os.remove(cluster_cache)

        if os.path.isfile(cluster_cache) and not args.regen_cache:
            print(f"\nLoading cached CAGS clusters from {cluster_cache}")
            with open(cluster_cache, "rb") as f:
                analyzer = pickle.load(f)
        else:
            print("\nComputing CAGS clusters...")
            analyzer = compute_clusters(args.spec, args.imagenet_dir, vae, device,
                                        nclass=args.nclass,
                                        sigmoid_slope=args.sigmoid_slope,
                                        sigmoid_center=args.sigmoid_center,
                                        complexity_k=args.complexity_k,
                                        alpha=args.alpha, beta=args.beta,
                                        gamma=args.gamma, delta=args.delta)
            with open(cluster_cache, "wb") as f:
                pickle.dump(analyzer, f)
            print(f"Cluster cache saved to {cluster_cache}")
        for c in analyzer.complexity_scores:
            print(f"  Class {c}: complexity={analyzer.complexity_scores[c]:.4f}, "
                  f"modes={analyzer.mode_counts[c]}")

    all_results = {}

    for duration in args.durations:
        config_name = f"{args.window}_{sched_tag}_d{duration}"
        save_dir = os.path.join(args.save_base, config_name)
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  Window: {args.window}, Duration: {duration} steps, "
              f"Schedule: {args.schedule}")
        print(f"{'='*60}")

        ds0_path = os.path.join(save_dir, "dataset_0")
        existing = 0
        if os.path.isdir(ds0_path):
            for d in os.listdir(ds0_path):
                cls_dir = os.path.join(ds0_path, d)
                if os.path.isdir(cls_dir):
                    existing += len([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        if existing >= args.ipc * len(sel_classes):
            print(f"  Already generated ({existing} images in dataset_0), skipping generation")
            gen_time = 0
        elif args.eval_only:
            print(f"  [eval-only] No images found for {config_name}, skipping config")
            continue
        else:
            gen_time = generate_config(
                model, vae, diffusion, latent_size, device,
                analyzer, class_labels, sel_classes, args.ipc,
                args.window, duration, args.num_datasets, save_dir,
                schedule=args.schedule,
                no_cags=args.no_cags,
                fixed_scale=args.fixed_scale,
                fixed_stop_t=args.fixed_stop_t,
                cags_min_scale=args.cags_min_scale,
                cags_max_scale=args.cags_max_scale,
                sigmoid_slope=args.sigmoid_slope,
                sigmoid_center=args.sigmoid_center,
                use_iast=args.use_iast,
            )

        print(f"\nEvaluating {config_name} over seeds {args.seeds}...")
        test_dir = os.path.join(args.imagenet_dir, "val")
        ds0_dir = os.path.join(save_dir, "dataset_0")
        class_names = sorted([d for d in os.listdir(ds0_dir)
                            if os.path.isdir(os.path.join(ds0_dir, d))
                            and d.startswith("n")])

        dataset_results = []
        for ds_idx in range(args.num_datasets):
            train_dir = os.path.join(save_dir, f"dataset_{ds_idx}")
            if not os.path.isdir(train_dir):
                print(f"  Dataset {ds_idx}: NOT FOUND, skipping")
                continue
            for seed in args.seeds:
                top1, top5 = evaluate_dataset(
                    train_dir, test_dir, class_names, args.nclass, device,
                    epochs=args.epochs, seed=seed, depth=args.depth,
                )
                print(f"  Dataset {ds_idx} seed {seed}: "
                      f"Top-1={top1:.2f}%, Top-5={top5:.2f}%")
                dataset_results.append({
                    "dataset": ds_idx, "seed": seed, "top1": top1, "top5": top5,
                })

        top1s = [r["top1"] for r in dataset_results]
        mean_top1 = float(np.mean(top1s)) if top1s else 0.0
        std_top1 = float(np.std(top1s)) if top1s else 0.0

        per_seed = {}
        for seed in args.seeds:
            vals = [r["top1"] for r in dataset_results if r["seed"] == seed]
            if vals:
                per_seed[str(seed)] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                }

        all_results[config_name] = {
            "window": args.window,
            "schedule": args.schedule,
            "duration": duration,
            "guidance_steps": duration,
            "gen_time_s": gen_time,
            "seeds": args.seeds,
            "runs": dataset_results,
            "per_seed": per_seed,
            "top1_mean": mean_top1,
            "top1_std": std_top1,
            "cags_min_scale": args.cags_min_scale,
            "cags_max_scale": args.cags_max_scale,
            "no_cags": args.no_cags,
            "fixed_scale": args.fixed_scale,
        }
        print(f"\n  {config_name}: Top-1 = {mean_top1:.2f} ± {std_top1:.2f}% "
              f"({len(top1s)} runs)")

    results_path = os.path.join(
        args.save_base, f"sweep_{args.window}_{sched_tag}_results.json")
    merged = {}
    if os.path.isfile(results_path):
        try:
            with open(results_path) as f:
                merged = json.load(f)
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged.update(all_results)
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved to {results_path} "
          f"({len(all_results)} new, {len(merged)} total configs)")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.window}, {args.schedule})")
    print(f"{'='*60}")
    for name, res in all_results.items():
        line = f"  {name}: {res['top1_mean']:.2f} ± {res['top1_std']:.2f}%"
        if len(res.get("per_seed", {})) > 1:
            parts = [f"s{s}={v['mean']:.2f}" for s, v in res["per_seed"].items()]
            line += "  [" + ", ".join(parts) + "]"
        print(line)

    del model, vae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
