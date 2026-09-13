"""
AGS-DD: Adaptive Guidance Scheduling for Training-Free Diffusion Dataset Distillation

Main entry point for generating distilled datasets with AGS guidance.

Usage:
    python sample_ags.py --spec nette --num-samples 10 --save-dir ./generated/ags_dd
    python sample_ags.py --spec woof --num-samples 50 --save-dir ./generated/ags_dd
    python sample_ags.py --spec imagenet100 --num-samples 50 --save-dir ./generated/ags_dd
    python sample_ags.py --spec imagenet1k --num-samples 50 --save-dir ./generated/ags_dd --phase 0

Ablation:
    python sample_ags.py --spec nette --num-samples 10 --no-cags  # Disable CAGS
    python sample_ags.py --spec nette --num-samples 10 --no-iast  # Disable IAST
    python sample_ags.py --spec nette --num-samples 10 --no-tags  # Disable TAGS
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from tsne_plots import define_model, get_loader, get_features_per_class
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from torchvision import io

from ags import (
    ClassComplexityAnalyzer,
    AdaptiveStopTiming,
    TimestepAdaptiveSchedule,
    AGSSampler,
)


def get_args():
    parser = argparse.ArgumentParser(description="AGS-DD Sampling")

    # Model
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to DiT checkpoint")

    # Dataset
    parser.add_argument("--spec", type=str, default="nette",
                        choices=["nette", "woof", "imagenet100", "imagenet1k"],
                        help="Dataset subset for generation")
    parser.add_argument("--save-dir", type=str, default="./generated/ags_dd",
                        help="Directory to save generated images")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Desired IPC (images per class)")
    parser.add_argument("--total-shift", type=int, default=0,
                        help="Index offset for file naming")
    parser.add_argument("--nclass", type=int, default=10,
                        help="Number of classes for generation")
    parser.add_argument("--phase", type=int, default=0,
                        help="Phase for generating large datasets (multi-processing)")
    parser.add_argument("--num-datasets", type=int, default=5,
                        help="Number of generated datasets (for multiple seeds)")
    parser.add_argument("--use-same-noise", action="store_true", default=False,
                        help="Use same noise across classes")

    # Data path
    parser.add_argument("--imagenet-dir", type=str, default="/ssd_data/imagenet/",
                        help="Path to ImageNet directory")
    parser.add_argument("--vae-ckpt", type=str, default=None, help="VAE checkpoint path")

    # Clustering
    parser.add_argument("--cluster-file", type=str, default=None,
                        help="Pre-computed cluster file")
    parser.add_argument("--use-pca", action="store_true", default=False,
                        help="Use PCA before clustering")
    parser.add_argument("--closest-point", action="store_true", default=True,
                        help="Use closest real point to cluster center")
    parser.add_argument("--real", action="store_true", default=False,
                        help="Use real data (no generation)")

    # AGS-DD specific arguments
    # CAGS
    parser.add_argument("--no-cags", action="store_true", default=False,
                        help="Disable CAGS (use fixed guidance scale)")
    parser.add_argument("--cags-alpha", type=float, default=0.5,
                        help="Weight for mode count in CAGS")
    parser.add_argument("--cags-beta", type=float, default=0.5,
                        help="Weight for entropy in CAGS")
    parser.add_argument("--cags-kmin", type=int, default=2,
                        help="Min clusters for CAGS")
    parser.add_argument("--cags-kmax", type=int, default=20,
                        help="Max clusters for CAGS")
    parser.add_argument("--guidance-scale-min", type=float, default=0.05,
                        help="Min guidance scale for CAGS output")
    parser.add_argument("--guidance-scale-max", type=float, default=0.5,
                        help="Max guidance scale for CAGS output")

    # IAST
    parser.add_argument("--no-iast", action="store_true", default=False,
                        help="Disable IAST (use fixed stop_t)")
    parser.add_argument("--iast-lambda", type=float, default=0.316,
                        help="Lambda for IAST stop timing")
    parser.add_argument("--iast-min-stop", type=int, default=5,
                        help="Min stop timestep")
    parser.add_argument("--iast-max-stop-ratio", type=float, default=0.9,
                        help="Max stop ratio of total timesteps")
    parser.add_argument("--iast-use-complexity", action="store_true", default=True,
                        help="Incorporate complexity in IAST")
    parser.add_argument("--default-stop-t", type=int, default=25,
                        help="Default stop_t when IAST is disabled")

    # TAGS
    parser.add_argument("--no-tags", action="store_true", default=False,
                        help="Disable TAGS (use constant guidance weight)")
    parser.add_argument("--guidance-window", type=str, default="low_noise",
                        choices=["low_noise", "high_noise"],
                        help="Guidance window: low_noise (default, best) or high_noise")
    parser.add_argument("--tags-schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "exponential", "step",
                                 "warmup_cosine", "adaptive"],
                        help="TAGS schedule type")
    parser.add_argument("--tags-warmup", type=int, default=0,
                        help="Warmup steps for warmup_cosine schedule")
    parser.add_argument("--tags-decay-rate", type=float, default=3.0,
                        help="Decay rate for exponential schedule")
    parser.add_argument("--default-guidance-scale", type=float, default=0.1,
                        help="Default guidance scale when CAGS is disabled")

    # Baseline comparison
    parser.add_argument("--baseline", action="store_true", default=False,
                        help="Run baseline MGD3 (no AGS, fixed parameters)")
    parser.add_argument("--baseline-stop-t", type=int, default=0,
                        help="Baseline stop_t parameter")
    parser.add_argument("--baseline-mode-scale", type=float, default=0.1,
                        help="Baseline mode_guidance_scale parameter")

    return parser.parse_args()


def setup_classes(args):
    """Load and parse class labels for the specified dataset."""
    with open("./misc/class_indices.txt", "r") as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]

    file_map = {
        "woof": "./misc/class_woof.txt",
        "nette": "./misc/class_nette.txt",
        "imagenet100": "./misc/class100.txt",
        "imagenet1k": "./misc/class_indices.txt",
    }
    with open(file_map[args.spec], "r") as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]

    class_labels = [all_classes.index(sel_class) for sel_class in sel_classes]
    return class_labels, sel_classes, all_classes


def load_model(args, device):
    """Load pre-trained DiT model and VAE."""
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if args.vae_ckpt is not None:
        ckpt = torch.load(args.vae_ckpt)
        vae.load_state_dict(ckpt["model"])

    diffusion = create_diffusion(str(args.num_sampling_steps))
    return model, vae, diffusion, latent_size


def compute_clusters(args, vae, class_labels, sel_classes):
    """Compute mode clusters using VAE features (CAGS integration)."""
    args_eval = argparse.Namespace(
        dataset="imagenet",
        net_type="resnet_ap",
        depth=10,
        width=1.0,
        norm_type="instance",
        nch=3,
        data_path=os.path.join(args.imagenet_dir, "train"),
        size=224,
        aug_type="color_crop_cutout",
        augment=True,
        dseed=0,
        global_batch_size=1,
        use_vae=True,
        finetune_ipc=-1,
        nclass=args.nclass,
        spec=args.spec,
        phase=args.phase,
        image_size=args.image_size,
    )

    original_loader = get_loader(args_eval, return_path=True)
    original_features_per_class, original_paths = get_features_per_class(
        args_eval, original_loader, vae
    )

    # CAGS: Analyze class complexity
    complexity_analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(args.cags_kmin, args.cags_kmax),
        alpha=args.cags_alpha,
        beta=args.cags_beta,
        use_pca=args.use_pca,
    )

    complexity_analyzer.analyze_all_classes(
        original_features_per_class, original_paths
    )

    return complexity_analyzer, original_features_per_class, original_paths


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("AGS-DD: Adaptive Guidance Scheduling for Dataset Distillation")
    print("=" * 70)
    print(f"Dataset: {args.spec}, IPC: {args.num_samples}")
    print(f"CAGS: {'enabled' if not args.no_cags else 'disabled'}")
    print(f"IAST: {'enabled' if not args.no_iast else 'disabled'}")
    print(f"TAGS: {'enabled' if not args.no_tags else 'disabled'}")
    print(f"Schedule: {args.tags_schedule}")
    print("=" * 70)

    # Setup classes
    class_labels, sel_classes, all_classes = setup_classes(args)
    print(f"Classes: {sel_classes}")

    # Load model
    print("\nLoading pre-trained DiT model...")
    model, vae, diffusion, latent_size = load_model(args, device)
    print(f"Model loaded. Latent size: {latent_size}")

    # Compute or load clusters
    if args.cluster_file is not None:
        print(f"\nLoading pre-computed clusters from {args.cluster_file}")
        with open(args.cluster_file, "rb") as f:
            complexity_analyzer = pickle.load(f)
    else:
        print("\nComputing class complexity (CAGS)...")
        complexity_analyzer, features_per_class, paths_per_class = compute_clusters(
            args, vae, class_labels, sel_classes
        )

    # IAST: Compute adaptive stop timings
    adaptive_stop = AdaptiveStopTiming(
        t_max=args.num_sampling_steps,
        lam=args.iast_lambda,
        min_stop=args.iast_min_stop,
        max_stop_ratio=args.iast_max_stop_ratio,
        use_complexity=args.iast_use_complexity,
    )

    # TAGS: Create guidance schedule
    guidance_schedule = TimestepAdaptiveSchedule(
        w_max=args.guidance_scale_max,
        schedule_type=args.tags_schedule,
        warmup_steps=args.tags_warmup,
        decay_rate=args.tags_decay_rate,
    )

    # Create AGS sampler
    sampler = AGSSampler(
        model=model,
        vae=vae,
        diffusion=diffusion,
        complexity_analyzer=complexity_analyzer,
        adaptive_stop=adaptive_stop,
        guidance_schedule=guidance_schedule,
        device=device,
        num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale,
        latent_size=latent_size,
        use_cags=not args.no_cags,
        use_iast=not args.no_iast,
        use_tags=not args.no_tags,
        guidance_scale_range=(args.guidance_scale_min, args.guidance_scale_max),
        guidance_window=args.guidance_window,
    )

    # Override defaults for ablation
    sampler.default_guidance_scale = args.default_guidance_scale
    sampler.default_stop_t = args.default_stop_t

    # Get cluster centers for generation
    # CAGS uses optimal K for complexity scoring, but generation needs IPC centers
    clusters_centers = complexity_analyzer.compute_clusters_for_ipc(
        args.num_samples, use_pca=True, closest_point=True
    )

    # Generate dataset
    print(f"\nGenerating {args.num_datasets} datasets with {args.num_samples} IPC...")
    print(f"Save directory: {args.save_dir}")

    if args.baseline:
        # Run baseline MGD3 for comparison
        print("\n[BASELINE MODE] Running MGD3 with fixed parameters...")
        sampler.use_cags = False
        sampler.use_iast = False
        sampler.use_tags = False
        sampler.default_guidance_scale = args.baseline_mode_scale
        sampler.default_stop_t = args.baseline_stop_t

    start_time = time.time()
    sampler.generate_dataset(
        args=args,
        class_labels=class_labels,
        sel_classes=sel_classes,
        clusters_centers=clusters_centers,
        save_dir=args.save_dir,
        num_datasets=args.num_datasets,
        use_same_noise=args.use_same_noise,
        total_shift=args.total_shift,
    )
    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"Generation complete!")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Output: {args.save_dir}")
    print(f"{'=' * 70}")

    # Save configuration
    config = sampler.get_config_summary()
    config["dataset"] = args.spec
    config["ipc"] = args.num_samples
    config["total_time"] = total_time
    config["num_datasets"] = args.num_datasets
    config_path = os.path.join(args.save_dir, "ags_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    main()
