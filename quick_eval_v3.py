import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import glob
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_models.resnet as RN
import train_models.convnet as CN


class Lighting:
    """AlexNet-style PCA lighting noise (batch version)."""
    def __init__(self, alphastd=0.1,
                 eigval=[0.2175, 0.0188, 0.0045],
                 eigvec=[[-0.5675, 0.7192, 0.4009],
                         [-0.5808, -0.0045, -0.8140],
                         [-0.5836, -0.6948, 0.4203]]):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval)
        self.eigvec = torch.tensor(eigvec)

    def __call__(self, imgs):
        if self.alphastd == 0:
            return imgs
        eigval = self.eigval.to(imgs.device)
        eigvec = self.eigvec.to(imgs.device)
        alpha = imgs.new_empty(imgs.size(0), 3).normal_(0, self.alphastd)
        rgb = torch.einsum('bi,ij->bj', alpha, eigvec) * eigval.unsqueeze(0)
        return imgs + rgb.unsqueeze(-1).unsqueeze(-1)


def create_model(arch, num_classes, depth=6, norm_type="instance", img_size=224):
    if arch == "convnet":
        return CN.ConvNet(
            num_classes,
            net_norm=norm_type,
            net_depth=depth,
            net_width=128,
            channel=3,
            im_size=(img_size, img_size),
        )
    elif arch == "resnet":
        return RN.ResNet(
            "imagenet", depth, num_classes,
            norm_type=norm_type, size=img_size, nch=3,
        )
    elif arch == "resnet_ap":
        import train_models.resnet_ap as RNAP
        return RNAP.ResNetAP(
            "imagenet", depth, num_classes,
            width=1.0, norm_type=norm_type, size=img_size, nch=3,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def rand_bbox(size, lam):
    W, H = size[2], size[3]
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


def load_images_to_tensor(data_dir, class_names, img_size=224):
    """Load training images as unnormalized tensors.
    
    Uses Resize + CenterCrop (preserving aspect ratio) to match MGD3 protocol.
    Normalization is deferred to the training loop (after augmentation).
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    images = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.png")) +
                       glob.glob(os.path.join(cls_dir, "*.JPEG")) +
                       glob.glob(os.path.join(cls_dir, "*.jpg")))
        for f in files:
            img = Image.open(f).convert("RGB")
            img = transform(img)
            images.append(img)
            labels.append(idx)
    if not images:
        raise RuntimeError(f"No images found in {data_dir}")
    return torch.stack(images), torch.tensor(labels)


def load_val_data(val_dir, class_names, img_size=224, batch_size=128):
    """Load validation images as normalized tensors.
    
    Uses Resize + CenterCrop (preserving aspect ratio) to match MGD3 protocol.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_files = []
    all_labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(glob.glob(os.path.join(cls_dir, "*.JPEG")) + glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png")))
        for f in files:
            all_files.append(f)
            all_labels.append(idx)

    def load_one(f):
        img = Image.open(f).convert("RGB")
        return transform(img)

    with ThreadPoolExecutor(max_workers=16) as executor:
        images = list(executor.map(load_one, all_files))

    return torch.stack(images), torch.tensor(all_labels)


def evaluate_fast(train_dir, val_dir, class_names, num_classes, device,
                  img_size=224, epochs=1000, batch_size=128, seed=0, depth=6,
                  arch="convnet", norm_type="instance", lr=0.1, weight_decay=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading training images from {train_dir}...")
    train_images, train_labels = load_images_to_tensor(train_dir, class_names, img_size)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    print(f"  Loaded {len(train_images)} training images")

    t_val_start = time.time()
    print(f"  Loading validation images from {val_dir}...")
    val_images, val_labels = load_val_data(val_dir, class_names, img_size)
    print(f"  Loaded {len(val_images)} validation images ({time.time()-t_val_start:.1f}s)")

    model = create_model(arch, num_classes, depth=depth, norm_type=norm_type, img_size=img_size)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_images)
    rrc = transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = Lighting(alphastd=0.1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t0 = time.time()
    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    eval_interval = max(epochs // 100, 1)

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            imgs = train_images[idx].clone()
            tgt = train_labels[idx]

            # Augmentation pipeline (matching MGD3):
            # RRC -> HFlip -> ColorJitter -> Lighting -> Normalize -> CutMix
            imgs = rrc(imgs)
            imgs = hflip(imgs)
            imgs = color_jitter(imgs)
            imgs = lighting(imgs)
            imgs = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # CutMix
            lam = np.random.beta(1.0, 1.0)
            rand_idx = torch.randperm(imgs.size(0), device=device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_idx, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))

            outputs = model(imgs)
            loss = criterion(outputs, tgt) * ratio + criterion(outputs, tgt[rand_idx]) * (1. - ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
            model.eval()
            correct = 0
            top5_correct = 0
            total = 0
            with torch.no_grad():
                for start in range(0, len(val_images), batch_size):
                    imgs = val_images[start:start + batch_size].to(device)
                    tgt = val_labels[start:start + batch_size].to(device)
                    outputs = model(imgs)
                    _, pred = outputs.max(1)
                    total += tgt.size(0)
                    correct += pred.eq(tgt).sum().item()
                    if outputs.size(1) >= 5:
                        _, pred5 = outputs.topk(5, 1, True, True)
                        top5_correct += pred5.eq(tgt.view(-1, 1)).any(dim=1).sum().item()
            cur_top1 = 100.0 * correct / total
            cur_top5 = 100.0 * top5_correct / total if total > 0 else 0
            if cur_top1 > best_top1:
                best_top1 = cur_top1
                best_top5 = cur_top5
                best_epoch = epoch + 1
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s) best={best_top1:.2f}%@ep{best_epoch}")

    final_top1 = best_top1
    final_top5 = best_top5
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s, best Top-1={best_top1:.2f}% @ epoch {best_epoch}")
    del model
    torch.cuda.empty_cache()
    return final_top1, final_top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--class-file", type=str, default="./misc/class100.txt")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--arch", type=str, default="convnet",
                        choices=["convnet", "resnet", "resnet_ap"],
                        help="Network architecture")
    parser.add_argument("--norm-type", type=str, default="instance",
                        choices=["instance", "batch", "none"],
                        help="Normalization type")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.class_file) as f:
        all_classes = [l.strip() for l in f.readlines()]
    class_names = all_classes[:args.nclass]
    print(f"Arch: {args.arch}, Norm: {args.norm_type}")
    print(f"Classes ({args.nclass}): {class_names}")

    results = []
    for seed in args.seeds:
        print(f"\nSeed {seed}:")
        top1, top5 = evaluate_fast(
            args.train_dir, args.val_dir, class_names, args.nclass, device,
            img_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, depth=args.depth, arch=args.arch, norm_type=args.norm_type,
            lr=args.lr, weight_decay=args.weight_decay,
        )
        print(f"  Seed {seed}: Best Top-1={top1:.2f}%, Best Top-5={top5:.2f}%")
        results.append({"seed": seed, "top1": top1, "top5": top5})

    top1s = [r["top1"] for r in results]
    top5s = [r["top5"] for r in results]
    print(f"\n{'='*60}")
    print(f"Arch: {args.arch}, Depth: {args.depth}")
    print(f"Mean Best Top-1: {np.mean(top1s):.2f} ± {np.std(top1s):.2f}%")
    print(f"Mean Best Top-5: {np.mean(top5s):.2f} ± {np.std(top5s):.2f}%")


if __name__ == "__main__":
    main()
