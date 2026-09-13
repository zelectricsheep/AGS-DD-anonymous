#!/usr/bin/env python3
"""
FID (Fréchet Inception Distance) computation for AGS-DD.

Computes FID between generated distilled images and real validation images.
Uses InceptionV3 (with proper FID pooling layer) for feature extraction.

Usage:
    python3 compute_fid.py \
        --gen-dir ./results/sweep_in10/high_noise_fixed_l0.1_d25/dataset_0 \
        --real-dir ./data/imagenet100/val \
        --class-file ./misc/class100.txt --nclass 10
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg


def get_inception_model(device="cuda"):
    """Load InceptionV3 with FID-specific pooling layer (2048-dim)."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = nn.Identity()
    model.eval()
    model = model.to(device)
    return model


def extract_features(model, image_dir, class_names, device, img_size=299, batch_size=64, max_images=None):
    """Extract InceptionV3 features from images in a directory."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = []
    for cls in class_names:
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(
            glob.glob(os.path.join(cls_dir, "*.png")) +
            glob.glob(os.path.join(cls_dir, "*.JPEG")) +
            glob.glob(os.path.join(cls_dir, "*.jpg"))
        )
        for f in files:
            img = Image.open(f).convert("RGB")
            img = transform(img)
            images.append(img)

    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    if max_images and len(images) > max_images:
        indices = np.random.choice(len(images), max_images, replace=False)
        images = [images[i] for i in indices]

    print(f"  Extracting features from {len(images)} images...")
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size]).to(device)
            feat = model(batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_statistics(features):
    """Compute mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Fréchet Inception Distance."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)


def main():
    parser = argparse.ArgumentParser(description="Compute FID between generated and real images")
    parser.add_argument("--gen-dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument("--real-dir", type=str, required=True, help="Directory of real validation images")
    parser.add_argument("--class-file", type=str, default="./misc/class100.txt")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=None, help="Max images per directory (for speed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.class_file) as f:
        class_names = [l.strip() for l in f.readlines()][:args.nclass]
    print(f"Classes: {class_names}")

    print("\nLoading InceptionV3 model...")
    model = get_inception_model(device)

    print(f"\nExtracting features from generated images ({args.gen_dir})...")
    gen_features = extract_features(model, args.gen_dir, class_names, device, max_images=args.max_images)
    print(f"  Generated features: {gen_features.shape}")

    print(f"\nExtracting features from real images ({args.real_dir})...")
    real_features = extract_features(model, args.real_dir, class_names, device, max_images=args.max_images)
    print(f"  Real features: {real_features.shape}")

    print("\nComputing FID...")
    mu_gen, sigma_gen = compute_statistics(gen_features)
    mu_real, sigma_real = compute_statistics(real_features)
    fid = compute_fid(mu_gen, sigma_gen, mu_real, sigma_real)

    print(f"\n{'='*50}")
    print(f"FID Score: {fid:.4f}")
    print(f"{'='*50}")
    print(f"Generated: {gen_features.shape[0]} images from {args.gen_dir}")
    print(f"Real: {real_features.shape[0]} images from {args.real_dir}")

    # Save result
    result = {
        "fid": fid,
        "gen_dir": args.gen_dir,
        "real_dir": args.real_dir,
        "n_gen": gen_features.shape[0],
        "n_real": real_features.shape[0],
    }
    import json
    result_path = os.path.join(os.path.dirname(args.gen_dir), "fid_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
