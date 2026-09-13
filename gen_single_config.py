#!/usr/bin/env python3
"""General-purpose dataset generation with configurable CAGS weights.

Generates a synthetic dataset using the DiT generator with specified CAGS
complexity weights. Reuses cached VAE features when available to avoid
re-extracting features.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 gen_single_config.py \
      --spec imagenet100 --nclass 100 \
      --imagenet-dir ./data/imagenet100/ \
      --save-base ./results/sweep_in100 \
      --tag entropy_only \
      --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 \
      --cags-min-scale 0.0 --cags-max-scale 0.06 \
      --sigmoid-slope 3.0 --sigmoid-center 0.6
"""
import os
import sys
import time
import copy
import argparse
import pickle

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duration_sweep import (
    load_model_and_vae, setup_classes, compute_clusters, generate_config,
)
from ags import ClassComplexityAnalyzer


def recompute_complexity(analyzer, alpha, beta, gamma, delta):
    """Recompute complexity scores with new weights using cached features."""
    analyzer = copy.deepcopy(analyzer)
    analyzer.alpha = alpha
    analyzer.beta = beta
    analyzer.gamma = gamma
    analyzer.delta = delta
    analyzer._compute_separability_and_finalize()
    return analyzer


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with single CAGS config")
    parser.add_argument("--spec", type=str, default="imagenet100",
                        choices=["nette", "woof", "imagenet100", "imagenet1k", "imagenet200", "animals", "objects"])
    parser.add_argument("--nclass", type=int, default=100)
    parser.add_argument("--imagenet-dir", type=str, required=True)
    parser.add_argument("--save-base", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True,
                        help="Output directory name suffix")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--window", type=str, default="high_noise",
                        choices=["high_noise", "low_noise"])
    parser.add_argument("--guidance-steps", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--cags-min-scale", type=float, default=0.0)
    parser.add_argument("--cags-max-scale", type=float, default=0.06)
    parser.add_argument("--sigmoid-slope", type=float, default=3.0)
    parser.add_argument("--sigmoid-center", type=float, default=0.6)
    parser.add_argument("--no-cags", action="store_true", default=False)
    parser.add_argument("--fixed-scale", type=float, default=0.0)
    parser.add_argument("--use-iast", action="store_true", default=False)
    parser.add_argument("--complexity-k", type=int, default=None)
    parser.add_argument("--regen-cache", action="store_true", default=False)
    parser.add_argument("--recompute-only", action="store_true", default=False,
                        help="Recompute complexity from cached features (skip feature extraction)")
    parser.add_argument("--cache-only", action="store_true", default=False,
                        help="Only generate cluster cache, skip dataset generation")
    args = parser.parse_args()

    device = "cuda:0"
    class_labels, sel_classes = setup_classes(spec=args.spec, nclass=args.nclass)
    print(f"Spec: {args.spec}, Nclass: {args.nclass}, Tag: {args.tag}")
    print(f"Classes: {sel_classes[:5]}... ({len(sel_classes)} total)")

    os.makedirs(args.save_base, exist_ok=True)
    cluster_cache = os.path.join(args.save_base, "cluster_cache.pkl")

    need_regen = args.regen_cache or not os.path.isfile(cluster_cache)

    if args.cache_only:
        if not need_regen:
            print(f"Cache already exists at {cluster_cache}, skipping.")
            return
        from diffusers.models import AutoencoderKL
        print("\nLoading VAE only (cache-only mode)...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        print("VAE loaded.")
        model = None
        diffusion = None
        latent_size = 32
    else:
        print("\nLoading DiT model and VAE...")
        model, vae, diffusion, latent_size = load_model_and_vae(device)
        print("Model loaded.")

    if need_regen and not args.recompute_only:
        print("\nComputing CAGS clusters (feature extraction + KMeans)...")
        analyzer = compute_clusters(
            args.spec, args.imagenet_dir, vae, device,
            nclass=args.nclass,
            sigmoid_slope=args.sigmoid_slope,
            sigmoid_center=args.sigmoid_center,
            complexity_k=args.complexity_k,
            alpha=args.alpha, beta=args.beta,
            gamma=args.gamma, delta=args.delta,
        )
        with open(cluster_cache, "wb") as f:
            pickle.dump(analyzer, f)
        print(f"Cluster cache saved to {cluster_cache}")
    else:
        print(f"\nLoading cached analyzer from {cluster_cache}...")
        with open(cluster_cache, "rb") as f:
            base_analyzer = pickle.load(f)

        if args.no_cags:
            analyzer = base_analyzer
        else:
            print(f"Recomputing complexity with alpha={args.alpha}, beta={args.beta}, "
                  f"gamma={args.gamma}, delta={args.delta}")
            analyzer = recompute_complexity(
                base_analyzer, args.alpha, args.beta, args.gamma, args.delta,
            )

    for c in sorted(analyzer.complexity_scores.keys())[:5]:
        print(f"  Class {c}: complexity={analyzer.complexity_scores[c]:.4f}, "
              f"modes={analyzer.mode_counts[c]}")

    if args.cache_only:
        print("\nCache-only mode: skipping dataset generation.")
        del vae
        torch.cuda.empty_cache()
        return

    save_dir = os.path.join(args.save_base, args.tag)
    print(f"\nGenerating dataset -> {save_dir}")

    generate_config(
        model=model, vae=vae, diffusion=diffusion, latent_size=latent_size,
        device=device,
        analyzer=analyzer, class_labels=class_labels, sel_classes=sel_classes,
        ipc=args.ipc, window=args.window, guidance_steps=args.guidance_steps,
        num_datasets=1, save_dir=save_dir,
        schedule="constant",
        no_cags=args.no_cags,
        fixed_scale=args.fixed_scale,
        cags_min_scale=args.cags_min_scale,
        cags_max_scale=args.cags_max_scale,
        sigmoid_slope=args.sigmoid_slope,
        sigmoid_center=args.sigmoid_center,
        use_iast=args.use_iast,
    )

    print(f"\nDone! Dataset saved to {save_dir}/dataset_0/")
    del model, vae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
