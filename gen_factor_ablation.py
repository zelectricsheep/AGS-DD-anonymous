#!/usr/bin/env python3
"""Generate factor ablation datasets for CAGS complexity factor analysis.

Tests:
  1. Single-factor variants: only intra-var, only separability, only mode-count, only entropy
  2. Weight sensitivity: equal weights vs current (0.15,0.15,0.35,0.35)
  3. No separability (remove the negative term)

All on ImageNet-100 (100-class), IPC=10, guidance range [0.0, 0.06].
Saves to ./results/sweep_in100/factor_ablation_*/dataset_0
"""
import os, sys, time, json, argparse, pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duration_sweep import load_model_and_vae, setup_classes, compute_clusters, generate_config

# Factor ablation configs: (tag, alpha, beta, gamma, delta, description)
# alpha=mode_count, beta=entropy, gamma=intra_var, delta=separability
FACTOR_CONFIGS = [
    # Single-factor: set one weight to 1.0, others to 0.0
    ("factor_intra_var_only",       0.0, 0.0, 1.0, 0.0, "Only intra-class variance"),
    ("factor_separability_only",    0.0, 0.0, 0.0, 1.0, "Only inter-class separability"),
    ("factor_mode_count_only",      1.0, 0.0, 0.0, 0.0, "Only mode count"),
    ("factor_entropy_only",         0.0, 1.0, 0.0, 0.0, "Only entropy"),
    # Weight sensitivity: equal weights
    ("factor_equal_weights",        0.25, 0.25, 0.25, 0.25, "Equal weights"),
    # No separability (remove delta term)
    ("factor_no_separability",      0.25, 0.25, 0.5, 0.0, "No separability, redistribute to variance"),
    # Variance-dominated (closer to current but shifted)
    ("factor_variance_dominated",   0.10, 0.10, 0.40, 0.40, "Variance-dominated (reduced mode/entropy)"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-group", type=int, default=0)
    parser.add_argument("--compute-only", action="store_true")
    args = parser.parse_args()

    spec = "imagenet100"
    nclass = 100
    imagenet_dir = "./data/imagenet100/"
    save_base = "./results/sweep_in100"
    ipc = 10
    cags_min = 0.0
    cags_max = 0.06

    # Assign configs to GPUs (4 configs per GPU, 7 total)
    # GPU0: first 4 configs, GPU1: last 3 configs
    if args.gpu_group == 0:
        configs = FACTOR_CONFIGS[:4]
    else:
        configs = FACTOR_CONFIGS[4:]

    device = f"cuda:{0}"  # CUDA_VISIBLE_DEVICES handles which physical GPU
    model, vae, diffusion, latent_size = load_model_and_vae(device)
    class_labels, sel_classes = setup_classes(spec, nclass)

    # We need to recompute clusters for each weight config since alpha/beta/gamma/delta
    # change the complexity scores and thus the per-class guidance
    for tag, alpha, beta, gamma, delta, desc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {tag} (alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})")
        print(f"  {desc}")
        print(f"{'='*60}")

        # Recompute clusters with these weights
        analyzer = compute_clusters(
            spec, imagenet_dir, vae, device, nclass=nclass,
            sigmoid_slope=5.0, sigmoid_center=0.5, complexity_k=10,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        )

        save_dir = os.path.join(save_base, f"factor_ablation_{tag}")
        print(f"  Saving to {save_dir}")

        generate_config(
            model=model, vae=vae, diffusion=diffusion, latent_size=latent_size,
            device=device,
            analyzer=analyzer, class_labels=class_labels, sel_classes=sel_classes,
            ipc=ipc, window="high_noise", guidance_steps=25,
            num_datasets=1, save_dir=save_dir,
            schedule="constant",
            no_cags=False, fixed_scale=0.1,
            cags_min_scale=cags_min, cags_max_scale=cags_max,
            sigmoid_slope=5.0, sigmoid_center=0.5,
            use_iast=False,
        )

        # Save complexity stats
        stats_path = os.path.join(save_base, f"factor_ablation_{tag}_stats.json")
        stats = {
            "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                       "sigmoid_slope": 5.0, "sigmoid_center": 0.5, "complexity_k": 10},
            "complexity_scores": analyzer.complexity_scores,
            "mode_counts": analyzer.mode_counts,
            "intra_variances": analyzer.intra_variances,
            "separabilities": analyzer.separabilities,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats to {stats_path}")

    print("\n\nAll factor ablation generation complete!")

if __name__ == "__main__":
    main()
