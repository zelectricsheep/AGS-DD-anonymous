#!/usr/bin/env python3
"""Generate factor ablation datasets v2 — matches main CAGS hyperparameters.

Key fixes from v1:
  - complexity_k=None (auto-K via silhouette, NOT fixed=10)
  - sigmoid_slope=3.0, sigmoid_center=0.6 (matching main CAGS)
  - Includes Full CAGS config (alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)

All on ImageNet-100 (100-class), IPC=10, guidance range [0.0, 0.06].
Saves to ./results/sweep_in100/factor_ablation_v2_*/dataset_0
"""
import os, sys, time, json, argparse, pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duration_sweep import load_model_and_vae, setup_classes, compute_clusters, generate_config

# Factor ablation configs: (tag, alpha, beta, gamma, delta, description)
# alpha=mode_count, beta=entropy, gamma=intra_var, delta=separability
# ALL use complexity_k=None (auto-K) and sigmoid 3.0/0.6 — matching main CAGS
FACTOR_CONFIGS = [
    # Full CAGS reference (same weights as main experiment)
    ("v2_full_cags",                0.3, 0.3, 0.2, 0.2, "Full CAGS (main weights)"),
    # Single-factor: set one weight to 1.0, others to 0.0
    ("v2_intra_var_only",           0.0, 0.0, 1.0, 0.0, "Only intra-class variance"),
    ("v2_separability_only",        0.0, 0.0, 0.0, 1.0, "Only inter-class separability"),
    ("v2_mode_count_only",          1.0, 0.0, 0.0, 0.0, "Only mode count (auto-K)"),
    ("v2_entropy_only",             0.0, 1.0, 0.0, 0.0, "Only entropy (auto-K)"),
    # Weight sensitivity: equal weights
    ("v2_equal_weights",            0.25, 0.25, 0.25, 0.25, "Equal weights"),
    # No separability (remove delta term)
    ("v2_no_separability",          0.25, 0.25, 0.5, 0.0, "No separability"),
    # Separability-dominated (high delta, matching our finding)
    ("v2_separability_dominated",   0.10, 0.10, 0.20, 0.60, "Separability-dominated"),
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

    # 8 configs total, split across 2 GPU groups
    if args.gpu_group == 0:
        configs = FACTOR_CONFIGS[:4]
    else:
        configs = FACTOR_CONFIGS[4:]

    device = f"cuda:{0}"
    model, vae, diffusion, latent_size = load_model_and_vae(device)
    class_labels, sel_classes = setup_classes(spec, nclass)

    for tag, alpha, beta, gamma, delta, desc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {tag} (alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})")
        print(f"  {desc}")
        print(f"  complexity_k=None (auto-K), sigmoid 3.0/0.6")
        print(f"{'='*60}")

        # Use auto-K (complexity_k=None) and sigmoid 3.0/0.6 — matching main CAGS
        analyzer = compute_clusters(
            spec, imagenet_dir, vae, device, nclass=nclass,
            sigmoid_slope=3.0, sigmoid_center=0.6, complexity_k=None,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        )

        # Print mode count distribution for this config
        mc = analyzer.mode_counts
        for k in sorted(set(mc.values())):
            count = sum(1 for v in mc.values() if v == k)
            print(f"  K={k}: {count} classes")
        cs = analyzer.complexity_scores
        print(f"  Complexity: min={min(cs.values()):.4f}, max={max(cs.values()):.4f}")

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
            sigmoid_slope=3.0, sigmoid_center=0.6,
            use_iast=False,
        )

        # Save complexity stats
        stats_path = os.path.join(save_base, f"factor_ablation_{tag}_stats.json")
        stats = {
            "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                       "sigmoid_slope": 3.0, "sigmoid_center": 0.6, "complexity_k": "auto"},
            "complexity_scores": analyzer.complexity_scores,
            "mode_counts": analyzer.mode_counts,
            "intra_variances": analyzer.intra_variances,
            "separabilities": analyzer.separabilities,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats to {stats_path}")

    print("\n\nAll factor ablation v2 generation complete!")

if __name__ == "__main__":
    main()
