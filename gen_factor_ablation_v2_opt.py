#!/usr/bin/env python3
"""Optimized factor ablation v2 — reuses cached features, only recomputes complexity scores.

This script loads the existing cluster_cache.pkl (which already has features,
mode counts, intra-variances, separabilities) and only recomputes complexity
scores with different alpha/beta/gamma/delta weights. This saves ~24 min
(feature extraction + KMeans) per config.

Usage: CUDA_VISIBLE_DEVICES=X python3 gen_factor_ablation_v2_optimized.py --gpu-group N
"""
import os, sys, time, json, argparse, pickle, copy
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duration_sweep import load_model_and_vae, setup_classes, generate_config
from ags import ClassComplexityAnalyzer

FACTOR_CONFIGS = [
    ("v2_full_cags",                0.3, 0.3, 0.2, 0.2, "Full CAGS (main weights)"),
    ("v2_intra_var_only",           0.0, 0.0, 1.0, 0.0, "Only intra-class variance"),
    ("v2_separability_only",        0.0, 0.0, 0.0, 1.0, "Only inter-class separability"),
    ("v2_mode_count_only",          1.0, 0.0, 0.0, 0.0, "Only mode count (auto-K)"),
    ("v2_entropy_only",             0.0, 1.0, 0.0, 0.0, "Only entropy (auto-K)"),
    ("v2_equal_weights",            0.25, 0.25, 0.25, 0.25, "Equal weights"),
    ("v2_no_separability",          0.25, 0.25, 0.5, 0.0, "No separability"),
    ("v2_separability_dominated",   0.10, 0.10, 0.20, 0.60, "Separability-dominated"),
]

def recompute_complexity(analyzer, alpha, beta, gamma, delta):
    """Recompute complexity scores with new weights using cached data."""
    analyzer = copy.deepcopy(analyzer)
    analyzer.alpha = alpha
    analyzer.beta = beta
    analyzer.gamma = gamma
    analyzer.delta = delta
    analyzer._compute_separability_and_finalize()
    return analyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-group", type=int, default=0)
    args = parser.parse_args()

    spec = "imagenet100"
    nclass = 100
    save_base = "./results/sweep_in100"
    ipc = 10
    cags_min = 0.0
    cags_max = 0.06

    if args.gpu_group == 0:
        configs = FACTOR_CONFIGS[:4]
    else:
        configs = FACTOR_CONFIGS[4:]

    device = f"cuda:{0}"
    model, vae, diffusion, latent_size = load_model_and_vae(device)
    class_labels, sel_classes = setup_classes(spec, nclass)

    # Load cached analyzer (has features, mode counts, etc.)
    cache_path = os.path.join(save_base, "cluster_cache.pkl")
    print(f"Loading cached analyzer from {cache_path}...")
    with open(cache_path, "rb") as f:
        base_analyzer = pickle.load(f)
    print(f"  Loaded. Mode counts: {sorted(set(base_analyzer.mode_counts.values()))}")
    print(f"  alpha={base_analyzer.alpha}, beta={base_analyzer.beta}, "
          f"gamma={base_analyzer.gamma}, delta={base_analyzer.delta}")

    for tag, alpha, beta, gamma, delta, desc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {tag} (alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})")
        print(f"  {desc}")
        print(f"{'='*60}")

        # Recompute complexity with new weights (seconds, not minutes)
        analyzer = recompute_complexity(base_analyzer, alpha, beta, gamma, delta)

        cs = analyzer.complexity_scores
        print(f"  Complexity: min={min(cs.values()):.4f}, max={max(cs.values()):.4f}")
        mc = analyzer.mode_counts
        for k in sorted(set(mc.values())):
            count = sum(1 for v in mc.values() if v == k)
            print(f"  K={k}: {count} classes")

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

    print("\n\nAll optimized factor ablation v2 generation complete!")

if __name__ == "__main__":
    main()
