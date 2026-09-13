#!/usr/bin/env python3
"""Regenerate cluster cache with CAGS v2 and print complexity stats."""

import os, sys, pickle, numpy as np
sys.path.insert(0, '.')
os.chdir('.')

import torch
from diffusers.models import AutoencoderKL
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from tsne_plots import get_loader, get_features_per_class
from ags.class_complexity import ClassComplexityAnalyzer
import argparse as ap

def regen_cache(spec, nclass, save_base, sigmoid_slope=3.0, sigmoid_center=0.6,
                complexity_k=None, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    device = "cuda"
    imagenet_dir = "./data/imagenet100"

    # Load model
    latent_size = 32
    model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000).to(device)
    state_dict = find_model("DiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Feature extraction
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

    # Create analyzer with v2
    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(2, 20), alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        use_pca=True, sigmoid_slope=sigmoid_slope, sigmoid_center=sigmoid_center,
    )
    analyzer.analyze_all_classes(features_per_class, paths_per_class, fixed_k=complexity_k)

    # Print stats
    scores = analyzer.complexity_scores
    mode_counts = analyzer.mode_counts
    intra_vars = analyzer.intra_variances
    separabilities = analyzer.separabilities

    print(f"\n{'='*70}")
    print(f"CAGS v2 Complexity Stats (spec={spec}, nclass={nclass})")
    print(f"Params: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")
    print(f"        sigmoid_slope={sigmoid_slope}, sigmoid_center={sigmoid_center}")
    print(f"        complexity_k={complexity_k}")
    print(f"{'='*70}")

    print(f"\n{'Class':>6} {'Complexity':>12} {'Modes':>6} {'IntraVar':>10} {'Sep':>8} {'Guidance[0.02,0.08]':>20}")
    print("-" * 70)
    for cid in sorted(scores.keys()):
        gs = 0.02 + scores[cid] * (0.08 - 0.02)
        print(f"{cid:>6} {scores[cid]:>12.4f} {mode_counts[cid]:>6} {intra_vars[cid]:>10.4f} {separabilities[cid]:>8.4f} {gs:>20.4f}")

    score_vals = list(scores.values())
    print(f"\nComplexity stats:")
    print(f"  Range: [{min(score_vals):.4f}, {max(score_vals):.4f}]")
    print(f"  Mean:  {np.mean(score_vals):.4f}")
    print(f"  Std:   {np.std(score_vals):.4f}")
    print(f"  Spread (max-min): {max(score_vals)-min(score_vals):.4f}")

    mode_vals = list(mode_counts.values())
    print(f"\nMode count distribution: {dict(sorted(mode_counts.items()))}")
    print(f"  Unique K values: {sorted(set(mode_vals))}")

    var_vals = list(intra_vars.values())
    print(f"\nIntra-variance: [{min(var_vals):.4f}, {max(var_vals):.4f}], std={np.std(var_vals):.4f}")

    sep_vals = list(separabilities.values())
    print(f"Separability: [{min(sep_vals):.4f}, {max(sep_vals):.4f}], std={np.std(sep_vals):.4f}")

    # Save cache
    cache_path = os.path.join(save_base, "cluster_cache.pkl")
    os.makedirs(save_base, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(analyzer, f)
    print(f"\nCache saved to {cache_path}")

    # Also save stats
    stats = {
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                   "sigmoid_slope": sigmoid_slope, "sigmoid_center": sigmoid_center,
                   "complexity_k": complexity_k},
        "complexity_scores": scores,
        "mode_counts": mode_counts,
        "intra_variances": intra_vars,
        "separabilities": separabilities,
    }
    stats_path = os.path.join(save_base, "complexity_stats.json")
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats saved to {stats_path}")

    del model, vae
    torch.cuda.empty_cache()
    return analyzer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="imagenet100")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--save-base", default="./results/sweep_in10")
    parser.add_argument("--sigmoid-slope", type=float, default=3.0)
    parser.add_argument("--sigmoid-center", type=float, default=0.6)
    parser.add_argument("--complexity-k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.2)
    args = parser.parse_args()

    regen_cache(args.spec, args.nclass, args.save_base,
                args.sigmoid_slope, args.sigmoid_center,
                args.complexity_k, args.alpha, args.beta, args.gamma, args.delta)
