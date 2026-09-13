#!/usr/bin/env python3
"""Generate CAGS sensitivity datasets (different guidance ranges) on 100-class IPC=10."""
import os, sys, time, argparse, pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duration_sweep import load_model_and_vae, setup_classes, compute_clusters, generate_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmin', type=float, default=0.0)
    parser.add_argument('--lmax', type=float, default=0.06)
    parser.add_argument('--tag', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model, vae, diffusion, latent_size = load_model_and_vae(device)

    spec = "imagenet100"
    nclass = 100
    ipc = 10
    save_base = "./results/sweep_in100"

    # Load cluster cache
    cache_path = os.path.join(save_base, "cluster_cache.pkl")
    with open(cache_path, "rb") as f:
        analyzer = pickle.load(f)

    class_labels, sel_classes = setup_classes(spec=spec, nclass=nclass)

    save_dir = os.path.join(save_base, f"high_noise_{args.tag}_d25")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {args.tag} with [{args.lmin}, {args.lmax}]...")
    t0 = time.time()
    generate_config(
        model, vae, diffusion, latent_size, device,
        analyzer, class_labels, sel_classes, ipc,
        "high_noise", 25, 1, save_dir,
        schedule="constant",
        no_cags=False,
        fixed_scale=0.1,
        cags_min_scale=args.lmin,
        cags_max_scale=args.lmax,
        use_iast=False,
    )
    elapsed = time.time() - t0
    print(f"Done generating {args.tag} in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
