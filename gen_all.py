#!/usr/bin/env python3
"""Generate all datasets for AGS-DD experiments on a single GPU.

Phase 1 (sequential): Compute all CAGS clusters
  python3 gen_all.py --compute-clusters-only

Phase 2 (parallel, 4 GPUs): Generate datasets
  CUDA_VISIBLE_DEVICES=0 python3 gen_all.py --gpu-group 0
  CUDA_VISIBLE_DEVICES=1 python3 gen_all.py --gpu-group 1
  CUDA_VISIBLE_DEVICES=2 python3 gen_all.py --gpu-group 2
  CUDA_VISIBLE_DEVICES=3 python3 gen_all.py --gpu-group 3
"""
import os, sys, time, json, argparse, pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duration_sweep import load_model_and_vae, setup_classes, compute_clusters, generate_config

# Each config: (spec, nclass, imagenet_dir, save_base, tag, ipc,
#                no_cags, fixed_scale, cags_min, cags_max, use_iast, schedule)
CONFIGS = [
    # === ImageNet-100 (100 classes) — 9 configs ===
    # IPC=50
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "ipc50_unguided", 50, True, 0.0, 0.05, 0.3, False, "constant"),
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "ipc50_cagsv2_0.0_0.06", 50, False, 0.1, 0.0, 0.06, False, "constant"),
    # IPC=10
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "unguided", 10, True, 0.0, 0.05, 0.3, False, "constant"),
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "fixed_l0.1", 10, True, 0.1, 0.05, 0.3, False, "constant"),
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "cagsv2_0.0_0.06", 10, False, 0.1, 0.0, 0.06, False, "constant"),
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "tags_linear_cagsv2", 10, False, 0.1, 0.0, 0.06, False, "linear"),
    # IPC=1
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "ipc1_unguided", 1, True, 0.0, 0.05, 0.3, False, "constant"),
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "ipc1_cagsv2_0.0_0.06", 1, False, 0.1, 0.0, 0.06, False, "constant"),
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100", "iast_cagsv2_0.0_0.06", 1, False, 0.1, 0.0, 0.06, True, "constant"),
    # === ImageNet-10 (10 classes from IN100) — 8 configs ===
    # IPC=50
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "ipc50_unguided", 50, True, 0.0, 0.05, 0.3, False, "constant"),
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "ipc50_cagsv2_0.0_0.06", 50, False, 0.1, 0.0, 0.06, False, "constant"),
    # IPC=10
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "fixed_l0.0", 10, True, 0.0, 0.05, 0.3, False, "constant"),
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "fixed_l0.05", 10, True, 0.05, 0.05, 0.3, False, "constant"),
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "cagsv2_0.0_0.08", 10, False, 0.1, 0.0, 0.08, False, "constant"),
    # IPC=1
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "ipc1_unguided", 1, True, 0.0, 0.05, 0.3, False, "constant"),
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "ipc1_cagsv2_0.0_0.06", 1, False, 0.1, 0.0, 0.06, False, "constant"),
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10", "iast_cagsv2_0.0_0.08", 1, False, 0.1, 0.0, 0.08, True, "constant"),
    # === ImageNette (10 classes) — 2 configs ===
    ("nette", 10, "./data/imagenette2/", "./results/sweep_nette", "unguided", 10, True, 0.0, 0.05, 0.3, False, "constant"),
    ("nette", 10, "./data/imagenette2/", "./results/sweep_nette", "cagsv2_0.0_0.06", 10, False, 0.1, 0.0, 0.06, False, "constant"),
    # === ImageWoof (10 classes) — 2 configs ===
    ("woof", 10, "./data/imagewoof2/", "./results/sweep_woof", "unguided", 10, True, 0.0, 0.05, 0.3, False, "constant"),
    ("woof", 10, "./data/imagewoof2/", "./results/sweep_woof", "cagsv2_0.0_0.06", 10, False, 0.1, 0.0, 0.06, False, "constant"),
]

# GPU group assignments (indices into CONFIGS)
# Balanced: GPU0/GPU1 handle 100-class (heaviest), GPU2 handles 10-class+Nette+Woof, GPU3 gets Woof
GPU_GROUPS = {
    0: [0, 2, 4, 6, 7],           # IN100 IPC=50 unguided + IN100 IPC=10 unguided + CAGS + IPC=1 unguided + CAGS
    1: [1, 3, 5, 8],              # IN100 IPC=50 CAGS + fixed_l0.1 + TAGS + IAST
    2: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # IN10 all + Nette
    3: [19, 20],                   # Woof
}

# Unique spec/nclass combos for cluster computation
CLUSTER_SPECS = [
    ("imagenet100", 100, "./data/imagenet100/", "./results/sweep_in100"),
    ("imagenet100", 10, "./data/imagenet100/", "./results/sweep_in10"),
    ("nette", 10, "./data/imagenette2/", "./results/sweep_nette"),
    ("woof", 10, "./data/imagewoof2/", "./results/sweep_woof"),
]


def check_existing(save_base, tag, ipc, nclass):
    config_name = f"high_noise_{tag}_d25"
    ds0 = os.path.join(save_base, config_name, "dataset_0")
    if not os.path.isdir(ds0):
        return 0
    total = 0
    for d in os.listdir(ds0):
        cls_dir = os.path.join(ds0, d)
        if os.path.isdir(cls_dir):
            total += len([f for f in os.listdir(cls_dir) if f.endswith((".png", ".jpg", ".JPEG"))])
    return total


def compute_all_clusters(device):
    model, vae, diffusion, latent_size = load_model_and_vae(device)
    for spec, nclass, imagenet_dir, save_base in CLUSTER_SPECS:
        cluster_cache = os.path.join(save_base, "cluster_cache.pkl")
        if os.path.isfile(cluster_cache):
            print(f"  [skip] {spec}/{nclass} clusters already cached at {cluster_cache}")
            continue
        print(f"\n  Computing clusters for {spec}/{nclass}...")
        os.makedirs(save_base, exist_ok=True)
        t0 = time.time()
        analyzer = compute_clusters(spec, imagenet_dir, vae, device, nclass=nclass)
        with open(cluster_cache, "wb") as f:
            pickle.dump(analyzer, f)
        print(f"  Done in {time.time()-t0:.1f}s -> {cluster_cache}")
    del model, vae, diffusion
    torch.cuda.empty_cache()
    print("\nAll clusters computed.")


def generate_all_for_group(gpu_group, device):
    indices = GPU_GROUPS[gpu_group]
    print(f"GPU group {gpu_group}: {len(indices)} configs to generate")

    model, vae, diffusion, latent_size = load_model_and_vae(device)
    loaded_spec = None
    analyzer = None

    for idx in indices:
        cfg = CONFIGS[idx]
        (spec, nclass, imagenet_dir, save_base, tag, ipc,
         no_cags, fixed_scale, cags_min, cags_max, use_iast, schedule) = cfg

        config_name = f"high_noise_{tag}_d25"
        save_dir = os.path.join(save_base, config_name)

        existing = check_existing(save_base, tag, ipc, nclass)
        if existing >= ipc * nclass:
            print(f"\n[{idx}] {config_name}: SKIP (already has {existing} images)")
            continue

        print(f"\n[{idx}] {config_name}: generating {ipc}×{nclass} images...")

        spec_key = (spec, nclass)
        if loaded_spec != spec_key:
            cluster_cache = os.path.join(save_base, "cluster_cache.pkl")
            if not os.path.isfile(cluster_cache):
                print(f"  WARNING: cluster cache not found at {cluster_cache}, computing...")
                analyzer = compute_clusters(spec, imagenet_dir, vae, device, nclass=nclass)
                with open(cluster_cache, "wb") as f:
                    pickle.dump(analyzer, f)
            else:
                print(f"  Loading clusters from {cluster_cache}")
                with open(cluster_cache, "rb") as f:
                    analyzer = pickle.load(f)
            loaded_spec = spec_key

        class_labels, sel_classes = setup_classes(spec=spec, nclass=nclass)

        t0 = time.time()
        generate_config(
            model, vae, diffusion, latent_size, device,
            analyzer, class_labels, sel_classes, ipc,
            "high_noise", 25, 1, save_dir,
            schedule=schedule,
            no_cags=no_cags,
            fixed_scale=fixed_scale,
            cags_min_scale=cags_min,
            cags_max_scale=cags_max,
            use_iast=use_iast,
        )
        elapsed = time.time() - t0
        print(f"  Generated in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    del model, vae, diffusion
    torch.cuda.empty_cache()
    print(f"\nGPU group {gpu_group} done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-clusters-only", action="store_true")
    parser.add_argument("--gpu-group", type=int, default=None, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.compute_clusters_only:
        compute_all_clusters(device)
    elif args.gpu_group is not None:
        generate_all_for_group(args.gpu_group, device)
    else:
        parser.error("Use --compute-clusters-only or --gpu-group N")


if __name__ == "__main__":
    main()
