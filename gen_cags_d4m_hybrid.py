#!/usr/bin/env python3
"""
CAGS+D4M Hybrid: Real image initialization + CAGS adaptive guidance.

Combines D4M's real-image initialization (VAE encode -> add noise at t_start -> denoise)
with CAGS's per-class adaptive mode guidance (steer x_start toward cluster centers
with per-class adaptive strength during high-noise window).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 gen_cags_d4m_hybrid.py \
        --spec imagenet100 --nclass 100 \
        --imagenet-dir ./data/imagenet100 \
        --save-base ./results/sweep_in100 \
        --tag cags_d4m_t40 \
        --ipc 10 --t-start 40 \
        --alpha 0.0 --beta 0.5 --gamma 0.5 --delta 0.0 \
        --cags-min-scale 0.0 --cags-max-scale 0.06 \
        --sigmoid-slope 3.0 --sigmoid-center 0.6 \
        --num-datasets 3
"""

import argparse
import os
import time
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from duration_sweep import load_model_and_vae, setup_classes, compute_clusters
from gen_d4m import load_real_images


def generate_hybrid(args):
    device = "cuda"
    latent_size = args.image_size // 8

    print("Loading DiT model and VAE...")
    model, vae, diffusion, latent_size = load_model_and_vae(device, args.image_size)
    model.eval()
    vae.eval()

    class_labels, sel_classes = setup_classes(args.spec, args.nclass)
    cfg_scale = 4.0
    t_start = args.t_start
    t_stop = args.guidance_stop

    guidance_steps = max(0, t_start - t_stop)
    print(f"\nCAGS+D4M Hybrid Configuration:")
    print(f"  t_start={t_start}, t_stop={t_stop}, cfg={cfg_scale}")
    print(f"  Guidance window: t={t_start} to t={t_stop+1} ({guidance_steps} steps)")
    print(f"  No-guidance window: t={t_stop} to t=0 ({t_stop+1} steps)")
    print(f"  IPC={args.ipc}, datasets={args.num_datasets}")
    print(f"  CAGS weights: alpha={args.alpha}, beta={args.beta}, "
          f"gamma={args.gamma}, delta={args.delta}")
    print(f"  Guidance range: [{args.cags_min_scale}, {args.cags_max_scale}]")

    cluster_cache = os.path.join(args.save_base, "cluster_cache.pkl")
    if os.path.isfile(cluster_cache) and not args.regen_cache:
        print(f"\nLoading cached CAGS clusters from {cluster_cache}")
        with open(cluster_cache, "rb") as f:
            analyzer = pickle.load(f)
    else:
        print("\nComputing CAGS clusters...")
        analyzer = compute_clusters(
            args.spec, args.imagenet_dir, vae, device,
            nclass=args.nclass,
            sigmoid_slope=args.sigmoid_slope,
            sigmoid_center=args.sigmoid_center,
            alpha=args.alpha, beta=args.beta,
            gamma=args.gamma, delta=args.delta,
        )
        os.makedirs(args.save_base, exist_ok=True)
        with open(cluster_cache, "wb") as f:
            pickle.dump(analyzer, f)
        print(f"Cluster cache saved to {cluster_cache}")

    clusters_centers = analyzer.compute_clusters_for_ipc(
        args.ipc, use_pca=False, closest_point=False
    )

    per_class_guidance = {}
    for class_label, sel_class in zip(class_labels, sel_classes):
        class_idx = sel_classes.index(sel_class)
        guidance_strength = analyzer.get_guidance_strength(
            class_idx, (args.cags_min_scale, args.cags_max_scale)
        )
        per_class_guidance[class_label] = guidance_strength

    guidance_values = [per_class_guidance[cl] for cl in class_labels]
    print(f"\nPer-class guidance: min={min(guidance_values):.4f}, "
          f"max={max(guidance_values):.4f}, "
          f"mean={np.mean(guidance_values):.4f}, "
          f"std={np.std(guidance_values):.4f}")

    for dataset_idx in range(args.num_datasets):
        save_dir = os.path.join(args.save_base, args.tag, f"dataset_{dataset_idx}")
        os.makedirs(save_dir, exist_ok=True)

        torch.manual_seed(dataset_idx * 1000 + 42)
        np.random.seed(dataset_idx * 1000 + 42)

        print(f"\n{'='*60}")
        print(f"Dataset {dataset_idx} (seed={dataset_idx * 1000 + 42})")
        print(f"{'='*60}")

        for class_label, sel_class in zip(class_labels, sel_classes):
            os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
            t0 = time.time()

            class_idx = sel_classes.index(sel_class)
            real_imgs = load_real_images(
                args.imagenet_dir, sel_class, args.ipc, args.image_size, device
            )
            guidance_strength = per_class_guidance[class_label]

            for img_idx in range(args.ipc):
                with torch.no_grad():
                    real_img = real_imgs[img_idx:img_idx+1]

                    latent = vae.encode(real_img).latent_dist.mean * 0.18215

                    noise = torch.randn_like(latent)
                    t = torch.tensor([t_start], device=device)
                    noisy = diffusion.q_sample(latent, t, noise=noise)

                    z = torch.cat([noisy, noisy], 0)
                    y_cond = torch.tensor([class_label], device=device)
                    y_null = torch.tensor([1000], device=device)
                    y = torch.cat([y_cond, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

                    mode_feat = clusters_centers[class_idx][img_idx]
                    mode_features = torch.tensor(
                        mode_feat.reshape(1, 4, latent_size, latent_size),
                        device=device,
                    )

                    indices = list(range(t_start + 1))[::-1]
                    img = z

                    for i in indices:
                        t_step = torch.tensor([i] * 2, device=device)

                        out = diffusion.p_mean_variance(
                            model.forward_with_cfg, img, t_step,
                            clip_denoised=False, model_kwargs=model_kwargs,
                        )

                        step_noise = torch.randn_like(img)
                        nonzero_mask = (
                            t_step != 0
                        ).float().view(-1, *([1] * (img.ndim - 1)))

                        if i > t_stop and guidance_strength > 0:
                            xstart = out["pred_xstart"]
                            xstart_cond, _ = xstart.chunk(2, dim=0)

                            guidance_score = -(
                                xstart_cond - mode_features
                            ) * guidance_strength * torch.exp(
                                0.5 * out["log_variance"]
                            )

                            img = (
                                out["mean"]
                                + guidance_score
                                + nonzero_mask * torch.exp(
                                    0.5 * out["log_variance"]
                                ) * step_noise
                            )
                        else:
                            img = (
                                out["mean"]
                                + nonzero_mask * torch.exp(
                                    0.5 * out["log_variance"]
                                ) * step_noise
                            )

                    samples, _ = img.chunk(2, dim=0)
                    samples = vae.decode(samples / 0.18215).sample

                save_path = os.path.join(
                    save_dir, sel_class, f"{img_idx}.png"
                )
                save_image(
                    samples[0], save_path, normalize=True, value_range=(-1, 1)
                )

            elapsed = time.time() - t0
            print(f"  {sel_class} (g={guidance_strength:.4f}): {elapsed:.1f}s")

    final_dir = os.path.join(args.save_base, args.tag)
    print(f"\nDone! {args.num_datasets} datasets saved to {final_dir}")
    print(f"Total images: {args.num_datasets * args.ipc * len(sel_classes)}")


def main():
    parser = argparse.ArgumentParser(
        description="CAGS+D4M Hybrid: real image init + CAGS adaptive guidance"
    )
    parser.add_argument("--spec", type=str, default="imagenet100",
                        choices=["nette", "woof", "imagenet100",
                                 "imagenet1k", "imagenet200"])
    parser.add_argument("--nclass", type=int, default=100)
    parser.add_argument("--imagenet-dir", type=str, required=True)
    parser.add_argument("--save-base", type=str, default="./results/sweep_in100")
    parser.add_argument("--tag", type=str, default="cags_d4m_hybrid")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--t-start", type=int, default=40,
                        help="Starting timestep (must be > guidance_stop for "
                             "guidance window to exist)")
    parser.add_argument("--guidance-stop", type=int, default=25,
                        help="Stop timestep for guidance (high-noise window "
                             "boundary, default 25)")
    parser.add_argument("--num-datasets", type=int, default=3,
                        help="Number of datasets (seeds) to generate")
    parser.add_argument("--cags-min-scale", type=float, default=0.0)
    parser.add_argument("--cags-max-scale", type=float, default=0.06)
    parser.add_argument("--sigmoid-slope", type=float, default=3.0)
    parser.add_argument("--sigmoid-center", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--regen-cache", action="store_true", default=False)
    args = parser.parse_args()

    if args.t_start <= args.guidance_stop:
        print(f"WARNING: t_start ({args.t_start}) <= guidance_stop "
              f"({args.guidance_stop}). No guidance window exists. "
              f"This is equivalent to pure D4M.")

    generate_hybrid(args)


if __name__ == "__main__":
    main()
