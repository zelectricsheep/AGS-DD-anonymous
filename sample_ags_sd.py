"""
AGS-DD with Stable Diffusion: Non-ImageNet Dataset Distillation

Extends the AGS-DD framework to non-ImageNet datasets by using Stable Diffusion
(text-to-image) instead of class-conditional DiT for generation. The CAGS complexity
analysis and TAGS guidance schedule remain unchanged — only the backbone generator
differs. Mode guidance operates in the same VAE latent space (sd-vae-ft-mse).

Usage:
    python sample_ags_sd.py --dataset food101 --num-samples 10 --save-dir ./generated/sd_food101
    python sample_ags_sd.py --dataset food101 --num-samples 10 --no-cags  # Unguided baseline
    python sample_ags_sd.py --dataset cifar10 --num-samples 10 --save-dir ./generated/sd_cifar10

Supported datasets: food101, cifar10, cifar100, flowers102, caltech101, pets
Requires: real images in ImageFolder format at ./data/{dataset}/train/
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder as TVImageFolder
from torch.utils.data import DataLoader
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel

from ags import (
    ClassComplexityAnalyzer,
    AdaptiveStopTiming,
    TimestepAdaptiveSchedule,
)


DATASET_CONFIGS = {
    "food101": {
        "data_dir": "./data/food101",
        "class_file": "./misc/class_food101.txt",
        "prompt_template": "a high-quality photo of {}, professional food photography, close-up, on a plate, appetizing, realistic, sharp focus",
        "negative_prompt": "blurry, low quality, distorted, cartoon, illustration, painting, text, watermark, dark, overexposed, blurry background",
        "nclass": 101,
    },
    "cifar10": {
        "data_dir": "./data/cifar10",
        "class_file": "./misc/class_cifar10.txt",
        "prompt_template": "a photo of a {}",
        "negative_prompt": "blurry, low quality, distorted, cartoon, illustration, painting, text, watermark, dark, overexposed",
        "nclass": 10,
    },
    "cifar100": {
        "data_dir": "./data/cifar100",
        "class_file": "./misc/class_cifar100.txt",
        "prompt_template": "a photo of a {}",
        "negative_prompt": "blurry, low quality, distorted, cartoon, illustration, painting, text, watermark",
        "nclass": 100,
    },
}


def get_args():
    parser = argparse.ArgumentParser(description="AGS-DD with Stable Diffusion")

    # Dataset
    parser.add_argument("--dataset", type=str, default="food101",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Non-ImageNet dataset for distillation")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (must contain train/ with class subdirs)")
    parser.add_argument("--save-dir", type=str, default="./generated/sd_food101")
    parser.add_argument("--num-samples", type=int, default=10, help="IPC (images per class)")
    parser.add_argument("--num-datasets", type=int, default=3, help="Number of generated datasets (seeds)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gen-resolution", type=int, default=512, help="Generation resolution (native SD is 512)")
    parser.add_argument("--seed", type=int, default=0)

    # SD model
    parser.add_argument("--sd-model", type=str, default=None,
                        help="Path to SD 1.5 model (auto-detected if not set)")
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)

    # CAGS
    parser.add_argument("--no-cags", action="store_true", default=False)
    parser.add_argument("--cags-alpha", type=float, default=0.0)
    parser.add_argument("--cags-beta", type=float, default=0.5)
    parser.add_argument("--cags-gamma", type=float, default=0.5)
    parser.add_argument("--cags-delta", type=float, default=0.0)
    parser.add_argument("--cags-kmin", type=int, default=2)
    parser.add_argument("--cags-kmax", type=int, default=20)
    parser.add_argument("--guidance-scale-min", type=float, default=0.05)
    parser.add_argument("--guidance-scale-max", type=float, default=0.5)

    # IAST
    parser.add_argument("--no-iast", action="store_true", default=False)
    parser.add_argument("--iast-lambda", type=float, default=0.316)
    parser.add_argument("--iast-min-stop", type=int, default=5)
    parser.add_argument("--iast-max-stop-ratio", type=float, default=0.9)

    # TAGS
    parser.add_argument("--no-tags", action="store_true", default=False)
    parser.add_argument("--guidance-window", type=str, default="low_noise",
                        choices=["low_noise", "high_noise"])
    parser.add_argument("--tags-schedule", type=str, default="cosine")
    parser.add_argument("--default-guidance-scale", type=float, default=0.1)

    return parser.parse_args()


def find_sd_model():
    """Auto-detect SD 1.5 model path from HF cache."""
    import glob
    patterns = [
        "./data/hf_cache/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/*",
    ]
    for pat in patterns:
        paths = glob.glob(pat)
        if paths:
            return paths[0]
    return None


def load_class_names(dataset, class_file=None):
    """Load class names for the dataset."""
    if class_file and os.path.exists(class_file):
        with open(class_file) as f:
            return [l.strip() for l in f if l.strip()]

    cfg = DATASET_CONFIGS.get(dataset, {})
    cf = cfg.get("class_file")
    if cf and os.path.exists(cf):
        with open(cf) as f:
            return [l.strip() for l in f if l.strip()]

    # Fallback: list directories
    data_dir = cfg.get("data_dir", "./data/" + dataset)
    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir):
        return sorted(os.listdir(train_dir))
    return []


def make_prompt(class_name, template):
    """Convert class name to text prompt."""
    readable = class_name.replace("_", " ").replace("-", " ")
    return template.format(readable)


def extract_vae_features(data_dir, class_names, vae, device, image_size=256, max_per_class=50):
    """Extract VAE latent features for each class from real images.

    This is the non-ImageNet equivalent of get_features_per_class() in tsne_plots.py.
    Uses the same VAE (sd-vae-ft-mse) so latents are compatible with SD generation.
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.center_crop(
            img, min(img.size))),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    features_per_class = {}
    paths_per_class = {}

    for class_name in tqdm(class_names, desc="Extracting VAE features"):
        class_dir = os.path.join(data_dir, "train", class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: {class_dir} not found, skipping")
            continue

        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))
        ])[:max_per_class]

        if len(image_files) == 0:
            print(f"  Warning: no images in {class_dir}")
            continue

        feats = []
        paths = []
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    latent = vae.encode(img_tensor).latent_dist.sample()
                    latent = latent * 0.18215  # SD scaling factor
                    # Flatten to feature vector
                    feat = latent.flatten().cpu().numpy()
                    feats.append(feat)
                    paths.append(img_path)
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
                continue

        class_idx = class_names.index(class_name)
        features_per_class[class_idx] = np.array(feats)
        paths_per_class[class_idx] = paths
        print(f"  {class_name}: {len(feats)} features, dim={feats[0].shape}")

    return features_per_class, paths_per_class


def compute_cags_complexity(features_per_class, paths_per_class, args):
    """Compute CAGS complexity scores and cluster centers."""
    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(args.cags_kmin, args.cags_kmax),
        alpha=args.cags_alpha,
        beta=args.cags_beta,
        gamma=args.cags_gamma,
        delta=args.cags_delta,
        use_pca=True,
        sigmoid_slope=3.0,
        sigmoid_center=0.6,
    )

    analyzer.analyze_all_classes(features_per_class, paths_per_class)
    return analyzer


@torch.no_grad()
def sample_with_ags_sd(
    unet, scheduler, text_embeddings, uncond_embeddings, mode_features,
    guidance_strength, t_stop, guidance_schedule,
    use_cags, use_tags, guidance_window,
    cfg_scale, num_steps, latent_size, device, seed=0,
    default_guidance_scale=0.1,
):
    """DDIM sampling with AGS mode guidance using Stable Diffusion.

    This is the SD equivalent of AGSSampler.sample_with_ags_guidance().
    Mode guidance steers predicted x0 toward cluster centers in VAE latent space.
    """
    batch_size = 1
    generator = torch.Generator(device=device).manual_seed(seed)

    # Start from pure noise
    latents = torch.randn(
        (batch_size, 4, latent_size, latent_size),
        device=device, generator=generator,
    )

    # Set timesteps
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    t_max = num_steps - 1

    for i, t in enumerate(timesteps):
        timestep_idx = num_steps - 1 - i  # 0 = lowest noise

        # TAGS: Compute adaptive guidance weight
        if use_tags:
            if guidance_window == "high_noise":
                w_t = guidance_schedule.get_weight(
                    t=timestep_idx, t_start=t_stop, t_stop=t_max,
                    w_max=guidance_strength, reverse=True,
                )
            else:
                w_t = guidance_schedule.get_weight(
                    t=timestep_idx, t_start=0, t_stop=t_stop,
                    w_max=guidance_strength, reverse=False,
                )
        else:
            if guidance_window == "high_noise":
                w_t = guidance_strength if timestep_idx > t_stop else 0.0
            else:
                w_t = guidance_strength if timestep_idx < t_stop else 0.0

        # Expand latents for CFG
        latent_input = torch.cat([latents, latents])
        text_input = torch.cat([uncond_embeddings, text_embeddings])

        # Predict noise
        noise_pred = unet(latent_input, t, encoder_hidden_states=text_input).sample

        # Apply CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        # Get DDIM step components
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[timesteps[i - 1]] if i > 0 else scheduler.final_alpha_cumprod
        beta_prod_t = 1.0 - alpha_prod_t

        # Predict x0 (epsilon parameterization)
        pred_x0 = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # Apply mode guidance: steer pred_x0 toward mode features
        if w_t > 0 and mode_features is not None:
            mode_feat = mode_features.reshape(1, 4, latent_size, latent_size)
            pred_x0_cond = pred_x0  # Already after CFG
            pred_x0 = pred_x0_cond - w_t * (pred_x0_cond - mode_feat)

        # Direction pointing to x_t
        pred_dir = (1.0 - alpha_prod_t_prev) ** 0.5 * noise_pred

        # Compute x_{t-1}
        latents = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

    return latents


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DATASET_CONFIGS[args.dataset]
    data_dir = args.data_dir or cfg["data_dir"]
    class_names = load_class_names(args.dataset)
    nclass = len(class_names) if class_names else cfg["nclass"]

    print("=" * 70)
    print(f"AGS-DD + Stable Diffusion: {args.dataset}")
    print(f"Classes: {nclass}, IPC: {args.num_samples}")
    print(f"CAGS: {'enabled' if not args.no_cags else 'disabled'}")
    print(f"TAGS: {'enabled' if not args.no_tags else 'disabled'}")
    print(f"IAST: {'enabled' if not args.no_iast else 'disabled'}")
    print("=" * 70)

    # Load SD 1.5 components
    sd_path = args.sd_model or find_sd_model()
    if not sd_path:
        print("ERROR: Could not find SD 1.5 model. Download it first.")
        sys.exit(1)
    print(f"\nLoading SD 1.5 from {sd_path}")

    vae = AutoencoderKL.from_pretrained(os.path.join(sd_path, "vae")).to(device)
    unet = UNet2DConditionModel.from_pretrained(os.path.join(sd_path, "unet")).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(sd_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(sd_path, "text_encoder")).to(device)
    scheduler = DDIMScheduler.from_pretrained(os.path.join(sd_path, "scheduler"))
    scheduler.set_timesteps(args.num_sampling_steps, device=device)

    latent_size = args.gen_resolution // 8
    print(f"Model loaded. Latent size: {latent_size}")

    # Compute CAGS complexity from real images
    if not args.no_cags:
        print("\nExtracting VAE features from real images...")
        features_per_class, paths_per_class = extract_vae_features(
            data_dir, class_names, vae, device, args.gen_resolution, max_per_class=50
        )

        print("\nComputing class complexity (CAGS)...")
        complexity_analyzer = compute_cags_complexity(features_per_class, paths_per_class, args)
        clusters_centers = complexity_analyzer.compute_clusters_for_ipc(
            args.num_samples, use_pca=True, closest_point=True
        )
    else:
        complexity_analyzer = None
        clusters_centers = None

    # IAST
    adaptive_stop = AdaptiveStopTiming(
        t_max=args.num_sampling_steps,
        lam=args.iast_lambda,
        min_stop=args.iast_min_stop,
        max_stop_ratio=args.iast_max_stop_ratio,
        use_complexity=not args.no_iast,
    )

    # TAGS
    guidance_schedule = TimestepAdaptiveSchedule(
        w_max=args.guidance_scale_max,
        schedule_type=args.tags_schedule,
    )

    # Generate datasets
    print(f"\nGenerating {args.num_datasets} datasets with {args.num_samples} IPC...")
    print(f"Save directory: {args.save_dir}")

    template = cfg["prompt_template"]

    # Pre-compute unconditional embeddings for CFG (empty string)
    uncond_input = tokenizer(
        "", padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    uncond_embeddings_default = uncond_embeddings

    for dataset_idx in range(args.num_datasets):
        ds_dir = os.path.join(args.save_dir, f"dataset_{dataset_idx}")
        os.makedirs(ds_dir, exist_ok=True)

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(ds_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Compute per-class guidance parameters
            if not args.no_cags and complexity_analyzer:
                # IAST range scaling
                ipc_ref = 10
                iast_scale = min(1.0, np.log(args.num_samples + 1) / np.log(ipc_ref + 1))
                scaled_range = (
                    args.guidance_scale_min * iast_scale,
                    args.guidance_scale_max * iast_scale,
                )
                guidance_strength = complexity_analyzer.get_guidance_strength(class_idx, scaled_range)

                n_modes = complexity_analyzer.mode_counts.get(class_idx, 5)
                complexity = complexity_analyzer.complexity_scores.get(class_idx, 0.5)
                t_stop = adaptive_stop.compute_stop(args.num_samples, n_modes, complexity)
            else:
                guidance_strength = args.default_guidance_scale
                t_stop = 25

            print(f"  [{dataset_idx}] {class_name}: guidance={guidance_strength:.4f}, stop_t={t_stop}")

            # Encode text prompt
            prompt = make_prompt(class_name, template)
            neg_prompt = cfg.get("negative_prompt", "")
            text_input = tokenizer(
                prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

            # Encode negative prompt for uncond
            if neg_prompt:
                neg_input = tokenizer(
                    neg_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                )
                uncond_embeddings = text_encoder(neg_input.input_ids.to(device))[0]
            else:
                uncond_embeddings = uncond_embeddings_default

            # Get mode features for this class
            if clusters_centers and class_idx in clusters_centers:
                mode_feats = torch.tensor(
                    clusters_centers[class_idx], device=device, dtype=torch.float32
                )
            else:
                mode_feats = None

            start_time = time.time()

            for img_idx in range(args.num_samples):
                # Generate with AGS guidance
                latent = sample_with_ags_sd(
                    unet=unet,
                    scheduler=scheduler,
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    mode_features=mode_feats[img_idx] if mode_feats is not None else None,
                    guidance_strength=guidance_strength,
                    t_stop=t_stop,
                    guidance_schedule=guidance_schedule,
                    use_cags=not args.no_cags,
                    use_tags=not args.no_tags,
                    guidance_window=args.guidance_window,
                    cfg_scale=args.cfg_scale,
                    num_steps=args.num_sampling_steps,
                    latent_size=latent_size,
                    device=device,
                    seed=dataset_idx * 10000 + class_idx * 100 + img_idx,
                    default_guidance_scale=args.default_guidance_scale,
                )

                # Decode latent to image
                latent = latent / 0.18215
                image = vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)

                # Resize to target image_size if different from gen_resolution
                if args.gen_resolution != args.image_size:
                    image = torch.nn.functional.interpolate(
                        image, size=(args.image_size, args.image_size),
                        mode="bilinear", align_corners=False
                    )

                save_path = os.path.join(class_dir, f"{img_idx}.png")
                save_image(image, save_path)

            elapsed = time.time() - start_time
            print(f"    Time: {elapsed:.1f}s ({elapsed / args.num_samples:.1f}s/img)")

    total_time = time.time() - start_time if 'start_time' in dir() else 0
    print(f"\n{'=' * 70}")
    print(f"Generation complete! Total time: {total_time:.1f}s")
    print(f"Output: {args.save_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
