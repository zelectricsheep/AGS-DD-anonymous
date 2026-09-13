import argparse
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from duration_sweep import load_model_and_vae, setup_classes
from tsne_plots import center_crop_arr


def load_real_images(imagenet_dir, sel_class, ipc, image_size, device):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    class_dir = os.path.join(imagenet_dir, "train", sel_class)
    if not os.path.isdir(class_dir):
        class_dir = os.path.join(imagenet_dir, "val", sel_class)
    img_files = sorted(os.listdir(class_dir))[:ipc]
    images = []
    for fname in img_files:
        img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
        img = transform(img)
        images.append(img)
    return torch.stack(images).to(device)


def generate_d4m(args):
    device = "cuda"
    print("Loading DiT model and VAE...")
    model, vae, diffusion, latent_size = load_model_and_vae(device, args.image_size)
    model.eval()
    vae.eval()

    class_labels, sel_classes = setup_classes(args.spec, args.nclass)
    cfg_scale = 4.0
    t_start = args.t_start

    save_dir = os.path.join(args.save_base, args.tag, "dataset_0")
    os.makedirs(save_dir, exist_ok=True)

    print(f"D4M: t_start={t_start}, cfg_scale={cfg_scale}, ipc={args.ipc}")
    print(f"Saving to {save_dir}")

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
        t0 = time.time()

        real_imgs = load_real_images(args.imagenet_dir, sel_class, args.ipc, args.image_size, device)
        batch_size = min(4, args.ipc)

        for batch_start in range(0, args.ipc, batch_size):
            batch_end = min(batch_start + batch_size, args.ipc)
            batch = real_imgs[batch_start:batch_end]
            cur_bs = batch.shape[0]

            with torch.no_grad():
                latent = vae.encode(batch).latent_dist.mean * 0.18215

                noise = torch.randn_like(latent)
                t = torch.tensor([t_start] * cur_bs, device=device)
                noisy = diffusion.q_sample(latent, t, noise=noise)

                z = torch.cat([noisy, noisy], 0)
                y_cond = torch.tensor([class_label] * cur_bs, device=device)
                y_null = torch.tensor([1000] * cur_bs, device=device)
                y = torch.cat([y_cond, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale)

                indices = list(range(t_start + 1))[::-1]
                img = z
                for i in indices:
                    t_step = torch.tensor([i] * z.shape[0], device=device)
                    out = diffusion.p_mean_variance(
                        model.forward_with_cfg, img, t_step,
                        clip_denoised=False, model_kwargs=model_kwargs,
                    )
                    step_noise = torch.randn_like(img)
                    nonzero_mask = (t_step != 0).float().view(-1, *([1] * (img.ndim - 1)))
                    img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * step_noise

                samples, _ = img.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample

            for j, image in enumerate(samples):
                idx = batch_start + j
                save_path = os.path.join(save_dir, sel_class, f"{idx}.png")
                save_image(image, save_path, normalize=True, value_range=(-1, 1))

        elapsed = time.time() - t0
        print(f"  {sel_class}: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="D4M baseline: real image -> VAE encode -> add noise -> DiT denoise")
    parser.add_argument("--spec", type=str, default="imagenet100", choices=["nette", "woof", "imagenet100", "imagenet1k", "imagenet200", "animals", "objects"])
    parser.add_argument("--nclass", type=int, default=100)
    parser.add_argument("--imagenet-dir", type=str, required=True)
    parser.add_argument("--save-base", type=str, default="./results/sweep_in100")
    parser.add_argument("--tag", type=str, default="d4m")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--t-start", type=int, default=25, help="Starting timestep for partial denoising (0-49)")
    args = parser.parse_args()
    generate_d4m(args)


if __name__ == "__main__":
    main()
