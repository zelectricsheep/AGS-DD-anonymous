"""
AGS Sampler: Adaptive Guidance Scheduling Sampler

Integrates CAGS, IAST, and TAGS into a unified sampling pipeline that replaces
the fixed-parameter mode guidance in MGD3 with adaptive, per-class, per-timestep
guidance scheduling.

The sampler modifies the diffusion sampling loop to:
  1. Use CAGS to determine per-class guidance strength (replaces fixed mode_guidance_scale)
  2. Use IAST to determine per-class stop timing (replaces fixed stop_t)
  3. Use TAGS to schedule guidance weight across timesteps (replaces constant guidance)
"""

import os
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torchvision.utils import save_image

from .class_complexity import ClassComplexityAnalyzer
from .adaptive_stop import AdaptiveStopTiming
from .guidance_schedule import TimestepAdaptiveSchedule


class AGSSampler:
    """
    Main AGS-DD sampler that orchestrates the three adaptive modules.

    Usage:
        sampler = AGSSampler(model, vae, diffusion, complexity_analyzer, ...)
        sampler.generate_dataset(args, class_labels, sel_classes)
    """

    def __init__(
        self,
        model,
        vae,
        diffusion,
        complexity_analyzer=None,
        adaptive_stop=None,
        guidance_schedule=None,
        device="cuda",
        num_sampling_steps=50,
        cfg_scale=4.0,
        latent_size=32,
        use_cags=True,
        use_iast=True,
        use_tags=True,
        guidance_scale_range=(0.05, 0.5),
        guidance_window="low_noise",
    ):
        """
        Args:
            model: Pre-trained diffusion model (DiT)
            vae: VAE decoder
            diffusion: Diffusion process (from create_diffusion)
            complexity_analyzer: CAGS module (ClassComplexityAnalyzer)
            adaptive_stop: IAST module (AdaptiveStopTiming)
            guidance_schedule: TAGS module (TimestepAdaptiveSchedule)
            device: torch device
            num_sampling_steps: Number of DDIM sampling steps
            cfg_scale: Classifier-free guidance scale
            latent_size: Latent spatial size
            use_cags: Enable/disable CAGS module (for ablation)
            use_iast: Enable/disable IAST module (for ablation)
            use_tags: Enable/disable TAGS module (for ablation)
            guidance_scale_range: (min, max) range for CAGS output
            guidance_window: "low_noise" (default, experimentally best) or
                            "high_noise" — controls which end of the denoising
                            process receives mode guidance
        """
        self.model = model
        self.vae = vae
        self.diffusion = diffusion
        self.device = device
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale
        self.latent_size = latent_size
        self.guidance_scale_range = guidance_scale_range
        self.guidance_window = guidance_window

        # AGS modules
        self.complexity_analyzer = complexity_analyzer or ClassComplexityAnalyzer()
        self.adaptive_stop = adaptive_stop or AdaptiveStopTiming(
            t_max=num_sampling_steps, lam=0.316, window=guidance_window
        )
        self.guidance_schedule = guidance_schedule or TimestepAdaptiveSchedule(
            w_max=0.3, schedule_type="cosine"
        )

        # Ablation flags
        self.use_cags = use_cags
        self.use_iast = use_iast
        self.use_tags = use_tags

        # Default values when a module is disabled
        self.default_guidance_scale = 0.1
        self.default_stop_t = 25

        # Storage for computed parameters
        self.per_class_params = {}

    def precompute_class_params(self, ipc, class_labels, sel_classes):
        """
        Precompute per-class guidance parameters using CAGS and IAST.

        This is called once before sampling to determine:
          - Per-class guidance strength (from CAGS)
          - Per-class stop timing (from IAST)
        """
        self.per_class_params = {}

        # IAST: IPC-adaptive guidance range scaling
        # When IPC is low, the per-class complexity estimates are noisy and
        # the guidance range should be shrunk to avoid over-sharpening.
        # range_scale = min(1.0, log(IPC+1) / log(IPC_ref+1))
        iast_range_scale = 1.0
        if self.use_iast:
            ipc_ref = 10  # full CAGS range at IPC>=10
            iast_range_scale = min(1.0, np.log(ipc + 1) / np.log(ipc_ref + 1))
            scaled_range = (
                self.guidance_scale_range[0] * iast_range_scale,
                self.guidance_scale_range[1] * iast_range_scale,
            )
            print(f"  IAST range scaling: IPC={ipc}, scale={iast_range_scale:.4f}, "
                  f"range {self.guidance_scale_range} -> {scaled_range}")
        else:
            scaled_range = self.guidance_scale_range

        for class_label, sel_class in zip(class_labels, sel_classes):
            class_idx = sel_classes.index(sel_class)

            # CAGS: Class-adaptive guidance strength (with IAST range scaling)
            if self.use_cags:
                guidance_strength = self.complexity_analyzer.get_guidance_strength(
                    class_idx, scaled_range
                )
            else:
                guidance_strength = self.default_guidance_scale

            # IAST: stop timing (kept for backward compat, but primary effect
            # is now via range scaling above)
            if self.use_iast:
                n_modes = self.complexity_analyzer.mode_counts.get(class_idx, 5)
                complexity = self.complexity_analyzer.complexity_scores.get(class_idx, 0.5)
                t_stop = self.adaptive_stop.compute_stop(ipc, n_modes, complexity)
            else:
                t_stop = self.default_stop_t

            self.per_class_params[class_label] = {
                "guidance_strength": guidance_strength,
                "t_stop": t_stop,
                "guidance_window": self.guidance_window,
                "n_modes": self.complexity_analyzer.mode_counts.get(class_idx, 5),
                "complexity": self.complexity_analyzer.complexity_scores.get(class_idx, 0.5),
            }

            print(
                f"  Class {sel_class} (label={class_label}): "
                f"guidance={guidance_strength:.4f}, stop_t={t_stop}, "
                f"modes={self.per_class_params[class_label]['n_modes']}, "
                f"complexity={self.per_class_params[class_label]['complexity']:.4f}"
            )

    @torch.no_grad()
    def sample_with_ags_guidance(
        self,
        z,
        class_label,
        mode_features,
        model_kwargs,
    ):
        """
        Core sampling function with AGS guidance.

        This replaces MGD3's p_sample_loop_with_mode_guidance with an
        adaptive version that varies guidance strength and stop timing
        per class and per timestep.

        Args:
            z: Initial noise tensor (already doubled for CFG)
            class_label: Integer class label
            mode_features: Mode features for guidance (latent space)
            model_kwargs: Model keyword arguments (y, cfg_scale)

        Returns:
            samples: Generated latent samples
        """
        params = self.per_class_params.get(class_label, {})
        guidance_strength = params.get("guidance_strength", self.default_guidance_scale)
        t_stop = params.get("t_stop", self.default_stop_t)

        img = z
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        t_max = self.num_sampling_steps - 1

        for i in indices:
            t = torch.tensor([i] * z.shape[0], device=self.device)

            # TAGS: Compute adaptive guidance weight for this timestep
            if self.use_tags:
                if self.guidance_window == "high_noise":
                    w_t = self.guidance_schedule.get_weight(
                        t=i, t_start=t_stop, t_stop=t_max,
                        w_max=guidance_strength, reverse=True,
                    )
                else:
                    w_t = self.guidance_schedule.get_weight(
                        t=i, t_start=0, t_stop=t_stop,
                        w_max=guidance_strength, reverse=False,
                    )
            else:
                # Constant guidance (like MGD3)
                if self.guidance_window == "high_noise":
                    w_t = guidance_strength if i > t_stop else 0.0
                else:
                    w_t = guidance_strength if i < t_stop else 0.0

            # If no guidance at this step, use standard p_sample
            if w_t <= 0:
                out = self.diffusion.p_mean_variance(
                    self.model.forward_with_cfg, img, t,
                    clip_denoised=False, model_kwargs=model_kwargs,
                )
                noise = torch.randn_like(img)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                continue

            # Compute p_mean_variance (predicts x_start and mean/variance)
            out = self.diffusion.p_mean_variance(
                self.model.forward_with_cfg, img, t,
                clip_denoised=False, model_kwargs=model_kwargs,
            )

            noise = torch.randn_like(img)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))

            # Extract predicted x_start (conditional part only)
            xstart = out["pred_xstart"]
            xstart_cond, _ = xstart.chunk(2, dim=0)

            # AGS guidance: steer x_start toward mode features with adaptive weight
            # Guidance score broadcasts from [1,...] to [2,...] matching log_variance
            guidance_score = -(
                xstart_cond - mode_features
            ) * w_t * torch.exp(0.5 * out["log_variance"])

            img = (
                out["mean"]
                + guidance_score
                + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            )

        return img

    @torch.no_grad()
    def generate_dataset(
        self,
        args,
        class_labels,
        sel_classes,
        clusters_centers,
        save_dir,
        num_datasets=5,
        batch_size=1,
        use_same_noise=False,
        total_shift=0,
    ):
        """
        Generate the full distilled dataset using AGS guidance.

        Args:
            args: Configuration arguments
            class_labels: List of integer class labels
            sel_classes: List of class name strings
            clusters_centers: dict {class_idx: numpy array of mode features}
            save_dir: Base directory to save generated images
            num_datasets: Number of datasets to generate (for multiple seeds)
            batch_size: Batch size for generation
            use_same_noise: Whether to use same noise across classes
            total_shift: Index offset for file naming
        """
        # Precompute per-class parameters
        ipc = args.num_samples
        self.precompute_class_params(ipc, class_labels, sel_classes)

        for dataset_idx in range(num_datasets):
            dataset_dir = os.path.join(save_dir, f"dataset_{dataset_idx}")

            # Pre-generate noises if using same noise
            noises = []
            if use_same_noise:
                for _ in range(ipc // batch_size):
                    z = torch.randn(
                        batch_size, 4, self.latent_size, self.latent_size, device=self.device
                    )
                    noises.append(z)

            for class_label, sel_class in zip(class_labels, sel_classes):
                print(f"\nGenerating class: {sel_class} (label={class_label})")
                class_idx = sel_classes.index(sel_class)
                os.makedirs(os.path.join(dataset_dir, sel_class), exist_ok=True)

                start_time = time.time()

                for shift in tqdm(range(ipc // batch_size)):
                    # Create sampling noise
                    if use_same_noise:
                        z = noises[shift]
                    else:
                        z = torch.randn(
                            batch_size, 4, self.latent_size, self.latent_size, device=self.device
                        )

                    y = torch.tensor([class_label], device=self.device)

                    # Setup classifier-free guidance (double the batch)
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * batch_size, device=self.device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=self.cfg_scale)

                    # Get mode features for this sample
                    mode_feat = clusters_centers[class_idx][shift]
                    mode_features = torch.tensor(
                        mode_feat.reshape(1, 4, self.latent_size, self.latent_size),
                        device=self.device,
                    )

                    # Sample with AGS guidance
                    samples = self.sample_with_ags_guidance(
                        z, class_label, mode_features, model_kwargs
                    )

                    # Remove null class samples (take conditional part only)
                    samples, _ = samples.chunk(2, dim=0)

                    # Decode from latent space
                    samples = self.vae.decode(samples / 0.18215).sample

                    # Save images
                    for image_index, image in enumerate(samples):
                        save_path = os.path.join(
                            dataset_dir, sel_class,
                            f"{image_index + shift * batch_size + total_shift}.png",
                        )
                        save_image(image, save_path, normalize=True, value_range=(-1, 1))

                elapsed = time.time() - start_time
                print(f"  Time: {elapsed:.1f}s")

    def get_config_summary(self):
        """Return a summary of the current AGS configuration for logging."""
        return {
            "use_cags": self.use_cags,
            "use_iast": self.use_iast,
            "use_tags": self.use_tags,
            "guidance_window": self.guidance_window,
            "guidance_scale_range": self.guidance_scale_range,
            "num_sampling_steps": self.num_sampling_steps,
            "cfg_scale": self.cfg_scale,
            "schedule_type": self.guidance_schedule.schedule_type,
            "iast_lambda": self.adaptive_stop.lam,
            "cags_alpha": self.complexity_analyzer.alpha,
            "cags_beta": self.complexity_analyzer.beta,
        }
