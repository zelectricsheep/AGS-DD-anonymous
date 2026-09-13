"""
IAST: IPC-Adaptive Stop Timing

Determines when to stop mode guidance during the diffusion sampling process,
adaptively based on the Images-Per-Class (IPC) setting and class complexity.

Key insight: 
  - Low IPC (e.g., 1): The synthetic dataset has very limited capacity, so we
    should stop guidance early to ensure representativeness (avoid overfitting
    to specific modes at the cost of overall quality).
  - High IPC (e.g., 50-100): More capacity allows prolonged guidance to increase
    diversity and cover all sub-modes within each class.

Inspired by DATM (ICLR 2024) difficulty-aligned trajectory matching, but applied
to the diffusion guidance stopping criterion.
"""

import numpy as np
import torch


class AdaptiveStopTiming:
    """
    Computes adaptive stop timing for mode guidance in diffusion sampling.

    The stop timestep t_stop determines when to transition from guided diffusion
    (mode guidance active) to unguided diffusion (prior takes over for detail
    refinement).

    t_stop(c, IPC) = t_max * (1 - exp(-lambda * sqrt(IPC) / K(c)))

    where K(c) is the number of modes in class c.

    Uses sqrt(IPC) scaling instead of linear IPC to prevent overly aggressive
    late stopping at high IPC values. This ensures that the stop timing
    increases gently with IPC, preserving image representativeness on harder
    datasets while still allowing some diversity increase.
    """

    def __init__(
        self,
        t_max=50,
        lam=0.316,
        min_stop=5,
        max_stop_ratio=0.9,
        use_complexity=True,
        complexity_weight=0.3,
        window="low_noise",
    ):
        """
        Args:
            t_max: Maximum number of diffusion timesteps (e.g., 50 for DDIM)
            lam: Lambda parameter controlling the exponential decay rate.
                 Adjusted to 0.316 for sqrt scaling (preserves t_stop at IPC=10
                 compared to the original linear formula with lam=0.1).
            min_stop: Minimum guidance steps (never guide fewer than this)
            max_stop_ratio: Maximum ratio of t_max for guidance (e.g., 0.9 means
                           never guide beyond 90% of timesteps)
            use_complexity: Whether to incorporate class complexity into stop timing
            complexity_weight: Weight of complexity in stop timing adjustment
            window: "low_noise" (default) — guidance in [0, t_stop], t_stop =
                    guidance_steps. "high_noise" — guidance in [t_stop, t_max],
                    t_stop = t_max - 1 - guidance_steps.
        """
        self.t_max = t_max
        self.lam = lam
        self.min_stop = min_stop
        self.max_stop_ratio = max_stop_ratio
        self.use_complexity = use_complexity
        self.complexity_weight = complexity_weight
        self.window = window

        self.stop_timings = {}

    def compute_guidance_steps(self, ipc, n_modes, complexity_score=None):
        """
        Compute the number of guidance steps for a specific class and IPC setting.

        This is the primary quantity: it is monotonically non-decreasing with
        IPC, n_modes, and complexity_score. The t_stop used by the sampler is
        derived from this value based on the guidance window direction.

        Args:
            ipc: Images Per Class
            n_modes: Number of detected modes in this class
            complexity_score: Optional complexity score from CAGS [0, 1]

        Returns:
            guidance_steps: int — number of timesteps with active guidance
        """
        ratio = 1.0 - np.exp(-self.lam * np.sqrt(ipc) / max(n_modes, 1))
        guidance_steps = int(self.t_max * ratio)

        if self.use_complexity and complexity_score is not None:
            complexity_adjustment = self.complexity_weight * complexity_score * self.t_max
            guidance_steps += int(complexity_adjustment)

        guidance_steps = int(np.clip(guidance_steps, self.min_stop, int(self.t_max * self.max_stop_ratio)))

        return guidance_steps

    def compute_stop(self, ipc, n_modes, complexity_score=None):
        """
        Compute the t_stop cutoff for the sampler, derived from guidance_steps.

        For low_noise window: guidance in [0, t_stop], so t_stop = guidance_steps.
        For high_noise window: guidance in [t_stop, t_max-1], so
        t_stop = t_max - 1 - guidance_steps.

        Args:
            ipc: Images Per Class
            n_modes: Number of detected modes in this class
            complexity_score: Optional complexity score from CAGS [0, 1]

        Returns:
            t_stop: int — the timestep cutoff used by the sampler
        """
        guidance_steps = self.compute_guidance_steps(ipc, n_modes, complexity_score)

        if self.window == "high_noise":
            t_stop = self.t_max - 1 - guidance_steps
        else:
            t_stop = guidance_steps

        return t_stop

    def compute_all_stops(self, ipc, mode_counts, complexity_scores=None):
        """
        Compute stop timings for all classes.

        Args:
            ipc: Images Per Class
            mode_counts: dict {class_id: int}
            complexity_scores: dict {class_id: float} from CAGS

        Returns:
            stop_timings: dict {class_id: int}
        """
        for class_id, n_modes in mode_counts.items():
            complexity = complexity_scores.get(class_id, 0.5) if complexity_scores else None
            t_stop = self.compute_stop(ipc, n_modes, complexity)
            self.stop_timings[class_id] = t_stop

        return self.stop_timings

    def get_stop(self, class_id):
        """Get precomputed stop timing for a class."""
        return self.stop_timings.get(class_id, self.min_stop)

    def analyze_sensitivity(self, ipc_range=(1, 10, 50, 100, 200), n_modes_range=(2, 5, 10, 20)):
        """
        Analyze stop timing sensitivity across different IPC and mode counts.
        Useful for hyperparameter analysis in the paper.

        Returns:
            results: dict with sensitivity analysis data
        """
        results = {}
        for ipc in ipc_range:
            for n_modes in n_modes_range:
                key = f"ipc{ipc}_modes{n_modes}"
                guidance_steps = self.compute_guidance_steps(ipc, n_modes, complexity_score=0.5)
                t_stop = self.compute_stop(ipc, n_modes, complexity_score=0.5)
                results[key] = {
                    "ipc": ipc,
                    "n_modes": n_modes,
                    "guidance_steps": guidance_steps,
                    "t_stop": t_stop,
                    "stop_ratio": t_stop / self.t_max,
                }
        return results
