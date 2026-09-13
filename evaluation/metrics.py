"""
Evaluation metrics for AGS-DD dataset distillation.

Computes:
  - Top-1 / Top-5 accuracy
  - FID (Fréchet Inception Distance) for generation quality
  - Cross-architecture generalization gap
  - Training efficiency metrics
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import os
import json
import time


class DistillationMetrics:
    """Compute metrics for distilled dataset evaluation."""

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Compute top-k accuracy."""
        max_k = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    @staticmethod
    def compute_accuracy_stats(accuracies):
        """Compute mean and std from a list of accuracy values."""
        accs = np.array(accuracies)
        return {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "median": float(np.median(accs)),
            "values": [float(a) for a in accs],
        }

    @staticmethod
    def cross_architecture_gap(results):
        """
        Compute the gap between distillation architecture and
        cross-architecture evaluation.

        Args:
            results: dict {arch_name: accuracy_stats}

        Returns:
            gap: dict with gap statistics
        """
        if not results:
            return {}

        archs = list(results.keys())
        accs = [results[a]["mean"] for a in archs]
        base_acc = accs[0]
        gaps = [base_acc - a for a in accs[1:]]

        return {
            "base_arch": archs[0],
            "base_acc": base_acc,
            "cross_arch_accs": {archs[i]: accs[i] for i in range(1, len(archs))},
            "cross_arch_gaps": {archs[i]: gaps[i - 1] for i in range(len(gaps))},
            "avg_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap": float(np.max(gaps)) if gaps else 0.0,
        }

    @staticmethod
    def efficiency_metrics(gen_time, eval_time, gpu_memory, num_images):
        """Compute efficiency metrics."""
        return {
            "generation_time_s": gen_time,
            "eval_time_s": eval_time,
            "total_time_s": gen_time + eval_time,
            "gpu_memory_mb": gpu_memory,
            "images_per_second": num_images / max(gen_time, 1e-6),
            "total_gpu_hours": (gen_time + eval_time) / 3600,
        }

    @staticmethod
    def save_results(results, save_path):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def load_results(load_path):
        """Load results from JSON file."""
        with open(load_path, "r") as f:
            return json.load(f)

    @staticmethod
    def format_results_table(results, method_name="AGS-DD"):
        """Format results as a markdown table for the paper."""
        lines = []
        lines.append(f"## {method_name} Results\n")
        lines.append("| Dataset | IPC | Architecture | Top-1 Acc (%) | Std |")
        lines.append("|---------|-----|-------------|-------------|-----|")

        for dataset_name, dataset_results in results.items():
            for ipc, ipc_results in dataset_results.items():
                for arch, arch_results in ipc_results.items():
                    mean = arch_results["mean"]
                    std = arch_results["std"]
                    lines.append(
                        f"| {dataset_name} | {ipc} | {arch} | {mean:.1f} | {std:.1f} |"
                    )

        return "\n".join(lines)

    @staticmethod
    def format_comparison_table(all_results, baseline_name="MGD3"):
        """Format comparison table between methods."""
        lines = []
        lines.append("## Method Comparison\n")
        lines.append("| Dataset | IPC | Method | Top-1 Acc (%) | Std |")
        lines.append("|---------|-----|--------|-------------|-----|")

        for method_name, method_results in all_results.items():
            for dataset_name, dataset_results in method_results.items():
                for ipc, ipc_results in dataset_results.items():
                    for arch, arch_results in ipc_results.items():
                        mean = arch_results["mean"]
                        std = arch_results["std"]
                        lines.append(
                            f"| {dataset_name} | {ipc} | {method_name} | {mean:.1f} | {std:.1f} |"
                        )

        return "\n".join(lines)


class AblationResults:
    """Store and format ablation study results."""

    def __init__(self):
        self.configs = {}

    def add_config(self, config_name, use_cags, use_iast, use_tags):
        """Add an ablation configuration."""
        self.configs[config_name] = {
            "use_cags": use_cags,
            "use_iast": use_iast,
            "use_tags": use_tags,
            "results": {},
        }

    def add_result(self, config_name, dataset, ipc, arch, accuracy, std):
        """Add a result for a configuration."""
        if config_name not in self.configs:
            return
        key = f"{dataset}_{ipc}_{arch}"
        self.configs[config_name]["results"][key] = {
            "accuracy": accuracy,
            "std": std,
        }

    def format_table(self, dataset, ipc, arch):
        """Format ablation table for a specific setting."""
        lines = []
        lines.append(f"## Ablation Study ({dataset}, IPC={ipc}, {arch})\n")
        lines.append("| Config | CAGS | IAST | TAGS | Acc (%) | Std | Delta |")
        lines.append("|--------|------|------|------|---------|-----|-------|")

        baseline_acc = None
        key = f"{dataset}_{ipc}_{arch}"

        for config_name, config_data in self.configs.items():
            if key in config_data["results"]:
                acc = config_data["results"][key]["accuracy"]
                std = config_data["results"][key]["std"]
                cags = "Y" if config_data["use_cags"] else "N"
                iast = "Y" if config_data["use_iast"] else "N"
                tags = "Y" if config_data["use_tags"] else "N"

                if config_name == "baseline":
                    baseline_acc = acc
                    delta = "-"
                else:
                    delta = f"+{acc - baseline_acc:.1f}" if baseline_acc else "-"

                lines.append(
                    f"| {config_name} | {cags} | {iast} | {tags} | {acc:.1f} | {std:.1f} | {delta} |"
                )

        return "\n".join(lines)
