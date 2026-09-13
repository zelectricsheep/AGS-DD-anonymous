"""
Cross-architecture evaluation for AGS-DD.

Evaluates distilled datasets across multiple architectures to measure
cross-architecture generalization, a key metric for ICLR reviewers.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch

from .train_eval import Trainer, ARCH_REGISTRY
from .metrics import DistillationMetrics


def run_cross_arch_evaluation(
    train_dir,
    test_dir,
    class_names,
    num_classes,
    arch_list,
    img_size=224,
    epochs=2000,
    num_seeds=5,
    save_dir="./results/cross_arch",
    dataset_name="imagenette",
    ipc=50,
):
    """
    Run cross-architecture evaluation.

    Args:
        train_dir: Path to generated dataset
        test_dir: Path to real test dataset
        class_names: List of class name strings
        num_classes: Number of classes
        arch_list: List of architecture names to evaluate
        img_size: Image size
        epochs: Training epochs
        num_seeds: Number of random seeds
        save_dir: Directory to save results
        dataset_name: Dataset name for result files
        ipc: Images per class

    Returns:
        all_results: dict {arch_name: accuracy_stats}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(device=device, save_dir=save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    for arch_name in arch_list:
        print(f"\n{'='*70}")
        print(f"Cross-architecture evaluation: {arch_name}")
        print(f"{'='*70}")

        if arch_name not in ARCH_REGISTRY:
            print(f"  Warning: Architecture {arch_name} not found, skipping.")
            continue

        result = trainer.run_multi_seed(
            arch_name=arch_name,
            train_data_dir=train_dir,
            test_data_dir=test_dir,
            class_names=class_names,
            num_classes=num_classes,
            num_seeds=num_seeds,
            img_size=img_size,
            epochs=epochs,
        )

        all_results[arch_name] = result["top1_stats"]

        # Save individual result
        result_path = os.path.join(
            save_dir, f"{dataset_name}_ipc{ipc}_{arch_name}.json"
        )
        DistillationMetrics.save_results(result, result_path)

        print(f"\n{arch_name}: Top-1 = {result['top1_stats']['mean']:.1f} ± "
              f"{result['top1_stats']['std']:.1f}%")

    # Compute cross-architecture gap
    gap_analysis = DistillationMetrics.cross_architecture_gap(all_results)

    # Save combined results
    combined = {
        "dataset": dataset_name,
        "ipc": ipc,
        "arch_results": all_results,
        "gap_analysis": gap_analysis,
        "arch_list": arch_list,
    }

    combined_path = os.path.join(
        save_dir, f"{dataset_name}_ipc{ipc}_cross_arch.json"
    )
    DistillationMetrics.save_results(combined, combined_path)

    # Print summary table
    print(f"\n{'='*70}")
    print("Cross-Architecture Summary")
    print(f"{'='*70}")
    print(f"{'Architecture':<20} {'Top-1 (%)':<15} {'Std':<10}")
    print(f"{'-'*45}")
    for arch, stats in all_results.items():
        print(f"{arch:<20} {stats['mean']:<15.1f} {stats['std']:<10.1f}")

    if gap_analysis:
        print(f"\nAverage cross-architecture gap: {gap_analysis['avg_gap']:.1f}%")
        print(f"Maximum cross-architecture gap: {gap_analysis['max_gap']:.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Cross-architecture evaluation")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="./results/cross_arch")
    parser.add_argument("--dataset-name", type=str, default="imagenette")
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--arch-list", type=str, nargs="+",
                        default=["convnet7", "resnet18", "vit_tiny", "swin_tiny", "deit_tiny"])

    args = parser.parse_args()

    # Get class names
    class_names = sorted(os.listdir(args.train_dir))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(args.train_dir, c))]

    run_cross_arch_evaluation(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        class_names=class_names,
        num_classes=args.num_classes,
        arch_list=args.arch_list,
        img_size=args.img_size,
        epochs=args.epochs,
        num_seeds=args.num_seeds,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name,
        ipc=args.ipc,
    )


if __name__ == "__main__":
    main()
