#!/usr/bin/env python3
"""
Collect and summarize all experiment results from logs and result directories.
Outputs a comprehensive JSON summary and prints a formatted table.
"""

import os
import re
import json
import glob
from collections import defaultdict

BASE_DIR = "."
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def parse_eval_log(log_path):
    """Parse a quick_eval log file to extract Top-1 and Top-5 results."""
    if not os.path.isfile(log_path):
        return None
    
    with open(log_path, "r") as f:
        content = f.read()
    
    # Look for "Mean Top-1: XX.XX ± Y.YY%"
    m1 = re.search(r"Mean Top-1:\s*([\d.]+)\s*±\s*([\d.]+)%", content)
    m5 = re.search(r"Mean Top-5:\s*([\d.]+)\s*±\s*([\d.]+)%", content)
    
    # Also extract per-seed results
    seeds = []
    for sm in re.finditer(r"Seed\s+(\d+):\s*Top-1=([\d.]+)%,\s*Top-5=([\d.]+)%", content):
        seeds.append({
            "seed": int(sm.group(1)),
            "top1": float(sm.group(2)),
            "top5": float(sm.group(3)),
        })
    
    if m1:
        return {
            "top1_mean": float(m1.group(1)),
            "top1_std": float(m1.group(2)),
            "top5_mean": float(m5.group(1)) if m5 else 0.0,
            "top5_std": float(m5.group(2)) if m5 else 0.0,
            "seeds": seeds,
        }
    return None


def main():
    results = {"10class": {}, "100class": {}}
    
    # Parse all eval logs
    for log_file in sorted(glob.glob(os.path.join(LOGS_DIR, "*_eval.log"))):
        basename = os.path.basename(log_file)
        # Parse: 10class_fixed_l0.0_eval.log or 100class_unguided_eval.log
        m = re.match(r"(10class|100class)_(.+)_eval\.log", basename)
        if not m:
            continue
        
        nclass_key = m.group(1)
        tag = m.group(2)
        eval_result = parse_eval_log(log_file)
        
        if eval_result:
            results[nclass_key][tag] = eval_result
            print(f"  {nclass_key}/{tag}: Top-1={eval_result['top1_mean']:.2f}±{eval_result['top1_std']:.2f}%")
    
    # Also collect any sweep result JSONs
    for sweep_dir in ["sweep_in10", "sweep_in100"]:
        sweep_path = os.path.join(RESULTS_DIR, sweep_dir)
        if not os.path.isdir(sweep_path):
            continue
        for json_file in sorted(glob.glob(os.path.join(sweep_path, "sweep_*.json"))):
            with open(json_file) as f:
                try:
                    data = json.load(f)
                    nclass_key = "10class" if "in10" in sweep_dir else "100class"
                    for config_name, config_data in data.items():
                        if "top1_mean" in config_data:
                            tag = config_name.replace("high_noise_", "").replace("_d25", "")
                            if tag not in results[nclass_key]:
                                results[nclass_key][tag] = {
                                    "top1_mean": config_data.get("top1_mean", 0),
                                    "top1_std": config_data.get("top1_std", 0),
                                    "top5_mean": 0,
                                    "top5_std": 0,
                                    "seeds": [],
                                }
                except json.JSONDecodeError:
                    pass
    
    # Save summary JSON
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    # Print formatted table
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    for nclass_key, label in [("10class", "10-class ImageNet-100"), ("100class", "100-class ImageNet-100")]:
        print(f"\n--- {label} ---")
        print(f"{'Config':<30} {'Top-1 (%)':<20} {'Top-5 (%)':<20}")
        print("-" * 70)
        
        configs = results[nclass_key]
        # Sort: unguided first, then fixed_l*, then cags_*, then tags_*
        def sort_key(item):
            tag = item[0]
            if "unguided" in tag:
                return (0, tag)
            elif "fixed_l" in tag:
                return (1, tag)
            elif "cags" in tag:
                return (2, tag)
            elif "tags" in tag:
                return (3, tag)
            return (4, tag)
        
        for tag, data in sorted(configs.items(), key=sort_key):
            t1 = f"{data['top1_mean']:.2f} ± {data['top1_std']:.2f}"
            t5 = f"{data['top5_mean']:.2f} ± {data['top5_std']:.2f}"
            print(f"  {tag:<28} {t1:<20} {t5:<20}")
    
    # Find best configs
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    
    for nclass_key, label in [("10class", "10-class"), ("100class", "100-class")]:
        configs = results[nclass_key]
        if not configs:
            continue
        
        best_tag = max(configs, key=lambda k: configs[k]["top1_mean"])
        best = configs[best_tag]
        print(f"  {label}: {best_tag} — Top-1={best['top1_mean']:.2f}±{best['top1_std']:.2f}%")
        
        # Best fixed lambda
        fixed_configs = {k: v for k, v in configs.items() if "fixed_l" in k and "unguided" not in k}
        if fixed_configs:
            best_fixed = max(fixed_configs, key=lambda k: fixed_configs[k]["top1_mean"])
            print(f"    Best fixed λ: {best_fixed} — {fixed_configs[best_fixed]['top1_mean']:.2f}%")
        
        # Best CAGS
        cags_configs = {k: v for k, v in configs.items() if "cags" in k}
        if cags_configs:
            best_cags = max(cags_configs, key=lambda k: cags_configs[k]["top1_mean"])
            print(f"    Best CAGS: {best_cags} — {cags_configs[best_cags]['top1_mean']:.2f}%")
    
    print()


if __name__ == "__main__":
    main()
