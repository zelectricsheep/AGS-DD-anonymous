#!/usr/bin/env python3
"""Generate D4M sensitivity curve figure.

Reads D4M evaluation results from logs and creates a publication-ready figure
showing D4M accuracy vs t_start, with reference lines for unguided and CAGS.
"""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# D4M results (t_start: (mean, std, n_seeds))
results = {}

# Parse existing 3-seed results
results[25] = (20.21, 0.39, 3)
results[40] = (23.07, 0.59, 3)

# Parse new 1-seed results
log_dir = "./logs"
for t in [5, 10, 15, 20, 30, 35, 45]:
    log_path = os.path.join(log_dir, f"eval_d4m_t{t}.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        # Look for "Mean Best Top-1: XX.XX ± YY.YY%"
        m = re.search(r"Mean Best Top-1:\s*([\d.]+)\s*±\s*([\d.]+)", content)
        if m:
            results[t] = (float(m.group(1)), float(m.group(2)), 1)
        else:
            # Try single seed format: "Best Top-1=XX.XX%"
            m2 = re.search(r"Best Top-1=([\d.]+)%", content)
            if m2:
                results[t] = (float(m2.group(1)), 0.0, 1)

# Sort by t_start
t_values = sorted(results.keys())
accs = [results[t][0] for t in t_values]
stds = [results[t][1] for t in t_values]

# Reference lines
unguided = 22.79
unguided_std = 0.18
cags = 25.70
cags_std = 0.07

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot D4M curve
ax.errorbar(t_values, accs, yerr=stds, fmt='o-', color='#d62728', markersize=7,
            linewidth=2, capsize=4, label='D4M (real-image init)', zorder=3)

# Reference lines
ax.axhline(y=unguided, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'Unguided ({unguided}%)')
ax.axhline(y=cags, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'CAGS optimal ({cags}%)')

# Shade reference regions
ax.fill_between(t_values, cags - cags_std, cags + cags_std, color='#1f77b4', alpha=0.1)
ax.fill_between(t_values, unguided - unguided_std, unguided + unguided_std, color='#2ca02c', alpha=0.1)

# Labels
ax.set_xlabel(r'Denoising start timestep ($t_{\mathrm{start}}$)', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('D4M Sensitivity to Denoising Start Timestep', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 50)
ax.set_ylim(14, 27)

plt.tight_layout()
os.makedirs("./figures", exist_ok=True)
plot_path = "./figures/d4m_sensitivity.pdf"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved to {plot_path}")
plt.close()

# Also print results table
print(f"\n{'t_start':>8} {'Top-1 (%)':>12} {'±':>6} {'seeds':>6}")
print("-" * 40)
for t in t_values:
    mean, std, n = results[t]
    print(f"{t:>8} {mean:>12.2f} {std:>6.2f} {n:>6}")
