import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE = {
    "low_const": "#0072B2",
    "low_cosine": "#D55E00",
    "high_const": "#009E73",
    "high_cosine": "#CC79A7",
}


def series(cfgs, window, schedule):
    pts = [(c["duration"], c["top1_mean"], c["top1_std"])
           for c in cfgs.values()
           if c["window"] == window and c["schedule"] == schedule]
    pts.sort()
    if not pts:
        return None
    d, m, s = zip(*pts)
    return np.array(d), np.array(m), np.array(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str,
                        default="./results/sweep/summary.json")
    parser.add_argument("--out", type=str, default="./figures")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    cfgs = summary["configs"]
    tests = summary["paired_tests"]
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for w, s, key, lbl, mk in [
        ("low_noise", "constant", "low_const", "Low-noise, constant", "o"),
        ("low_noise", "cosine", "low_cosine", "Low-noise, TAGS (cosine)", "s"),
        ("high_noise", "constant", "high_const", "High-noise, constant", "^"),
        ("high_noise", "cosine", "high_cosine", "High-noise, TAGS (cosine)", "v"),
    ]:
        r = series(cfgs, w, s)
        if r is None:
            continue
        d, m, sd = r
        ax.errorbar(d, m, yerr=sd, marker=mk, capsize=3, lw=1.8,
                    ms=6, color=OKABE[key], label=lbl,
                    ls="-" if s == "constant" else "--")
    ax.axvspan(23, 27, color="0.85", alpha=0.5, zorder=0)
    ax.text(25, ax.get_ylim()[0] + 0.6, "crossover", ha="center",
            fontsize=8, color="0.3")
    ax.set_xlabel("Guidance duration (denoising steps)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("(a) Duration $\\times$ window $\\times$ schedule")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    ds, deltas, errs, cols = [], [], [], []
    for d in sorted({c["duration"] for c in cfgs.values()}):
        k = f"low_cosine_vs_const_d{d}"
        if k not in tests:
            continue
        t = tests[k]
        ds.append(d)
        deltas.append(t["delta"])
        errs.append(abs(t["delta"] / t["t"]) if t["t"] else 0)
        cols.append(OKABE["low_cosine"] if t["p"] < 0.05 else "0.6")
    x = np.arange(len(ds))
    ax.bar(x, deltas, yerr=errs, capsize=4, color=cols, width=0.6)
    for i, d in enumerate(ds):
        t = tests[f"low_cosine_vs_const_d{d}"]
        star = "***" if t["p"] < 0.001 else "**" if t["p"] < 0.01 \
            else "*" if t["p"] < 0.05 else "n.s."
        ax.text(i, deltas[i] + errs[i] + 0.12, star, ha="center", fontsize=8)
    hk = [f"high_cosine_vs_const_d{d}" for d in ds
          if f"high_cosine_vs_const_d{d}" in tests]
    if hk:
        hx = [ds.index(int(k.split("_d")[1])) for k in hk]
        hv = [tests[k]["delta"] for k in hk]
        ax.plot(hx, hv, "x", ms=9, mew=2, color=OKABE["high_const"],
                label="High-noise (no gain)")
        ax.legend(fontsize=7.5, loc="upper left")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in ds])
    ax.set_xlabel("Guidance duration (denoising steps)")
    ax.set_ylabel("TAGS gain over constant (pp)")
    ax.set_title("(b) TAGS gain grows with duration")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(args.out, f"duration_sweep.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
