import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def load_all(base):
    configs = {}
    for path in sorted(glob.glob(os.path.join(base, "sweep_*_results.json"))):
        if path.endswith("_noise_results.json"):
            continue
        with open(path) as f:
            data = json.load(f)
        for name, res in data.items():
            configs[name] = res
    return configs


def runs_map(res):
    out = {}
    for r in res.get("runs", res.get("datasets", [])):
        key = (r["dataset"], r.get("seed", 0))
        out[key] = r["top1"]
    return out


def paired_test(a_res, b_res):
    am, bm = runs_map(a_res), runs_map(b_res)
    keys = sorted(set(am) & set(bm))
    if len(keys) < 3:
        return None
    a = np.array([am[k] for k in keys])
    b = np.array([bm[k] for k in keys])
    t, p = stats.ttest_rel(a, b)
    d = a - b
    return {
        "n": len(keys),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(d.mean()),
        "t": float(t),
        "p": float(p),
        "wins": int((d > 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="./results/sweep")
    parser.add_argument("--out", type=str, default="./results/sweep/summary.json")
    args = parser.parse_args()

    configs = load_all(args.base)

    print("=" * 74)
    print("ALL CONFIGS (multi-seed)")
    print("=" * 74)
    print(f"{'config':<30} {'Top-1':>16} {'runs':>6}  per-seed")
    for name in sorted(configs, key=lambda n: (configs[n].get("window", ""),
                                               configs[n].get("schedule", ""),
                                               configs[n].get("duration", 0))):
        res = configs[name]
        n = len(res.get("runs", res.get("datasets", [])))
        ps = res.get("per_seed", {})
        ps_str = ", ".join(f"s{k}={v['mean']:.2f}" for k, v in sorted(ps.items()))
        print(f"{name:<30} {res['top1_mean']:>7.2f} ± {res['top1_std']:<5.2f} "
              f"{n:>6}  {ps_str}")

    print()
    print("=" * 74)
    print("DURATION CURVE BY WINDOW x SCHEDULE")
    print("=" * 74)
    groups = {}
    for name, res in configs.items():
        key = (res.get("window", "?"), res.get("schedule", "constant"))
        groups.setdefault(key, []).append((res.get("duration", 0), res))
    for key in sorted(groups):
        items = sorted(groups[key])
        cells = [f"d{d}={r['top1_mean']:.2f}" for d, r in items]
        best_d, best_r = max(items, key=lambda x: x[1]["top1_mean"])
        print(f"  {key[0]:<11} {key[1]:<9} " + "  ".join(cells)
              + f"   -> peak d{best_d} ({best_r['top1_mean']:.2f}%)")

    print()
    print("=" * 74)
    print("PAIRED TESTS")
    print("=" * 74)
    tests = {}
    for d in sorted({r.get("duration") for r in configs.values()}):
        for w in ("low", "high"):
            cos, con = f"{w}_noise_cosine_d{d}", f"{w}_noise_const_d{d}"
            if cos in configs and con in configs:
                r = paired_test(configs[cos], configs[con])
                if r:
                    tests[f"{w}_cosine_vs_const_d{d}"] = r
                    sig = "SIGNIFICANT" if r["p"] < 0.05 else "not significant"
                    print(f"  TAGS @ {w:>4}-noise d{d}: "
                          f"{r['mean_a']:.2f} vs {r['mean_b']:.2f} "
                          f"= {r['delta']:+.2f} pp, p={r['p']:.4f} [{sig}] "
                          f"wins {r['wins']}/{r['n']}")
        lo, hi = f"low_noise_const_d{d}", f"high_noise_const_d{d}"
        if lo in configs and hi in configs:
            r = paired_test(configs[lo], configs[hi])
            if r:
                tests[f"low_vs_high_d{d}"] = r
                sig = "SIGNIFICANT" if r["p"] < 0.05 else "not significant"
                print(f"  low vs high noise @ d{d}:        "
                      f"{r['mean_a']:.2f} vs {r['mean_b']:.2f} "
                      f"= {r['delta']:+.2f} pp, p={r['p']:.4f} [{sig}] "
                      f"wins {r['wins']}/{r['n']}")

    summary = {
        "configs": {
            n: {
                "window": r.get("window"),
                "schedule": r.get("schedule", "constant"),
                "duration": r.get("duration"),
                "top1_mean": r["top1_mean"],
                "top1_std": r["top1_std"],
                "n_runs": len(r.get("runs", r.get("datasets", []))),
                "per_seed": r.get("per_seed", {}),
                "gen_time_s": r.get("gen_time_s", 0),
            }
            for n, r in configs.items()
        },
        "paired_tests": tests,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
