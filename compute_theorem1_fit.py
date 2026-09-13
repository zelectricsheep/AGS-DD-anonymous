#!/usr/bin/env python3
"""Compute Theorem 1 adaptivity gap using RAW (unnormalized) values.

The CAGS pipeline applies min-max normalization, which makes lambda distributions
not comparable across class counts. Instead, we compute the theoretical optimal
lambda* = v_c / (beta * sigma^2_c) directly from raw features, and compute the
adaptivity gap Delta = (1/C) sum (sigma^2_c / 2)(lambda*_c - bar_lambda)^2.

We also compute several scale-invariant proxies:
  - Fraction of classes with K > 2 (mode structure diversity)
  - Coefficient of variation of raw entropy
  - Coefficient of variation of sigma^2
  - Gini coefficient of sigma^2
"""
import pickle, numpy as np, os, json

def compute_raw_stats(cache, eps=0.01):
    """Compute raw (unnormalized) per-class statistics."""
    class_ids = list(cache.features_per_class.keys())
    
    sigma2 = {}
    H_raw = {}  # Raw normalized entropy (0 to 1)
    v_raw = {}  # Raw intra-class variance
    K = {}
    log_pi_var = {}  # Alternative: variance of log proportions
    
    for cls_id in class_ids:
        if cls_id in cache.mode_id_per_class:
            labels = cache.mode_id_per_class[cls_id]
            k = cache.mode_counts.get(cls_id, 2)
            proportions = np.bincount(labels, minlength=k) / len(labels)
            proportions = proportions[proportions > 0]
            
            log_pi = np.log(proportions + 1e-10)
            s2 = float(np.mean(log_pi**2) - np.mean(log_pi)**2)
            sigma2[cls_id] = max(s2, eps)  # Floor to avoid div-by-zero
            
            entropy = -np.sum(proportions * np.log(proportions + 1e-8))
            H_raw[cls_id] = entropy / np.log(k) if k > 1 else 0.0
            K[cls_id] = k
        else:
            sigma2[cls_id] = eps
            H_raw[cls_id] = 0.5
            K[cls_id] = 2
        
        v_raw[cls_id] = max(cache.intra_variances.get(cls_id, 0.0), eps)
    
    return class_ids, sigma2, H_raw, v_raw, K

def gini(x):
    """Compute Gini coefficient."""
    x = np.sort(np.array(x))
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1)) * x)) / (n * np.sum(x)) - (n + 1) / n

def main():
    cache_paths = {
        10: 'results/sweep_in10/cluster_cache.pkl',
        100: 'results/sweep_in100/cluster_cache.pkl',
        200: 'results/sweep_in200/cluster_cache.pkl',
        1000: 'results/in1k/cluster_cache.pkl',
    }
    
    actual_gains = {
        10: 0.0,
        100: 2.91,
        200: 3.30,
        1000: 3.58,
    }
    
    results = {}
    
    for n_classes, path in cache_paths.items():
        if not os.path.exists(path):
            continue
        
        print(f"\n=== {n_classes} classes ===")
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        
        class_ids, sigma2, H_raw, v_raw, K = compute_raw_stats(cache)
        
        s2_vals = np.array([sigma2[c] for c in class_ids])
        H_vals = np.array([H_raw[c] for c in class_ids])
        v_vals = np.array([v_raw[c] for c in class_ids])
        K_vals = np.array([K[c] for c in class_ids])
        
        # Theoretical optimal lambda* = v_c / (beta * sigma^2_c), beta=1
        lambda_star = v_vals / s2_vals  # beta=1
        lambda_bar = lambda_star.mean()
        
        # Adaptivity gap: Delta = (1/C) sum (sigma^2/2)(lambda* - bar)^2
        delta = np.mean(s2_vals / 2.0 * (lambda_star - lambda_bar)**2)
        
        # Also compute with beta implicit (scale-invariant version)
        # Delta / (v_mean^2 / 2) = mean[sigma^2 * (1/sigma^2 - mean[1/sigma^2])^2] 
        # This is scale-invariant in v
        inv_s2 = 1.0 / s2_vals
        delta_v_invariant = np.mean(s2_vals * (inv_s2 - inv_s2.mean())**2) / 2.0
        
        # Scale-invariant proxies
        frac_K_gt2 = float(np.mean(K_vals > 2))
        cv_H = H_vals.std() / H_vals.mean()  # Coefficient of variation
        cv_s2 = s2_vals.std() / s2_vals.mean()
        gini_s2 = gini(s2_vals)
        gini_H = gini(H_vals)
        entropy_var = H_vals.var()
        s2_mean = s2_vals.mean()
        
        print(f"  K>2 fraction: {frac_K_gt2:.3f}")
        print(f"  sigma^2: mean={s2_mean:.4f}, std={s2_vals.std():.4f}, CV={cv_s2:.3f}, Gini={gini_s2:.3f}")
        print(f"  H: mean={H_vals.mean():.4f}, std={H_vals.std():.4f}, CV={cv_H:.3f}")
        print(f"  lambda*: mean={lambda_bar:.2f}, std={lambda_star.std():.2f}, CV={lambda_star.std()/lambda_bar:.3f}")
        print(f"  Delta (raw): {delta:.4f}")
        print(f"  Delta (v-invariant): {delta_v_invariant:.6f}")
        print(f"  Actual gain: {actual_gains[n_classes]:.2f}%")
        
        results[n_classes] = {
            'n_classes': len(class_ids),
            'frac_K_gt2': frac_K_gt2,
            's2_mean': float(s2_mean),
            's2_std': float(s2_vals.std()),
            'cv_s2': float(cv_s2),
            'gini_s2': float(gini_s2),
            'cv_H': float(cv_H),
            'gini_H': float(gini_H),
            'entropy_var': float(entropy_var),
            'cv_lambda_star': float(lambda_star.std() / lambda_bar),
            'delta_raw': float(delta),
            'delta_v_invariant': float(delta_v_invariant),
            'actual_gain': actual_gains[n_classes],
            'K_dist': {int(k): int(c) for k, c in zip(*np.unique(K_vals, return_counts=True))},
        }
    
    # Find best predictor
    print("\n\n=== CORRELATION WITH ACTUAL GAINS ===")
    Cs = sorted(results.keys())
    gains = np.array([results[c]['actual_gain'] for c in Cs])
    
    predictors = ['frac_K_gt2', 's2_mean', 'cv_s2', 'gini_s2', 'cv_H', 'gini_H',
                  'entropy_var', 'cv_lambda_star', 'delta_raw', 'delta_v_invariant']
    
    print(f"{'Predictor':>20} {'r':>8} {'r^2':>8} {'Monotonic?':>12}")
    print("-" * 55)
    
    for pred in predictors:
        vals = np.array([results[c][pred] for c in Cs])
        r = np.corrcoef(vals, gains)[0, 1]
        # Check monotonicity
        mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1)) or \
               all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        print(f"{pred:>20} {r:>8.3f} {r**2:>8.3f} {'YES' if mono else 'no':>12}")
    
    with open('theorem1_fit_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to theorem1_fit_data.json")

if __name__ == '__main__':
    main()
