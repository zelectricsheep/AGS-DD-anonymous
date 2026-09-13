#!/usr/bin/env python3
"""Enhanced class-level mechanism analysis with 2-factor optimal CAGS data.

Trains unguided and CAGS models on the OPTIMAL 2-factor data (wl_r2_c2) at 2000 epochs,
extracts per-class accuracy, then computes correlations and quartile analysis.

Key improvements over the original:
1. Uses correct data: wl_r2_c2 (alpha=0, beta=0.5, gamma=0.5, delta=0) and unguided
2. Uses 2000 epochs (matching main results)
3. Computes 2-factor complexity correlations
4. Adds quartile binning analysis
5. Compares lambda distributions (2-factor vs 4-factor)
6. Generates publication-ready plots

Outputs:
  - class_level_analysis_v2.json
  - figures/class_level_mechanism.pdf (4-panel figure)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_eval_v3 import (
    create_model, load_images_to_tensor, load_val_data, rand_bbox, Lighting
)


def train_and_eval_per_class(train_dir, val_dir, class_names, num_classes, device,
                              img_size=224, epochs=2000, batch_size=128, seed=0,
                              depth=6, arch="convnet", norm_type="instance",
                              lr=0.1, weight_decay=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading training images from {train_dir}...")
    train_images, train_labels = load_images_to_tensor(train_dir, class_names, img_size)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    print(f"  Loaded {len(train_images)} training images")

    print(f"  Loading validation images from {val_dir}...")
    val_images, val_labels = load_val_data(val_dir, class_names, img_size)
    val_images = val_images.to(device)
    val_labels = val_labels.to(device)
    print(f"  Loaded {len(val_images)} validation images")

    model = create_model(arch, num_classes, depth=depth, norm_type=norm_type, img_size=img_size)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_images)
    rrc = transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = Lighting(alphastd=0.1)

    best_top1 = 0.0
    best_state = None
    eval_interval = max(epochs // 100, 1)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            imgs = train_images[idx].clone()
            tgt = train_labels[idx]
            imgs = rrc(imgs)
            imgs = hflip(imgs)
            imgs = color_jitter(imgs)
            imgs = lighting(imgs)
            imgs = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            lam = np.random.beta(1.0, 1.0)
            rand_idx = torch.randperm(imgs.size(0), device=device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_idx, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
            outputs = model(imgs)
            loss = criterion(outputs, tgt) * ratio + criterion(outputs, tgt[rand_idx]) * (1. - ratio)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for start in range(0, len(val_images), batch_size):
                    imgs = val_images[start:start + batch_size]
                    tgt = val_labels[start:start + batch_size]
                    outputs = model(imgs)
                    _, pred = outputs.max(1)
                    total += tgt.size(0)
                    correct += pred.eq(tgt).sum().item()
            cur_top1 = 100.0 * correct / total
            if cur_top1 > best_top1:
                best_top1 = cur_top1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 400 == 0:
                print(f"    Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s) best={best_top1:.2f}%")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    with torch.no_grad():
        for start in range(0, len(val_images), batch_size):
            imgs = val_images[start:start + batch_size]
            tgt = val_labels[start:start + batch_size]
            outputs = model(imgs)
            _, pred = outputs.max(1)
            for i in range(tgt.size(0)):
                label = tgt[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1

    per_class_acc = {}
    for c in range(num_classes):
        per_class_acc[c] = class_correct[c] / max(class_total[c], 1)

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s, best Top-1={best_top1:.2f}%")
    del model
    torch.cuda.empty_cache()
    return per_class_acc, best_top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="./class_level_analysis_v2.json")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("results/sweep_in100/complexity_stats_2factor.json") as f:
        stats = json.load(f)

    classes = sorted(int(k) for k in stats["complexity_scores"].keys())
    new_complexity = np.array([stats["complexity_scores"][str(c)] for c in classes])
    new_lambda = np.array([stats["lambda_values"][str(c)] for c in classes])
    old_complexity = np.array([stats["old_4factor_complexity"][str(c)] for c in classes])
    old_lambda = np.array([stats["old_4factor_lambda"][str(c)] for c in classes])
    entropy = np.array([stats["entropy_values"][str(c)] for c in classes])
    intra_var_norm = np.array([stats["intra_var_norms"][str(c)] for c in classes])
    mode_counts = np.array([stats["mode_counts"][str(c)] for c in classes])
    separabilities = np.array([stats["separabilities"][str(c)] for c in classes])

    with open("misc/class100.txt") as f:
        all_classes = [l.strip() for l in f.readlines()]
    class_names = all_classes[:100]

    print(f"\n=== Training Unguided model (seed {args.seed}, {args.epochs} epochs) ===")
    unguided_data = "./results/sweep_in100/unguided/dataset_0"
    unguided_acc, unguided_top1 = train_and_eval_per_class(
        unguided_data, "./data/imagenet100/val", class_names, 100, device,
        epochs=args.epochs, seed=args.seed)

    print(f"\n=== Training CAGS 2-factor model (seed {args.seed}, {args.epochs} epochs) ===")
    cags_data = "./results/sweep_in100/wl_r2_c2/dataset_0"
    cags_acc, cags_top1 = train_and_eval_per_class(
        cags_data, "./data/imagenet100/val", class_names, 100, device,
        epochs=args.epochs, seed=args.seed)

    from scipy.stats import pearsonr, spearmanr

    unguided_accs = np.array([unguided_acc[c] for c in classes])
    cags_accs = np.array([cags_acc[c] for c in classes])
    gains = cags_accs - unguided_accs

    results = {}
    results["unguided_top1"] = float(unguided_top1)
    results["cags_top1"] = float(cags_top1)
    results["overall_gain"] = float(cags_top1 - unguided_top1)
    results["seed"] = args.seed
    results["epochs"] = args.epochs

    print(f"\n{'='*60}")
    print(f"Unguided Top-1: {unguided_top1:.2f}%")
    print(f"CAGS Top-1:     {cags_top1:.2f}%")
    print(f"Overall gain:   {cags_top1 - unguided_top1:.2f}%")
    print(f"{'='*60}")

    # Correlations
    print(f"\n=== Correlations ===")
    for name, arr in [("2-factor complexity", new_complexity),
                       ("2-factor lambda", new_lambda),
                       ("4-factor complexity", old_complexity),
                       ("4-factor lambda", old_lambda),
                       ("entropy", entropy),
                       ("intra_var_norm", intra_var_norm),
                       ("separability", separabilities),
                       ("mode_count", mode_counts)]:
        r, p = pearsonr(arr, gains)
        rho, p_rho = spearmanr(arr, gains)
        results[f"{name}_vs_gain"] = {"pearson_r": float(r), "pearson_p": float(p),
                                      "spearman_rho": float(rho), "spearman_p": float(p_rho)}
        print(f"  {name}: r={r:.4f} (p={p:.4f}), rho={rho:.4f} (p={p_rho:.4f})")

    # Quartile analysis
    print(f"\n=== Quartile analysis (by 2-factor complexity) ===")
    sorted_idx = np.argsort(new_complexity)
    n = len(sorted_idx)
    q_size = n // 4
    quartile_results = []
    for q in range(4):
        if q < 3:
            q_idx = sorted_idx[q*q_size:(q+1)*q_size]
        else:
            q_idx = sorted_idx[3*q_size:]
        q_gains = gains[q_idx]
        q_ung = unguided_accs[q_idx]
        q_cags = cags_accs[q_idx]
        q_cx = new_complexity[q_idx]
        q_lam = new_lambda[q_idx]
        q_res = {
            "quartile": q + 1,
            "n_classes": len(q_idx),
            "mean_complexity": float(q_cx.mean()),
            "mean_lambda": float(q_lam.mean()),
            "mean_unguided_acc": float(q_ung.mean()),
            "mean_cags_acc": float(q_cags.mean()),
            "mean_gain": float(q_gains.mean()),
            "std_gain": float(q_gains.std()),
            "n_positive": int((q_gains > 0).sum()),
            "n_negative": int((q_gains < 0).sum()),
            "n_zero": int((q_gains == 0).sum()),
            "class_ids": [int(classes[i]) for i in q_idx],
        }
        quartile_results.append(q_res)
        print(f"  Q{q+1} (complexity={q_cx.mean():.4f}, lambda={q_lam.mean():.4f}): "
              f"ung={q_ung.mean():.4f} -> cags={q_cags.mean():.4f}, "
              f"gain={q_gains.mean():.4f}±{q_gains.std():.4f}, "
              f"+{q_res['n_positive']}/-{q_res['n_negative']}/={q_res['n_zero']}")
    results["quartile_analysis"] = quartile_results

    # Gain distribution
    print(f"\n=== Gain distribution ===")
    print(f"  Mean: {gains.mean():.4f}, Median: {np.median(gains):.4f}, Std: {gains.std():.4f}")
    print(f"  Positive: {(gains > 0).sum()}, Negative: {(gains < 0).sum()}, Zero: {(gains == 0).sum()}")
    print(f"  Max gain: {gains.max():.4f} (class {classes[gains.argmax()]})")
    print(f"  Max loss: {gains.min():.4f} (class {classes[gains.argmin()]})")
    results["gain_distribution"] = {
        "mean": float(gains.mean()),
        "median": float(np.median(gains)),
        "std": float(gains.std()),
        "n_positive": int((gains > 0).sum()),
        "n_negative": int((gains < 0).sum()),
        "n_zero": int((gains == 0).sum()),
        "max_gain": float(gains.max()),
        "max_gain_class": int(classes[gains.argmax()]),
        "max_loss": float(gains.min()),
        "max_loss_class": int(classes[gains.argmin()]),
    }

    # Lambda distribution comparison
    print(f"\n=== Lambda distribution comparison ===")
    print(f"  2-factor: range=[{new_lambda.min():.4f}, {new_lambda.max():.4f}], "
          f"std={new_lambda.std():.4f}, spread={new_lambda.max()-new_lambda.min():.4f}")
    print(f"  4-factor: range=[{old_lambda.min():.4f}, {old_lambda.max():.4f}], "
          f"std={old_lambda.std():.4f}, spread={old_lambda.max()-old_lambda.min():.4f}")
    print(f"  2-factor/4-factor std ratio: {new_lambda.std()/old_lambda.std():.2f}x")
    results["lambda_comparison"] = {
        "twofactor": {"min": float(new_lambda.min()), "max": float(new_lambda.max()),
                       "mean": float(new_lambda.mean()), "std": float(new_lambda.std()),
                       "spread": float(new_lambda.max() - new_lambda.min())},
        "fourfactor": {"min": float(old_lambda.min()), "max": float(old_lambda.max()),
                        "mean": float(old_lambda.mean()), "std": float(old_lambda.std()),
                        "spread": float(old_lambda.max() - old_lambda.min())},
        "std_ratio": float(new_lambda.std() / old_lambda.std()),
    }

    # Save raw data
    results["raw_data"] = {
        "classes": [int(c) for c in classes],
        "complexity_2factor": [float(x) for x in new_complexity],
        "lambda_2factor": [float(x) for x in new_lambda],
        "complexity_4factor": [float(x) for x in old_complexity],
        "lambda_4factor": [float(x) for x in old_lambda],
        "entropy": [float(x) for x in entropy],
        "intra_var_norm": [float(x) for x in intra_var_norm],
        "mode_counts": [int(x) for x in mode_counts],
        "separabilities": [float(x) for x in separabilities],
        "unguided_acc": [float(x) for x in unguided_accs],
        "cags_acc": [float(x) for x in cags_accs],
        "accuracy_gain": [float(x) for x in gains],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Generate plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Complexity vs Gain scatter
    ax = axes[0, 0]
    ax.scatter(new_complexity, gains, alpha=0.6, s=30, c='steelblue')
    z = np.polyfit(new_complexity, gains, 1)
    ax.plot(new_complexity, np.polyval(z, new_complexity), "r--", alpha=0.8)
    r, p = pearsonr(new_complexity, gains)
    ax.set_xlabel("Class Complexity Score (2-factor)")
    ax.set_ylabel("Per-class Accuracy Gain")
    ax.set_title(f"(a) Complexity vs Gain ($r$={r:.3f}, $p$={p:.3f})")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # Panel 2: Gain distribution histogram
    ax = axes[0, 1]
    ax.hist(gains, bins=20, color='seagreen', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
    ax.axvline(x=np.mean(gains), color='blue', linestyle='-', alpha=0.7,
               label=f'Mean={np.mean(gains):.3f}')
    ax.set_xlabel("Per-class Accuracy Gain")
    ax.set_ylabel("Number of Classes")
    ax.set_title(f"(b) Distribution of Per-class Gains "
                 f"(+{(gains > 0).sum()}/-{(gains < 0).sum()}/={(gains == 0).sum()})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: Quartile bar chart
    ax = axes[1, 0]
    q_means = [q["mean_gain"] for q in quartile_results]
    q_stds = [q["std_gain"] for q in quartile_results]
    q_labels = [f"Q{q['quartile']}\n(λ={q['mean_lambda']:.3f})" for q in quartile_results]
    bars = ax.bar(range(4), q_means, yerr=q_stds, capsize=5,
                  color=['#440154', '#3b528b', '#21908d', '#5dc863'],
                  alpha=0.8, edgecolor='white')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(q_labels)
    ax.set_xlabel("Complexity Quartile (low → high)")
    ax.set_ylabel("Mean Accuracy Gain")
    ax.set_title("(c) Gain by Complexity Quartile")
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 4: Lambda distribution comparison
    ax = axes[1, 1]
    bins = np.linspace(min(old_lambda.min(), new_lambda.min()) - 0.001,
                       max(old_lambda.max(), new_lambda.max()) + 0.001, 20)
    ax.hist(old_lambda, bins=bins, alpha=0.5, color='orange', label=f'4-factor (std={old_lambda.std():.4f})', edgecolor='white')
    ax.hist(new_lambda, bins=bins, alpha=0.5, color='steelblue', label=f'2-factor (std={new_lambda.std():.4f})', edgecolor='white')
    ax.set_xlabel("Per-class Guidance Strength ($\\lambda_c$)")
    ax.set_ylabel("Number of Classes")
    ax.set_title(f"(d) $\\lambda_c$ Distribution: 2-factor vs 4-factor "
                 f"({new_lambda.std()/old_lambda.std():.1f}× wider)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs("./figures", exist_ok=True)
    plot_path = "./figures/class_level_mechanism.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
