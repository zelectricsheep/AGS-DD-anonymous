#!/usr/bin/env python3
"""Prepare real images baseline: copy 10 real images per class from IN-100 train."""
import os, shutil, random

src_base = "./data/imagenet100/train"
dst_base = "./results/sweep_in100/real_images/dataset_0"

with open("./misc/class100.txt") as f:
    classes = [l.strip() for l in f.readlines() if l.strip()]

os.makedirs(dst_base, exist_ok=True)
total = 0
for cls in classes:
    src_dir = os.path.join(src_base, cls)
    dst_dir = os.path.join(dst_base, cls)
    os.makedirs(dst_dir, exist_ok=True)
    
    files = sorted(os.listdir(src_dir))
    selected = files[:10]  # Take first 10 images per class
    
    for fname in selected:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            total += 1

print(f"Copied {total} real images ({len(classes)} classes × 10 IPC)")
print(f"Saved to {dst_base}")
