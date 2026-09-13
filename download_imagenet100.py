import os
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

OUT_DIR = "./data/imagenet100"
CLASSES_FILE = "./misc/class100.txt"

with open(CLASSES_FILE) as f:
    classes = [c.strip() for c in f.readlines() if c.strip()]
print(f"Classes: {len(classes)}")
cls_to_idx = {c: i for i, c in enumerate(classes)}

os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading validation parquet...")
val_path = hf_hub_download(
    "clane9/imagenet-100", "data/validation-00000-of-00001.parquet", repo_type="dataset"
)
print(f"  Downloaded: {val_path}")

val_out = os.path.join(OUT_DIR, "val")
for c in classes:
    os.makedirs(os.path.join(val_out, c), exist_ok=True)

print("Extracting validation images...")
df = pd.read_parquet(val_path)
print(f"  Val rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")

saved = 0
for idx, row in df.iterrows():
    img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
    label = row["label"]
    cls_name = classes[label]
    img = Image.open(io.BytesIO(img_bytes)) if isinstance(img_bytes, bytes) else Image.open(io.BytesIO(img_bytes))
    img.save(os.path.join(val_out, cls_name, f"{cls_name}_{idx:05d}.JPEG"))
    saved += 1
    if saved % 1000 == 0:
        print(f"  Saved {saved}/{len(df)}")

print(f"Validation: {saved} images extracted to {val_out}")

print("\nDownloading train parquet files (for CAGS clustering)...")
train_out = os.path.join(OUT_DIR, "train")
for c in classes:
    os.makedirs(os.path.join(train_out, c), exist_ok=True)

total_train = 0
for i in range(17):
    fname = f"data/train-{i:05d}-of-00017.parquet"
    local = hf_hub_download("clane9/imagenet-100", fname, repo_type="dataset")
    df = pd.read_parquet(local)
    for idx, row in df.iterrows():
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        label = row["label"]
        cls_name = classes[label]
        img = Image.open(io.BytesIO(img_bytes)) if isinstance(img_bytes, bytes) else Image.open(io.BytesIO(img_bytes))
        count = len(os.listdir(os.path.join(train_out, cls_name)))
        img.save(os.path.join(train_out, cls_name, f"{cls_name}_{count:05d}.JPEG"))
        total_train += 1
    print(f"  Shard {i}: total {total_train} images")
    os.remove(local)

print(f"\nTrain: {total_train} images extracted to {train_out}")
print(f"Val: {saved} images extracted to {val_out}")

for c in classes:
    n_train = len(os.listdir(os.path.join(train_out, c)))
    n_val = len(os.listdir(os.path.join(val_out, c)))
    if n_train < 10 or n_val < 1:
        print(f"  WARNING: {c} train={n_train} val={n_val}")

print("\nDone!")
