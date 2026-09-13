"""Extract CIFAR-10 to ImageFolder format for SD-based generation."""
import os
import pickle
import numpy as np
from PIL import Image

cifar_dir = "./data/cifar10"
out_dir = "./data/cifar10_img"

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Extract tar.gz
import tarfile
tar_path = os.path.join(cifar_dir, "cifar-10-python.tar.gz")
if os.path.exists(tar_path):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(cifar_dir)
    print("Extraction complete")

# Load training data
train_dir = os.path.join(out_dir, "train")
val_dir = os.path.join(out_dir, "val")

for d in [train_dir, val_dir]:
    for c in class_names:
        os.makedirs(os.path.join(d, c), exist_ok=True)

# Load training batches
cifar_batches_dir = os.path.join(cifar_dir, "cifar-10-batches-py")
train_files = [f"data_batch_{i}" for i in range(1, 6)]
test_file = "test_batch"

# Extract training images (use first 4 batches for train, last batch for val)
for batch_idx, batch_file in enumerate(train_files[:4]):
    print(f"Loading {batch_file}...")
    with open(os.path.join(cifar_batches_dir, batch_file), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']  # (10000, 3072)
    labels = batch[b'labels']
    for i in range(len(data)):
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img_pil = Image.fromarray(img)
        cls = class_names[labels[i]]
        img_pil.save(os.path.join(train_dir, cls, f"{batch_idx * 10000 + i}.png"))

# Use 5th batch + test batch as validation
val_offset = 0
for batch_file in [train_files[4], test_file]:
    print(f"Loading {batch_file} for validation...")
    with open(os.path.join(cifar_batches_dir, batch_file), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    for i in range(len(data)):
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img_pil = Image.fromarray(img)
        cls = class_names[labels[i]]
        img_pil.save(os.path.join(val_dir, cls, f"{val_offset + i}.png"))
    val_offset += len(data)

print(f"Done! Train: {sum(len(os.listdir(os.path.join(train_dir, c))) for c in class_names)} images")
print(f"Val: {sum(len(os.listdir(os.path.join(val_dir, c))) for c in class_names)} images")
