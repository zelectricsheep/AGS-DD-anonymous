"""
Training evaluation for AGS-DD.

Trains evaluation networks (ConvNet, ResNet, ViT, etc.) on the distilled
dataset and evaluates on the real test set. Supports multiple random seeds
and cross-architecture evaluation.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from tqdm import tqdm

from .metrics import DistillationMetrics


# ---------------------------------------------------------------------------
# Network architectures for evaluation
# ---------------------------------------------------------------------------

class ConvNet(nn.Module):
    """Standard ConvNet used in dataset distillation literature."""

    def __init__(self, channel=128, num_classes=10, depth=6, norm="instance"):
        super().__init__()
        self.features = nn.ModuleList()
        in_ch = 3
        for i in range(depth):
            out_ch = channel
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            if norm == "instance":
                norm_layer = nn.InstanceNorm2d(out_ch)
            elif norm == "batch":
                norm_layer = nn.BatchNorm2d(out_ch)
            else:
                norm_layer = nn.Identity()
            block = nn.Sequential(
                conv, norm_layer, nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
            )
            self.features.append(block)
            in_ch = out_ch
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def get_resnet18(num_classes, pretrained=False):
    """ResNet-18 for cross-architecture evaluation."""
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet10_ap(num_classes):
    """ResNet-10 with average pooling (used in MGD3)."""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_vit_tiny(num_classes, img_size=224):
    """ViT-Tiny for cross-architecture evaluation."""
    try:
        from torchvision.models.vision_transformer import vit_t_16
        model = vit_t_16(pretrained=False, image_size=img_size, num_classes=num_classes)
        return model
    except ImportError:
        return SimpleViT(num_classes, img_size)


class SimpleViT(nn.Module):
    """Simple ViT implementation for cross-architecture evaluation."""

    def __init__(self, num_classes=10, img_size=224, patch_size=16, dim=192, depth=4, heads=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def get_swin_tiny(num_classes, img_size=224):
    """Swin-Transformer-Tiny for cross-architecture evaluation."""
    try:
        from torchvision.models.swin_transformer import swin_t
        model = swin_t(pretrained=False)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    except ImportError:
        return SimpleViT(num_classes, img_size)


def get_deit_tiny(num_classes, img_size=224):
    """DeiT-Tiny for cross-architecture evaluation."""
    try:
        from torchvision.models.vision_transformer import deit_tiny_patch16_224
        model = deit_tiny_patch16_224(pretrained=False, img_size=img_size, num_classes=num_classes)
        return model
    except ImportError:
        return SimpleViT(num_classes, img_size)


ARCH_REGISTRY = {
    "convnet": lambda nc, **kw: ConvNet(num_classes=nc, depth=kw.get("depth", 6)),
    "convnet6": lambda nc, **kw: ConvNet(num_classes=nc, depth=6),
    "convnet7": lambda nc, **kw: ConvNet(num_classes=nc, depth=7),
    "resnet18": lambda nc, **kw: get_resnet18(nc, pretrained=kw.get("pretrained", False)),
    "resnet10_ap": lambda nc, **kw: get_resnet10_ap(nc),
    "vit_tiny": lambda nc, **kw: get_vit_tiny(nc, img_size=kw.get("img_size", 224)),
    "swin_tiny": lambda nc, **kw: get_swin_tiny(nc, img_size=kw.get("img_size", 224)),
    "deit_tiny": lambda nc, **kw: get_deit_tiny(nc, img_size=kw.get("img_size", 224)),
}


# ---------------------------------------------------------------------------
# Dataset loader for generated images
# ---------------------------------------------------------------------------

class GeneratedDataset(Dataset):
    """Load generated images from directory structure."""

    def __init__(self, root, class_names, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        for idx, cls_name in enumerate(class_names):
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

class Trainer:
    """Train and evaluate models on distilled datasets."""

    def __init__(self, device="cuda", save_dir="./results"):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def get_train_transform(self, img_size=224, dsa=True):
        """Get training data augmentation transforms."""
        transforms_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=int(img_size * 0.125)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ]
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transforms_list)

    def get_test_transform(self, img_size=224):
        """Get test transforms."""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_mixup_cutmix(self, beta=1.0, prob=0.5):
        """Mixup/CutMix augmentation."""
        def mixup_fn(x, y):
            if np.random.rand() > prob:
                return x, y
            lam = np.random.beta(beta, beta)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return mixed_x, (y_a, y_b, lam)
        return mixup_fn

    def train_one_epoch(self, model, train_loader, optimizer, criterion, mixup_fn=None):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            if mixup_fn:
                images, targets = mixup_fn(images, targets)
                if isinstance(targets, tuple):
                    y_a, y_b, lam = targets
                    outputs = model(images)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0) if not isinstance(targets, tuple) else targets[0].size(0)
            if isinstance(targets, tuple):
                correct += lam * predicted.eq(targets[0]).sum().item()
                correct += (1 - lam) * predicted.eq(targets[1]).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(train_loader), 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        top5_correct = 0

        for images, targets in test_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if outputs.size(1) >= 5:
                _, pred5 = outputs.topk(5, 1, True, True)
                top5_correct += pred5.eq(targets.view(-1, 1)).any(dim=1).sum().item()

        top1 = 100.0 * correct / total
        top5 = 100.0 * top5_correct / total if total > 0 else 0
        return top1, top5

    def train_and_eval(
        self,
        arch_name,
        train_data_dir,
        test_data_dir,
        class_names,
        num_classes,
        img_size=224,
        epochs=2000,
        batch_size=128,
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        use_mixup=True,
        seed=0,
    ):
        """Train a model on distilled data and evaluate on real test set."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        model = ARCH_REGISTRY[arch_name](num_classes, img_size=img_size, depth=7)
        model = model.to(self.device)

        # Data
        train_transform = self.get_train_transform(img_size)
        test_transform = self.get_test_transform(img_size)

        train_dataset = GeneratedDataset(train_data_dir, class_names, train_transform)
        eval_batch_size = min(batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=True,
                                   num_workers=4, drop_last=False)

        test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # Mixup
        mixup_fn = self.get_mixup_cutmix() if use_mixup else None

        # Training loop
        print(f"\nTraining {arch_name} on distilled data ({len(train_dataset)} samples)...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

        best_acc = 0
        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(
                model, train_loader, optimizer, criterion, mixup_fn
            )
            scheduler.step()

            if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
                test_acc, test_top5 = self.evaluate(model, test_loader)
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                      f"train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%")
                if test_acc > best_acc:
                    best_acc = test_acc

        train_time = time.time() - start_time

        # Final evaluation
        final_acc, final_top5 = self.evaluate(model, test_loader)

        return {
            "arch": arch_name,
            "top1": float(final_acc),
            "top5": float(final_top5),
            "best_top1": float(best_acc),
            "train_time_s": train_time,
            "epochs": epochs,
            "seed": seed,
        }

    def run_multi_seed(
        self,
        arch_name,
        train_data_dir,
        test_data_dir,
        class_names,
        num_classes,
        num_seeds=5,
        **kwargs,
    ):
        """Run training with multiple random seeds."""
        all_results = []
        for seed in range(num_seeds):
            print(f"\n{'='*50}")
            print(f"Seed {seed + 1}/{num_seeds}")
            print(f"{'='*50}")
            result = self.train_and_eval(
                arch_name, train_data_dir, test_data_dir,
                class_names, num_classes, seed=seed, **kwargs
            )
            all_results.append(result)

        # Aggregate
        top1_accs = [r["top1"] for r in all_results]
        stats = DistillationMetrics.compute_accuracy_stats(top1_accs)
        return {
            "arch": arch_name,
            "top1_stats": stats,
            "all_results": all_results,
            "num_seeds": num_seeds,
        }


def main():
    parser = argparse.ArgumentParser(description="AGS-DD Evaluation")
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Path to generated dataset directory")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to real test dataset directory")
    parser.add_argument("--arch", type=str, default="convnet7",
                        choices=list(ARCH_REGISTRY.keys()))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--dataset-name", type=str, default="imagenette")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--no-mixup", action="store_true")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get class names from train directory
    class_names = sorted(os.listdir(args.train_dir))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(args.train_dir, c))]

    trainer = Trainer(device=device, save_dir=args.save_dir)

    results = trainer.run_multi_seed(
        arch_name=args.arch,
        train_data_dir=args.train_dir,
        test_data_dir=args.test_dir,
        class_names=class_names,
        num_classes=args.num_classes,
        num_seeds=args.num_seeds,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_mixup=not args.no_mixup,
    )

    # Save results
    result_path = os.path.join(
        args.save_dir,
        f"{args.dataset_name}_ipc{args.ipc}_{args.arch}.json"
    )
    DistillationMetrics.save_results(results, result_path)
    print(f"\nResults saved to {result_path}")
    print(f"Top-1 Accuracy: {results['top1_stats']['mean']:.1f} ± {results['top1_stats']['std']:.1f}%")


if __name__ == "__main__":
    main()
