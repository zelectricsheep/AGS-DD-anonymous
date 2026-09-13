#!/usr/bin/env python3
"""Herding selection for orthogonality experiment.

Selects IPC images per class from a larger generated dataset using
herding (greedy feature matching to the class mean).
"""
import os, sys, glob, argparse, numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

def extract_features(image_paths, device='cuda'):
    """Extract features using ResNet-18 pretrained on ImageNet."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC
    model = model.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            feat = model(img_t).squeeze().cpu().numpy()
            features.append(feat)
    return np.array(features)

def herding_select(features, n_select):
    """Greedy herding: select n_select points that best approximate the mean."""
    n = len(features)
    if n <= n_select:
        return list(range(n))
    
    mean = features.mean(axis=0)
    selected = []
    remaining = list(range(n))
    summed = np.zeros_like(mean)
    
    for _ in range(n_select):
        best_idx = None
        best_dist = float('inf')
        for i in remaining:
            candidate = (summed + features[i]) / (len(selected) + 1)
            dist = np.linalg.norm(candidate - mean)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        selected.append(best_idx)
        summed += features[best_idx]
        remaining.remove(best_idx)
    
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', required=True, help='Source dataset directory')
    parser.add_argument('--dst-dir', required=True, help='Destination dataset directory')
    parser.add_argument('--ipc', type=int, default=10, help='Images per class to select')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.dst_dir, exist_ok=True)
    
    class_dirs = sorted([d for d in os.listdir(args.src_dir) 
                         if os.path.isdir(os.path.join(args.src_dir, d))])
    
    for cls_idx, cls_dir in enumerate(class_dirs):
        cls_path = os.path.join(args.src_dir, cls_dir)
        image_paths = sorted(glob.glob(os.path.join(cls_path, '*.png')) + 
                            glob.glob(os.path.join(cls_path, '*.JPEG')) +
                            glob.glob(os.path.join(cls_path, '*.jpg')))
        
        if len(image_paths) <= args.ipc:
            # If already <= IPC, just copy all
            dst_cls = os.path.join(args.dst_dir, cls_dir)
            os.makedirs(dst_cls, exist_ok=True)
            for p in image_paths:
                dst = os.path.join(dst_cls, os.path.basename(p))
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(p), dst)
            print(f"Class {cls_dir}: {len(image_paths)} images (<= {args.ipc}, copied all)")
            continue
        
        # Extract features
        features = extract_features(image_paths, args.device)
        
        # Herding selection
        selected_indices = herding_select(features, args.ipc)
        
        # Copy selected images
        dst_cls = os.path.join(args.dst_dir, cls_dir)
        os.makedirs(dst_cls, exist_ok=True)
        for idx in selected_indices:
            src = image_paths[idx]
            dst = os.path.join(dst_cls, os.path.basename(src))
            os.symlink(os.path.abspath(src), dst)
        
        print(f"Class {cls_dir}: selected {args.ipc}/{len(image_paths)} images")
        
        if (cls_idx + 1) % 10 == 0:
            print(f"  Processed {cls_idx + 1}/{len(class_dirs)} classes")
    
    print(f"\nDone! Herded dataset saved to {args.dst_dir}")

if __name__ == '__main__':
    main()
