"""
CAGS v2: Class-Adaptive Guidance Strength (Redesigned)

Key improvements over v1:
  1. Configurable sigmoid parameters (slope, center) — allows fine-tuning
  2. New complexity factors:
     - Intra-class feature variance (compactness)
     - Inter-class separability (how distinguishable this class is)
  3. Calibrated mode: centers guidance strength around a known-good λ

Complexity(c) = α·mode_count + β·entropy + γ·intra_variance + δ·(1−separability)
guidance_strength(c) = min_s + sigmoid(slope·(complexity − center)) · (max_s − min_s)
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict
from tqdm import tqdm
import time
import pickle
import os


class ClassComplexityAnalyzer:
    """
    Analyzes per-class complexity to determine adaptive guidance strength.

    The complexity score combines:
      1. Mode count: Number of distinct clusters (sub-modes) in feature space
      2. Intra-class entropy: Distribution uniformity across clusters
      3. Intra-class variance: Average distance of features to their centroid
      4. Inter-class separability: How distinguishable this class is from others

    A higher complexity score leads to stronger guidance during diffusion sampling.
    """

    def __init__(
        self,
        feature_extractor=None,
        n_clusters_range=(2, 20),
        alpha=0.3,
        beta=0.3,
        gamma=0.2,
        delta=0.2,
        use_pca=True,
        pca_components=4,
        use_silhouette=True,
        max_k_method="silhouette",
        sigmoid_slope=3.0,
        sigmoid_center=0.6,
    ):
        self.feature_extractor = feature_extractor
        self.n_clusters_range = n_clusters_range
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_silhouette = use_silhouette
        self.max_k_method = max_k_method
        self.max_k = n_clusters_range[1]
        self.sigmoid_slope = sigmoid_slope
        self.sigmoid_center = sigmoid_center

        self.complexity_scores = {}
        self.mode_counts = {}
        self.cluster_centers = {}
        self.cluster_centers_path = {}
        self.mode_id_per_class = {}
        self.intra_variances = {}
        self.separabilities = {}
        self.all_centroids = {}
        self.features_per_class = {}

    def _find_optimal_k(self, X, k_min=2, k_max=20):
        k_max = min(k_max, len(X) - 1)
        if k_max < k_min:
            return k_min, np.zeros(len(X))

        if self.use_pca and X.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X

        if self.max_k_method == "silhouette" and self.use_silhouette:
            best_k = k_min
            best_score = -1
            for k in range(k_min, k_max + 1):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(X_pca)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_pca, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            return best_k, KMeans(n_clusters=best_k, random_state=0, n_init=10).fit_predict(X_pca)
        else:
            inertias = []
            for k in range(k_min, k_max + 1):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                kmeans.fit(X_pca)
                inertias.append(kmeans.inertia_)
            diffs = np.diff(inertias)
            if len(diffs) > 1:
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(np.abs(second_diffs)) + 1
                best_k = k_min + elbow_idx
            else:
                best_k = k_min
            best_k = max(k_min, min(best_k, k_max))
            kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
            return best_k, kmeans.fit_predict(X_pca)

    def compute_complexity(self, features, class_id, all_class_features=None, class_ids_list=None):
        """
        Compute complexity score for a single class.

        Args:
            features: numpy array (N, D) for this class
            class_id: class identifier
            all_class_features: dict {class_id: features} for separability computation
            class_ids_list: list of all class ids (for separability)
        """
        X = np.stack(features) if isinstance(features, list) else features

        optimal_k, labels = self._find_optimal_k(
            X, k_min=self.n_clusters_range[0], k_max=self.n_clusters_range[1]
        )

        kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_

        closest_points = []
        for center in centers:
            closest_idx = np.argmin(np.sum((X - center) ** 2, axis=1))
            closest_points.append(X[closest_idx])
        cluster_centers = np.stack(closest_points)

        # Factor 1: Mode count score
        mode_count_score = optimal_k / self.max_k

        # Factor 2: Normalized entropy
        proportions = np.bincount(labels, minlength=optimal_k) / len(labels)
        proportions = proportions[proportions > 0]
        entropy = -np.sum(proportions * np.log(proportions + 1e-8))
        normalized_entropy = entropy / np.log(optimal_k) if optimal_k > 1 else 0.0

        # Factor 3: Intra-class variance (mean distance to centroid, normalized)
        centroid = np.mean(X, axis=0)
        intra_var = np.mean(np.sqrt(np.sum((X - centroid) ** 2, axis=1)))
        self.intra_variances[class_id] = float(intra_var)
        # Normalize later across all classes

        # Factor 4: Inter-class separability (computed in analyze_all_classes)
        # Placeholder here, filled later
        self.separabilities[class_id] = 0.5

        # Store for later normalization
        self.all_centroids[class_id] = centroid

        # Base complexity (without separability, will be updated)
        complexity_score = self.alpha * mode_count_score + self.beta * normalized_entropy

        self.complexity_scores[class_id] = float(complexity_score)  # temporary
        self.mode_counts[class_id] = optimal_k
        self.cluster_centers[class_id] = cluster_centers
        self.mode_id_per_class[class_id] = labels

        return complexity_score, optimal_k, cluster_centers

    def _compute_separability_and_finalize(self):
        """Compute inter-class separability and finalize complexity scores."""
        if len(self.all_centroids) < 2:
            for class_id in self.complexity_scores:
                self.complexity_scores[class_id] = 0.5
            return

        # Normalize intra-class variance
        var_values = list(self.intra_variances.values())
        var_min, var_max = min(var_values), max(var_values)
        var_range = var_max - var_min if var_max > var_min else 1.0

        # Compute separability for each class
        class_ids = list(self.all_centroids.keys())
        centroids = np.stack([self.all_centroids[cid] for cid in class_ids])

        for i, class_id in enumerate(class_ids):
            # Mean distance to own centroid
            own_features = np.stack(self.features_per_class[class_id])
            own_centroid = self.all_centroids[class_id]
            own_dist = np.mean(np.sqrt(np.sum((own_features - own_centroid) ** 2, axis=1)))

            # Mean distance to other centroids
            other_dists = []
            for j, other_id in enumerate(class_ids):
                if j == i:
                    continue
                other_centroid = self.all_centroids[other_id]
                d = np.sqrt(np.sum((own_centroid - other_centroid) ** 2))
                other_dists.append(d)
            mean_other_dist = np.mean(other_dists) if other_dists else own_dist

            # Separability: ratio of inter-class distance to intra-class distance
            # Higher = more separable = easier class = less guidance needed
            separability = mean_other_dist / (own_dist + 1e-8)
            # Normalize to [0, 1] using sigmoid
            separability_norm = 1.0 / (1.0 + np.exp(-(separability - 2.0)))
            self.separabilities[class_id] = float(separability_norm)

            # Normalize intra-class variance to [0, 1]
            intra_var_norm = (self.intra_variances[class_id] - var_min) / var_range

            # Recompute mode count and entropy from stored values
            optimal_k = self.mode_counts[class_id]
            mode_count_score = optimal_k / self.max_k

            labels = self.mode_id_per_class[class_id]
            proportions = np.bincount(labels, minlength=optimal_k) / len(labels)
            proportions = proportions[proportions > 0]
            entropy = -np.sum(proportions * np.log(proportions + 1e-8))
            normalized_entropy = entropy / np.log(optimal_k) if optimal_k > 1 else 0.0

            # Final complexity: higher variance + lower separability = more complex
            raw_complexity = (
                self.alpha * mode_count_score
                + self.beta * normalized_entropy
                + self.gamma * intra_var_norm
                + self.delta * (1.0 - separability_norm)
            )

            # Sigmoid normalization
            complexity_score = 1.0 / (1.0 + np.exp(-self.sigmoid_slope * (raw_complexity - self.sigmoid_center)))

            self.complexity_scores[class_id] = float(complexity_score)

            print(
                f"  Class {class_id}: complexity={complexity_score:.4f}, "
                f"modes={optimal_k}, entropy={normalized_entropy:.3f}, "
                f"intra_var={intra_var_norm:.3f}, sep={separability_norm:.3f}"
            )

    def analyze_all_classes(self, features_per_class, paths_per_class=None):
        self.features_per_class = features_per_class
        for class_id in tqdm(features_per_class.keys(), desc="Computing class complexity"):
            start_time = time.time()
            features = features_per_class[class_id]
            complexity, n_modes, centers = self.compute_complexity(features, class_id)

            if paths_per_class is not None and class_id in paths_per_class:
                self.cluster_centers_path[class_id] = [
                    paths_per_class[class_id][
                        np.argmin(np.sum(
                            np.stack(features) - centers[i], axis=1
                        ) ** 2)
                    ]
                    for i in range(n_modes)
                ]

            end_time = time.time()
            print(
                f"Class {class_id}: modes={n_modes}, time={end_time - start_time:.2f}s"
            )

        # Compute separability and finalize
        print("\nComputing inter-class separability and finalizing complexity scores...")
        self._compute_separability_and_finalize()

        return self.complexity_scores, self.mode_counts, self.cluster_centers

    def get_guidance_strength(self, class_id, scale_range=(0.05, 0.15)):
        if class_id not in self.complexity_scores:
            return (scale_range[0] + scale_range[1]) / 2
        complexity = self.complexity_scores[class_id]
        min_s, max_s = scale_range
        return min_s + complexity * (max_s - min_s)

    def save(self, path):
        data = {
            "complexity_scores": self.complexity_scores,
            "mode_counts": self.mode_counts,
            "cluster_centers": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.cluster_centers.items()
            },
            "cluster_centers_path": self.cluster_centers_path,
            "mode_id_per_class": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.mode_id_per_class.items()
            },
            "intra_variances": self.intra_variances,
            "separabilities": self.separabilities,
            "sigmoid_slope": self.sigmoid_slope,
            "sigmoid_center": self.sigmoid_center,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.complexity_scores = data["complexity_scores"]
        self.mode_counts = data["mode_counts"]
        self.cluster_centers = {k: np.array(v) for k, v in data["cluster_centers"].items()}
        self.cluster_centers_path = data.get("cluster_centers_path", {})
        self.mode_id_per_class = data.get("mode_id_per_class", {})
        self.intra_variances = data.get("intra_variances", {})
        self.separabilities = data.get("separabilities", {})
        self.sigmoid_slope = data.get("sigmoid_slope", 3.0)
        self.sigmoid_center = data.get("sigmoid_center", 0.6)

    def compute_clusters_for_ipc(self, ipc, use_pca=True, closest_point=True):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        ipc_clusters = {}
        for class_label, features in self.features_per_class.items():
            X = np.stack(features)
            if len(X) < ipc:
                indices = np.random.choice(len(X), ipc, replace=True)
                ipc_clusters[class_label] = X[indices]
                continue

            if use_pca and X.shape[1] > 4:
                pca = PCA(n_components=4)
                X_pca = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=ipc, random_state=0, n_init="auto").fit(X_pca)
            else:
                X_pca = X
                kmeans = KMeans(n_clusters=ipc, random_state=0, n_init="auto").fit(X)

            if closest_point:
                centers = kmeans.cluster_centers_
                closest = []
                for center in centers:
                    idx = np.argmin(np.sum((X_pca - center) ** 2, axis=1))
                    closest.append(X[idx])
                ipc_clusters[class_label] = np.stack(closest)
            else:
                ipc_clusters[class_label] = kmeans.cluster_centers_

        return ipc_clusters
