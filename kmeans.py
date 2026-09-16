"""
K-Means Clustering and Vector Quantization (VQ) Module
Implemented in pure NumPy for speech recognition codebooks.
"""

import numpy as np

class KMeans:
    """
    K-Means clustering algorithm using K-Means++ initialization
    and Lloyd's iterative refinement.
    """
    def __init__(self, n_clusters=16, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None

    def fit(self, X):
        """
        Fits K cluster centroids to data matrix X of shape (N, D).
        """
        N, D = X.shape
        if N < self.n_clusters:
            # If fewer samples than clusters, replicate samples
            repeats = int(np.ceil(self.n_clusters / N))
            X = np.tile(X, (repeats, 1))[:self.n_clusters]
            N = self.n_clusters

        rng = np.random.RandomState(self.random_state)

        # 1. K-Means++ Initialization
        centroids = np.empty((self.n_clusters, D), dtype=X.dtype)
        first_idx = rng.randint(0, N)
        centroids[0] = X[first_idx]

        for k in range(1, self.n_clusters):
            # Compute distance from all points to nearest existing centroid
            dist_sq = np.min(np.sum((X[:, np.newaxis, :] - centroids[:k, :]) ** 2, axis=2), axis=1)
            dist_sq = np.maximum(dist_sq, 1e-10)
            probs = dist_sq / np.sum(dist_sq)
            next_idx = rng.choice(N, p=probs)
            centroids[k] = X[next_idx]

        # 2. Lloyd's Iterations
        for iteration in range(self.max_iter):
            # Distances from all samples to all centroids: (N, K)
            dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                cluster_members = X[labels == k]
                if len(cluster_members) > 0:
                    new_centroids[k] = np.mean(cluster_members, axis=0)
                else:
                    # Handle empty cluster by re-seeding from random sample
                    new_centroids[k] = X[rng.randint(0, N)]

            # Check convergence
            shift = np.sum((centroids - new_centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids = centroids
        # Compute final inertia
        final_dists = np.min(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        self.inertia_ = np.sum(final_dists)
        return self

def compute_vq_distortion(mfcc_frames, codebook_centroids):
    """
    Computes average minimum Euclidean Vector Quantization (VQ) distortion
    between an utterance's MFCC frames (T, D) and a codebook (K, D).
    
    Formula: D(X, C) = (1 / T) * sum_{t=1}^T min_{k} ||x_t - c_k||^2
    """
    # mfcc_frames: (T, D)
    # codebook_centroids: (K, D)
    # Pairwise squared Euclidean distances: (T, K)
    dists = np.sum((mfcc_frames[:, np.newaxis, :] - codebook_centroids[np.newaxis, :, :]) ** 2, axis=2)
    # Min distance for each frame to nearest centroid
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))

class DigitVQClassifier:
    """
    Multi-class Digit Classifier using K-Means Vector Quantization Codebooks.
    Contains 10 codebooks (one for each digit 0 through 9).
    """
    def __init__(self, n_clusters_per_digit=16):
        self.n_clusters_per_digit = n_clusters_per_digit
        self.codebooks = {} # digit (int) -> centroids ndarray (K, D)

    def train_digit(self, digit, all_mfcc_frames):
        """
        Trains the K-Means codebook for a single digit class using pooled frames.
        """
        km = KMeans(n_clusters=self.n_clusters_per_digit, max_iter=100, random_state=42 + digit)
        km.fit(all_mfcc_frames)
        self.codebooks[digit] = km.centroids

    def predict(self, test_mfcc_frames):
        """
        Classifies test MFCC frames by computing VQ distortion to all digit codebooks.
        Returns:
            predicted_digit: int (0-9)
            distortions: dict {digit: distortion_score}
            confidence: float (0.0 to 1.0)
        """
        distortions = {}
        for digit in sorted(self.codebooks.keys()):
            distortions[digit] = compute_vq_distortion(test_mfcc_frames, self.codebooks[digit])

        predicted_digit = min(distortions.keys(), key=lambda d: distortions[d])

        # Convert inverse distortions to pseudo-confidence probabilities via softmax
        vals = np.array([distortions[d] for d in sorted(distortions.keys())])
        # Invert: smaller distortion is better
        inv_vals = -vals / (np.std(vals) + 1e-6)
        exp_vals = np.exp(inv_vals - np.max(inv_vals))
        probs = exp_vals / np.sum(exp_vals)
        confidence = float(probs[predicted_digit])

        return predicted_digit, distortions, confidence

    def to_dict(self):
        """Exports codebooks as JSON-serializable dictionary."""
        return {
            "n_clusters": self.n_clusters_per_digit,
            "digits": {
                str(digit): self.codebooks[digit].tolist()
                for digit in self.codebooks
            }
        }

    def from_dict(self, data):
        """Loads codebooks from dictionary."""
        self.n_clusters_per_digit = data["n_clusters"]
        self.codebooks = {
            int(d): np.array(centroids, dtype=np.float32)
            for d, centroids in data["digits"].items()
        }
        return self
