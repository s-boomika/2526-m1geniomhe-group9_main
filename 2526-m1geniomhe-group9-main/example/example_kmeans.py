"""
Example: K-Means Clustering on Synthetic Blobs Dataset
this example demonstrates how to use the KMeans clustering algorithm
from the upsaclay_ml library on a synthetic blobs dataset.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from upsaclay_ml.models.cluster import KMeans
from upsaclay_ml.datasets import Blobs
from upsaclay_ml.metrics import silhouette_score


def main():
    """Run K-Means clustering example."""
    print("=" * 60)
    print("K-Means Clustering on Blobs Dataset")
    print("=" * 60)
    print()
    
    # Create synthetic blobs dataset
    print("Creating synthetic dataset with 3 clusters...")
    dataset = Blobs(n_samples=300, n_features=2, centers=3, 
                   cluster_std=0.6, random_state=42)
    
    X = dataset.X
    y_true = dataset.y
    
    print(f"Dataset shape: {X.shape}")
    print(f"True labels: {np.unique(y_true)}")
    print()
    
    # Apply K-Means
    print("Fitting K-Means with 3 clusters...")
    kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"K-Means converged in {kmeans.n_iter_} iterations")
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    print()
    
    # Evaluate clustering quality
    silhouette = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette:.3f}")
    print("(Closer to 1 indicates better-defined clusters)")
    print()
    
    # Visualize results
    print("Generating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot true labels
    ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
    ax1.set_title('True Labels', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    
    # Plot predicted labels
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', marker='X', s=300, edgecolors='black', linewidths=2,
               label='Centroids')
    ax2.set_title(f'K-Means Predictions (Silhouette: {silhouette:.3f})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_clustering_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'kmeans_clustering_results.png'")
    print()
    
    # Try different numbers of clusters
    print("Testing different numbers of clusters...")
    print("-" * 60)
    print(f"{'K':>3s} | {'Inertia':>12s} | {'Silhouette':>11s} | {'Iterations':>11s}")
    print("-" * 60)
    
    for k in range(2, 7):
        kmeans_k = KMeans(n_clusters=k, random_state=42)
        labels_k = kmeans_k.fit_predict(X)
        sil_k = silhouette_score(X, labels_k)
        print(f"{k:3d} | {kmeans_k.inertia_:12.2f} | {sil_k:11.3f} | {kmeans_k.n_iter_:11d}")
    
    print("-" * 60)
    print()
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
