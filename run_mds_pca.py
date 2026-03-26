#!/usr/bin/env python
"""
PCA and MDS comparison using project data loaders.

Usage:
    python run_mds_pca.py --dataset spheres
    python run_mds_pca.py --dataset concentric_spheres
    python run_mds_pca.py --dataset mnist --n_samples 2000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import time
import os

from config import get_config, DATASET_CONFIGS
from data import load_data


def run_comparison(dataset_name, n_samples=None, output_dir='results/mds_pca', seed=42):
    """Run PCA and MDS on a dataset using project data loaders."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print('='*60)
    
    # Build config
    config = get_config(dataset_name, 'vanilla')
    config['seed'] = seed
    if n_samples is not None:
        config['n_samples'] = n_samples
    
    # Load data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        dataset_name, config, with_embeddings=False
    )
    
    # Get data as numpy
    X = train_dataset.data.numpy()
    labels = train_dataset.labels.numpy()
    
    # Flatten if needed
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {len(np.unique(labels))} unique values")
    
    # Normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    
    # PCA
    print("\nRunning PCA...")
    start = time.time()
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X)
    pca_time = time.time() - start
    print(f"  Time: {pca_time:.3f}s")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    X_mds = X
    labels_mds = labels
    
    print(f"\nRunning Metric MDS on {X_mds.shape[0]} samples...")
    start = time.time()
    mds = MDS(n_components=2, metric=True, n_init=1, max_iter=300, 
              random_state=seed, normalized_stress='auto')
    Z_mds = mds.fit_transform(X_mds)
    mds_time = time.time() - start
    print(f"  Time: {mds_time:.3f}s")
    print(f"  Stress: {mds.stress_:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_unique = len(np.unique(labels))
    cmap = 'Spectral' if n_unique > 10 else 'Spectral'
    
    # PCA plot
    scatter = axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=labels, cmap=cmap,
                              s=10, alpha=0.8, edgecolors='black', linewidths=0.1)
    axes[0].set_title(f'PCA\n(time: {pca_time:.2f}s, n={X.shape[0]})', fontsize=12)
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # MDS plot
    scatter = axes[1].scatter(Z_mds[:, 0], Z_mds[:, 1], c=labels_mds, cmap=cmap,
                              s=10, alpha=0.8, edgecolors='black', linewidths=0.1)
    axes[1].set_title(f'Metric MDS\n(time: {mds_time:.2f}s, n={X.shape[0]})', fontsize=12)
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes[1], shrink=0.8)
    plt.suptitle(f'Dataset: {dataset_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{dataset_name}_pca_mds.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='PCA vs MDS comparison')
    parser.add_argument('--dataset', type=str, default='spheres',
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples (overrides dataset default)')
    parser.add_argument('--output_dir', type=str, default='results/mds_pca')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_comparison(args.dataset, args.n_samples, args.output_dir, args.seed)


if __name__ == '__main__':
    main()