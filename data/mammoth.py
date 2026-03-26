"""Mammoth 3D dataset for topology-preserving dimensionality reduction.

Dataset source: https://github.com/MNoichl/UMAP-examples-mammoth
Original 3D scan: Smithsonian Institution

To download manually:
    wget https://raw.githubusercontent.com/MNoichl/UMAP-examples-mammoth/master/mammoth_a.csv -O data/mammoth.csv
"""

import numpy as np
import pandas as pd
import os

from .base import register_dataset, create_dataloaders, compute_pca_embeddings, split_train_val_test

MAMMOTH_URL = "https://raw.githubusercontent.com/MNoichl/UMAP-examples-mammoth/master/mammoth_a.csv"


def download_mammoth(cache_path):
    """Download mammoth dataset."""
    import urllib.request
    print(f"Downloading Mammoth dataset from {MAMMOTH_URL}...")
    urllib.request.urlretrieve(MAMMOTH_URL, cache_path)


def load_mammoth_data(data_dir="./data", n_samples=None, seed=42):
    """Load Mammoth 3D point cloud dataset.
    
    Returns:
        data: (N, 3) array of 3D coordinates
        labels: (N,) array of pseudo-labels based on y-coordinate (body segments)
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "mammoth.csv")
    
    if not os.path.exists(cache_path):
        try:
            download_mammoth(cache_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download Mammoth dataset: {e}\n"
                f"Please download manually:\n"
                f"  wget {MAMMOTH_URL} -O {cache_path}"
            )
    
    df = pd.read_csv(cache_path)
    data = df[['x', 'y', 'z']].values.astype(np.float32)
    
    # Subsample if requested
    if n_samples is not None and n_samples < len(data):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(data), n_samples, replace=False)
        data = data[indices]
    
    # Create pseudo-labels based on y-coordinate quantiles (body segments)
    n_segments = 10
    y_vals = data[:, 1]
    labels = pd.qcut(y_vals, q=n_segments, labels=False, duplicates='drop')
    labels = labels.astype(np.float32)
    
    return data, labels


@register_dataset("mammoth")
def load_mammoth(config, with_embeddings=False, return_indices=False):
    """Load Mammoth dataset."""
    data, labels = load_mammoth_data(
        n_samples=config.get("n_samples", 10000),
        seed=config.get("seed", 42)
    )
    
    seed = config.get("seed", 42)
    # Fallback to non-stratified if label bins are too small
    try:
        train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
            data, labels, seed=seed
        )
    except ValueError:
        train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
            data, labels, seed=seed, stratify=False
        )

    # Center and scale uniformly (preserves shape proportions)
    mean = train_data.mean(axis=0)
    scale = np.abs(train_data - mean).max()
    train_data = ((train_data - mean) / scale).astype(np.float32)
    val_data = ((val_data - mean) / scale).astype(np.float32)
    test_data = ((test_data - mean) / scale).astype(np.float32)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", 3)
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_data, test_data, n_components, seed=seed, val_data=val_data
        )
        print(f"Computed PCA embeddings with {n_components} components")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )