"""Klein bottle synthetic dataset."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


def generate_klein_bottle(n_samples=2000, noise=0.05, seed=42):
    """
    Generate Klein bottle embedded in 4D.
    
    Parametric equations for Klein bottle in R^4:
    x = (a + b*cos(v)) * cos(u)
    y = (a + b*cos(v)) * sin(u)
    z = b * sin(v) * cos(u/2)
    w = b * sin(v) * sin(u/2)
    
    where u, v ∈ [0, 2π), a > b > 0
    """
    np.random.seed(seed)
    
    a, b = 3.0, 1.0
    
    u = np.random.uniform(0, 2 * np.pi, n_samples)
    v = np.random.uniform(0, 2 * np.pi, n_samples)
    
    x = (a + b * np.cos(v)) * np.cos(u)
    y = (a + b * np.cos(v)) * np.sin(u)
    z = b * np.sin(v) * np.cos(u / 2)
    w = b * np.sin(v) * np.sin(u / 2)
    
    data = np.stack([x, y, z, w], axis=1)
    
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    
    # Labels based on position along u parameter (creates bands)
    labels = np.floor(u / (2 * np.pi / 4)).astype(np.float32)
    
    return data.astype(np.float32), labels


@register_dataset("klein_bottle")
def load_klein_bottle(config, with_embeddings=False, return_indices=False):
    """Load Klein bottle dataset."""
    data, labels = generate_klein_bottle(
        n_samples=config.get("n_samples", 2000),
        noise=config.get("noise", 0.05),
        seed=config.get("seed", 42)
    )
    
    seed = config.get("seed", 42)
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        data, labels, seed=seed
    )

    train_data, val_data, test_data = normalize_features(train_data, test_data, val_data=val_data)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", 4)
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