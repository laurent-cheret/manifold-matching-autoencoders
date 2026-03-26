"""Spheres synthetic dataset."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


def generate_spheres(n_samples=500, d=100, n_spheres=11, r=5, seed=42):
    """Generate synthetic spheres dataset."""
    np.random.seed(seed)
    
    variance = 10 / np.sqrt(d)
    shift_matrix = np.random.normal(0, variance, [n_spheres, d + 1])
    
    spheres = []
    n_datapoints = 0
    
    for i in range(n_spheres - 1):
        data = np.random.randn(n_samples, d + 1)
        data = r * data / np.sqrt(np.sum(data ** 2, axis=1, keepdims=True))
        spheres.append(data + shift_matrix[i, :])
        n_datapoints += n_samples
    
    n_samples_big = 10 * n_samples
    big = np.random.randn(n_samples_big, d + 1)
    big = r * 5 * big / np.sqrt(np.sum(big ** 2, axis=1, keepdims=True))
    spheres.append(big)
    n_datapoints += n_samples_big
    
    dataset = np.concatenate(spheres, axis=0)
    
    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples
    
    return dataset.astype(np.float32), labels.astype(np.float32)


@register_dataset("spheres")
def load_spheres(config, with_embeddings=False, return_indices=False):
    """Load Spheres dataset."""
    data, labels = generate_spheres(
        n_samples=config.get("n_samples", 500),
        d=config.get("d", 100),
        n_spheres=config.get("n_spheres", 11),
        r=config.get("r", 5),
        seed=config.get("seed", 42)
    )
    
    seed = config.get("seed", 42)
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        data, labels, seed=seed
    )

    train_data, val_data, test_data = normalize_features(train_data, test_data, val_data=val_data)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", 80)
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