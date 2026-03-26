"""Concentric Spheres dataset in 1000D."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


def generate_concentric_spheres(n_samples_per_shell=2000, n_shells=5, d=100,
                                 r_min=1.0, r_max=5.0, noise=0.01, seed=42):
    """Generate nested spherical shells in high-D."""
    np.random.seed(seed)
    
    radii_values = np.linspace(r_min, r_max, n_shells)
    
    all_data = []
    all_labels = []
    all_radii = []
    
    for shell_idx, r in enumerate(radii_values):
        points = np.random.randn(n_samples_per_shell, d)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = r * points
        
        radial_noise = 1 + noise * np.random.randn(n_samples_per_shell, 1)
        points = points * radial_noise
        
        all_data.append(points)
        all_labels.append(np.full(n_samples_per_shell, shell_idx))
        all_radii.append(np.full(n_samples_per_shell, r))
    
    data = np.vstack(all_data).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.float32)
    radii = np.concatenate(all_radii).astype(np.float32)
    
    return data, labels, radii


@register_dataset("concentric_spheres")
def load_concentric_spheres(config, with_embeddings=False, return_indices=False):
    """Load Concentric Spheres dataset (1000D)."""
    seed = config.get("seed", 42)
    n_samples_per_shell = config.get("n_samples_per_shell", 400)
    n_shells = config.get("n_shells", 5)
    d = config.get("d", 1000)
    
    data, labels, _ = generate_concentric_spheres(
        n_samples_per_shell=n_samples_per_shell,
        n_shells=n_shells,
        d=d,
        seed=seed
    )
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        data, labels, seed=seed
    )

    train_data, val_data, test_data = normalize_features(train_data, test_data, val_data=val_data)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", config.get("input_dim"))
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_data, test_data, n_components, seed=seed, val_data=val_data
        )
        print(f"Computed PCA embeddings with {n_components} components")

    print(f"Concentric Spheres: {data.shape[0]} samples, {data.shape[1]}D, {n_shells} shells")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )



