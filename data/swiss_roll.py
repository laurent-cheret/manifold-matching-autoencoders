"""Rotated Swiss Roll dataset in 100D."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


# def generate_swiss_roll(n_samples=5000, noise=0.01, seed=42):
#     """Generate 3D Swiss roll, then rotate into 100D."""
#     np.random.seed(seed)
    
#     t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
#     height = 6 * np.random.rand(n_samples)
    
#     x = t * np.cos(t)
#     y = height
#     z = t * np.sin(t)
    
#     original_3d = np.column_stack([x, y, z])
#     original_3d += noise * np.random.randn(n_samples, 3)
    
#     d_high = 100
#     random_matrix = np.random.randn(3, d_high)
#     Q, _ = np.linalg.qr(random_matrix.T)
    
#     data = original_3d @ Q.T
#     data += 0.01 * np.random.randn(n_samples, d_high)
    
#     return data.astype(np.float32), t.astype(np.float32), original_3d.astype(np.float32)

def generate_swiss_roll(n_samples=5000, noise=0.01, seed=42):
    """Generate 3D Swiss roll with a hole, then rotate into 100D."""
    np.random.seed(seed)
    
    # Oversample to account for hole removal
    n_oversample = int(n_samples * 1.8)
    
    # Generate in normalized [0,1] space first
    t_raw = np.random.rand(n_oversample)
    height_raw = np.random.rand(n_oversample)
    
    # Remove points in the hole region (in normalized coords)
    hole_t = (0.1, 0.9)
    hole_h = (0.3, 0.7)
    in_hole = (
        (t_raw > hole_t[0]) & (t_raw < hole_t[1]) &
        (height_raw > hole_h[0]) & (height_raw < hole_h[1])
    )
    t_raw = t_raw[~in_hole][:n_samples]
    height_raw = height_raw[~in_hole][:n_samples]
    
    # Scale to actual Swiss roll coordinates
    t = 1.5 * np.pi * (1 + 2 * t_raw)
    height = 15 * height_raw
    
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    
    original_3d = np.column_stack([x, y, z])
    original_3d += noise * np.random.randn(len(t), 3)
    
    d_high = 100
    random_matrix = np.random.randn(3, d_high)
    Q, _ = np.linalg.qr(random_matrix.T)
    
    data = original_3d @ Q.T
    data += 0.01 * np.random.randn(len(t), d_high)
    
    return data.astype(np.float32), t.astype(np.float32), original_3d.astype(np.float32)


@register_dataset("swiss_roll")
def load_swiss_roll(config, with_embeddings=False, return_indices=False):
    """Load Rotated Swiss Roll dataset (100D)."""
    seed = config.get("seed", 42)
    n_samples = config.get("n_samples", 2000)
    
    data, t_param, _ = generate_swiss_roll(n_samples=n_samples, seed=seed)
    
    n_bins = config.get("n_classes", 10)
    labels = np.digitize(t_param, np.linspace(t_param.min(), t_param.max(), n_bins)) - 1
    labels = labels.astype(np.float32)
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        data, labels, seed=seed, stratify=False
    )

    train_data, val_data, test_data = normalize_features(train_data, test_data, val_data=val_data)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", config.get("input_dim"))
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_data, test_data, n_components, seed=seed, val_data=val_data
        )
        print(f"Computed PCA embeddings with {n_components} components")

    print(f"Swiss Roll: {data.shape[0]} samples, {data.shape[1]}D, {n_bins} classes")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )



