"""Linked Tori dataset in high dimensions."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


def gauss_linking_number(curve1, curve2):
    """
    Compute linking number via Gauss linking integral.
    Lk = 0 means unlinked, Lk ≠ 0 means linked.
    """
    n1, n2 = len(curve1), len(curve2)
    
    dr1 = np.roll(curve1, -1, axis=0) - curve1
    dr2 = np.roll(curve2, -1, axis=0) - curve2
    
    linking_sum = 0.0
    for i in range(n1):
        for j in range(n2):
            r = curve1[i] - curve2[j]
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-10:
                continue
            cross = np.cross(dr1[i], dr2[j])
            linking_sum += np.dot(r, cross) / (r_norm ** 3)
    
    return linking_sum / (4 * np.pi)


def generate_linked_tori(n_samples=2000, d=100, R=3.0, r=1.0, noise=0.02, seed=42, verify=True):
    """
    Generate two linked tori (chain link configuration) in high-D.
    
    Construction:
    - Torus 1: Standard torus in 3D
    - Torus 2: Rotated 90° and translated to pass through hole of Torus 1
    - Both lifted to d dimensions via random orthogonal rotation
    
    Args:
        n_samples: Total number of points (split between two tori)
        d: Ambient dimension
        R: Major radius (center of tube to center of torus)
        r: Minor radius (tube radius)
        noise: Gaussian noise level
        seed: Random seed
        verify: Whether to verify linking via Gauss integral
    
    Returns:
        data: (n_samples, d) array
        labels: (n_samples,) array with 0 for torus 1, 1 for torus 2
    """
    np.random.seed(seed)
    
    n_per_torus = n_samples // 2
    
    # Torus 1: Standard position in 3D
    theta1 = np.random.uniform(0, 2 * np.pi, n_per_torus)
    phi1 = np.random.uniform(0, 2 * np.pi, n_per_torus)
    
    torus1 = np.zeros((n_per_torus, 3))
    torus1[:, 0] = (R + r * np.cos(theta1)) * np.cos(phi1)
    torus1[:, 1] = (R + r * np.cos(theta1)) * np.sin(phi1)
    torus1[:, 2] = r * np.sin(theta1)
    
    # Torus 2: Rotated 90° around x-axis and translated to link through Torus 1
    theta2 = np.random.uniform(0, 2 * np.pi, n_per_torus)
    phi2 = np.random.uniform(0, 2 * np.pi, n_per_torus)
    
    # Standard torus coordinates
    x2 = (R + r * np.cos(theta2)) * np.cos(phi2)
    y2 = (R + r * np.cos(theta2)) * np.sin(phi2)
    z2 = r * np.sin(theta2)
    
    # Rotate 90° around x-axis: (x, y, z) -> (x, -z, y), then translate by (R, 0, 0)
    torus2 = np.zeros((n_per_torus, 3))
    torus2[:, 0] = x2 + R
    torus2[:, 1] = -z2
    torus2[:, 2] = y2
    
    if verify:
        # Extract core circles and verify linking
        t_core = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        
        core1 = np.stack([
            R * np.cos(t_core),
            R * np.sin(t_core),
            np.zeros(100)
        ], axis=1)
        
        core2 = np.stack([
            R + R * np.cos(t_core),
            np.zeros(100),
            R * np.sin(t_core)
        ], axis=1)
        
        lk = gauss_linking_number(core1, core2)
        assert abs(abs(lk) - 1.0) < 0.15, f"Linking number {lk} != ±1, construction may be wrong"
        print(f"Linked Tori: Verified linking number = {lk:.3f}")
    
    # Combine into 3D
    data_3d = np.vstack([torus1, torus2])
    data_3d += noise * np.random.randn(n_samples, 3)
    
    # Lift to d dimensions via random orthogonal rotation
    if d > 3:
        random_matrix = np.random.randn(3, d)
        Q, _ = np.linalg.qr(random_matrix.T)
        data = data_3d @ Q.T
        data += 0.01 * np.random.randn(n_samples, d)
    else:
        data = data_3d
    
    labels = np.array([0] * n_per_torus + [1] * n_per_torus)
    
    return data.astype(np.float32), labels.astype(np.float32)


@register_dataset("linked_tori")
def load_linked_tori(config, with_embeddings=False, return_indices=False):
    """Load Linked Tori dataset."""
    data, labels = generate_linked_tori(
        n_samples=config.get("n_samples", 2000),
        d=config.get("d", 100),
        R=config.get("R", 3.0),
        r=config.get("r", 1.0),
        noise=config.get("noise", 0.02),
        seed=config.get("seed", 42),
        verify=config.get("verify", True)
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

    print(f"Linked Tori: {data.shape[0]} samples, {data.shape[1]}D, 2 linked tori")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )