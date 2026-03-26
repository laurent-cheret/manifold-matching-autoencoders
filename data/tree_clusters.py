"""Clusters on Tree dataset in 100D."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


def generate_tree_clusters(n_samples_per_cluster=200, d=100, n_levels=3,
                           branch_factor=2, cluster_std=0.3,
                           level_distance=3.0, seed=42):
    """Generate clusters arranged on a binary tree in high-D."""
    np.random.seed(seed)
    
    n_clusters = branch_factor ** n_levels
    centers = np.zeros((n_clusters, d))
    
    def assign_center(cluster_idx, level, center):
        if level == n_levels:
            centers[cluster_idx] = center
            return cluster_idx + 1
        
        dim = level * 2
        for branch in range(branch_factor):
            new_center = center.copy()
            offset = (branch - (branch_factor - 1) / 2) * level_distance
            new_center[dim] = center[dim] + offset
            cluster_idx = assign_center(cluster_idx, level + 1, new_center)
        return cluster_idx
    
    assign_center(0, 0, np.zeros(d))
    
    random_matrix = np.random.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)
    centers = centers @ Q
    
    all_data = []
    all_labels = []
    
    for cluster_idx in range(n_clusters):
        points = centers[cluster_idx] + cluster_std * np.random.randn(n_samples_per_cluster, d)
        all_data.append(points)
        all_labels.append(np.full(n_samples_per_cluster, cluster_idx))
    
    data = np.vstack(all_data).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.float32)
    
    tree_info = {
        'centers': centers,
        'n_levels': n_levels,
        'branch_factor': branch_factor,
        'n_clusters': n_clusters
    }
    
    return data, labels, tree_info


@register_dataset("tree_clusters")
def load_tree_clusters(config, with_embeddings=False, return_indices=False):
    """Load Clusters on Tree dataset (100D)."""
    seed = config.get("seed", 42)
    n_samples_per_cluster = config.get("n_samples_per_cluster", 200)
    n_levels = config.get("n_levels", 3)
    branch_factor = config.get("branch_factor", 2)
    d = config.get("d", 100)
    
    data, labels, tree_info = generate_tree_clusters(
        n_samples_per_cluster=n_samples_per_cluster,
        d=d,
        n_levels=n_levels,
        branch_factor=branch_factor,
        seed=seed
    )
    
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

    n_clusters = tree_info['n_clusters']
    print(f"Tree Clusters: {data.shape[0]} samples, {data.shape[1]}D, {n_clusters} clusters")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )


