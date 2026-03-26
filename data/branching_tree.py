"""Branching tree synthetic dataset in high-dimensional space."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


def generate_branching_tree(n_samples=5000, d=100, noise=0.08, seed=42):
    """
    Generate complex branching tree in high-dimensional space.
    
    Structure:
    - Main trunk (depth 0)
    - Branches from trunk (depth 1)
    - Sub-branches from branches (depth 2)
    - Sub-sub-branches (depth 3)
    
    Each level has branches rotated around the parent axis for higher intrinsic dim.
    """
    rng = np.random.default_rng(seed)
    
    points = []
    labels = []
    
    def add_segment(start, direction, length, n_pts, label):
        t = np.linspace(0, length, n_pts)
        segment = start + np.outer(t, direction)
        segment += rng.normal(0, noise, segment.shape)
        points.append(segment)
        labels.extend([label] * n_pts)
        return start + length * direction
    
    def random_orthogonal_dir(parent_dir):
        """Random direction perpendicular to parent."""
        v = rng.normal(0, 1, d)
        v = v - np.dot(v, parent_dir) * parent_dir
        return v / np.linalg.norm(v)
    
    def rotate_in_plane(vec, axis, angle):
        """Rotate vec around axis by angle in high-D (rotation in the plane spanned by vec and a perpendicular)."""
        # Get a direction perpendicular to both axis and vec
        perp = rng.normal(0, 1, d)
        perp = perp - np.dot(perp, axis) * axis  # Remove axis component
        perp = perp - np.dot(perp, vec) * vec    # Remove vec component
        if np.linalg.norm(perp) < 1e-10:
            perp = rng.normal(0, 1, d)
            perp = perp - np.dot(perp, axis) * axis
        perp = perp / np.linalg.norm(perp)
        
        # Rotate in the plane spanned by vec and perp
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = vec * cos_a + perp * sin_a
        # Keep perpendicular to axis
        rotated = rotated - np.dot(rotated, axis) * axis
        return rotated / np.linalg.norm(rotated)
    
    label = 0
    n_pts = n_samples // 40
    
    # Main trunk
    trunk_dir = np.zeros(d)
    trunk_dir[0] = 1.0
    trunk_start = np.zeros(d)
    trunk_start[0] = -3
    add_segment(trunk_start, trunk_dir, 6, n_pts * 3, label)
    label += 1
    
    # Branches from trunk
    for i, tb_pos in enumerate([0.25, 0.5, 0.75]):
        branch_origin = trunk_start + tb_pos * 6 * trunk_dir
        base_branch_dir = random_orthogonal_dir(trunk_dir)
        
        for j in range(2):
            angle = j * np.pi + rng.uniform(-0.2, 0.2)
            branch_dir = rotate_in_plane(base_branch_dir, trunk_dir, angle)
            
            add_segment(branch_origin, branch_dir, 2.5, n_pts * 2, label)
            label += 1
            
            # Sub-branches
            base_sub_dir = random_orthogonal_dir(branch_dir)
            
            for k, sb_pos in enumerate([0.4, 0.7]):
                sub_origin = branch_origin + sb_pos * 2.5 * branch_dir
                
                for m in range(2):
                    angle = m * np.pi + rng.uniform(-0.3, 0.3)
                    sub_dir = rotate_in_plane(base_sub_dir, branch_dir, angle)
                    
                    add_segment(sub_origin, sub_dir, 1.2, n_pts, label)
                    label += 1
                    
                    # Sub-sub-branches
                    subsub_origin = sub_origin + 0.6 * 1.2 * sub_dir
                    base_ss_dir = random_orthogonal_dir(sub_dir)
                    
                    for p in range(2):
                        angle = p * np.pi + rng.uniform(-0.4, 0.4)
                        ss_dir = rotate_in_plane(base_ss_dir, sub_dir, angle)
                        
                        add_segment(subsub_origin, ss_dir, 0.6, n_pts // 2, label)
                        label += 1
    
    dataset = np.vstack(points)
    labels = np.array(labels)
    
    return dataset.astype(np.float32), labels.astype(np.float32)


@register_dataset("branching_tree")
def load_branching_tree(config, with_embeddings=False, return_indices=False):
    """Load Branching Tree dataset."""
    data, labels = generate_branching_tree(
        n_samples=config.get("n_samples", 5000),
        d=config.get("d", 100),
        noise=config.get("noise", 0.08),
        seed=config.get("seed", 42)
    )
    
    seed = config.get("seed", 42)
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

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )