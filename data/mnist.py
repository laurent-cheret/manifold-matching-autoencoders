"""MNIST and Fashion-MNIST datasets."""

import numpy as np
from sklearn.model_selection import train_test_split
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings


def load_mnist_base(config, dataset_cls, with_embeddings=False, return_indices=False):
    """Base loader for MNIST-like datasets.

    Uses the official torchvision test split as the held-out test set.
    Carves a validation set from the official training split.
    """
    from torchvision import datasets

    train_set = dataset_cls(root='./data/raw', train=True, download=True)
    test_set = dataset_cls(root='./data/raw', train=False, download=True)

    arch_type = config.get('arch_type', 'mlp')
    n_samples = config.get('n_samples', None)
    seed = config.get('seed', 42)

    # Raw data
    train_data = train_set.data.numpy().astype(np.float32) / 255.0
    test_data = test_set.data.numpy().astype(np.float32) / 255.0
    train_labels = train_set.targets.numpy().astype(np.float32)
    test_labels = test_set.targets.numpy().astype(np.float32)

    # Subsample training pool if requested
    if n_samples is not None and n_samples < len(train_data):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_data), n_samples, replace=False)
        train_data = train_data[idx]
        train_labels = train_labels[idx]

    # Subsample test set proportionally
    n_test = min(max(len(train_data) // 5, 2000), len(test_data))
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(len(test_data), n_test, replace=False)
    test_data = test_data[test_idx]
    test_labels = test_labels[test_idx]

    # Carve validation set from training pool (stratified)
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels,
        test_size=0.15,
        random_state=seed,
        stratify=train_labels
    )

    if arch_type == 'conv':
        train_data = train_data[:, np.newaxis, :, :]
        val_data = val_data[:, np.newaxis, :, :]
        test_data = test_data[:, np.newaxis, :, :]
    else:
        train_data = train_data.reshape(-1, 784)
        val_data = val_data.reshape(-1, 784)
        test_data = test_data.reshape(-1, 784)

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


@register_dataset("mnist")
def load_mnist(config, with_embeddings=False, return_indices=False):
    """Load MNIST dataset."""
    from torchvision import datasets
    return load_mnist_base(config, datasets.MNIST, with_embeddings, return_indices)


@register_dataset("fmnist")
def load_fashion_mnist(config, with_embeddings=False, return_indices=False):
    """Load Fashion-MNIST dataset."""
    from torchvision import datasets
    return load_mnist_base(config, datasets.FashionMNIST, with_embeddings, return_indices)
