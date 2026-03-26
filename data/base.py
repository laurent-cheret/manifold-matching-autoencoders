"""Dataset registry and base class."""

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Registry to store dataset classes
DATASET_REGISTRY = {}


def register_dataset(name):
    """Decorator to register a dataset class."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name):
    """Get dataset class by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


def list_datasets():
    """List all registered datasets."""
    return list(DATASET_REGISTRY.keys())


def set_global_seed(seed):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_val_test(data, labels, test_size=0.15, val_size=0.15, seed=42, stratify=True):
    """Split data into train / val / test sets (default ~70 / 15 / 15).

    Args:
        data: Feature array.
        labels: Label array.
        test_size: Fraction of full dataset held out as test.
        val_size: Fraction of full dataset used for validation.
        seed: Random seed.
        stratify: Whether to use stratified splitting.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    stratify_arr = labels if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        data, labels, test_size=test_size,
        random_state=seed, stratify=stratify_arr
    )
    # val_size expressed as a fraction of the remaining train+val pool
    val_frac = val_size / (1.0 - test_size)
    stratify_arr = y_trainval if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac,
        random_state=seed, stratify=stratify_arr
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class BaseDataset(Dataset):
    """Base dataset class."""

    def __init__(self, data, labels, embeddings=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32) if embeddings is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.data[idx], self.embeddings[idx], self.labels[idx]
        return self.data[idx], self.labels[idx]

    def has_embeddings(self):
        return self.embeddings is not None


class IndexedDataset(Dataset):
    """Dataset that returns indices along with data (for GGAE)."""

    def __init__(self, data, labels, embeddings=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32) if embeddings is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.data[idx], self.embeddings[idx], idx, self.labels[idx]
        return self.data[idx], idx, self.labels[idx]


def normalize_features(train_data, test_data, val_data=None):
    """Normalize to zero mean and unit variance (per-feature, fit on train only)."""
    orig_shape = train_data.shape
    train_flat = train_data.reshape(train_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)

    mean = np.mean(train_flat, axis=0, keepdims=True)
    std = np.std(train_flat, axis=0, keepdims=True)
    std[std == 0] = 1.0

    train_norm = (train_flat - mean) / std
    test_norm = (test_flat - mean) / std

    if len(orig_shape) > 2:
        train_norm = train_norm.reshape(orig_shape)
        test_norm = test_norm.reshape(test_data.shape)

    if val_data is not None:
        val_flat = val_data.reshape(val_data.shape[0], -1)
        val_norm = (val_flat - mean) / std
        if len(orig_shape) > 2:
            val_norm = val_norm.reshape(val_data.shape)
        return train_norm, val_norm, test_norm

    return train_norm, test_norm


def compute_pca_embeddings(train_data, test_data, n_components, seed=42, val_data=None):
    """Compute PCA embeddings for MMAE reference (fit on train only).

    Args:
        train_data: Training data.
        test_data: Test data.
        n_components: Number of PCA components.
        seed: Random seed for PCA.
        val_data: Optional validation data.

    Returns:
        train_emb, test_emb  (and val_emb if val_data is provided)
    """
    train_flat = train_data.reshape(train_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)

    pca = PCA(n_components=n_components, random_state=seed)
    train_emb = pca.fit_transform(train_flat)
    test_emb = pca.transform(test_flat)

    if val_data is not None:
        val_flat = val_data.reshape(val_data.shape[0], -1)
        val_emb = pca.transform(val_flat)
        return train_emb.astype(np.float32), val_emb.astype(np.float32), test_emb.astype(np.float32)

    return train_emb.astype(np.float32), test_emb.astype(np.float32)


def collate_with_embeddings(batch):
    """Custom collate function for datasets with embeddings."""
    if len(batch[0]) == 3:  # (data, embedding, label)
        data, emb, labels = zip(*batch)
        return torch.stack(data), torch.stack(emb), torch.stack(labels)
    else:  # (data, label)
        data, labels = zip(*batch)
        return torch.stack(data), torch.stack(labels)


def collate_with_indices(batch):
    """Custom collate function for datasets with indices (GGAE)."""
    if len(batch[0]) == 4:  # (data, embedding, idx, label)
        data, emb, indices, labels = zip(*batch)
        return torch.stack(data), torch.stack(emb), torch.tensor(indices), torch.stack(labels)
    else:  # (data, idx, label)
        data, indices, labels = zip(*batch)
        return torch.stack(data), torch.tensor(indices), torch.stack(labels)


def create_dataloaders(train_data, val_data, test_data,
                       train_labels, val_labels, test_labels,
                       batch_size=64,
                       train_emb=None, val_emb=None, test_emb=None,
                       return_indices=False):
    """Create train, val, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset
    """
    if return_indices:
        DatasetCls = IndexedDataset
        collate_fn = collate_with_indices
    else:
        DatasetCls = BaseDataset
        collate_fn = collate_with_embeddings if train_emb is not None else None

    train_dataset = DatasetCls(train_data, train_labels, train_emb)
    val_dataset = DatasetCls(val_data, val_labels, val_emb)
    test_dataset = DatasetCls(test_data, test_labels, test_emb)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
