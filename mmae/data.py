"""Datasets for the simplified MMAE repo.

Four datasets are bundled:
    - mnist, fmnist : auto-downloaded via torchvision
    - spheres       : 11 nested spheres in 101D (TopoAE-style synthetic)
    - mammoth       : 3D point cloud (Smithsonian mammoth scan)

Each loader returns a `Bundle` namedtuple with arrays + metadata so callers can
attach a reference embedding on top.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.environ.get("MMAE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache"))


@dataclass
class Bundle:
    X_train: np.ndarray  # (N, D) float32
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    input_dim: int
    name: str


# --------------------------------------------------------------------------- #
# Per-dataset loaders                                                          #
# --------------------------------------------------------------------------- #

def _normalize(train, val, test):
    """Standardize each feature using train statistics."""
    flat_train = train.reshape(len(train), -1)
    mean = flat_train.mean(axis=0, keepdims=True)
    std = flat_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    out_shape = train.shape[1:]
    return (
        ((train.reshape(len(train), -1) - mean) / std).reshape(-1, *out_shape).astype(np.float32),
        ((val.reshape(len(val), -1) - mean) / std).reshape(-1, *out_shape).astype(np.float32),
        ((test.reshape(len(test), -1) - mean) / std).reshape(-1, *out_shape).astype(np.float32),
    )


def _split(data, labels, seed=42, stratify=True):
    strat = labels if stratify else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        data, labels, test_size=0.15, random_state=seed, stratify=strat,
    )
    strat = y_tv if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=seed, stratify=strat,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_mnist_like(name: str, n_samples: Optional[int] = 20000, seed: int = 42) -> Bundle:
    from torchvision import datasets

    cls = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST}[name]
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    train_set = cls(root=raw_dir, train=True, download=True)
    test_set = cls(root=raw_dir, train=False, download=True)

    X_train_full = train_set.data.numpy().astype(np.float32) / 255.0
    y_train_full = train_set.targets.numpy().astype(np.int64)
    X_test = test_set.data.numpy().astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    rng = np.random.RandomState(seed)
    if n_samples is not None and n_samples < len(X_train_full):
        idx = rng.choice(len(X_train_full), n_samples, replace=False)
        X_train_full = X_train_full[idx]
        y_train_full = y_train_full[idx]

    # Carve a validation set from the training pool (15%).
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=seed, stratify=y_train_full,
    )

    # Subsample test for speed; default ~one-fifth of training pool.
    n_test = min(max(len(X_train) // 5, 2000), len(X_test))
    test_idx = rng.choice(len(X_test), n_test, replace=False)
    X_test, y_test = X_test[test_idx], y_test[test_idx]

    X_train, X_val, X_test = _normalize(X_train, X_val, X_test)
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)
    X_test = X_test.reshape(len(X_test), -1)

    return Bundle(X_train, X_val, X_test, y_train, y_val, y_test, X_train.shape[1], name)


def load_spheres(n_samples_per_sphere: int = 500, n_spheres: int = 11,
                 d: int = 100, r: float = 5.0, seed: int = 42) -> Bundle:
    """Nested spheres (TopoAE-style): n_spheres-1 small spheres inside one big sphere in d+1 dims."""
    rng = np.random.RandomState(seed)
    variance = 10 / np.sqrt(d)
    shifts = rng.normal(0, variance, [n_spheres, d + 1])

    spheres, labels = [], []
    for i in range(n_spheres - 1):
        pts = rng.randn(n_samples_per_sphere, d + 1)
        pts = r * pts / np.sqrt((pts ** 2).sum(axis=1, keepdims=True))
        spheres.append(pts + shifts[i])
        labels.append(np.full(n_samples_per_sphere, i, dtype=np.int64))

    big_n = 10 * n_samples_per_sphere
    big = rng.randn(big_n, d + 1)
    big = r * 5 * big / np.sqrt((big ** 2).sum(axis=1, keepdims=True))
    spheres.append(big)
    labels.append(np.full(big_n, n_spheres - 1, dtype=np.int64))

    data = np.concatenate(spheres, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0)

    X_train, X_val, X_test, y_train, y_val, y_test = _split(data, labels, seed=seed, stratify=True)
    X_train, X_val, X_test = _normalize(X_train, X_val, X_test)
    return Bundle(X_train, X_val, X_test, y_train, y_val, y_test, X_train.shape[1], "spheres")


def load_mammoth(n_samples: Optional[int] = 50000, seed: int = 42) -> Bundle:
    """3D Smithsonian mammoth point cloud."""
    url = "https://raw.githubusercontent.com/MNoichl/UMAP-examples-mammoth/master/mammoth_a.csv"
    cache_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "mammoth.csv")
    if not os.path.exists(cache):
        print(f"Downloading mammoth dataset from {url} …")
        urllib.request.urlretrieve(url, cache)

    import pandas as pd
    df = pd.read_csv(cache)
    data = df[["x", "y", "z"]].values.astype(np.float32)

    rng = np.random.RandomState(seed)
    if n_samples is not None and n_samples < len(data):
        data = data[rng.choice(len(data), n_samples, replace=False)]

    # Pseudo-labels = vertical-axis quantile (handy for visualization color).
    y_vals = data[:, 1]
    labels = pd.qcut(y_vals, q=10, labels=False, duplicates="drop").astype(np.int64)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = _split(data, labels, seed=seed, stratify=True)
    except ValueError:
        X_train, X_val, X_test, y_train, y_val, y_test = _split(data, labels, seed=seed, stratify=False)

    # Center + uniformly scale (preserves shape proportions).
    mean = X_train.mean(axis=0)
    scale = np.abs(X_train - mean).max() + 1e-8
    X_train = ((X_train - mean) / scale).astype(np.float32)
    X_val = ((X_val - mean) / scale).astype(np.float32)
    X_test = ((X_test - mean) / scale).astype(np.float32)

    return Bundle(X_train, X_val, X_test, y_train, y_val, y_test, 3, "mammoth")


_LOADERS = {
    "mnist": lambda **kw: load_mnist_like("mnist", **kw),
    "fmnist": lambda **kw: load_mnist_like("fmnist", **kw),
    "spheres": load_spheres,
    "mammoth": load_mammoth,
}


def list_datasets():
    return list(_LOADERS.keys())


def load_dataset(name: str, **kwargs) -> Bundle:
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list_datasets()}")
    return _LOADERS[name](**kwargs)


# --------------------------------------------------------------------------- #
# DataLoader factory with optional reference embeddings                        #
# --------------------------------------------------------------------------- #

class _AEDataset(Dataset):
    def __init__(self, X, y, ref=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.ref = torch.from_numpy(ref).float() if ref is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.ref is not None:
            return self.X[idx], self.ref[idx], self.y[idx]
        return self.X[idx], torch.zeros(0), self.y[idx]


def make_loaders(bundle: Bundle, ref_train=None, ref_val=None, ref_test=None,
                 batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = _AEDataset(bundle.X_train, bundle.y_train, ref_train)
    val_ds = _AEDataset(bundle.X_val, bundle.y_val, ref_val)
    test_ds = _AEDataset(bundle.X_test, bundle.y_test, ref_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
