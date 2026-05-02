"""Training loop with optional per-epoch latent snapshots (for animations)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .data import Bundle, load_dataset, make_loaders
from .model import Autoencoder
from .reference import compute_reference


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Result:
    model: Autoencoder
    bundle: Bundle
    history: dict
    snapshots: list                       # list of (epoch, z, y) on snapshot_split
    device: str
    reference_method: Optional[str] = None  # 'pca' | 'umap' | 'tsne' | None
    ref_train: Optional[np.ndarray] = None  # reference embedding of training set
    ref_val: Optional[np.ndarray] = None
    ref_test: Optional[np.ndarray] = None
    snapshot_split: str = "test"


def _encode_dataset(model, X, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            out.append(model.encode(batch).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_run(
    dataset: str = "mnist",
    regularizer: str = "mmae",
    reference: str = "pca",
    ref_dim: int = 2,
    lam: float = 1.0,
    latent_dim: int = 2,
    hidden_dims=(512, 256, 128),
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: str = "auto",
    snapshot_every: int = 0,
    mm_normalize: bool = True,
    snapshot_split: str = "test",
    n_samples: Optional[int] = None,
    verbose: bool = True,
) -> Result:
    """Run a full training session and return model + history (+ snapshots)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    kwargs = {"seed": seed}
    if n_samples is not None and dataset in ("mnist", "fmnist", "mammoth"):
        # n_samples<=0 means "use the full dataset" -> pass None to skip subsampling.
        kwargs["n_samples"] = None if n_samples <= 0 else n_samples
    bundle = load_dataset(dataset, **kwargs)

    # Reference embedding (only when regularizer is on).
    ref_train, ref_val, ref_test = None, None, None
    ref_method_used = None
    if regularizer == "mmae":
        ref_method_used = reference
        if verbose:
            print(f"Computing {reference.upper()} reference (dim={ref_dim}) on training set ...")
        ref_train, ref_eval = compute_reference(
            bundle.X_train, method=reference, dim=ref_dim, seed=seed,
            X_eval=[bundle.X_val, bundle.X_test],
        )
        ref_val, ref_test = ref_eval[0], ref_eval[1]

    train_loader, val_loader, test_loader = make_loaders(
        bundle, ref_train=ref_train, ref_val=ref_val, ref_test=ref_test, batch_size=batch_size,
    )

    model = Autoencoder(
        input_dim=bundle.input_dim, latent_dim=latent_dim,
        hidden_dims=hidden_dims, regularizer=regularizer, lam=lam,
        mm_normalize=mm_normalize,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"Training {regularizer} AE on {dataset} | "
            f"latent_dim={latent_dim} | epochs={epochs} | "
            f"batch_size={batch_size} | lr={lr} | params={n_params:,} | device={device}"
        )

    snap_X, snap_y = (
        (bundle.X_test, bundle.y_test) if snapshot_split == "test"
        else (bundle.X_train, bundle.y_train)
    )
    snapshots = []
    if snapshot_every and snapshot_every > 0:
        z0 = _encode_dataset(model, snap_X, device)
        snapshots.append((0, z0, snap_y.copy()))

    history = {"train_total": [], "train_recon": [], "train_mm": [],
               "val_total": [], "val_recon": [], "val_mm": []}

    for epoch in range(1, epochs + 1):
        model.train()
        agg = {"total": [], "recon": [], "mm": []}
        for x, ref, _y in train_loader:
            x = x.to(device)
            ref = ref.to(device) if ref.numel() > 0 else None
            optimizer.zero_grad()
            loss, comps = model(x, ref)
            loss.backward()
            optimizer.step()
            for k, v in comps.items():
                if k in agg:
                    agg[k].append(v)

        history["train_total"].append(float(np.mean(agg["total"])) if agg["total"] else float("nan"))
        history["train_recon"].append(float(np.mean(agg["recon"])) if agg["recon"] else float("nan"))
        history["train_mm"].append(float(np.mean(agg["mm"])) if agg["mm"] else float("nan"))

        model.eval()
        v_agg = {"total": [], "recon": [], "mm": []}
        with torch.no_grad():
            for x, ref, _y in val_loader:
                x = x.to(device)
                ref = ref.to(device) if ref.numel() > 0 else None
                _loss, comps = model(x, ref)
                for k, v in comps.items():
                    if k in v_agg:
                        v_agg[k].append(v)
        history["val_total"].append(float(np.mean(v_agg["total"])) if v_agg["total"] else float("nan"))
        history["val_recon"].append(float(np.mean(v_agg["recon"])) if v_agg["recon"] else float("nan"))
        history["val_mm"].append(float(np.mean(v_agg["mm"])) if v_agg["mm"] else float("nan"))

        if snapshot_every and (epoch % snapshot_every == 0 or epoch == epochs):
            z = _encode_dataset(model, snap_X, device)
            snapshots.append((epoch, z, snap_y.copy()))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs):
            msg = f"  epoch {epoch:>4d}/{epochs}  train={history['train_total'][-1]:.4f}  val={history['val_total'][-1]:.4f}"
            if regularizer == "mmae":
                msg += f"  recon={history['train_recon'][-1]:.4f}  mm={history['train_mm'][-1]:.4f}"
            print(msg)

    return Result(
        model=model, bundle=bundle, history=history, snapshots=snapshots, device=device,
        reference_method=ref_method_used,
        ref_train=ref_train, ref_val=ref_val, ref_test=ref_test,
        snapshot_split=snapshot_split,
    )
