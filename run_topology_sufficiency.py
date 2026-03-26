#!/usr/bin/env python
"""
Topology Sufficiency Experiment.

Tests whether MM-Reg's distance preservation becomes sufficient for topology
preservation as latent dimension grows, validating the Whitney-Stability argument.

Key prediction: at low d, topoae/rtdae beat mmae on Wasserstein metrics.
At moderate d (>= 2k_eff + 1), mmae converges to topoae/rtdae topologically.

Usage:
    python run_topology_sufficiency.py --dataset spheres
    python run_topology_sufficiency.py --dataset spheres --latent_dims 2 3 5 10 20 50
    python run_topology_sufficiency.py --dataset concentric_spheres --epochs 150
    python run_topology_sufficiency.py --dataset spheres --models vanilla topoae mmae
    python run_topology_sufficiency.py --dataset spheres --n_runs 5 --epochs 200
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import traceback

from config import get_config, DATASET_CONFIGS
from data import load_data

try:
    from data.base import normalize_features, compute_pca_embeddings, create_dataloaders
except ImportError:
    from data import normalize_features, compute_pca_embeddings, create_dataloaders

from models import build_model
from models.base import list_models
from training import Trainer, get_latents, get_reconstructions
from evaluation import evaluate


# ---------------------------------------------------------------------------
# Dataset cache (avoid reloading / recomputing PCA across runs)
# ---------------------------------------------------------------------------
class IndexDataset(torch.utils.data.Dataset):
    """Wraps any dataset to also return sample indices (needed by GGAE)."""
    def __init__(self, dataset):
        self.dataset = dataset
        # Proxy attributes so the rest of the code can access .data / .labels
        self.data = dataset.data
        self.labels = dataset.labels
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        # Original returns (x, label); we insert index in the middle → (x, idx, label)
        items = self.dataset[idx]
        return items[0], idx, items[-1]


class DataCache:
    def __init__(self):
        self._data = {}
        self._emb = {}

    def get_data(self, dataset_name, config):
        key = dataset_name
        if key not in self._data:
            print(f"  Loading {dataset_name}...")
            loader = load_data(dataset_name, config, with_embeddings=False)
            train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = loader
            self._data[key] = (
                train_ds.data.numpy(), test_ds.data.numpy(),
                train_ds.labels.numpy(), test_ds.labels.numpy(),
            )
        return self._data[key]

    def get_embeddings(self, dataset_name, config, n_components):
        key = (dataset_name, n_components)
        if key not in self._emb:
            tr, te, _, _ = self.get_data(dataset_name, config)
            tr_flat = tr.reshape(tr.shape[0], -1)
            te_flat = te.reshape(te.shape[0], -1)
            max_comp = min(n_components, tr_flat.shape[1], tr_flat.shape[0])
            print(f"  Computing PCA embeddings (n_components={max_comp})...")
            self._emb[key] = compute_pca_embeddings(tr_flat, te_flat, max_comp)
        return self._emb[key]

    def get_loaders(self, dataset_name, config, model_name):
        is_mmae = model_name.startswith("mmae")
        is_ggae = model_name == "ggae"

        tr, te, tr_l, te_l = self.get_data(dataset_name, config)
        tr_emb, te_emb = None, None
        if is_mmae:
            n_comp = config.get("mmae_n_components", 80)
            tr_emb, te_emb = self.get_embeddings(dataset_name, config, n_comp)

        train_ld, test_ld, train_ds, test_ds = create_dataloaders(
            tr, te, tr_l, te_l,
            batch_size=config.get("batch_size", 64),
            train_emb=tr_emb, test_emb=te_emb,
        )

        # GGAE needs (x, index, label) batches — wrap datasets and rebuild loaders
        if is_ggae:
            bs = config.get("batch_size", 64)
            train_ds = IndexDataset(train_ds)
            test_ds = IndexDataset(test_ds)
            train_ld = torch.utils.data.DataLoader(
                train_ds, batch_size=bs, shuffle=True, drop_last=True)
            test_ld = torch.utils.data.DataLoader(
                test_ds, batch_size=bs, shuffle=False, drop_last=False)

        return train_ld, test_ld, train_ds, test_ds


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def run_one(model_name, dataset_name, latent_dim, config, device, cache):
    """Train one model and return metrics dict."""
    cfg = config.copy()
    cfg["latent_dim"] = latent_dim

    # Handle mmae PCA variant naming (e.g. mmae_pca10)
    actual_model = model_name
    if model_name.startswith("mmae") and "_pca" in model_name:
        pca_comp = int(model_name.split("_pca")[1])
        cfg["mmae_n_components"] = pca_comp
        actual_model = "mmae"
    cfg["model_name"] = actual_model

    # Data
    train_ld, test_ld, train_ds, test_ds = cache.get_loaders(
        dataset_name, cfg, model_name
    )

    # Model
    model = build_model(actual_model, cfg)

    # GGAE: compute bandwidth and precompute kernel
    if actual_model == "ggae":
        train_data_flat = train_ds.data.view(len(train_ds), -1)
        if cfg.get("ggae_bandwidth") is None:
            with torch.no_grad():
                n_sample = min(1000, len(train_data_flat))
                idx = torch.randperm(len(train_data_flat))[:n_sample]
                X_sample = train_data_flat[idx]
                dist_sq = torch.cdist(X_sample, X_sample).pow(2)
                mask = dist_sq > 0
                bandwidth = dist_sq[mask].median().item()
            cfg["ggae_bandwidth"] = bandwidth
        model.precompute_kernel(train_data_flat.to(device))

    # Train
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    trainer = Trainer(model, opt, device, model_name=actual_model)
    t0 = time.time()
    trainer.fit(train_ld, test_ld, n_epochs=cfg.get("n_epochs", 100), verbose=False)
    train_time = time.time() - t0

    # Evaluate
    latents, labels = get_latents(model, test_ld, device)
    orig, recon, _ = get_reconstructions(model, test_ld, device)
    orig_flat = orig.reshape(orig.shape[0], -1)
    recon_flat = recon.reshape(recon.shape[0], -1)

    metrics = evaluate(orig_flat, latents, labels, compute_wasserstein=True)
    metrics["reconstruction_error"] = float(np.mean((orig_flat - recon_flat) ** 2))
    metrics["train_time_seconds"] = float(train_time)
    return metrics


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_experiment(dataset_name, models, latent_dims, args):
    save_dir = os.path.join(
        args.results_dir or "results/topology_sufficiency", dataset_name
    )
    os.makedirs(save_dir, exist_ok=True)

    config = get_config(dataset_name)
    config["n_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr

    cache = DataCache()

    # Load existing results for resume support
    csv_path = os.path.join(save_dir, "results.csv")
    if os.path.exists(csv_path):
        all_results = pd.read_csv(csv_path).to_dict("records")
        print(f"  Resuming: {len(all_results)} existing results loaded")
    else:
        all_results = []

    def already_done(m, d):
        return any(
            r["model"] == m and r["latent_dim"] == d for r in all_results
        )

    total = len(models) * len(latent_dims)
    idx = 0

    for model_name in models:
        for latent_dim in latent_dims:
            idx += 1
            if already_done(model_name, latent_dim):
                print(f"[{idx}/{total}] {model_name} d={latent_dim} — SKIP (exists)")
                continue

            print(f"[{idx}/{total}] {model_name} d={latent_dim}")
            run_metrics = []

            for run in range(args.n_runs):
                seed = args.seed + run
                torch.manual_seed(seed)
                np.random.seed(seed)
                config["seed"] = seed

                try:
                    m = run_one(
                        model_name, dataset_name, latent_dim,
                        config, args.device, cache,
                    )
                    run_metrics.append(m)
                    wh0 = m.get("wasserstein_H0", float("nan"))
                    wh1 = m.get("wasserstein_H1", float("nan"))
                    print(
                        f"  run {run+1}/{args.n_runs}: "
                        f"dcorr={m['distance_correlation']:.4f}  "
                        f"W_H0={wh0:.4f}  W_H1={wh1:.4f}  "
                        f"recon={m['reconstruction_error']:.4f}  "
                        f"time={m['train_time_seconds']:.1f}s"
                    )
                except Exception as e:
                    print(f"  run {run+1}/{args.n_runs}: ERROR — {e}")
                    traceback.print_exc()

            if run_metrics:
                row = {
                    "model": model_name,
                    "latent_dim": latent_dim,
                    "dataset": dataset_name,
                }
                for key in run_metrics[0]:
                    vals = [r[key] for r in run_metrics if key in r]
                    row[key] = float(np.mean(vals))
                    if len(vals) > 1:
                        row[f"{key}_std"] = float(np.std(vals))
                all_results.append(row)

                # Save incrementally
                pd.DataFrame(all_results).to_csv(csv_path, index=False)

    # Save run config
    run_cfg = {
        "dataset": dataset_name,
        "models": models,
        "latent_dims": latent_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_runs": args.n_runs,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"\nResults saved to {csv_path}")
    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Topology Sufficiency Experiment: topology vs bottleneck dimension"
    )
    parser.add_argument("--dataset", type=str, default="spheres",
                        help="Dataset name (must be registered)")
    parser.add_argument("--latent_dims", type=int, nargs="+",
                        default=[2, 3, 5, 10, 20, 50],
                        help="Latent dimensions to sweep")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to test (default: all available)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve models: use CLI list, or discover all registered models
    if args.models is None:
        available = list_models()
        # Default priority order; keep only what's registered
        preferred = ["vanilla", "topoae", "rtdae", "mmae", "geomae", "ggae", "spae"]
        models = [m for m in preferred if m in available]
        # Add any registered models we missed
        for m in available:
            if m not in models:
                models.append(m)
    else:
        models = args.models

    print("=" * 70)
    print("TOPOLOGY SUFFICIENCY EXPERIMENT")
    print("=" * 70)
    print(f"Dataset:     {args.dataset}")
    print(f"Models:      {models}")
    print(f"Latent dims: {args.latent_dims}")
    print(f"Runs/config: {args.n_runs}")
    print(f"Epochs:      {args.epochs}")
    print("=" * 70)

    run_experiment(args.dataset, models, args.latent_dims, args)


if __name__ == "__main__":
    main()