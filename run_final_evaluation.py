#!/usr/bin/env python
"""
Final Training and Evaluation Script for Paper Results.

This script:
1. Trains models on FULL dataset with best hyperparameters
2. Runs training with multiple seeds for statistical robustness
3. Evaluates using fixed subsample size (500) with multiple repetitions
4. All metrics computed on SAME subsample per repetition for consistency
5. Reports mean ± std across seeds

Supports: vanilla, mmae, topoae, rtdae, geomae, ggae

Usage:
    # Run MMAE on MNIST with latent dim 2
    python run_final_evaluation.py \\
        --dataset mnist \\
        --best_configs_dir experiments/hyperparam_search/mnist/results \\
        --output_dir results/final/mnist_dim2 \\
        --latent_dim 2 \\
        --model mmae

    # Run all models on spheres with latent dim 2
    python run_final_evaluation.py \\
        --dataset spheres \\
        --best_configs_dir results/hyperparam_search/spheres \\
        --output_dir results/final/spheres_dim2 \\
        --latent_dim 2

    # High-dimensional spheres (10001D) with latent dim 2
    python run_final_evaluation.py \\
        --dataset spheres \\
        --spheres_dim 10000 \\
        --best_configs_dir results/hyperparam_search/spheres_10000d \\
        --output_dir results/final/spheres_10000d_dim2 \\
        --latent_dim 2 \\
        --model mmae ggae geomae \\
        --n_seeds 5

    # Higher latent dimension (64)
    python run_final_evaluation.py \\
        --dataset mnist \\
        --best_configs_dir results/hyperparam_search/mnist \\
        --output_dir results/final/mnist_dim64 \\
        --latent_dim 64 \\
        --model mmae topoae
"""

import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer, get_latents, get_reconstructions


# ============================================================
# EVALUATION FUNCTIONS (Paper-consistent)
# ============================================================

def distance_matrix(X):
    """Compute pairwise Euclidean distance matrix."""
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(X))


def neighbors_and_ranks(D, k):
    """Get k-nearest neighbors and ranks."""
    idx = np.argsort(D, axis=-1, kind='stable')
    return idx[:, 1:k+1], idx.argsort(axis=-1, kind='stable')


def trustworthiness(X, Z, k):
    """Trustworthiness metric."""
    n = X.shape[0]
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    Nx, Rx = neighbors_and_ranks(Dx, k)
    Nz, _ = neighbors_and_ranks(Dz, k)
    
    result = 0.0
    for i in range(n):
        for j in np.setdiff1d(Nz[i], Nx[i]):
            result += Rx[i, j] - k
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result


def continuity(X, Z, k):
    """Continuity metric."""
    return trustworthiness(Z, X, k)


def mrre(X, Z, k):
    """Mean Relative Rank Error."""
    n = X.shape[0]
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    
    Nx, Rx = neighbors_and_ranks(Dx, k)
    Nz, Rz = neighbors_and_ranks(Dz, k)
    
    mrre_zx = 0.0
    for i in range(n):
        for j in Nz[i]:
            mrre_zx += abs(Rx[i, j] - Rz[i, j]) / max(Rz[i, j], 1)
    
    mrre_xz = 0.0
    for i in range(n):
        for j in Nx[i]:
            mrre_xz += abs(Rx[i, j] - Rz[i, j]) / max(Rx[i, j], 1)
    
    C = n * sum(abs(2*j - n - 1) / j for j in range(1, k+1))
    
    return mrre_zx / C, mrre_xz / C


def distance_correlation(X, Z):
    """Distance correlation (Pearson correlation of pairwise distances)."""
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    mask = np.triu(np.ones_like(Dx), k=1) > 0
    return np.corrcoef(Dx[mask], Dz[mask])[0, 1]


def density_kl_divergence(X, Z, sigma=0.1):
    """
    KL divergence between density estimates.
    Distance normalized to [0,1] as per TopoAE paper.
    """
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    
    # Normalize distances to [0, 1]
    Dx = Dx / (Dx.max() + 1e-10)
    Dz = Dz / (Dz.max() + 1e-10)
    
    # Gaussian kernel density
    density_x = np.sum(np.exp(-(Dx ** 2) / sigma), axis=-1)
    density_x = density_x / (density_x.sum() + 1e-10)
    
    density_z = np.sum(np.exp(-(Dz ** 2) / sigma), axis=-1)
    density_z = density_z / (density_z.sum() + 1e-10)
    
    # KL divergence
    eps = 1e-10
    kl = np.sum(density_x * (np.log(density_x + eps) - np.log(density_z + eps)))
    
    return kl


def triplet_accuracy(X, Z, n_triplets=10000, seed=42):
    """Triplet distance ranking accuracy."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    
    if n < 3:
        return 0.0
    
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    
    n_triplets = min(n_triplets, n * (n-1) * (n-2) // 6)
    
    correct = 0
    for _ in range(n_triplets):
        i, j, k = rng.choice(n, 3, replace=False)
        x_order = Dx[i, j] < Dx[i, k]
        z_order = Dz[i, j] < Dz[i, k]
        if x_order == z_order:
            correct += 1
    
    return correct / n_triplets


def knn_accuracy(latent, labels, k=10):
    """kNN classification accuracy with cross-validation."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    # Check for degenerate cases
    n_unique_labels = len(np.unique(labels))
    n_splits = min(5, n_unique_labels)
    
    if n_splits < 2:
        return 0.0, 0.0
    
    if n_unique_labels > 100:
        # Labels look like indices, not classes
        return 0.0, 0.0
    
    # Check if latent representations have collapsed
    n_unique_points = len(np.unique(latent, axis=0))
    if n_unique_points < len(latent) * 0.5:
        print(f"  WARNING: Only {n_unique_points}/{len(latent)} unique latent points - representations may have collapsed")
    
    knn = KNeighborsClassifier(n_neighbors=min(k, len(latent)-1))
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(knn, latent, labels, cv=skf)
        return scores.mean(), scores.std()
    except Exception as e:
        print(f"  WARNING: kNN failed: {e}")
        return 0.0, 0.0


def clustering_metrics(latent, labels):
    """Compute clustering ARI, NMI, and purity."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    n_clusters = len(np.unique(labels))
    n_samples = len(latent)
    
    # Safety checks
    if n_clusters < 2:
        return 0.0, 0.0, 0.0
    
    if n_clusters > n_samples:
        print(f"  WARNING: More clusters ({n_clusters}) than samples ({n_samples}). Labels might be indices.")
        return 0.0, 0.0, 0.0
    
    if n_clusters > 100:
        print(f"  WARNING: Too many clusters ({n_clusters}). Skipping clustering metrics.")
        return 0.0, 0.0, 0.0
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(latent)
        
        ari = adjusted_rand_score(labels, pred_labels)
        nmi = normalized_mutual_info_score(labels, pred_labels, average_method='arithmetic')
        
        # Purity
        total_correct = 0
        for c in range(n_clusters):
            mask = pred_labels == c
            if mask.sum() > 0:
                total_correct += np.bincount(labels[mask].astype(int)).max()
        purity = total_correct / len(labels)
        
        return float(ari), float(nmi), float(purity)
    except Exception as e:
        print(f"  WARNING: Clustering failed: {e}")
        return 0.0, 0.0, 0.0


def silhouette_score_safe(latent, labels):
    """Compute silhouette score safely."""
    from sklearn.metrics import silhouette_score
    
    n_unique = len(np.unique(labels))
    if n_unique < 2 or len(latent) <= n_unique:
        return None
    
    try:
        return float(silhouette_score(latent, labels, metric='euclidean'))
    except:
        return None


def compute_persistence_diagrams(X, max_dim=1):
    """Compute persistence diagrams using gudhi."""
    try:
        import gudhi as gd
        D = distance_matrix(X)
        rips = gd.RipsComplex(distance_matrix=D, max_edge_length=np.inf)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        st.compute_persistence()
        
        diagrams = {}
        for dim in range(max_dim + 1):
            intervals = st.persistence_intervals_in_dimension(dim)
            finite = intervals[np.isfinite(intervals[:, 1])] if len(intervals) > 0 else np.array([]).reshape(0, 2)
            diagrams[dim] = finite
        return diagrams
    except ImportError:
        return None


def wasserstein_distance(dgm1, dgm2, order=1):
    """Compute Wasserstein distance between persistence diagrams."""
    try:
        import gudhi.wasserstein as gw
        return gw.wasserstein_distance(dgm1, dgm2, order=order)
    except ImportError:
        try:
            import gudhi.hera as hera
            return hera.wasserstein_distance(dgm1, dgm2, internal_p=order)
        except ImportError:
            return None


def evaluate_subsample(X, Z, labels, ks=[5, 10, 50, 100], sigmas=[0.01, 0.1, 1.0], seed=42):
    """
    Evaluate all metrics on a single subsample.
    All metrics computed on the SAME data for consistency.
    """
    results = {}
    n = X.shape[0]
    
    # Distance correlation (sample-size independent)
    results['distance_correlation'] = float(distance_correlation(X, Z))
    
    # Neighborhood metrics for each k
    for k in ks:
        if k < n:
            results[f'trustworthiness_{k}'] = float(trustworthiness(X, Z, k))
            results[f'continuity_{k}'] = float(continuity(X, Z, k))
            zx, xz = mrre(X, Z, k)
            results[f'mrre_zx_{k}'] = float(zx)
            results[f'mrre_xz_{k}'] = float(xz)
    
    # Density KL for each sigma
    for sigma in sigmas:
        sigma_str = str(sigma).replace('.', '_')
        results[f'density_kl_{sigma_str}'] = float(density_kl_divergence(X, Z, sigma))
    
    # Triplet accuracy
    results['triplet_accuracy'] = float(triplet_accuracy(X, Z, n_triplets=10000, seed=seed))
    
    # kNN accuracy for k=5, 10
    for k in [5, 10]:
        if k < n:
            mean_acc, std_acc = knn_accuracy(Z, labels, k)
            results[f'knn_accuracy_{k}'] = float(mean_acc)
    
    # Clustering metrics
    ari, nmi, purity = clustering_metrics(Z, labels)
    results['clustering_ari'] = ari
    results['clustering_nmi'] = nmi
    results['cluster_purity'] = purity
    
    # Silhouette
    sil = silhouette_score_safe(Z, labels)
    if sil is not None:
        results['silhouette_score'] = sil
    
    # Wasserstein distances on persistence diagrams
    X_dists = distance_matrix(X)
    Z_dists = distance_matrix(Z)
    X_norm = X / (np.percentile(X_dists, 90) + 1e-10)
    Z_norm = Z / (np.percentile(Z_dists, 90) + 1e-10)
    
    dgm_x = compute_persistence_diagrams(X_norm, max_dim=1)
    dgm_z = compute_persistence_diagrams(Z_norm, max_dim=1)
    
    if dgm_x is not None and dgm_z is not None:
        for dim in [0, 1]:
            w = wasserstein_distance(dgm_x[dim], dgm_z[dim], order=1)
            if w is not None:
                results[f'wasserstein_H{dim}'] = float(w)
    
    return results


def evaluate_with_subsampling(X_full, Z_full, labels_full, 
                               subsample_size=500, n_reps=10, seed=42):
    """
    Paper-consistent evaluation with subsampling.
    """
    n = len(X_full)
    rng = np.random.RandomState(seed)
    
    # If test set smaller than subsample size, use full test set
    if n <= subsample_size:
        print(f"  Test set ({n}) <= subsample size ({subsample_size}), using full test set")
        results = evaluate_subsample(X_full, Z_full, labels_full, seed=seed)
        return {k: (v, 0.0) for k, v in results.items()}
    
    # Collect results from each repetition
    all_results = []
    
    for rep in range(n_reps):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=subsample_size, 
                                         random_state=seed + rep)
            _, idx = next(sss.split(X_full, labels_full))
        except:
            idx = rng.choice(n, subsample_size, replace=False)
        
        X_sub = X_full[idx]
        Z_sub = Z_full[idx]
        labels_sub = labels_full[idx]
        
        rep_results = evaluate_subsample(X_sub, Z_sub, labels_sub, seed=seed+rep)
        all_results.append(rep_results)
    
    # Aggregate: mean and std for each metric
    final_results = {}
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    
    for key in all_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            final_results[key] = (float(np.mean(values)), float(np.std(values)))
    
    return final_results


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def load_best_config(best_configs_dir, model_name, latent_dim):
    """Load best hyperparameters from search results."""
    config_path = Path(best_configs_dir) / f"{model_name}_dim{latent_dim}" / "best_config.json"
    
    if not config_path.exists():
        print(f"Warning: No best config found at {config_path}")
        return None
    
    with open(config_path) as f:
        return json.load(f)


def compute_auto_bandwidth(train_data, n_sample=1000):
    """Compute bandwidth for GGAE based on median squared distance."""
    train_data_flat = train_data.view(len(train_data), -1)
    with torch.no_grad():
        n_sample = min(n_sample, len(train_data_flat))
        idx = torch.randperm(len(train_data_flat))[:n_sample]
        X_sample = train_data_flat[idx]
        dist_sq = torch.cdist(X_sample, X_sample).pow(2)
        mask = dist_sq > 0
        bandwidth = dist_sq[mask].median().item()
    return bandwidth


def train_single_seed(config, train_loader, test_loader, model_name, device, seed,
                      train_dataset=None):
    """Train a model with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config = config.copy()
    config['seed'] = seed
    
    # Build model
    model = build_model(model_name, config)
    
    # For GGAE, precompute kernel if needed
    if model_name == 'ggae' and train_dataset is not None:
        if hasattr(model, 'precompute_kernel'):
            train_data_flat = train_dataset.data.view(len(train_dataset), -1).to(device)
            model.precompute_kernel(train_data_flat)
            print(f"   GGAE kernel precomputed for {len(train_dataset)} samples")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Train
    trainer = Trainer(model, optimizer, device=device, model_name=model_name)
    n_epochs = config.get('n_epochs', config.get('epochs', 100))
    
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader, n_epochs=n_epochs, verbose=False)
    train_time = time.time() - start_time
    
    return model, train_time, history


def get_embeddings(model, data_loader, device):
    """Get latent embeddings for entire dataset."""
    model.eval()
    
    all_latents = []
    all_originals = []
    all_labels = []
    
    first_batch = True
    
    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch based on length and content
            # Possible formats:
            # - (data, labels) - vanilla, 2 elements
            # - (data, embeddings, labels) - MMAE, 3 elements, embeddings is 2D float
            # - (data, labels, indices) - GGAE, 3 elements, both labels and indices are 1D int
            # - (data, embeddings, labels, indices) - MMAE + indices, 4 elements
            
            if first_batch:
                print(f"  Batch format: {len(batch)} elements")
                for i, b in enumerate(batch):
                    if isinstance(b, torch.Tensor):
                        print(f"    [{i}] shape={b.shape}, dtype={b.dtype}")
                    else:
                        print(f"    [{i}] type={type(b)}")
                first_batch = False
            
            data = batch[0]
            
            if len(batch) == 2:
                labels = batch[1]
            elif len(batch) == 3:
                second, third = batch[1], batch[2]
                
                # Identify labels: should be 1D with integer-like dtype and few unique values
                # Embeddings: 2D float tensor
                # Indices: 1D int tensor with many unique values (0 to N-1)
                
                second_is_1d = len(second.shape) == 1
                third_is_1d = len(third.shape) == 1
                
                if not second_is_1d:
                    # second is 2D (embeddings), third must be labels
                    labels = third
                elif not third_is_1d:
                    # third is 2D (shouldn't happen), second is labels
                    labels = second
                else:
                    # Both are 1D - need to determine which is labels vs indices
                    # Labels typically have few unique values (classes)
                    # Indices have many unique values (0 to batch_size-1 or 0 to N-1)
                    second_unique = len(torch.unique(second))
                    third_unique = len(torch.unique(third))
                    
                    # The one with fewer unique values is likely labels
                    if second_unique <= third_unique:
                        labels = second
                    else:
                        labels = third
                        
            elif len(batch) == 4:
                # (data, embeddings, labels, indices)
                labels = batch[2]
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            
            data = data.to(device)
            latent = model.encode(data)
            
            all_originals.append(data.cpu().numpy())
            all_latents.append(latent.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.cpu().numpy())
            else:
                all_labels.append(np.array(labels))
    
    originals = np.vstack(all_originals)
    latents = np.vstack(all_latents)
    labels = np.concatenate(all_labels)
    
    # Flatten originals if needed
    originals = originals.reshape(originals.shape[0], -1)
    
    # Ensure labels are integers (for classification metrics)
    if labels.dtype in [np.float32, np.float64]:
        print(f"  WARNING: Labels are float type ({labels.dtype}), converting to int")
        labels = labels.astype(np.int64)
    
    # Verify labels look reasonable
    n_unique = len(np.unique(labels))
    print(f"  Labels: {len(labels)} samples, {n_unique} unique values, range [{labels.min()}, {labels.max()}]")
    
    if n_unique > 1000:
        print(f"  WARNING: Labels have {n_unique} unique values - might be indices instead of class labels!")
    
    return originals, latents, labels


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_final_evaluation(
    dataset_name,
    best_configs_dir,
    output_dir,
    models=None,
    latent_dim=2,
    n_seeds=3,
    eval_subsample=500,
    eval_reps=10,
    device='cuda',
    spheres_dim=None,
    epochs=None,
    seed=42
):
    """
    Run final training and evaluation pipeline.
    
    Args:
        dataset_name: Name of dataset
        best_configs_dir: Directory with best_config.json files from hyperparam search
        output_dir: Directory to save results
        models: List of models to evaluate (default: all with configs)
        latent_dim: Latent/bottleneck dimension (e.g., 2, 8, 32, 64)
        n_seeds: Number of training seeds
        eval_subsample: Subsample size for evaluation
        eval_reps: Number of evaluation repetitions
        device: torch device
        spheres_dim: For spheres dataset, the ambient dimension (e.g., 10000 for 10001D)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle spheres dimension override
    if dataset_name == 'spheres' and spheres_dim is not None:
        input_dim = spheres_dim + 1
        dataset_label = f"spheres_{spheres_dim}d"
    else:
        input_dim = DATASET_CONFIGS.get(dataset_name, {}).get('input_dim', 101)
        dataset_label = dataset_name
    
    # Discover available configs for the specified latent_dim
    best_configs_dir = Path(best_configs_dir)
    available = []
    
    # Look for configs matching the specified latent_dim
    for folder in best_configs_dir.iterdir():
        if folder.is_dir() and (folder / "best_config.json").exists():
            parts = folder.name.rsplit("_dim", 1)
            if len(parts) == 2:
                model_name = parts[0]
                config_dim = int(parts[1])
                if config_dim == latent_dim:
                    available.append(model_name)
    
    # Filter by specified models if provided
    if models:
        available = [m for m in available if m in models]
        # Also add models that were requested but don't have configs (will use defaults)
        for m in models:
            if m not in available:
                print(f"Note: No best config for {m} at dim{latent_dim}, will use defaults")
                available.append(m)
    
    if not available:
        print(f"No configs found in {best_configs_dir} for latent_dim={latent_dim}")
        if models:
            print(f"Will use default configs for specified models: {models}")
            available = models
        else:
            return
    
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION PIPELINE")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_label} (input_dim={input_dim})")
    print(f"Latent dimension: {latent_dim}")
    print(f"Models to evaluate: {available}")
    print(f"Seeds: {n_seeds}")
    print(f"Eval subsample: {eval_subsample}")
    print(f"Eval repetitions: {eval_reps}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for model_name in sorted(set(available)):
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name.upper()}, LATENT_DIM: {latent_dim}")
        print(f"{'='*60}")
        
        # Load best config (may be None if not found)
        best_config = load_best_config(best_configs_dir, model_name, latent_dim)
        if best_config is not None:
            best_params = best_config.get('hyperparameters', {})
            print(f"Best hyperparameters: {best_params}")
        else:
            best_params = {}
            print(f"Using default hyperparameters (no best config found)")
        
        # Build full config
        config = get_config(dataset_name, model_name)
        config['latent_dim'] = latent_dim
        config['device'] = device
        config['input_dim'] = input_dim
        
        # Apply spheres dimension override
        if dataset_name == 'spheres' and spheres_dim is not None:
            config['d'] = spheres_dim
        
        # Apply best hyperparameters
        config['learning_rate'] = best_params.get('learning_rate', config.get('learning_rate', 1e-3))
        config['batch_size'] = int(best_params.get('batch_size', config.get('batch_size', 64)))
        if epochs is not None:
            config['n_epochs'] = epochs
        
        # Model-specific hyperparameters
        if model_name == 'vanilla':
            pass  # No extra params
        
        elif model_name == 'mmae':
            config['mmae_n_components'] = int(best_params.get('mmae_n_components', 
                                              best_params.get('n_components', 80)))
            config['mmae_lambda'] = best_params.get('mmae_lambda', 
                                    best_params.get('lambda', 1.0))
        
        elif model_name == 'topoae':
            config['topo_lambda'] = best_params.get('topo_lambda', 
                                    best_params.get('lambda', 1.0))
        
        elif model_name == 'rtdae':
            config['rtd_lambda'] = best_params.get('rtd_lambda',
                                   best_params.get('lambda', 1.0))
            config['rtd_dim'] = int(best_params.get('rtd_dim', 1))
            config['rtd_card'] = int(best_params.get('rtd_card', 50))
        
        elif model_name == 'geomae':
            config['geom_lambda'] = best_params.get('geom_lambda',
                                    best_params.get('lambda', 1.0))
        
        elif model_name == 'ggae':
            config['ggae_lambda'] = best_params.get('ggae_lambda',
                                    best_params.get('lambda', 1.0))
            config['ggae_bandwidth'] = best_params.get('ggae_bandwidth',
                                       best_params.get('bandwidth', None))
        
        # Determine data loading options
        needs_embeddings = model_name == 'mmae'
        needs_indices = model_name == 'ggae'
        
        # Load data
        print(f"\nLoading dataset...")
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
            dataset_name, config,
            with_embeddings=needs_embeddings,
            return_indices=needs_indices
        )
        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        
        # For GGAE, compute bandwidth if not provided
        if model_name == 'ggae' and config.get('ggae_bandwidth') is None:
            bandwidth = compute_auto_bandwidth(train_dataset.data)
            config['ggae_bandwidth'] = bandwidth
            print(f"Auto-computed GGAE bandwidth: {bandwidth:.2f}")
        
        # Train with multiple seeds
        seed_results = []
        
        for seed_idx in range(n_seeds):
            run_seed = seed + seed_idx * 100
            print(f"\n--- Seed {seed_idx + 1}/{n_seeds} (seed={run_seed}) ---")

            # Train
            print(f"Training {model_name}...")
            model, train_time, history = train_single_seed(
                config, train_loader, val_loader, model_name, device, run_seed,
                train_dataset=train_dataset if model_name == 'ggae' else None
            )
            print(f"Training time: {train_time:.1f}s")
            
            # Get embeddings on test set
            print(f"Getting embeddings...")
            X_test, Z_test, labels_test = get_embeddings(model, test_loader, device)
            
            # Compute reconstruction error on full test set
            # Compute reconstruction error on full test set
            model.eval()
            with torch.no_grad():
                # Check if conv architecture - need to reshape
                arch_type = config.get('arch_type', 'mlp')
                if arch_type == 'conv':
                    input_shape = config.get('input_shape')  # e.g., (3, 32, 32)
                    test_tensor = torch.from_numpy(X_test).float().reshape(-1, *input_shape).to(device)
                else:
                    test_tensor = torch.from_numpy(X_test).float().to(device)
                
                recon = model.decode(model.encode(test_tensor)).cpu().numpy()
                recon_flat = recon.reshape(recon.shape[0], -1)
            recon_error = float(np.mean((X_test - recon_flat) ** 2))
            
            # Evaluate with subsampling
            print(f"Evaluating ({eval_reps} reps of {eval_subsample} samples)...")
            eval_results = evaluate_with_subsampling(
                X_test, Z_test, labels_test,
                subsample_size=eval_subsample,
                n_reps=eval_reps,
                seed=run_seed
            )
            
            # Add reconstruction error and train time
            eval_results['reconstruction_error'] = (recon_error, 0.0)
            eval_results['train_time'] = (train_time, 0.0)
            
            seed_results.append(eval_results)
            
            # Print key metrics for this seed
            print(f"  Recon error: {recon_error:.6f}")
            if 'distance_correlation' in eval_results:
                print(f"  Dist corr: {eval_results['distance_correlation'][0]:.4f}")
            if 'clustering_ari' in eval_results:
                print(f"  Clustering ARI: {eval_results['clustering_ari'][0]:.4f}")
            if 'knn_accuracy_10' in eval_results:
                print(f"  kNN acc (k=10): {eval_results['knn_accuracy_10'][0]:.4f}")
        
        # Aggregate across seeds
        print(f"\nAggregating {n_seeds} seeds...")
        final_metrics = aggregate_seed_results(seed_results)
        
        # Store result
        result_row = {
            'model': model_name,
            'latent_dim': latent_dim,
            'input_dim': input_dim,
            'n_seeds': n_seeds,
            'eval_subsample': eval_subsample,
            'eval_reps': eval_reps,
            **{k: v for k, v in best_params.items()},
        }
        
        # Add metrics (mean and std)
        for metric, (mean, std) in final_metrics.items():
            result_row[metric] = mean
            result_row[f'{metric}_std'] = std
        
        all_results.append(result_row)
        
        # Print final metrics for this model
        print(f"\n{'='*40}")
        print(f"FINAL: {model_name} dim={latent_dim}")
        print(f"{'='*40}")
        for metric in ['reconstruction_error', 'distance_correlation', 
                       'trustworthiness_10', 'continuity_10',
                       'knn_accuracy_10', 'clustering_ari', 
                       'density_kl_0_1', 'wasserstein_H0', 'wasserstein_H1',
                       'train_time']:
            if metric in final_metrics:
                mean, std = final_metrics[metric]
                print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'final_results_{dataset_label}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save summary JSON
    summary = {
        'dataset': dataset_name,
        'dataset_label': dataset_label,
        'input_dim': input_dim,
        'n_seeds': n_seeds,
        'eval_subsample': eval_subsample,
        'eval_reps': eval_reps,
        'timestamp': timestamp,
        'results': all_results
    }
    with open(os.path.join(output_dir, f'final_summary_{dataset_label}_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SAVED")
    print(f"{'='*80}")
    print(f"CSV: {results_path}")
    
    # Print summary table
    print_summary_table(results_df)
    
    return results_df


def aggregate_seed_results(seed_results):
    """Aggregate results across multiple seeds."""
    all_keys = set()
    for r in seed_results:
        all_keys.update(r.keys())
    
    final = {}
    for key in all_keys:
        values = [r[key][0] for r in seed_results if key in r]
        if values:
            final[key] = (float(np.mean(values)), float(np.std(values)))
    
    return final


def print_summary_table(df):
    """Print a formatted summary table."""
    
    display_metrics = [
        'reconstruction_error', 'distance_correlation',
        'trustworthiness_10', 'continuity_10',
        'knn_accuracy_10', 'clustering_ari',
        'density_kl_0_1', 'wasserstein_H0', 'train_time'
    ]
    
    print(f"\n{'='*140}")
    print("SUMMARY TABLE")
    print(f"{'='*140}")
    
    # Header
    header = f"{'Model':<12} {'Dim':>4}"
    for m in display_metrics:
        short_name = m.replace('reconstruction_', 'rec_').replace('distance_', 'dist_')
        short_name = short_name.replace('trustworthiness_', 'trust_').replace('continuity_', 'cont_')
        short_name = short_name.replace('knn_accuracy_', 'knn_').replace('clustering_', '')
        short_name = short_name.replace('density_kl_', 'kl_').replace('wasserstein_', 'W')
        short_name = short_name.replace('train_', 't_')
        header += f" {short_name[:12]:>12}"
    print(header)
    print("-" * 140)
    
    # Rows
    for _, row in df.sort_values(['latent_dim', 'model']).iterrows():
        line = f"{row['model']:<12} {int(row['latent_dim']):>4}"
        for m in display_metrics:
            if m in row and pd.notna(row[m]):
                std_key = f'{m}_std'
                if std_key in row and pd.notna(row[std_key]) and row[std_key] > 0.001:
                    line += f" {row[m]:>5.3f}±{row[std_key]:.2f}"
                else:
                    line += f" {row[m]:>12.4f}"
            else:
                line += f" {'-':>12}"
        print(line)
    
    print(f"{'='*140}")


def main():
    parser = argparse.ArgumentParser(description='Final Training and Evaluation for Paper')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--best_configs_dir', type=str, required=True,
                       help='Directory containing best_config.json files from hyperparam search')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results (e.g., results/final/spheres_10000d)')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                       help='Specific model(s) to evaluate (vanilla, mmae, topoae, rtdae, geomae, ggae)')
    parser.add_argument('--latent_dim', type=int, required=True,
                       help='Latent/bottleneck dimension to evaluate (e.g., 2, 8, 32, 64)')
    parser.add_argument('--n_seeds', type=int, default=3,
                       help='Number of training seeds (default: 3)')
    parser.add_argument('--eval_subsample', type=int, default=500,
                       help='Subsample size for evaluation (default: 500)')
    parser.add_argument('--eval_reps', type=int, default=10,
                       help='Number of evaluation repetitions (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--spheres_dim', type=int, default=None,
                       help='For spheres dataset: ambient dimension d (input_dim = d+1). '
                            'E.g., --spheres_dim 10000 for 10001D spheres')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override training epochs (default: read from best config, else 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    args = parser.parse_args()

    run_final_evaluation(
        dataset_name=args.dataset,
        best_configs_dir=args.best_configs_dir,
        output_dir=args.output_dir,
        models=args.model,
        latent_dim=args.latent_dim,
        n_seeds=args.n_seeds,
        eval_subsample=args.eval_subsample,
        eval_reps=args.eval_reps,
        device=args.device,
        spheres_dim=args.spheres_dim,
        epochs=args.epochs,
        seed=args.seed
    )


if __name__ == '__main__':
    main()