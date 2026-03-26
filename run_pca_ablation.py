#!/usr/bin/env python
"""
PCA Reference Dimensionality Ablation Study for MMAE.

Shows how Wasserstein distance varies with PCA reference dimensionality,
demonstrating that 100% PCA is not always optimal.

Usage:
    # Run all datasets with defaults
    python run_pca_ablation.py --output_dir results/pca_ablation
    
    # Run specific datasets
    python run_pca_ablation.py --datasets mnist fmnist --output_dir results/pca_ablation
    
    # Adjust training params
    python run_pca_ablation.py --epochs 30 --n_seeds 5 --subsample_size 2000
    
    # Resume interrupted run (automatic)
    python run_pca_ablation.py --output_dir results/pca_ablation
"""

import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from scipy.spatial.distance import pdist, squareform

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer


def distance_matrix(X):
    """Compute pairwise Euclidean distance matrix."""
    return squareform(pdist(X))


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
        raise ImportError("gudhi is required for persistence computation: pip install gudhi")


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
            raise ImportError("gudhi wasserstein/hera required")


def subsample_dataset(data, labels, n_samples, seed=42):
    """Stratified subsample of dataset."""
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
        idx, _ = next(sss.split(data, labels))
        return data[idx], labels[idx]
    except:
        # Fallback to random sampling
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(data), min(n_samples, len(data)), replace=False)
        return data[idx], labels[idx]


def get_result_path(output_dir, dataset, pca_pct, seed):
    """Get path for a specific result."""
    return Path(output_dir) / 'results' / f'{dataset}_pca{int(pca_pct*100)}_seed{seed}.json'


def load_result(output_dir, dataset, pca_pct, seed):
    """Load result if exists."""
    path = get_result_path(output_dir, dataset, pca_pct, seed)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_result(output_dir, dataset, pca_pct, seed, result):
    """Save result immediately."""
    path = get_result_path(output_dir, dataset, pca_pct, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


def train_and_evaluate(dataset_name, pca_pct, config, train_loader, test_loader, 
                       test_data_flat, device, seed):
    """Train MMAE and evaluate Wasserstein distance."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build model
    model = build_model('mmae', config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Train
    trainer = Trainer(model, optimizer, device=device, model_name='mmae')
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader, n_epochs=config['epochs'], verbose=False)
    train_time = time.time() - start_time
    
    # Get embeddings on test set
    model.eval()
    all_latents = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            latent = model.encode(data)
            all_latents.append(latent.cpu().numpy())
    
    Z_test = np.vstack(all_latents)
    X_test = test_data_flat[:len(Z_test)]  # Match in case of batch alignment
    
    # Normalize for persistence computation (as in your evaluation.py)
    X_dists = distance_matrix(X_test)
    Z_dists = distance_matrix(Z_test)
    X_norm = X_test / (np.percentile(X_dists, 90) + 1e-10)
    Z_norm = Z_test / (np.percentile(Z_dists, 90) + 1e-10)
    
    # Compute persistence diagrams
    print(f"      Computing persistence diagrams...")
    dgm_X = compute_persistence_diagrams(X_norm, max_dim=1)
    dgm_Z = compute_persistence_diagrams(Z_norm, max_dim=1)
    
    # Compute Wasserstein distances
    wass_H0 = wasserstein_distance(dgm_X[0], dgm_Z[0], order=1)
    wass_H1 = wasserstein_distance(dgm_X[1], dgm_Z[1], order=1)
    
    # Compute reconstruction error - handle CNN vs MLP
    with torch.no_grad():
        arch_type = config.get('arch_type', 'mlp')
        input_shape = config.get('input_shape')
        
        if arch_type == 'conv' and input_shape is not None:
            # CNN expects shaped input (N, C, H, W)
            test_tensor = torch.from_numpy(X_test).float().reshape(-1, *input_shape).to(device)
        else:
            # MLP expects flat input
            test_tensor = torch.from_numpy(X_test).float().to(device)
        
        recon = model.decode(model.encode(test_tensor)).cpu().numpy()
        recon_flat = recon.reshape(recon.shape[0], -1)
    
    recon_error = float(np.mean((X_test - recon_flat) ** 2))
    
    result = {
        'dataset': dataset_name,
        'pca_pct': float(pca_pct),
        'pca_n_components': int(config['mmae_n_components']),
        'input_dim': int(config['input_dim']),
        'seed': int(seed),
        'wasserstein_H0': float(wass_H0),
        'wasserstein_H1': float(wass_H1),
        'reconstruction_error': float(recon_error),
        'train_time': float(train_time),
        'final_loss': float(history['train_loss'][-1]) if history['train_loss'] else None
    }
    
    return result


def run_pca_ablation(
    datasets,
    output_dir,
    pca_percentages,
    n_seeds=3,
    subsample_size=3000,
    latent_dim=2,
    learning_rate=1e-4,
    batch_size=256,
    mmae_lambda=1.0,
    epochs=50,
    device='cuda'
):
    """Run PCA dimensionality ablation study."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"PCA REFERENCE DIMENSIONALITY ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Datasets: {datasets}")
    print(f"PCA percentages: {[f'{p*100:.0f}%' for p in pca_percentages]}")
    print(f"Seeds: {n_seeds}")
    print(f"Subsample size: {subsample_size}")
    print(f"Latent dim: {latent_dim}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"MMAE lambda: {mmae_lambda}")
    print(f"Epochs: {epochs}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    all_results = []
    total_runs = len(datasets) * len(pca_percentages) * n_seeds
    completed_runs = 0
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Get input dimension
        input_dim = DATASET_CONFIGS.get(dataset_name, {}).get('input_dim')
        if input_dim is None:
            print(f"ERROR: Unknown dataset {dataset_name}")
            continue
        
        print(f"Input dimension: {input_dim}")
        
        # Load full dataset (without PCA for now, just to get test data)
        print(f"Loading dataset...")
        config = get_config(dataset_name, 'mmae')
        config['batch_size'] = batch_size
        config['latent_dim'] = latent_dim
        config['device'] = device
        
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
            dataset_name, config, with_embeddings=False, return_indices=False
        )
        
        # Subsample test set and get flat version for persistence
        test_data = test_dataset.data.numpy()
        test_labels = test_dataset.labels.numpy()
        
        if len(test_data) > subsample_size:
            print(f"Subsampling test set from {len(test_data)} to {subsample_size}...")
            test_data, test_labels = subsample_dataset(test_data, test_labels, subsample_size)
        
        test_data_flat = test_data.reshape(len(test_data), -1)
        print(f"Test set size: {len(test_data)}")
        
        for pca_pct in pca_percentages:
            n_components = max(2, int(input_dim * pca_pct))
            
            print(f"\n  PCA: {pca_pct*100:.0f}% ({n_components}/{input_dim} components)")
            
            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx * 100
                completed_runs += 1
                
                print(f"    Seed {seed_idx+1}/{n_seeds} (seed={seed}) [{completed_runs}/{total_runs}]")
                
                # Check if already computed
                existing = load_result(output_dir, dataset_name, pca_pct, seed)
                if existing is not None:
                    print(f"      ✓ Already computed (Wass H0: {existing['wasserstein_H0']:.4f})")
                    all_results.append(existing)
                    continue
                
                # Build config for this run
                run_config = config.copy()
                run_config['seed'] = seed
                run_config['learning_rate'] = learning_rate
                run_config['batch_size'] = batch_size
                run_config['mmae_n_components'] = n_components
                run_config['mmae_lambda'] = mmae_lambda
                run_config['epochs'] = epochs
                run_config['input_dim'] = input_dim
                
                # Reload data with PCA for this specific n_components
                print(f"      Loading data with PCA({n_components})...")
                train_loader_pca, val_loader_pca, test_loader_pca, _, _, _ = load_data(
                    dataset_name, run_config, with_embeddings=True, return_indices=False
                )
                
                # Train and evaluate
                print(f"      Training MMAE...")
                try:
                    result = train_and_evaluate(
                        dataset_name, pca_pct, run_config,
                        train_loader_pca, test_loader_pca,
                        test_data_flat, device, seed
                    )
                    
                    print(f"      ✓ Wass H0: {result['wasserstein_H0']:.4f}, "
                          f"H1: {result['wasserstein_H1']:.4f}, "
                          f"Recon: {result['reconstruction_error']:.4f} "
                          f"({result['train_time']:.1f}s)")
                    
                    # Save immediately
                    save_result(output_dir, dataset_name, pca_pct, seed, result)
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"      ✗ FAILED: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save consolidated results
    print(f"\n{'='*80}")
    print(f"SAVING CONSOLIDATED RESULTS")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    csv_path = output_dir / f'pca_ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    generate_plots(df, output_dir)
    
    return df


def generate_plots(df, output_dir):
    """Generate separate plots for H0 and H1."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style('whitegrid')
    
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    
    # Aggregate across seeds
    df_agg = df.groupby(['dataset', 'pca_pct']).agg({
        'wasserstein_H0': ['mean', 'std'],
        'wasserstein_H1': ['mean', 'std'],
        'reconstruction_error': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns]
    
    # Plot H0
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), squeeze=False)
    axes = axes.flatten()
    
    for i, dataset in enumerate(sorted(datasets)):
        ax = axes[i]
        data = df_agg[df_agg['dataset'] == dataset]
        
        ax.errorbar(data['pca_pct']*100, data['wasserstein_H0_mean'], 
                   yerr=data['wasserstein_H0_std'], 
                   marker='o', linewidth=2, capsize=5, markersize=8)
        
        ax.set_xlabel('PCA Components (% of input dim)', fontsize=11)
        ax.set_ylabel('Wasserstein Distance (H0)', fontsize=11)
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Mark minimum
        min_idx = data['wasserstein_H0_mean'].idxmin()
        min_pct = data.loc[min_idx, 'pca_pct'] * 100
        min_val = data.loc[min_idx, 'wasserstein_H0_mean']
        ax.axvline(min_pct, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.scatter([min_pct], [min_val], color='red', s=150, marker='*', 
                  zorder=5, edgecolors='black', linewidths=1)
    
    plt.tight_layout()
    h0_path = output_dir / 'pca_ablation_H0.png'
    plt.savefig(h0_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {h0_path}")
    plt.close()
    
    # Plot H1
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), squeeze=False)
    axes = axes.flatten()
    
    for i, dataset in enumerate(sorted(datasets)):
        ax = axes[i]
        data = df_agg[df_agg['dataset'] == dataset]
        
        ax.errorbar(data['pca_pct']*100, data['wasserstein_H1_mean'], 
                   yerr=data['wasserstein_H1_std'], 
                   marker='s', linewidth=2, capsize=5, markersize=8, color='darkorange')
        
        ax.set_xlabel('PCA Components (% of input dim)', fontsize=11)
        ax.set_ylabel('Wasserstein Distance (H1)', fontsize=11)
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Mark minimum
        min_idx = data['wasserstein_H1_mean'].idxmin()
        min_pct = data.loc[min_idx, 'pca_pct'] * 100
        min_val = data.loc[min_idx, 'wasserstein_H1_mean']
        ax.axvline(min_pct, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.scatter([min_pct], [min_val], color='red', s=150, marker='*', 
                  zorder=5, edgecolors='black', linewidths=1)
    
    plt.tight_layout()
    h1_path = output_dir / 'pca_ablation_H1.png'
    plt.savefig(h1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {h1_path}")
    plt.close()
    
    # Combined plot (all datasets, both H0 and H1)
    fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 8), squeeze=False)
    
    for i, dataset in enumerate(sorted(datasets)):
        data = df_agg[df_agg['dataset'] == dataset]
        
        # H0
        ax = axes[0, i]
        ax.errorbar(data['pca_pct']*100, data['wasserstein_H0_mean'], 
                   yerr=data['wasserstein_H0_std'], 
                   marker='o', linewidth=2, capsize=5, markersize=8, label='H0')
        ax.set_ylabel('Wasserstein H0', fontsize=11)
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # H1
        ax = axes[1, i]
        ax.errorbar(data['pca_pct']*100, data['wasserstein_H1_mean'], 
                   yerr=data['wasserstein_H1_std'], 
                   marker='s', linewidth=2, capsize=5, markersize=8, 
                   label='H1', color='darkorange')
        ax.set_xlabel('PCA Components (% of input dim)', fontsize=11)
        ax.set_ylabel('Wasserstein H1', fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    combined_path = output_dir / 'pca_ablation_combined.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PCA Reference Dimensionality Ablation')
    
    # Datasets
    parser.add_argument('--datasets', nargs='+', 
                       default=['mnist', 'fmnist', 'cifar10', 'paul15', 'pbmc3k'],
                       help='Datasets to evaluate')
    
    # PCA percentages
    parser.add_argument('--pca_percentages', nargs='+', type=float,
                       default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
                       help='PCA component percentages to test')
    
    # Training params
    parser.add_argument('--n_seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--subsample_size', type=int, default=3000,
                       help='Subsample size for each dataset')
    parser.add_argument('--latent_dim', type=int, default=2,
                       help='Latent dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--mmae_lambda', type=float, default=1.0,
                       help='MMAE regularization weight')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    run_pca_ablation(
        datasets=args.datasets,
        output_dir=args.output_dir,
        pca_percentages=args.pca_percentages,
        n_seeds=args.n_seeds,
        subsample_size=args.subsample_size,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mmae_lambda=args.mmae_lambda,
        epochs=args.epochs,
        device=args.device
    )


if __name__ == '__main__':
    main()