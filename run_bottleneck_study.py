#!/usr/bin/env python
"""
Unified Bottleneck Study for all datasets.

Compares autoencoder methods across varying latent dimensions.

Usage:
    python run_bottleneck_study.py --dataset spheres
    python run_bottleneck_study.py --dataset mnist --epochs 50
    python run_bottleneck_study.py --dataset all --epochs 50
    python run_bottleneck_study.py --dataset cifar10 --latent_dims 2 8 32
    python run_bottleneck_study.py --dataset mnist --results_dir my_results/exp1
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from config import get_config, DATASET_CONFIGS
from data import load_data

try:
    from data.base import normalize_features, compute_pca_embeddings, create_dataloaders
except ImportError:
    from data import normalize_features, compute_pca_embeddings, create_dataloaders

from models import build_model
from training import Trainer, get_latents, get_reconstructions
from evaluation import evaluate


# PCA percentages for image datasets
PCA_PERCENTAGES = [0.10, 0.30, 0.50, 0.80]  # 10%, 30%, 50%, 80%

# Input dimensions for calculating percentages
INPUT_DIMS = {
    'mnist': 784,
    'fmnist': 784,
    'cifar10': 3072,
}


class DatasetCache:
    """Cache for dataset and PCA embeddings to avoid recomputation."""
    
    def __init__(self):
        self.cached_data = {}  # {(dataset_name, arch_type): (train_data, test_data, train_labels, test_labels)}
        self.cached_embeddings = {}  # {(dataset_name, n_components): (train_emb, test_emb)}
    
    def get_data(self, dataset_name, config):
        """Get raw dataset, loading if not cached."""
        arch_type = config.get('arch_type', 'mlp')
        cache_key = (dataset_name, arch_type)
        
        if cache_key not in self.cached_data:
            print(f"  Loading {dataset_name} dataset (arch_type={arch_type})...")
            # Load without embeddings first
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
                dataset_name, config, with_embeddings=False
            )
            # Extract raw data
            train_data = train_dataset.data.numpy()
            test_data = test_dataset.data.numpy()
            train_labels = train_dataset.labels.numpy()
            test_labels = test_dataset.labels.numpy()
            self.cached_data[cache_key] = (train_data, test_data, train_labels, test_labels)
        
        return self.cached_data[cache_key]
    
    def get_embeddings(self, dataset_name, config, n_components):
        """Get PCA embeddings, computing if not cached."""
        cache_key = (dataset_name, n_components)
        
        if cache_key not in self.cached_embeddings:
            train_data, test_data, _, _ = self.get_data(dataset_name, config)
            # Flatten for PCA
            train_flat = train_data.reshape(train_data.shape[0], -1)
            test_flat = test_data.reshape(test_data.shape[0], -1)
            print(f"  Computing PCA embeddings (n_components={n_components})...")
            train_emb, test_emb = compute_pca_embeddings(train_flat, test_flat, n_components)
            self.cached_embeddings[cache_key] = (train_emb, test_emb)
        
        return self.cached_embeddings[cache_key]
    
    def get_dataloaders(self, dataset_name, config, with_embeddings=False, n_components=None):
        """Get dataloaders with optional cached embeddings."""
        train_data, test_data, train_labels, test_labels = self.get_data(dataset_name, config)
        
        train_emb, test_emb = None, None
        if with_embeddings and n_components is not None:
            train_emb, test_emb = self.get_embeddings(dataset_name, config, n_components)
        
        return create_dataloaders(
            train_data, test_data, train_labels, test_labels,
            batch_size=config.get('batch_size', 64),
            train_emb=train_emb, test_emb=test_emb
        )


# Global cache
_dataset_cache = DatasetCache()


def get_mmae_variants(dataset_name):
    """Get MMAE variant names for a dataset."""
    if dataset_name == 'spheres':
        # Fixed components for spheres (low-dim data)
        return ['mmae_pca2', 'mmae_pca10', 'mmae_pca50', 'mmae_pca80']
    elif dataset_name in INPUT_DIMS:
        # Percentage-based for image datasets
        dim = INPUT_DIMS[dataset_name]
        variants = ['mmae_pca2']  # Always include 2D
        for pct in PCA_PERCENTAGES:
            n_comp = int(dim * pct)
            variants.append(f'mmae_pca{n_comp}')
        return variants
    else:
        return ['mmae_pca2', 'mmae_pca10', 'mmae_pca50']


def get_default_models(dataset_name):
    """Get default model list for a dataset."""
    base_models = ['vanilla', 'topoae', 'rtdae']
    mmae_variants = get_mmae_variants(dataset_name)
    return base_models + mmae_variants


def run_single_experiment(model_name, dataset_name, latent_dim, config, device, cache):
    """Run single experiment for a model/dataset/latent_dim combo."""
    
    # Update config
    config = config.copy()
    config['latent_dim'] = latent_dim
    config['dataset_name'] = dataset_name
    
    # Check if MMAE
    is_mmae = model_name.startswith('mmae')
    pca_comp = None
    if is_mmae:
        # Extract PCA components from model name (e.g., mmae_pca78)
        pca_comp = int(model_name.split('_pca')[1]) if '_pca' in model_name else 80
        config['mmae_n_components'] = pca_comp
        config['model_name'] = 'mmae'
    else:
        config['model_name'] = model_name
    
    # Load data using cache
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = cache.get_dataloaders(
        dataset_name, config, with_embeddings=is_mmae, n_components=pca_comp
    )
    
    # Build model
    model = build_model(config['model_name'], config)
    
    # Train
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    trainer = Trainer(model, optimizer, device, model_name=config['model_name'])
    
    start_time = time.time()
    history = trainer.fit(
        train_loader, test_loader,
        n_epochs=config.get('n_epochs', 100),
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate
    latents, labels = get_latents(model, test_loader, device)
    originals, reconstructions, _ = get_reconstructions(model, test_loader, device)
    
    # Flatten for evaluation if needed
    originals_flat = originals.reshape(originals.shape[0], -1)
    reconstructions_flat = reconstructions.reshape(reconstructions.shape[0], -1)
    
    metrics = evaluate(originals_flat, latents, labels)
    metrics['reconstruction_error'] = float(np.mean((originals_flat - reconstructions_flat) ** 2))
    metrics['train_time_seconds'] = float(train_time)
    
    return metrics, latents, labels


def save_results(results, save_dir, filename='results.csv'):
    """Save results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, filename), index=False)
    return df


def load_existing_results(save_dir, filename='results.csv'):
    """Load existing results if present."""
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.to_dict('records')
    return []


def result_exists(results, model, latent_dim, dataset):
    """Check if a result already exists."""
    for r in results:
        if r['model'] == model and r['latent_dim'] == latent_dim and r['dataset'] == dataset:
            return True
    return False


def run_dataset_experiment(dataset_name, args, save_dir, cache):
    """Run bottleneck study for a single dataset."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Base config
    config = get_config(dataset_name)
    config['n_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    # Get models for this dataset
    if args.models:
        models = args.models
    else:
        models = get_default_models(dataset_name)
    
    print("\n" + "=" * 70)
    print(f"BOTTLENECK STUDY - {dataset_name.upper()}")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Latent dims: {args.latent_dims}")
    print(f"Runs per config: {args.n_runs}")
    print(f"Output: {save_dir}")
    print("=" * 70)
    
    # Load existing results for incremental progress
    all_results = load_existing_results(save_dir)
    if all_results:
        print(f"Loaded {len(all_results)} existing results")
    
    total_experiments = len(models) * len(args.latent_dims)
    completed = 0
    
    for model_name in models:
        for latent_dim in args.latent_dims:
            completed += 1
            
            # Skip if already done
            if result_exists(all_results, model_name, latent_dim, dataset_name):
                print(f"\n[{completed}/{total_experiments}] {model_name} | latent_dim={latent_dim} - SKIPPED (exists)")
                continue
            
            print(f"\n[{completed}/{total_experiments}] {model_name} | latent_dim={latent_dim}")
            
            run_results = []
            for run in range(args.n_runs):
                config['seed'] = args.seed + run
                torch.manual_seed(config['seed'])
                np.random.seed(config['seed'])
                
                try:
                    metrics, latents, labels = run_single_experiment(
                        model_name, dataset_name, latent_dim, config, args.device, cache
                    )
                    run_results.append(metrics)
                    print(f"  Run {run+1}/{args.n_runs}: recon={metrics['reconstruction_error']:.4f}, "
                          f"dcorr={metrics.get('distance_correlation', 0):.4f}")
                except Exception as e:
                    print(f"  Run {run+1}/{args.n_runs}: ERROR - {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if run_results:
                # Average across runs
                avg_result = {'model': model_name, 'latent_dim': latent_dim, 'dataset': dataset_name}
                for key in run_results[0].keys():
                    values = [r[key] for r in run_results if key in r]
                    avg_result[key] = float(np.mean(values))
                    avg_result[f'{key}_std'] = float(np.std(values))
                all_results.append(avg_result)
                
                # Save incrementally
                save_results(all_results, save_dir)
                print(f"  -> Saved ({len(all_results)} total results)")
    
    # Final save
    df = save_results(all_results, save_dir)
    
    # Save config
    run_config = {
        'dataset': dataset_name,
        'models': models,
        'latent_dims': args.latent_dims,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'n_runs': args.n_runs,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Bottleneck Study')
    parser.add_argument('--dataset', type=str, default='spheres',
                       choices=list(DATASET_CONFIGS.keys()) + ['all'],
                       help='Dataset to use (or "all" for all datasets)')
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[2, 3, 16, 32, 64, 128],
                       help='Latent dimensions to test')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_runs', type=int, default=3, help='Runs per config for averaging')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory to save results (default: results/bottleneck_study/<dataset>)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to compare (default: auto-select based on dataset)')
    args = parser.parse_args()
    
    # Determine datasets to run
    if args.dataset == 'all':
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]
    
    print("=" * 70)
    print("BOTTLENECK STUDY")
    print("=" * 70)
    print(f"Datasets: {datasets}")
    print("=" * 70)
    
    # Print MMAE variants for each dataset
    print("\nMMAE variants per dataset:")
    for ds in datasets:
        variants = get_mmae_variants(ds)
        if ds in INPUT_DIMS:
            dim = INPUT_DIMS[ds]
            print(f"  {ds} (dim={dim}): {variants}")
        else:
            print(f"  {ds}: {variants}")
    
    # Create shared cache
    cache = DatasetCache()
    
    all_dfs = []
    for dataset_name in datasets:
        # Determine save directory
        if args.results_dir:
            if len(datasets) > 1:
                save_dir = os.path.join(args.results_dir, dataset_name)
            else:
                save_dir = args.results_dir
        else:
            save_dir = os.path.join('results/bottleneck_study', dataset_name)
        
        df = run_dataset_experiment(dataset_name, args, save_dir, cache)
        all_dfs.append(df)
    
    # If multiple datasets, save combined results
    if len(datasets) > 1 and args.results_dir:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(args.results_dir, 'combined_results.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()