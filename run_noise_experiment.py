#!/usr/bin/env python
"""
Noise robustness experiment: SPAE vs MMAE variants.

Tests hypothesis that SPAE degrades with noise (uses raw distances) 
while MMAE with aggressive PCA is robust (filters noise).

Usage:
    python run_noise_experiment.py --dataset spheres --pca_components 20 40 60 80 100
    python run_noise_experiment.py --dataset linked_tori --pca_components 20 50 80 100 --epochs 150
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer, get_latents, get_reconstructions
from evaluation import evaluate


NOISE_LEVELS = {
    'none': 0.0,
    'moderate': 0.2,
    'strong': 0.5,
    'extreme': 1.0,
}


def add_noise_to_loader(loader, noise_std, seed=None):
    """Create new loader with noise added to data."""
    if seed is not None:
        torch.manual_seed(seed)
    
    dataset = loader.dataset
    original_data = dataset.data.clone()
    
    if noise_std > 0:
        noise = torch.randn_like(original_data) * noise_std
        dataset.data = original_data + noise
    
    return loader, original_data


def restore_loader(loader, original_data):
    """Restore original data to loader."""
    loader.dataset.data = original_data


def get_data_std(loader):
    """Compute std of dataset for noise scaling."""
    data = loader.dataset.data
    return data.std().item()


def run_single_experiment(model_name, config, train_loader, test_loader, 
                          device, epochs, pca_components=None):
    """Run one model training and return metrics + latents."""
    
    # Update config for specific model
    cfg = config.copy()
    
    if model_name == 'mmae' and pca_components is not None:
        cfg['mmae_n_components'] = pca_components
    
    # Build model
    model = build_model(model_name, cfg)
    
    # Train
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.get('learning_rate', 1e-3),
        weight_decay=cfg.get('weight_decay', 1e-5)
    )
    trainer = Trainer(model, optimizer, device, model_name=model_name)
    
    history = trainer.fit(train_loader, test_loader, n_epochs=epochs, verbose=False)
    
    # Get latents
    latents, labels = get_latents(model, test_loader, device)
    
    # Evaluate
    originals, reconstructions, _ = get_reconstructions(model, test_loader, device)
    originals_flat = originals.reshape(originals.shape[0], -1)
    reconstructions_flat = reconstructions.reshape(reconstructions.shape[0], -1)
    
    metrics = evaluate(originals_flat, latents, labels)
    metrics['reconstruction_error'] = float(np.mean((originals_flat - reconstructions_flat) ** 2))
    
    return metrics, latents, labels


def main():
    parser = argparse.ArgumentParser(description='Noise Robustness Experiment')
    parser.add_argument('--dataset', type=str, default='spheres',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--pca_components', type=int, nargs='+', default=[20, 40, 60, 80, 100],
                        help='PCA component percentages for MMAE variants')
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/noise_experiment')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"NOISE ROBUSTNESS EXPERIMENT: {args.dataset.upper()}")
    print("=" * 70)
    print(f"PCA components: {args.pca_components}")
    print(f"Noise levels: {list(NOISE_LEVELS.keys())}")
    print(f"Epochs: {args.epochs}, Latent dim: {args.latent_dim}")
    print("=" * 70)
    
    # Build base config
    config = get_config(args.dataset, 'mmae')
    config['latent_dim'] = args.latent_dim
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    # Load clean data first to get dimensions and std
    train_loader, val_loader, test_loader, train_dataset, val_dataset, _ = load_data(
        args.dataset, config, with_embeddings=True
    )
    
    data_std = get_data_std(train_loader)
    input_dim = config['input_dim']
    print(f"Data std: {data_std:.4f}, Input dim: {input_dim}")
    
    # Build model list: SPAE + MMAE variants
    models_to_run = [('spae', None)]
    for pca_pct in args.pca_components:
        n_components = max(2, int(input_dim * pca_pct / 100))
        models_to_run.append((f'mmae_{pca_pct}pct', n_components))
    
    print(f"Models: {[m[0] for m in models_to_run]}")
    print("=" * 70)
    
    # Results storage
    all_results = []
    latent_cache = {}  # (model_name, noise_level) -> (latents, labels)
    
    # Run experiments
    for noise_name, noise_mult in NOISE_LEVELS.items():
        noise_std = noise_mult * data_std
        print(f"\n{'='*70}")
        print(f"NOISE LEVEL: {noise_name} (std={noise_std:.4f})")
        print("=" * 70)
        
        for model_label, pca_components in models_to_run:
            print(f"\n  Running {model_label}...", end=" ", flush=True)
            
            # Reload data fresh for each run
            train_loader, val_loader, test_loader, _, _, _ = load_data(
                args.dataset, config, with_embeddings=(pca_components is not None)
            )
            
            # Add noise
            train_loader, orig_train = add_noise_to_loader(
                train_loader, noise_std, seed=args.seed
            )
            test_loader, orig_test = add_noise_to_loader(
                test_loader, noise_std, seed=args.seed + 1
            )
            
            # Recompute PCA embeddings on noisy data if MMAE
            if pca_components is not None:
                from sklearn.decomposition import PCA
                train_data = train_loader.dataset.data.numpy()
                test_data = test_loader.dataset.data.numpy()
                
                train_flat = train_data.reshape(train_data.shape[0], -1)
                test_flat = test_data.reshape(test_data.shape[0], -1)
                
                n_comp = min(pca_components, train_flat.shape[1])
                pca = PCA(n_components=n_comp)
                train_emb = pca.fit_transform(train_flat).astype(np.float32)
                test_emb = pca.transform(test_flat).astype(np.float32)
                
                train_loader.dataset.embeddings = torch.from_numpy(train_emb)
                test_loader.dataset.embeddings = torch.from_numpy(test_emb)
                
                config['mmae_n_components'] = n_comp
            
            # Determine actual model type
            model_type = 'spae' if model_label == 'spae' else 'mmae'
            
            try:
                start_time = time.time()
                metrics, latents, labels = run_single_experiment(
                    model_type, config, train_loader, test_loader,
                    args.device, args.epochs, pca_components
                )
                train_time = time.time() - start_time
                
                # Store results
                result = {
                    'model': model_label,
                    'noise_level': noise_name,
                    'noise_std': noise_std,
                    'distance_correlation': metrics.get('distance_correlation', 0),
                    'triplet_accuracy': metrics.get('triplet_accuracy', 0),
                    'reconstruction_error': metrics.get('reconstruction_error', 0),
                    'train_time': train_time,
                }
                all_results.append(result)
                latent_cache[(model_label, noise_name)] = (latents, labels)
                
                print(f"dcorr={result['distance_correlation']:.3f}, "
                      f"triplet={result['triplet_accuracy']:.3f}")
                
            except Exception as e:
                print(f"FAILED: {e}")
                continue
    
    # Save results CSV
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    print(f"\nResults saved to {save_dir}/results.csv")
    
    # Create summary plots
    print("\nGenerating plots...")
    
    # Plot 1: Metric vs noise level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    noise_order = list(NOISE_LEVELS.keys())
    models = df['model'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for metric, ax in zip(['distance_correlation', 'triplet_accuracy'], axes):
        for i, model in enumerate(models):
            subset = df[df['model'] == model]
            subset = subset.set_index('noise_level').loc[noise_order].reset_index()
            ax.plot(noise_order, subset[metric], 'o-', label=model, color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Noise Robustness: {args.dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_vs_noise.png'), dpi=150)
    plt.close()
    
    # Plot 2: Latent space grid
    n_noise = len(NOISE_LEVELS)
    n_models = len(models)
    
    fig, axes = plt.subplots(n_noise, n_models, figsize=(3 * n_models, 3 * n_noise))
    if n_noise == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for i, noise_name in enumerate(noise_order):
        for j, model in enumerate(models):
            ax = axes[i, j]
            key = (model, noise_name)
            
            if key in latent_cache:
                latents, labels = latent_cache[key]
                ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='Spectral', 
                          s=5, alpha=0.7)
            
            if i == 0:
                ax.set_title(model, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{noise_name}', fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Latent Spaces: {args.dataset} (rows=noise, cols=model)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_grid.png'), dpi=150)
    plt.close()
    
    # Save config
    run_config = {
        'dataset': args.dataset,
        'pca_components': args.pca_components,
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'noise_levels': NOISE_LEVELS,
        'data_std': data_std,
        'input_dim': input_dim,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    pivot = df.pivot_table(index='noise_level', columns='model', 
                           values='distance_correlation', aggfunc='first')
    pivot = pivot.reindex(noise_order)
    print("\nDistance Correlation:")
    print(pivot.round(3).to_string())
    
    print(f"\nResults saved to {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()