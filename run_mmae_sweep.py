#!/usr/bin/env python
"""
Unified MMAE Hyperparameter Sweep for all datasets.

Tests various learning rates, batch sizes, and PCA components.

Usage:
    python run_mmae_sweep.py --dataset spheres
    python run_mmae_sweep.py --dataset mnist --epochs 50
    python run_mmae_sweep.py --dataset cifar10 --pca_components 10 50 80
"""

import argparse
import os
import json
import time
from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer, get_latents, get_reconstructions
from evaluation import evaluate


def plot_latent_space(latents, labels, title, save_path, metrics=None):
    """Plot 2D latent space with metrics overlay."""
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='Spectral', s=25, alpha=0.8, edgecolors='none')
    plt.colorbar(scatter)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel('z1', fontsize=12)
    plt.ylabel('z2', fontsize=12)
    plt.axis('equal')
    
    # Add metrics text in bottom-left corner
    if metrics is not None:
        # Distance: distance correlation
        dc = metrics.get('distance_correlation', 0)
        # Neighborhood: average trustworthiness
        t10 = metrics.get('trustworthiness_10', 0)
        t50 = metrics.get('trustworthiness_50', 0)
        trust = (t10 + t50) / 2 if (t10 and t50) else (t10 or t50 or 0)
        # Topology: density KL
        kl = metrics.get('density_kl_0_1', 0)
        
        metrics_text = f"DC={dc:.3f}\nT={trust:.3f}\nKL={kl:.3f}"
        
        plt.text(0.03, 0.03, metrics_text, transform=plt.gca().transAxes,
                fontsize=13, fontweight='bold', verticalalignment='bottom',
                fontfamily='monospace', color='black',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='black', linewidth=1.5, alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_single_experiment(dataset_name, config, device):
    """Run single MMAE experiment."""
    
    # Load data with embeddings
    train_loader, val_loader, test_loader, _, _, _ = load_data(
        dataset_name, config, with_embeddings=True
    )
    
    # Build MMAE model
    model = build_model('mmae', config)
    
    # Train
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    trainer = Trainer(model, optimizer, device, model_name='mmae')
    
    start_time = time.time()
    history = trainer.fit(
        train_loader, test_loader,
        n_epochs=config['n_epochs'],
        verbose=False
    )
    train_time = time.time() - start_time
    
    # Evaluate
    latents, labels = get_latents(model, test_loader, device)
    originals, reconstructions, _ = get_reconstructions(model, test_loader, device)
    
    originals_flat = originals.reshape(originals.shape[0], -1)
    reconstructions_flat = reconstructions.reshape(reconstructions.shape[0], -1)
    
    metrics = evaluate(originals_flat, latents, labels)
    metrics['reconstruction_error'] = float(np.mean((originals_flat - reconstructions_flat) ** 2))
    metrics['train_time_seconds'] = float(train_time)
    
    return metrics, latents, labels, history


def main():
    parser = argparse.ArgumentParser(description='MMAE Hyperparameter Sweep')
    parser.add_argument('--dataset', type=str, default='spheres',
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[1e-4, 5e-4, 1e-3])
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--pca_components', type=int, nargs='+', default=[2, 10, 30, 50, 80])
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/mmae_sweep')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    os.makedirs(os.path.join(save_dir, 'latents'), exist_ok=True)
    
    # Base config
    config = get_config(args.dataset, 'mmae')
    config['latent_dim'] = args.latent_dim
    config['n_epochs'] = args.epochs
    config['seed'] = args.seed
    
    print("=" * 70)
    print(f"MMAE HYPERPARAMETER SWEEP - {args.dataset.upper()}")
    print("=" * 70)
    print(f"Learning rates: {args.learning_rates}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"PCA components: {args.pca_components}")
    print(f"Output: {save_dir}")
    print("=" * 70)
    
    all_results = []
    all_latents = {}
    
    combos = list(product(args.learning_rates, args.batch_sizes, args.pca_components))
    
    for i, (lr, bs, pca) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] lr={lr}, bs={bs}, pca={pca}")
        
        # Update config
        config['learning_rate'] = lr
        config['batch_size'] = bs
        config['mmae_n_components'] = pca
        
        try:
            metrics, latents, labels, history = run_single_experiment(
                args.dataset, config, args.device
            )
            
            result = {
                'learning_rate': lr,
                'batch_size': bs,
                'pca_components': pca,
                'dataset': args.dataset,
                **metrics
            }
            all_results.append(result)
            all_latents[(lr, bs, pca)] = (latents, labels)
            
            # Save latent plot with metrics
            if args.latent_dim == 2:
                plot_title = f'MMAE: lr={lr}, bs={bs}, pca={pca}'
                plot_path = os.path.join(save_dir, 'latents', f'latent_lr{lr}_bs{bs}_pca{pca}.png')
                plot_latent_space(latents, labels, plot_title, plot_path, metrics=metrics)
            
            print(f"  → recon={metrics['reconstruction_error']:.4f}, "
                  f"dcorr={metrics.get('distance_correlation', 0):.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    
    # Save config
    run_config = {
        'dataset': args.dataset,
        'learning_rates': args.learning_rates,
        'batch_sizes': args.batch_sizes,
        'pca_components': args.pca_components,
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'seed': args.seed,
        'timestamp': timestamp
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if len(df) > 0:
        print("\nBest by distance correlation:")
        best = df.loc[df['distance_correlation'].idxmax()]
        print(f"  lr={best['learning_rate']}, bs={int(best['batch_size'])}, "
              f"pca={int(best['pca_components'])} → dcorr={best['distance_correlation']:.4f}")
        
        print("\nBest by reconstruction error:")
        best = df.loc[df['reconstruction_error'].idxmin()]
        print(f"  lr={best['learning_rate']}, bs={int(best['batch_size'])}, "
              f"pca={int(best['pca_components'])} → recon={best['reconstruction_error']:.6f}")
    
    print(f"\nResults saved to {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()