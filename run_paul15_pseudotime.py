#!/usr/bin/env python
"""
Paul15 Hematopoiesis: Latent space visualization with pseudotime coloring.
Trains all 6 autoencoder models and creates a 3x2 figure with continuous 
pseudotime gradient (progenitor → mature cells).

Usage:
    python run_paul15_pseudotime.py
    python run_paul15_pseudotime.py --epochs 50 --batch_size 256
    python run_paul15_pseudotime.py --best_configs_dir experiments/hyperparam_search/paul15/results
    python run_paul15_pseudotime.py --output_dir results/paul15_pseudotime
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Project imports
from config import get_config, DATASET_CONFIGS, MODEL_CONFIGS
from models import build_model
from training import Trainer, get_latents


def load_best_config(best_configs_dir, model_name, latent_dim=2):
    """Load best hyperparameters from hyperparameter search results."""
    if best_configs_dir is None:
        return None
    
    config_path = os.path.join(best_configs_dir, f'{model_name}_dim{latent_dim}', 'best_config.json')
    
    if not os.path.exists(config_path):
        print(f"    [Config] No best config at {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            best_config = json.load(f)
        params = best_config.get('hyperparameters', {})
        print(f"    [Config] Loaded: {params}")
        return params
    except Exception as e:
        print(f"    [Config] Failed to load: {e}")
        return None


def apply_best_config(config, best_params, model_name):
    """Apply best hyperparameters to config."""
    if best_params is None:
        return config
    
    # Common params
    if 'learning_rate' in best_params:
        config['learning_rate'] = best_params['learning_rate']
    if 'batch_size' in best_params:
        config['batch_size'] = int(best_params['batch_size'])
    
    # Model-specific params
    if model_name == 'mmae':
        if 'mmae_lambda' in best_params:
            config['mmae_lambda'] = best_params['mmae_lambda']
        if 'mmae_n_components' in best_params:
            config['mmae_n_components'] = int(best_params['mmae_n_components'])
    
    elif model_name == 'topoae':
        if 'topo_lambda' in best_params:
            config['topo_lambda'] = best_params['topo_lambda']
    
    elif model_name == 'rtdae':
        if 'rtd_lambda' in best_params:
            config['rtd_lambda'] = best_params['rtd_lambda']
        if 'rtd_dim' in best_params:
            config['rtd_dim'] = int(best_params['rtd_dim'])
        if 'rtd_card' in best_params:
            config['rtd_card'] = int(best_params['rtd_card'])
    
    elif model_name == 'geomae':
        if 'geom_lambda' in best_params:
            config['geom_lambda'] = best_params['geom_lambda']
    
    elif model_name == 'ggae':
        if 'gg_lambda' in best_params:
            config['ggae_lambda'] = best_params['gg_lambda']
        if 'gg_bandwidth' in best_params:
            config['ggae_bandwidth'] = best_params['gg_bandwidth']
    
    return config


def load_paul15_with_pseudotime(config, with_embeddings=False, return_indices=False):
    """Load Paul15 and compute diffusion pseudotime."""
    import scanpy as sc
    from sklearn.model_selection import train_test_split
    from data.base import normalize_features, compute_pca_embeddings, create_dataloaders
    
    seed = config.get("seed", 42)
    n_top_genes = config.get("n_top_genes", 2000)
    
    # Load and preprocess
    adata = sc.datasets.paul15()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    # Compute diffusion pseudotime
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.diffmap(adata)
    
    # Find root cell (likely progenitor - cluster with highest average of first diffusion component)
    cluster_labels = adata.obs['paul15_clusters'].cat.codes.values
    unique_clusters = np.unique(cluster_labels)
    dc1 = adata.obsm['X_diffmap'][:, 0]
    
    # Progenitors tend to be at one extreme of DC1
    cluster_means = [dc1[cluster_labels == c].mean() for c in unique_clusters]
    root_cluster = unique_clusters[np.argmin(cluster_means)]  # Most negative = progenitor
    root_cell = np.where(cluster_labels == root_cluster)[0][np.argmin(dc1[cluster_labels == root_cluster])]
    
    adata.uns['iroot'] = root_cell
    sc.tl.dpt(adata)
    
    # Extract data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    
    pseudotime = adata.obs['dpt_pseudotime'].values.astype(np.float32)
    pseudotime = np.nan_to_num(pseudotime, nan=np.nanmax(pseudotime))
    
    print(f"Paul15: {X.shape[0]} cells, {X.shape[1]} genes")
    print(f"Pseudotime range: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")
    
    # Split
    train_data, test_data, train_pt, test_pt = train_test_split(
        X, pseudotime, test_size=config.get("val_size", 0.15),
        random_state=seed
    )
    
    train_data, test_data = normalize_features(train_data, test_data)
    
    # PCA embeddings for MMAE
    train_emb, test_emb = None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", 50)
        train_emb, test_emb = compute_pca_embeddings(train_data, test_data, n_components)
    
    return create_dataloaders(
        train_data, test_data, train_pt, test_pt,
        batch_size=config.get("batch_size", 256),
        train_emb=train_emb, test_emb=test_emb,
        return_indices=return_indices
    )


def train_model(model_name, config, train_loader, test_loader, device, epochs):
    """Train a single model and return latents for full dataset (train + test)."""
    model = build_model(model_name, config)
    
    # GGAE kernel precomputation
    if model_name == 'ggae':
        train_data = train_loader.dataset.data.view(len(train_loader.dataset), -1)
        model.precompute_kernel(train_data.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                  weight_decay=config.get('weight_decay', 1e-5))
    trainer = Trainer(model, optimizer, device, model_name=model_name)
    
    start = time.time()
    trainer.fit(train_loader, test_loader, n_epochs=epochs, verbose=False)
    train_time = time.time() - start
    
    # Get latents for both train and test sets
    latents_train, pt_train = get_latents(model, train_loader, device)
    latents_test, pt_test = get_latents(model, test_loader, device)
    
    # Concatenate for full dataset visualization
    latents = np.concatenate([latents_train, latents_test], axis=0)
    pseudotime = np.concatenate([pt_train, pt_test], axis=0)
    
    return latents, pseudotime, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/paul15_pseudotime')
    parser.add_argument('--pca_components', type=int, default=50, help='PCA components for MMAE (if no best config)')
    parser.add_argument('--best_configs_dir', type=str, default=None,
                       help='Directory with best configs from hyperparam search')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Models to train
    MODELS = ['vanilla', 'mmae', 'topoae', 'rtdae', 'geomae', 'ggae']
    MODEL_DISPLAY = {
        'vanilla': 'Vanilla AE',
        'mmae': 'MMAE (Ours)',
        'topoae': 'TopoAE',
        'rtdae': 'RTD-AE', 
        'geomae': 'GeomAE',
        'ggae': 'GGAE'
    }
    
    # Build base config for paul15
    base_config = {
        'input_dim': 2000,
        'latent_dim': args.latent_dim,
        'hidden_dims': [512, 256, 128],
        'arch_type': 'mlp',
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'seed': args.seed,
        'val_size': 0.15,
        'n_top_genes': 2000,
        # Model-specific defaults
        'mmae_lambda': 1.0,
        'mmae_n_components': args.pca_components,
        'topo_lambda': 1.0,
        'rtd_lambda': 1.0,
        'rtd_dim': 1,
        'rtd_card': 50,
        'geom_lambda': 1.0,
        'ggae_lambda': 1.0,
        'ggae_bandwidth': None,  # Auto-compute
    }
    
    if args.best_configs_dir:
        print(f"Loading best configs from: {args.best_configs_dir}")
    
    results = {}
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Training {MODEL_DISPLAY[model_name]}")
        print('='*60)
        
        config = base_config.copy()
        
        # Load and apply best config if available
        best_params = load_best_config(args.best_configs_dir, model_name, args.latent_dim)
        config = apply_best_config(config, best_params, model_name)
        
        # Load data (with embeddings for MMAE, indices for GGAE)
        is_mmae = model_name == 'mmae'
        is_ggae = model_name == 'ggae'
        
        try:
            train_loader, test_loader, train_ds, test_ds = load_paul15_with_pseudotime(
                config, with_embeddings=is_mmae, return_indices=is_ggae
            )
        except Exception as e:
            print(f"  Failed to load data: {e}")
            continue
        
        # GGAE bandwidth (auto if not in best config)
        if is_ggae and config.get('ggae_bandwidth') is None:
            train_data = train_ds.data.view(len(train_ds), -1)
            with torch.no_grad():
                n_sample = min(1000, len(train_data))
                idx = torch.randperm(len(train_data))[:n_sample]
                X_sample = train_data[idx]
                dist_sq = torch.cdist(X_sample, X_sample).pow(2)
                mask = dist_sq > 0
                config['ggae_bandwidth'] = dist_sq[mask].median().item()
            print(f"  GGAE auto bandwidth: {config['ggae_bandwidth']:.2f}")
        
        try:
            latents, pseudotime, train_time = train_model(
                model_name, config, train_loader, test_loader, args.device, args.epochs
            )
            results[model_name] = {
                'latents': latents,
                'pseudotime': pseudotime,
                'train_time': train_time
            }
            print(f"  Done in {train_time:.1f}s")
        except Exception as e:
            print(f"  Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) == 0:
        print("No models trained successfully!")
        return
    
    # === Create Figure ===
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    axes = axes.flatten()
    
    # Custom colormap: blue (progenitor) → red (mature)
    cmap = plt.cm.coolwarm
    
    for i, model_name in enumerate(MODELS):
        ax = axes[i]
        
        if model_name not in results:
            ax.text(0.5, 0.5, f'{MODEL_DISPLAY[model_name]}\n(failed)', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        latents = results[model_name]['latents']
        pseudotime = results[model_name]['pseudotime']
        train_time = results[model_name]['train_time']
        
        # Normalize pseudotime to [0, 1]
        pt_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-8)
        
        scatter = ax.scatter(
            latents[:, 0], latents[:, 1],
            c=pt_norm, cmap=cmap, s=8, alpha=0.8,
            edgecolors='none', rasterized=True
        )
        
        ax.set_title(f'{MODEL_DISPLAY[model_name]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('$z_1$', fontsize=10)
        ax.set_ylabel('$z_2$', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(labelsize=8)
        
        # Add training time annotation
        ax.text(0.02, 0.98, f'{train_time:.0f}s', transform=ax.transAxes,
               fontsize=8, va='top', ha='left', color='gray')
    
    # Add single colorbar
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Pseudotime\n(Progenitor → Mature)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Title
    fig.suptitle('Paul15 Hematopoiesis: Latent Space with Differentiation Pseudotime', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    save_path = os.path.join(args.output_dir, 'paul15_pseudotime_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    main()