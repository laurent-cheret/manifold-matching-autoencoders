#!/usr/bin/env python
"""
Visualize Latent Spaces with Best Hyperparameters.

Generates 2D/3D scatter plots of latent representations colored by class.

Usage:
    # Visualize all models for a dataset
    python visualize_latent_spaces.py --dataset spheres --best_configs_dir experiments/hyperparam_search/spheres/results
    
    # Specific model and latent dim
    python visualize_latent_spaces.py --dataset mnist --best_configs_dir experiments/hyperparam_search/mnist/results --model mmae --latent_dim 2
    
    # 3D visualization
    python visualize_latent_spaces.py --dataset spheres --best_configs_dir experiments/hyperparam_search/spheres/results --latent_dim 3
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer


# Color palettes
COLORS_10 = plt.cm.tab10.colors
COLORS_20 = plt.cm.tab20.colors


def load_best_config(best_configs_dir, model_name, latent_dim):
    """Load best hyperparameters from search results."""
    config_path = Path(best_configs_dir) / f"{model_name}_dim{latent_dim}" / "best_config.json"
    
    if not config_path.exists():
        print(f"Warning: No best config found at {config_path}")
        return None
    
    with open(config_path) as f:
        return json.load(f)


def train_model(config, train_loader, test_loader, model_name, device, seed=42):
    """Train a model and return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = build_model(model_name, config)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    trainer = Trainer(model, optimizer, device=device, model_name=model_name)
    n_epochs = config.get('n_epochs', config.get('epochs', 100))
    
    print(f"  Training for {n_epochs} epochs...")
    trainer.fit(train_loader, test_loader, n_epochs=n_epochs, verbose=False)
    
    return model


def get_embeddings(model, data_loader, device):
    """Get latent embeddings for dataset."""
    model.eval()
    
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                data, _, labels = batch
            else:
                data, labels = batch
            
            data = data.to(device)
            latent = model.encode(data)
            
            all_latents.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())
    
    latents = np.vstack(all_latents)
    labels = np.concatenate(all_labels)
    
    return latents, labels


def plot_latent_2d(latents, labels, title, save_path, figsize=(8, 8)):
    """Create 2D scatter plot of latent space."""
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = COLORS_20 if n_classes > 10 else COLORS_10
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            latents[mask, 0], latents[mask, 1],
            c=[colors[i % len(colors)]],
            label=f'Class {int(label)}',
            alpha=0.8,
            s=25,
            edgecolors='none'
        )
    
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_title(title)
    
    if n_classes <= 15:
        ax.legend(loc='best', markerscale=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latent_3d(latents, labels, title, save_path, figsize=(10, 8)):
    """Create 3D scatter plot of latent space."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = COLORS_20 if n_classes > 10 else COLORS_10
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            latents[mask, 0], latents[mask, 1], latents[mask, 2],
            c=[colors[i % len(colors)]],
            label=f'Class {int(label)}',
            alpha=0.8,
            s=25,
            edgecolors='none'
        )
    
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    ax.set_title(title)
    
    if n_classes <= 15:
        ax.legend(loc='best', markerscale=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latent_grid(all_results, dataset_name, latent_dim, save_path, figsize=None):
    """Create grid of all models' latent spaces."""
    n_models = len(all_results)
    
    if n_models == 0:
        return
    
    # Determine grid size
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)
    
    is_3d = latent_dim == 3
    
    fig = plt.figure(figsize=figsize)
    
    for idx, (model_name, latents, labels) in enumerate(all_results):
        if is_3d:
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        else:
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        colors = COLORS_20 if n_classes > 10 else COLORS_10
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if is_3d:
                ax.scatter(
                    latents[mask, 0], latents[mask, 1], latents[mask, 2],
                    c=[colors[i % len(colors)]],
                    alpha=0.8, s=15, edgecolors='none'
                )
            else:
                ax.scatter(
                    latents[mask, 0], latents[mask, 1],
                    c=[colors[i % len(colors)]],
                    alpha=0.8, s=15, edgecolors='none'
                )
        
        ax.set_title(model_name.upper())
        ax.set_xticks([])
        ax.set_yticks([])
        if is_3d:
            ax.set_zticks([])
    
    plt.suptitle(f'{dataset_name.upper()} - Latent Dim {latent_dim}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid: {save_path}")


def visualize_latent_spaces(
    dataset_name,
    best_configs_dir,
    output_dir,
    models=None,
    latent_dims=None,
    device='cuda',
    seed=42,
    n_samples_viz=2000
):
    """
    Train models and visualize latent spaces.
    
    Args:
        dataset_name: Name of dataset
        best_configs_dir: Directory with best_config.json files
        output_dir: Directory to save visualizations
        models: List of models to visualize (default: all with configs)
        latent_dims: List of latent dims (default: [2, 3] from configs)
        device: torch device
        seed: Random seed
        n_samples_viz: Max samples to plot (for clarity)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover available configs
    best_configs_dir = Path(best_configs_dir)
    available = []
    for folder in best_configs_dir.iterdir():
        if folder.is_dir() and (folder / "best_config.json").exists():
            parts = folder.name.rsplit("_dim", 1)
            if len(parts) == 2:
                available.append((parts[0], int(parts[1])))
    
    if not available:
        print(f"No best configs found in {best_configs_dir}")
        return
    
    # Filter to 2D and 3D only for visualization
    available = [(m, d) for m, d in available if d in [2, 3]]
    
    if models:
        available = [(m, d) for m, d in available if m in models]
    if latent_dims:
        available = [(m, d) for m, d in available if d in latent_dims]
    
    print(f"\n{'='*60}")
    print(f"LATENT SPACE VISUALIZATION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Available: {available}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Group by latent dim for grid plots
    by_dim = {}
    for model_name, latent_dim in available:
        if latent_dim not in by_dim:
            by_dim[latent_dim] = []
        by_dim[latent_dim].append(model_name)
    
    for latent_dim, model_list in by_dim.items():
        print(f"\n--- Latent Dim {latent_dim} ---")
        
        grid_results = []
        
        for model_name in sorted(model_list):
            print(f"\nProcessing {model_name.upper()}...")
            
            # Load best config
            best_config = load_best_config(best_configs_dir, model_name, latent_dim)
            if best_config is None:
                continue
            
            best_params = best_config.get('hyperparameters', {})
            
            # Build config
            config = get_config(dataset_name, model_name)
            config['latent_dim'] = latent_dim
            config['device'] = device
            config['seed'] = seed
            
            # Apply best hyperparameters
            config['learning_rate'] = best_params.get('learning_rate', config.get('learning_rate', 1e-3))
            config['batch_size'] = int(best_params.get('batch_size', config.get('batch_size', 64)))
            
            if model_name == 'mmae':
                n_components = int(best_params.get('mmae_n_components', 80))
                # Cap to input dimensionality - 1
                max_components = config.get('input_dim', 1000) - 1
                config['mmae_n_components'] = min(n_components, max_components)
                config['mmae_lambda'] = best_params.get('mmae_lambda', 1.0)
                if n_components > max_components:
                    print(f"  Note: Capped PCA components from {n_components} to {max_components}")
            elif model_name == 'topoae':
                config['topo_lambda'] = best_params.get('topo_lambda', 1.0)
            elif model_name == 'rtdae':
                config['rtd_lambda'] = best_params.get('rtd_lambda', 1.0)
                config['rtd_dim'] = int(best_params.get('rtd_dim', 1))
                config['rtd_card'] = int(best_params.get('rtd_card', 50))
            
            # Load data
            needs_embeddings = model_name == 'mmae'
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
                dataset_name, config, with_embeddings=needs_embeddings
            )
            
            # Train model
            model = train_model(config, train_loader, test_loader, model_name, device, seed)
            
            # Get embeddings
            latents, labels = get_embeddings(model, test_loader, device)
            
            # Subsample for visualization if needed
            if len(latents) > n_samples_viz:
                np.random.seed(seed)
                idx = np.random.choice(len(latents), n_samples_viz, replace=False)
                latents = latents[idx]
                labels = labels[idx]
            
            # Individual plot
            title = f'{dataset_name.upper()} - {model_name.upper()} (d={latent_dim})'
            save_name = f'{dataset_name}_{model_name}_dim{latent_dim}.png'
            save_path = os.path.join(output_dir, save_name)
            
            if latent_dim == 2:
                plot_latent_2d(latents, labels, title, save_path)
            elif latent_dim == 3:
                plot_latent_3d(latents, labels, title, save_path)
            
            grid_results.append((model_name, latents, labels))
        
        # Grid plot comparing all models
        if len(grid_results) > 1:
            grid_path = os.path.join(output_dir, f'{dataset_name}_comparison_dim{latent_dim}.png')
            plot_latent_grid(grid_results, dataset_name, latent_dim, grid_path)
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize latent spaces')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--best_configs_dir', type=str, required=True,
                       help='Directory containing best_config.json files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/visualizations/{dataset})')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                       help='Specific model(s) to visualize')
    parser.add_argument('--latent_dim', type=int, nargs='+', default=None,
                       help='Specific latent dim(s) to visualize (2 or 3)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Max samples to visualize (default: 2000)')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f'results/visualizations/{args.dataset}'
    
    visualize_latent_spaces(
        dataset_name=args.dataset,
        best_configs_dir=args.best_configs_dir,
        output_dir=args.output_dir,
        models=args.model,
        latent_dims=args.latent_dim,
        device=args.device,
        seed=args.seed,
        n_samples_viz=args.n_samples
    )


if __name__ == '__main__':
    main()