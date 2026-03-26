#!/usr/bin/env python
"""
Run a single experiment.

Usage:
    python run_experiment.py --dataset spheres --model vanilla
    python run_experiment.py --dataset mnist --model mmae --pca_components 50
    python run_experiment.py --dataset cifar10 --model topoae --epochs 100
    python run_experiment.py --dataset spheres --model ggae --ggae_bandwidth 50
    python run_experiment.py --dataset spheres --model mmae --latent_dim 3 --plot_3d
    python run_experiment.py --dataset klein_bottle --model mmae --latent_dim 3 --plot_3d --interactive
    python run_experiment.py --dataset klein_bottle --model mmae --no_eval --plot_3d --interactive
    
    # With best configs from hyperparameter search:
    python run_experiment.py --dataset spheres --model mmae --best_configs_dir results/hyperparam_search/spheres
    python run_experiment.py --dataset spheres --model topoae --best_configs_dir results/hyperparam_search/spheres --no_eval
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import get_config, DATASET_CONFIGS, MODEL_CONFIGS
from data import load_data, set_global_seed
from models import build_model
from training import Trainer, get_latents, get_reconstructions
from evaluation import evaluate


def load_best_config(best_configs_dir, model_name, latent_dim=2):
    """Load best hyperparameters from hyperparameter search results."""
    if best_configs_dir is None:
        return None
    
    config_path = os.path.join(best_configs_dir, f'{model_name}_dim{latent_dim}', 'best_config.json')
    
    if not os.path.exists(config_path):
        print(f"  [Config] No best config found at {config_path}, using defaults")
        return None
    
    try:
        with open(config_path, 'r') as f:
            best_config = json.load(f)
        print(f"  [Config] Loaded best config for {model_name}: {best_config.get('hyperparameters', {})}")
        return best_config.get('hyperparameters', {})
    except Exception as e:
        print(f"  [Config] Failed to load {config_path}: {e}")
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


def plot_latent_space_2d(latents, labels, title, save_path=None):
    """Plot 2D latent space."""
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='Spectral', s=20, alpha=1.0, edgecolors='black', linewidths=0.1)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_latent_space_3d(latents, labels, title, save_path=None, elev=30, azim=45):
    """Plot 3D latent space."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        latents[:, 0], latents[:, 1], latents[:, 2],
        c=labels, cmap='Spectral', s=10, alpha=0.7
    )
    
    fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    ax.set_title(title)
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_latent_space_3d_multiview(latents, labels, title, save_path=None):
    """Plot 3D latent space from multiple viewing angles."""
    fig = plt.figure(figsize=(16, 5))
    
    views = [
        (30, 45, 'View 1'),
        (30, 135, 'View 2'),
        (0, 0, 'Front'),
        (90, 0, 'Top'),
    ]
    
    for i, (elev, azim, view_name) in enumerate(views):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        scatter = ax.scatter(
            latents[:, 0], latents[:, 1], latents[:, 2],
            c=labels, cmap='Spectral', s=5, alpha=0.6
        )
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        ax.set_zlabel('z3')
        ax.set_title(view_name)
        ax.view_init(elev=elev, azim=azim)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_latent_space_3d_interactive(latents, labels, title, save_path=None):
    """Interactive 3D plot using plotly."""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Scatter3d(
        x=latents[:, 0],
        y=latents[:, 1],
        z=latents[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=labels,
            colorscale='Spectral',
            opacity=0.8,
            colorbar=dict(title='Label')
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='z1',
            yaxis_title='z2',
            zaxis_title='z3',
            aspectmode='data'
        ),
        width=900,
        height=700,
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")
    
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Run Single Experiment')
    parser.add_argument('--dataset', type=str, default='spheres',
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--model', type=str, default='vanilla',
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides best config if set)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides best config if set)')
    parser.add_argument('--pca_components', type=int, default=None,
                       help='For MMAE (overrides best config if set)')
    parser.add_argument('--ggae_bandwidth', type=float, default=None,
                       help='For GGAE (auto if not set)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/single_runs')
    parser.add_argument('--no_save', action='store_true', help='Do not save results')
    parser.add_argument('--no_eval', action='store_true', help='Skip evaluation, only visualize')
    parser.add_argument('--plot_3d', action='store_true', 
                       help='Generate 3D plots (requires latent_dim >= 3)')
    parser.add_argument('--interactive', action='store_true',
                       help='Show interactive 3D plot (requires plotly)')
    parser.add_argument('--best_configs_dir', type=str, default=None,
                       help='Directory with best configs from hyperparam search')
    args = parser.parse_args()
    
    # Validate 3D plotting
    if (args.plot_3d or args.interactive) and args.latent_dim < 3:
        print(f"Warning: 3D plotting requires latent_dim >= 3, but got {args.latent_dim}. "
              f"Setting latent_dim to 3.")
        args.latent_dim = 3
    
    # Setup — full determinism
    set_global_seed(args.seed)
    
    # Build config
    config = get_config(args.dataset, args.model)
    config['latent_dim'] = args.latent_dim
    config['n_epochs'] = args.epochs
    config['seed'] = args.seed
    
    # Load and apply best config if available
    best_params = load_best_config(args.best_configs_dir, args.model, args.latent_dim)
    config = apply_best_config(config, best_params, args.model)
    
    # Override with CLI args if explicitly provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.model == 'mmae' and args.pca_components is not None:
        config['mmae_n_components'] = args.pca_components
    
    # Set defaults if still not set
    if 'batch_size' not in config:
        config['batch_size'] = 64
    if 'learning_rate' not in config:
        config['learning_rate'] = 1e-3
    
    print("=" * 70)
    print(f"SINGLE EXPERIMENT: {args.model.upper()} on {args.dataset.upper()}")
    print("=" * 70)
    print(f"Config: latent_dim={args.latent_dim}, epochs={args.epochs}, "
          f"batch_size={config['batch_size']}, lr={config['learning_rate']}")
    if args.model == 'mmae':
        print(f"MMAE PCA components: {config.get('mmae_n_components', 'N/A')}")
        print(f"MMAE lambda: {config.get('mmae_lambda', 'N/A')}")
    if args.best_configs_dir:
        print(f"Best configs dir: {args.best_configs_dir}")
    if args.plot_3d:
        print("3D visualization enabled")
    if args.interactive:
        print("Interactive 3D plot enabled")
    if args.no_eval:
        print("Evaluation SKIPPED (visualization only)")
    print("=" * 70)
    
    # Load data
    is_mmae = args.model == 'mmae'
    is_ggae = args.model == 'ggae'
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        args.dataset, config,
        with_embeddings=is_mmae,
        return_indices=is_ggae
    )
    
    # For GGAE: compute bandwidth if not specified
    if is_ggae:
        train_data_flat = train_dataset.data.view(len(train_dataset), -1)
        
        if args.ggae_bandwidth is not None:
            bandwidth = args.ggae_bandwidth
        elif config.get('ggae_bandwidth') is not None:
            bandwidth = config['ggae_bandwidth']
        else:
            with torch.no_grad():
                n_sample = min(1000, len(train_data_flat))
                idx = torch.randperm(len(train_data_flat))[:n_sample]
                X_sample = train_data_flat[idx]
                dist_sq = torch.cdist(X_sample, X_sample).pow(2)
                mask = dist_sq > 0
                bandwidth = dist_sq[mask].median().item()
            print(f"GGAE auto bandwidth: {bandwidth:.2f}")
        
        config['ggae_bandwidth'] = bandwidth
        print(f"GGAE bandwidth: {bandwidth:.2f}")
    
    # Build model
    model = build_model(args.model, config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # GGAE: precompute kernel matrix
    if is_ggae:
        print("\nPrecomputing GGAE kernel matrix...")
        train_data_flat = train_dataset.data.view(len(train_dataset), -1)
        model.precompute_kernel(train_data_flat.to(args.device))
    
    # Train
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5)
    )
    trainer = Trainer(model, optimizer, args.device, model_name=args.model)
    
    print("\nTraining...")
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, n_epochs=args.epochs, verbose=True)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")

    # Get latents on held-out test set
    print("\nExtracting latent representations...")
    latents, labels = get_latents(model, test_loader, args.device)
    latents_train, labels_train = get_latents(model, train_loader, args.device)
    # Evaluate (unless skipped) — always on held-out test set
    metrics = {}
    if not args.no_eval:
        print("\nEvaluating on held-out test set...")
        originals, reconstructions, _ = get_reconstructions(model, test_loader, args.device)
        
        originals_flat = originals.reshape(originals.shape[0], -1)
        reconstructions_flat = reconstructions.reshape(reconstructions.shape[0], -1)
        
        metrics = evaluate(originals_flat, latents, labels)
        metrics['reconstruction_error'] = float(np.mean((originals_flat - reconstructions_flat) ** 2))
        metrics['train_time_seconds'] = float(train_time)
        
        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Reconstruction error: {metrics['reconstruction_error']:.6f}")
        print(f"Distance correlation: {metrics['distance_correlation']:.4f}")
        print(f"Triplet accuracy: {metrics['triplet_accuracy']:.4f}")
        if 'trustworthiness_10' in metrics:
            print(f"Trustworthiness (k=10): {metrics['trustworthiness_10']:.4f}")
        if 'continuity_10' in metrics:
            print(f"Continuity (k=10): {metrics['continuity_10']:.4f}")
        if 'knn_accuracy_5' in metrics:
            print(f"kNN accuracy (k=5): {metrics['knn_accuracy_5']:.4f}")
        if 'wasserstein_H0' in metrics:
            print(f"Wasserstein H0: {metrics['wasserstein_H0']:.4f}")
        if 'wasserstein_H1' in metrics:
            print(f"Wasserstein H1: {metrics['wasserstein_H1']:.4f}")
    else:
        metrics['train_time_seconds'] = float(train_time)
    
    # Prepare save directory
    save_dir = None
    if not args.no_save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.output_dir, args.dataset, args.model, timestamp)
        os.makedirs(save_dir, exist_ok=True)
    
    # Interactive plot
    if args.interactive:
        plot_title = f'{args.model.upper()} on {args.dataset}'
        interactive_path = os.path.join(save_dir, 'latent_space_interactive.html') if save_dir else None
        plot_latent_space_3d_interactive(latents[:, :3], labels, plot_title, interactive_path)
    
    # Save results
    if not args.no_save:
        # Save metrics
        if metrics:
            metrics_json = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_json, f, indent=2)
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({k: v for k, v in config.items() if not callable(v)}, f, indent=2)
        
        # Save latent plot
        plot_title = f'{args.model.upper()} on {args.dataset}'
        
        if args.latent_dim == 2:
            plot_latent_space_2d(
                latents_train, labels_train, 
                plot_title,
                os.path.join(save_dir, 'latent_space.png')
            )
        elif args.latent_dim >= 3 or args.plot_3d:
            plot_latent_space_3d(
                latents[:, :3], labels, 
                plot_title,
                os.path.join(save_dir, 'latent_space_3d.png')
            )
            plot_latent_space_3d_multiview(
                latents[:, :3], labels, 
                plot_title,
                os.path.join(save_dir, 'latent_space_3d_multiview.png')
            )
        
        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        
        print(f"\nResults saved to {save_dir}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()