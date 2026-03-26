#!/usr/bin/env python
"""
MMAE Lambda Relaxation Experiment.

Tests how latent space evolves as regularization strength decreases.
Starts at lambda=1.0, decreases by 0.1 every N epochs until 0.1,
then decreases by factor of 10 (0.01, 0.001, etc.)

Outputs a GIF showing the evolution of the latent space (every epoch).

Usage:
    python run_lambda_relaxation.py --dataset earth --epochs_per_stage 10
    python run_lambda_relaxation.py --dataset earth --epochs_per_stage 10 --final_epochs 50
    python run_lambda_relaxation.py --dataset spheres --epochs_per_stage 20 --pca_components 50
"""

import argparse
import os
import json
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer, get_latents


LAMBDA_SCHEDULE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.0]


def get_all_latents(model, dataset, device, batch_size=256):
    """Get latents for entire dataset without shuffling."""
    model.eval()
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            
            x = x.to(device)
            z = model.encode(x)
            latents.append(z.cpu().numpy())
            labels.append(y.cpu().numpy() if hasattr(y, 'numpy') else y)
    
    return np.concatenate(latents), np.concatenate(labels)


def plot_latent_2d(latents, labels, title, save_path=None):
    """Plot 2D latent space. Returns PIL Image."""
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab20', 
                         s=10, alpha=1.0, edgecolors='none')
    plt.colorbar(scatter, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close()
    return img


def plot_latent_3d(latents, labels, title, save_path=None, elev=30, azim=45):
    """Plot 3D latent space. Returns PIL Image."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2],
                         c=labels, cmap='Spectral', s=3, alpha=0.6)
    fig.colorbar(scatter, ax=ax, shrink=0.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close()
    return img


def create_gif(images, save_path, duration=100):
    """Create GIF from list of PIL Images."""
    if not images:
        return
    
    frames = [images[0]] * 3 + images + [images[-1]] * 5
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved: {save_path} ({len(images)} frames)")


def create_summary_grid(latent_cache, labels, save_path):
    """Create grid of all latent spaces."""
    n = len(latent_cache)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (lam, latents) in enumerate(latent_cache.items()):
        ax = axes[i]
        ax.scatter(latents[:, 0], latents[:, 1], c=labels, 
                   cmap='Spectral', s=2, alpha=0.6)
        ax.set_title(f'λ = {lam}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    
    for i in range(len(latent_cache), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('MMAE Lambda Relaxation', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MMAE Lambda Relaxation Experiment')
    parser.add_argument('--dataset', type=str, default='earth',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs_per_stage', type=int, default=10,
                        help='Epochs to train at each lambda value')
    parser.add_argument('--final_epochs', type=int, default=None,
                        help='Epochs to train at lambda=0.0 (default: same as epochs_per_stage)')
    parser.add_argument('--pca_components', type=int, default=None,
                        help='PCA components for MMAE')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/lambda_relaxation')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Generate 3D plots (requires latent_dim >= 3)')
    parser.add_argument('--gif_duration', type=int, default=100,
                        help='Duration per frame in ms (default 100 for smooth GIF)')
    args = parser.parse_args()
    
    # Default final_epochs to epochs_per_stage if not specified
    if args.final_epochs is None:
        args.final_epochs = args.epochs_per_stage
    
    if args.plot_3d and args.latent_dim < 3:
        print(f"Warning: --plot_3d requires latent_dim >= 3. Setting latent_dim=3.")
        args.latent_dim = 3
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'latents'), exist_ok=True)
    
    print("=" * 70)
    print(f"MMAE LAMBDA RELAXATION: {args.dataset.upper()}")
    print("=" * 70)
    print(f"Lambda schedule: {LAMBDA_SCHEDULE}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print(f"Final epochs (λ=0.0): {args.final_epochs}")
    print(f"Latent dim: {args.latent_dim}")
    print("=" * 70)
    
    config = get_config(args.dataset, 'mmae')
    config['latent_dim'] = args.latent_dim
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    if args.pca_components is not None:
        config['mmae_n_components'] = args.pca_components
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, _ = load_data(
        args.dataset, config, with_embeddings=True
    )
    
    print(f"PCA components: {config.get('mmae_n_components', 'auto')}")
    
    config['mmae_lambda'] = LAMBDA_SCHEDULE[0]
    model = build_model('mmae', config)
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.get('weight_decay', 1e-5)
    )
    
    trainer = Trainer(model, optimizer, args.device, model_name='mmae')
    
    latent_cache = {}
    metrics_history = []
    gif_frames = []
    total_epochs = 0
    
    # Get fixed labels (consistent ordering)
    _, labels = get_all_latents(model, train_dataset, args.device)
    
    for lam in LAMBDA_SCHEDULE:
        print(f"\n{'='*50}")
        print(f"Lambda = {lam}")
        print("=" * 50)
        
        model.lam = lam
        
        # Use final_epochs for lambda=0.0, otherwise epochs_per_stage
        stage_epochs = args.final_epochs if lam == 0.0 else args.epochs_per_stage
        
        for epoch in range(1, stage_epochs + 1):
            train_loss = trainer.train_epoch(train_loader)
            total_epochs += 1
            
            # Get latents with consistent ordering
            latents, _ = get_all_latents(model, train_dataset, args.device)
            
            title = f'MMAE λ={lam} (epoch {total_epochs})'
            
            if args.latent_dim >= 3 and args.plot_3d:
                frame = plot_latent_3d(latents, labels, title)
            else:
                frame = plot_latent_2d(latents, labels, title)
            
            gif_frames.append(frame)
            
            # Print more frequently for final stage
            print_interval = max(1, stage_epochs // 5) if lam == 0.0 else max(1, stage_epochs // 2)
            if epoch % print_interval == 0 or epoch == stage_epochs:
                print(f"  Epoch {epoch}/{stage_epochs} "
                      f"(total: {total_epochs}) | "
                      f"loss: {train_loss['total_loss']:.4f}, "
                      f"recon: {train_loss.get('recon_loss', 0):.4f}, "
                      f"dist: {train_loss.get('dist_loss', 0):.4f}")
        
        latent_cache[lam] = latents
        
        lam_str = f"{lam:.4f}".replace('.', 'p')
        save_path = os.path.join(save_dir, 'latents', f'latent_lam{lam_str}.png')
        
        if args.latent_dim >= 3 and args.plot_3d:
            plot_latent_3d(latents, labels, title, save_path)
        else:
            plot_latent_2d(latents, labels, title, save_path)
        
        metrics_history.append({
            'lambda': lam,
            'total_epochs': total_epochs,
            'stage_epochs': stage_epochs,
            'final_loss': train_loss['total_loss'],
            'recon_loss': train_loss.get('recon_loss', 0),
            'dist_loss': train_loss.get('dist_loss', 0),
        })
        
        print(f"  Saved PNG for λ={lam}")
    
    print(f"\nCreating GIF with {len(gif_frames)} frames...")
    gif_path = os.path.join(save_dir, 'lambda_relaxation.gif')
    create_gif(gif_frames, gif_path, duration=args.gif_duration)
    
    print("Creating summary grid...")
    create_summary_grid(latent_cache, labels, os.path.join(save_dir, 'lambda_grid.png'))
    
    print("Creating loss plots...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    lambdas = [m['lambda'] for m in metrics_history]
    
    axes[0].plot(lambdas, [m['final_loss'] for m in metrics_history], 'o-', linewidth=2)
    axes[0].set_xlabel('Lambda')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss vs Lambda')
    axes[0].set_xscale('symlog', linthresh=0.01)
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(lambdas, [m['recon_loss'] for m in metrics_history], 'o-', linewidth=2, color='tab:orange')
    axes[1].set_xlabel('Lambda')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction vs Lambda')
    axes[1].set_xscale('symlog', linthresh=0.01)
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lambdas, [m['dist_loss'] for m in metrics_history], 'o-', linewidth=2, color='tab:green')
    axes[2].set_xlabel('Lambda')
    axes[2].set_ylabel('Distance Loss')
    axes[2].set_title('Distance Loss vs Lambda')
    axes[2].set_xscale('symlog', linthresh=0.01)
    axes[2].invert_xaxis()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'MMAE Lambda Relaxation: {args.dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    
    run_config = {
        'dataset': args.dataset,
        'latent_dim': args.latent_dim,
        'epochs_per_stage': args.epochs_per_stage,
        'final_epochs': args.final_epochs,
        'pca_components': config.get('mmae_n_components'),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'lambda_schedule': LAMBDA_SCHEDULE,
        'total_epochs': total_epochs,
        'gif_frames': len(gif_frames),
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pt'))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total epochs trained: {total_epochs}")
    print(f"GIF frames: {len(gif_frames)}")
    print(f"Lambda stages: {len(LAMBDA_SCHEDULE)}")
    print(f"Final stage epochs: {args.final_epochs}")
    print(f"GIF: {gif_path}")
    print(f"\nResults saved to {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()