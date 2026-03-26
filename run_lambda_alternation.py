#!/usr/bin/env python
"""
MMAE Lambda Alternation Experiment.

Tests how latent space evolves when alternating between regularized (lambda=1.0)
and unregularized (lambda=0.0) training phases.

Investigates whether geometric structure is recovered after being relaxed,
or whether different configurations emerge.

Usage:
    python run_lambda_alternation.py --dataset earth --epochs_per_phase 20 --num_alternations 5
    python run_lambda_alternation.py --dataset spheres --epochs_per_phase 30 --num_alternations 4 --pca_components 50
    python run_lambda_alternation.py --dataset mammoth --epochs_per_phase 15 --num_alternations 6 --lambda_high 0.5
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
    scatter = ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='Spectral', 
                         s=5, alpha=0.7, edgecolors='none')
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


def create_summary_grid(latent_cache, labels, save_path, epochs_per_phase):
    """Create grid of latent spaces at phase transitions."""
    n = len(latent_cache)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (epoch, (latents, lam)) in enumerate(latent_cache.items()):
        ax = axes[i]
        ax.scatter(latents[:, 0], latents[:, 1], c=labels, 
                   cmap='Spectral', s=2, alpha=0.6)
        phase = epoch // epochs_per_phase
        phase_type = "ON" if lam > 0 else "OFF"
        ax.set_title(f'Epoch {epoch}\nPhase {phase} (λ={phase_type})', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    
    for i in range(len(latent_cache), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('MMAE Lambda Alternation', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MMAE Lambda Alternation Experiment')
    parser.add_argument('--dataset', type=str, default='earth',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs_per_phase', type=int, default=20,
                        help='Epochs per lambda phase')
    parser.add_argument('--num_alternations', type=int, default=5,
                        help='Number of alternations (e.g., 5 = ON, OFF, ON, OFF, ON)')
    parser.add_argument('--lambda_high', type=float, default=1.0,
                        help='Lambda value for "ON" phases')
    parser.add_argument('--lambda_low', type=float, default=0.0,
                        help='Lambda value for "OFF" phases')
    parser.add_argument('--pca_components', type=int, default=None,
                        help='PCA components for MMAE')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/lambda_alternation')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Generate 3D plots (requires latent_dim >= 3)')
    parser.add_argument('--gif_duration', type=int, default=100,
                        help='Duration per frame in ms')
    args = parser.parse_args()
    
    if args.plot_3d and args.latent_dim < 3:
        print(f"Warning: --plot_3d requires latent_dim >= 3. Setting latent_dim=3.")
        args.latent_dim = 3
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'latents'), exist_ok=True)
    
    # Build lambda schedule: alternating high/low
    lambda_schedule = []
    for i in range(args.num_alternations):
        lam = args.lambda_high if i % 2 == 0 else args.lambda_low
        lambda_schedule.append(lam)
    
    total_epochs = args.epochs_per_phase * args.num_alternations
    
    print("=" * 70)
    print(f"MMAE LAMBDA ALTERNATION: {args.dataset.upper()}")
    print("=" * 70)
    print(f"Lambda schedule: {lambda_schedule}")
    print(f"Epochs per phase: {args.epochs_per_phase}")
    print(f"Total epochs: {total_epochs}")
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
    
    config['mmae_lambda'] = lambda_schedule[0]
    model = build_model('mmae', config)
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.get('weight_decay', 1e-5)
    )
    
    trainer = Trainer(model, optimizer, args.device, model_name='mmae')
    
    # Storage
    latent_cache = {}  # epoch -> (latents, lambda)
    metrics_history = []
    gif_frames = []
    epoch_counter = 0
    
    # Get fixed labels
    _, labels = get_all_latents(model, train_dataset, args.device)
    
    # Run through alternating phases
    for phase_idx, lam in enumerate(lambda_schedule):
        phase_type = "ON" if lam == args.lambda_high else "OFF"
        print(f"\n{'='*50}")
        print(f"Phase {phase_idx + 1}/{args.num_alternations}: λ = {lam} ({phase_type})")
        print("=" * 50)
        
        model.lam = lam
        
        for epoch in range(1, args.epochs_per_phase + 1):
            train_loss = trainer.train_epoch(train_loader)
            epoch_counter += 1
            
            # Get latents with consistent ordering
            latents, _ = get_all_latents(model, train_dataset, args.device)
            
            title = f'MMAE λ={lam} (epoch {epoch_counter}, phase {phase_idx + 1})'
            
            if args.latent_dim >= 3 and args.plot_3d:
                frame = plot_latent_3d(latents, labels, title)
            else:
                frame = plot_latent_2d(latents, labels, title)
            
            gif_frames.append(frame)
            
            # Print progress
            print_interval = max(1, args.epochs_per_phase // 4)
            if epoch % print_interval == 0 or epoch == args.epochs_per_phase:
                print(f"  Epoch {epoch}/{args.epochs_per_phase} "
                      f"(total: {epoch_counter}) | "
                      f"loss: {train_loss['total_loss']:.4f}, "
                      f"recon: {train_loss.get('recon_loss', 0):.4f}, "
                      f"dist: {train_loss.get('dist_loss', 0):.4f}")
        
        # Save snapshot at end of phase
        latent_cache[epoch_counter] = (latents, lam)
        
        save_path = os.path.join(save_dir, 'latents', f'latent_epoch{epoch_counter:04d}_phase{phase_idx + 1}.png')
        if args.latent_dim >= 3 and args.plot_3d:
            plot_latent_3d(latents, labels, title, save_path)
        else:
            plot_latent_2d(latents, labels, title, save_path)
        
        metrics_history.append({
            'phase': phase_idx + 1,
            'lambda': lam,
            'phase_type': phase_type,
            'epoch_start': epoch_counter - args.epochs_per_phase + 1,
            'epoch_end': epoch_counter,
            'final_loss': train_loss['total_loss'],
            'recon_loss': train_loss.get('recon_loss', 0),
            'dist_loss': train_loss.get('dist_loss', 0),
        })
        
        print(f"  Saved PNG for phase {phase_idx + 1}")
    
    # Create GIF
    print(f"\nCreating GIF with {len(gif_frames)} frames...")
    gif_path = os.path.join(save_dir, 'lambda_alternation.gif')
    create_gif(gif_frames, gif_path, duration=args.gif_duration)
    
    # Create summary grid
    print("Creating summary grid...")
    create_summary_grid(latent_cache, labels, 
                       os.path.join(save_dir, 'phase_grid.png'),
                       args.epochs_per_phase)
    
    # Plot loss curves
    print("Creating loss plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    phases = [m['phase'] for m in metrics_history]
    colors = ['tab:green' if m['lambda'] > 0 else 'tab:red' for m in metrics_history]
    
    # Total loss
    axes[0].bar(phases, [m['final_loss'] for m in metrics_history], color=colors)
    axes[0].set_xlabel('Phase')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss per Phase')
    axes[0].set_xticks(phases)
    
    # Reconstruction loss
    axes[1].bar(phases, [m['recon_loss'] for m in metrics_history], color=colors)
    axes[1].set_xlabel('Phase')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss per Phase')
    axes[1].set_xticks(phases)
    
    # Distance loss
    axes[2].bar(phases, [m['dist_loss'] for m in metrics_history], color=colors)
    axes[2].set_xlabel('Phase')
    axes[2].set_ylabel('Distance Loss')
    axes[2].set_title('Distance Loss per Phase')
    axes[2].set_xticks(phases)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:green', label=f'λ={args.lambda_high} (ON)'),
                       Patch(facecolor='tab:red', label=f'λ={args.lambda_low} (OFF)')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle(f'MMAE Lambda Alternation: {args.dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    
    # Plot continuous loss over epochs
    # Re-run to get per-epoch losses (we only stored per-phase)
    # For now, create a phase timeline plot
    fig, ax = plt.subplots(figsize=(12, 4))
    
    for m in metrics_history:
        color = 'tab:green' if m['lambda'] > 0 else 'tab:red'
        ax.axvspan(m['epoch_start'] - 0.5, m['epoch_end'] + 0.5, 
                   alpha=0.3, color=color)
        ax.axvline(m['epoch_end'] + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlim(0.5, epoch_counter + 0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Phase')
    ax.set_title('Lambda Alternation Timeline')
    
    legend_elements = [Patch(facecolor='tab:green', alpha=0.3, label=f'λ={args.lambda_high} (ON)'),
                       Patch(facecolor='tab:red', alpha=0.3, label=f'λ={args.lambda_low} (OFF)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timeline.png'), dpi=150)
    plt.close()
    
    # Save config and metrics
    run_config = {
        'dataset': args.dataset,
        'latent_dim': args.latent_dim,
        'epochs_per_phase': args.epochs_per_phase,
        'num_alternations': args.num_alternations,
        'lambda_high': args.lambda_high,
        'lambda_low': args.lambda_low,
        'lambda_schedule': lambda_schedule,
        'pca_components': config.get('mmae_n_components'),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'total_epochs': epoch_counter,
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
    print(f"Total epochs trained: {epoch_counter}")
    print(f"GIF frames: {len(gif_frames)}")
    print(f"Phases: {args.num_alternations} ({' → '.join([f'λ={l}' for l in lambda_schedule])})")
    print(f"GIF: {gif_path}")
    print(f"\nResults saved to {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()