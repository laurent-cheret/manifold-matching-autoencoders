#!/usr/bin/env python
"""
Enhanced Progressive vs Fixed MMAE Comparison.

Features:
- Side-by-side latent space evolution with tight layout
- GIF animations and static checkpoints
- Quantitative metrics overlay with large, readable fonts
- Support for 2D and 3D latent spaces
- Multiple fixed variants comparison

Usage:
    # Quick test
    python experiment_progressive_mmae_v2.py --dataset spheres --epochs_per_step 5 --max_pca 30
    
    # Full comparison
    python experiment_progressive_mmae_v2.py --dataset spheres --fixed_pca 5 10 20 50 --epochs_per_step 15
    
    # MNIST comparison
    python experiment_progressive_mmae_v2.py --dataset mnist --start_pca 10 --max_pca 200 --pca_step 20
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from evaluation import evaluate


# ============== Styling ==============
COLORS = {
    'Progressive': '#e41a1c',  # Red
    'Fixed-5D': '#377eb8',     # Blue
    'Fixed-10D': '#4daf4a',    # Green
    'Fixed-20D': '#984ea3',    # Purple
    'Fixed-50D': '#ff7f00',    # Orange
    'Fixed-100D': '#a65628',   # Brown
}

METRIC_DISPLAY = {
    'distance_correlation': ('DC', '↑', '.3f'),
    'knn_avg': ('kNN', '↑', '.3f'),
    'reconstruction_error': ('Rec', '↓', '.4f'),
    'wasserstein_H0': ('W₀', '↓', '.3f'),
    'wasserstein_H1': ('W₁', '↓', '.3f'),
}


# ============== Helper Functions ==============

def compute_pca_embeddings(train_data, test_data, n_components):
    """Compute PCA embeddings."""
    train_flat = train_data.reshape(train_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)
    pca = PCA(n_components=n_components)
    train_emb = pca.fit_transform(train_flat)
    test_emb = pca.transform(test_flat)
    return train_emb.astype(np.float32), test_emb.astype(np.float32)


def get_latents(model, data, device):
    """Extract latent representations."""
    model.eval()
    with torch.no_grad():
        data_t = torch.from_numpy(data).float().to(device)
        return model.encode(data_t).cpu().numpy()


def quick_metrics(X, Z, labels):
    """Fast metrics computation."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    
    X_flat = X.reshape(len(X), -1)
    
    # Distance correlation (Spearman on distances)
    n_sample = min(500, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)
    dx = pdist(X_flat[idx])
    dz = pdist(Z[idx])
    dc = spearmanr(dx, dz)[0]
    
    # kNN accuracy
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Z, labels)
    knn_acc = knn.score(Z, labels)
    
    # Reconstruction (approximated if we have access)
    return {
        'distance_correlation': float(dc) if not np.isnan(dc) else 0.0,
        'knn_avg': float(knn_acc),
    }


def full_metrics(X, Z, labels, compute_wass=True):
    """Full metrics computation."""
    metrics = evaluate(X.reshape(len(X), -1), Z, labels, ks=[5, 10], compute_wasserstein=compute_wass)
    knn_keys = [k for k in metrics if k.startswith('knn_accuracy_') and not k.endswith('_std')]
    metrics['knn_avg'] = np.mean([metrics[k] for k in knn_keys]) if knn_keys else 0.0
    return metrics


# ============== Training Classes ==============

class MMAETrainer:
    """Base MMAE trainer with PCA reference."""
    
    def __init__(self, model, optimizer, device, train_data, test_data):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_data_flat = train_data.reshape(len(train_data), -1)
        self.test_data_flat = test_data.reshape(len(test_data), -1)
        self.train_indices = np.arange(len(train_data))
    
    def set_reference(self, train_emb, test_emb):
        """Set reference PCA embeddings."""
        self.train_emb = train_emb
        self.test_emb = test_emb
    
    def train_epoch(self, batch_size=64, shuffle=True):
        """Train one epoch."""
        self.model.train()
        
        if shuffle:
            np.random.shuffle(self.train_indices)
        
        total_loss = 0
        n_batches = 0
        
        for start_idx in range(0, len(self.train_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(self.train_indices))
            batch_idx = self.train_indices[start_idx:end_idx]
            
            data = torch.from_numpy(self.train_data_flat[batch_idx]).float().to(self.device)
            ref = torch.from_numpy(self.train_emb[batch_idx]).float().to(self.device)
            
            self.optimizer.zero_grad()
            loss, _ = self.model(data, ref)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)


class ProgressiveSchedule:
    """Manages progressive PCA schedule."""
    
    def __init__(self, train_data, test_data, start_pca, max_pca, pca_step):
        self.train_data = train_data.reshape(len(train_data), -1)
        self.test_data = test_data.reshape(len(test_data), -1)
        
        # Build schedule
        self.schedule = list(range(start_pca, max_pca + 1, pca_step))
        if self.schedule[-1] != max_pca:
            self.schedule.append(max_pca)
        
        # Precompute all embeddings
        print(f"Precomputing PCA embeddings for schedule: {self.schedule}")
        self.train_embs = {}
        self.test_embs = {}
        for n in self.schedule:
            tr, te = compute_pca_embeddings(self.train_data, self.test_data, n)
            self.train_embs[n] = tr
            self.test_embs[n] = te
    
    def get_embeddings(self, n_components):
        """Get precomputed embeddings."""
        return self.train_embs[n_components], self.test_embs[n_components]


# ============== Visualization ==============

def create_tight_comparison_figure(results_dict, title_info, latent_dim=2):
    """Create tightly-packed comparison figure with large fonts."""
    n = len(results_dict)
    
    # Compact figure size
    panel_w, panel_h = 2.4, 2.4
    fig = plt.figure(figsize=(panel_w * n + 0.5, panel_h + 0.8))
    
    # Use GridSpec for tight control
    gs = GridSpec(1, n, figure=fig, wspace=0.08, left=0.02, right=0.98, bottom=0.12, top=0.88)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    
    # Shared colormap normalization
    all_labels = np.concatenate([d['labels'] for d in results_dict.values()])
    norm = Normalize(vmin=all_labels.min(), vmax=all_labels.max())
    cmap = cm.Spectral
    
    for idx, (name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        Z = data['latents']
        labels = data['labels']
        metrics = data.get('metrics', {})
        
        # Scatter with consistent styling
        ax.scatter(
            Z[:, 0], Z[:, 1],
            c=labels, cmap=cmap, norm=norm,
            s=6, alpha=0.85,
            edgecolors='white', linewidths=0.08,
            rasterized=True
        )
        
        # Metrics box - top left, large bold font
        lines = []
        for key, (short_name, arrow, fmt) in METRIC_DISPLAY.items():
            if key in metrics:
                val = metrics[key]
                if val is not None and not np.isnan(val):
                    lines.append(f'{short_name}{arrow}:{val:{fmt}}')
        
        if lines:
            text = '\n'.join(lines[:3])  # Max 3 metrics
            ax.text(0.04, 0.96, text, transform=ax.transAxes,
                   fontsize=7, fontweight='bold', fontfamily='monospace',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                            alpha=0.92, edgecolor='#666666', linewidth=0.5))
        
        # Title below plot - large and bold
        ax.set_xlabel(name, fontsize=9, fontweight='bold', labelpad=3)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        
        # Subtle border
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color('#aaaaaa')
    
    # Suptitle with epoch/PCA info
    fig.suptitle(title_info, fontsize=11, fontweight='bold', y=0.96)
    
    return fig


def create_evolution_grid(history, n_cols=5):
    """Create grid showing evolution at key stages."""
    n_stages = len(history)
    n_rows = (n_stages + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    axes = np.atleast_2d(axes)
    
    for idx, stage_data in enumerate(history):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        Z = stage_data['latents']
        labels = stage_data['labels']
        
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='Spectral', s=4, alpha=0.8)
        ax.set_title(stage_data.get('title', f'Stage {idx+1}'), fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
    
    # Hide unused axes
    for idx in range(n_stages, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def create_metrics_comparison_plot(all_histories, output_path):
    """Create multi-panel metrics comparison."""
    metrics_keys = ['distance_correlation', 'knn_avg', 'reconstruction_error']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, metric in zip(axes, metrics_keys):
        for name, history in all_histories.items():
            epochs = [h['epoch'] for h in history]
            values = [h.get(metric, np.nan) for h in history]
            
            color = COLORS.get(name, '#333333')
            ax.plot(epochs, values, label=name, color=color, linewidth=2, marker='o', markersize=3)
        
        display = METRIC_DISPLAY.get(metric, (metric, '', '.3f'))
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(f'{display[0]} {display[1]}', fontsize=11)
        ax.set_title(f'{display[0]} over Training', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_gif_from_frames(frames_dir, output_path, duration_ms=150):
    """Create GIF from PNG frames."""
    try:
        from PIL import Image
        import glob
        
        frames = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if not frames:
            print("No frames found")
            return
        
        images = [Image.open(f) for f in frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"Created GIF: {output_path}")
    except ImportError:
        print("PIL not available - skipping GIF creation")


# ============== Main Experiment ==============

def run_experiment(args):
    """Run progressive vs fixed MMAE comparison."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    frames_dir = os.path.join(output_dir, 'frames')
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Config
    config = get_config(args.dataset)
    config['latent_dim'] = args.latent_dim
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['mmae_lambda'] = args.mmae_lambda
    config['seed'] = args.seed
    
    if args.dataset == 'spheres' and args.spheres_dim:
        config['d'] = args.spheres_dim
        config['input_dim'] = args.spheres_dim + 1
    
    input_dim = config['input_dim']
    max_pca = args.max_pca if args.max_pca else min(input_dim - 1, 100)
    
    print("=" * 70)
    print("PROGRESSIVE vs FIXED MMAE COMPARISON")
    print("=" * 70)
    print(f"Dataset: {args.dataset} (input_dim={input_dim})")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Progressive: {args.start_pca} → {max_pca} (step={args.pca_step})")
    print(f"Fixed variants: {[p for p in args.fixed_pca if p <= input_dim]}")
    print(f"Epochs per step: {args.epochs_per_step}")
    print("=" * 70)
    
    # Load data
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = load_data(
        args.dataset, config, with_embeddings=False
    )
    
    train_data = train_ds.data.numpy()
    test_data = test_ds.data.numpy()
    test_labels = test_ds.labels.numpy()
    
    # Setup progressive schedule
    prog_schedule = ProgressiveSchedule(
        train_data, test_data, args.start_pca, max_pca, args.pca_step
    )
    
    # Initialize all trainers
    trainers = {}
    histories = {}
    
    # Progressive
    print("\nInitializing Progressive MMAE...")
    model_prog = build_model('mmae', config).to(device)
    opt_prog = torch.optim.Adam(model_prog.parameters(), lr=args.lr)
    trainer_prog = MMAETrainer(model_prog, opt_prog, device, train_data, test_data)
    trainers['Progressive'] = trainer_prog
    histories['Progressive'] = []
    
    # Fixed variants
    for n_pca in args.fixed_pca:
        if n_pca > input_dim - 1:
            continue
        
        name = f'Fixed-{n_pca}D'
        print(f"Initializing {name}...")
        
        model = build_model('mmae', config).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainer = MMAETrainer(model, opt, device, train_data, test_data)
        
        # Set fixed embeddings
        tr_emb, te_emb = compute_pca_embeddings(train_data, test_data, n_pca)
        trainer.set_reference(tr_emb, te_emb)
        
        trainers[name] = trainer
        histories[name] = []
    
    # Training loop
    total_epochs = len(prog_schedule.schedule) * args.epochs_per_step
    epoch = 0
    frame_idx = 0
    
    print(f"\nTraining for {total_epochs} total epochs...")
    print(f"Progressive schedule: {prog_schedule.schedule}\n")
    
    start_time = time.time()
    progressive_stages = []  # For evolution grid
    
    for step_idx, current_pca in enumerate(prog_schedule.schedule):
        # Update progressive reference
        tr_emb, te_emb = prog_schedule.get_embeddings(current_pca)
        trainers['Progressive'].set_reference(tr_emb, te_emb)
        
        print(f"--- Step {step_idx+1}/{len(prog_schedule.schedule)}: Progressive PCA = {current_pca} ---")
        
        for step_epoch in range(args.epochs_per_step):
            epoch += 1
            
            # Train all
            losses = {}
            for name, trainer in trainers.items():
                loss = trainer.train_epoch(args.batch_size)
                losses[name] = loss
            
            # Record metrics and save frame
            if epoch % args.save_interval == 0 or epoch == 1:
                results = {}
                
                for name, trainer in trainers.items():
                    Z = get_latents(trainer.model, test_data.reshape(len(test_data), -1), device)
                    metrics = quick_metrics(test_data, Z, test_labels)
                    metrics['loss'] = losses[name]
                    metrics['epoch'] = epoch
                    
                    if name == 'Progressive':
                        metrics['current_pca'] = current_pca
                        display = f'Prog (PCA={current_pca})'
                    else:
                        display = name
                    
                    results[display] = {
                        'latents': Z,
                        'labels': test_labels,
                        'metrics': metrics
                    }
                    histories[name].append(metrics.copy())
                
                # Create and save frame
                title = f'Epoch {epoch} | Progressive PCA: {current_pca}'
                fig = create_tight_comparison_figure(results, title, args.latent_dim)
                frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
                fig.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                frame_idx += 1
            
            if epoch % 10 == 0:
                loss_str = ' | '.join([f'{k[:8]}: {v:.4f}' for k, v in losses.items()])
                print(f"  Epoch {epoch}/{total_epochs}: {loss_str}")
        
        # Save checkpoint at end of each progressive step
        print(f"\n  Checkpoint: Progressive PCA = {current_pca}")
        
        checkpoint_results = {}
        for name, trainer in trainers.items():
            Z = get_latents(trainer.model, test_data.reshape(len(test_data), -1), device)
            metrics = full_metrics(test_data, Z, test_labels, compute_wass=args.compute_wass)
            
            if name == 'Progressive':
                display = f'Prog (PCA={current_pca})'
                progressive_stages.append({
                    'latents': Z.copy(),
                    'labels': test_labels.copy(),
                    'title': f'Prog PCA={current_pca}',
                    'metrics': metrics.copy()
                })
            else:
                display = name
            
            checkpoint_results[display] = {
                'latents': Z,
                'labels': test_labels,
                'metrics': metrics
            }
            
            print(f"    {display}: DC={metrics['distance_correlation']:.4f}, kNN={metrics['knn_avg']:.4f}")
        
        # Save checkpoint figure
        fig = create_tight_comparison_figure(
            checkpoint_results, 
            f'Checkpoint: Epoch {epoch}, Progressive PCA={current_pca}',
            args.latent_dim
        )
        cp_path = os.path.join(checkpoints_dir, f'checkpoint_e{epoch:04d}_pca{current_pca}.png')
        fig.savefig(cp_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Save checkpoint data
        cp_data = {
            'epoch': epoch,
            'progressive_pca': current_pca,
            'metrics': {k: v['metrics'] for k, v in checkpoint_results.items()}
        }
        with open(cp_path.replace('.png', '.json'), 'w') as f:
            json.dump(cp_data, f, indent=2, default=float)
        
        print()
    
    train_time = time.time() - start_time
    
    # Create outputs
    print("\nCreating outputs...")
    
    # GIF
    gif_path = os.path.join(output_dir, 'evolution.gif')
    create_gif_from_frames(frames_dir, gif_path, args.gif_duration)
    
    # Metrics plot
    metrics_path = os.path.join(output_dir, 'metrics_comparison.png')
    create_metrics_comparison_plot(histories, metrics_path)
    
    # Progressive evolution grid
    if progressive_stages:
        fig = create_evolution_grid(progressive_stages, n_cols=min(5, len(progressive_stages)))
        fig.suptitle('Progressive MMAE Evolution', fontsize=14, fontweight='bold')
        fig.savefig(os.path.join(output_dir, 'progressive_evolution.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    # Final comparison
    final_results = {}
    for name, trainer in trainers.items():
        Z = get_latents(trainer.model, test_data.reshape(len(test_data), -1), device)
        metrics = full_metrics(test_data, Z, test_labels, compute_wass=True)
        
        if name == 'Progressive':
            display = f'Progressive (final)'
        else:
            display = name
        
        final_results[display] = {
            'latents': Z,
            'labels': test_labels,
            'metrics': metrics
        }
    
    fig = create_tight_comparison_figure(final_results, 'Final Comparison', args.latent_dim)
    fig.savefig(os.path.join(output_dir, 'final_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Summary
    summary = {
        'dataset': args.dataset,
        'input_dim': input_dim,
        'latent_dim': args.latent_dim,
        'total_epochs': total_epochs,
        'progressive_schedule': prog_schedule.schedule,
        'fixed_variants': [p for p in args.fixed_pca if p <= input_dim - 1],
        'epochs_per_step': args.epochs_per_step,
        'train_time_seconds': train_time,
        'final_metrics': {k: v['metrics'] for k, v in final_results.items()},
        'histories': histories
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"  evolution.gif - Training animation")
    print(f"  metrics_comparison.png - Metrics over time")
    print(f"  progressive_evolution.png - Progressive stages grid")
    print(f"  final_comparison.png - Final latent spaces")
    print(f"  checkpoints/ - Snapshots at each PCA transition")
    print(f"\nTotal training time: {train_time:.1f}s")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Progressive vs Fixed MMAE Comparison')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='spheres',
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--spheres_dim', type=int, default=None)
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--mmae_lambda', type=float, default=1.0)
    
    # Progressive schedule
    parser.add_argument('--start_pca', type=int, default=2)
    parser.add_argument('--max_pca', type=int, default=None)
    parser.add_argument('--pca_step', type=int, default=5)
    parser.add_argument('--epochs_per_step', type=int, default=10)
    
    # Fixed variants
    parser.add_argument('--fixed_pca', type=int, nargs='+', default=[10, 20, 50])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/progressive_mmae')
    parser.add_argument('--save_interval', type=int, default=3)
    parser.add_argument('--gif_duration', type=int, default=150)
    parser.add_argument('--compute_wass', action='store_true')
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()