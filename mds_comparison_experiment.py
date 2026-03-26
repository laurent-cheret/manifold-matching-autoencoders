#!/usr/bin/env python
"""
MDS vs MMAE Comparison Experiment (Robust Version)
==================================================
Compares MDS variants and MMAE across multiple dataset sizes with averaging.

Features:
- Multiple dataset sizes (5k, 10k, 20k) to show scaling
- Multiple seeds for robust averaging
- Optional cuML GPU-accelerated MDS
- Clean vs Noisy comparison

Usage:
    python mds_comparison_experiment.py
    python mds_comparison_experiment.py --device cuda
    python mds_comparison_experiment.py --sizes 5000 10000 20000 --n_seeds 3
"""

import argparse
import os
import time
import tracemalloc
import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Project imports
from config import get_config
from data.spheres import generate_spheres
from data.base import normalize_features
from models import build_model
from training import Trainer

# Optional cuML - handle both import and runtime errors
CUML_AVAILABLE = False
try:
    from cuml.manifold import MDS as cuMDS
    # Test that it actually works (catches CUDA runtime issues)
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceCount()
    CUML_AVAILABLE = True
    print("✓ cuML available - GPU MDS enabled")
except ImportError:
    print("✗ cuML not installed")
    print("  To install on Colab: !pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12")
except Exception as e:
    print(f"✗ cuML installed but not working: {e}")
    print("  Try: !pip uninstall cuml-cu11 -y && pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12")


# ============== Landmark MDS ==============

def landmark_mds(data, n_components=2, n_landmarks=200, seed=42):
    """Landmark MDS (Nyström approximation)."""
    np.random.seed(seed)
    n = len(data)
    n_landmarks = min(n_landmarks, n)
    
    landmark_idx = np.random.choice(n, n_landmarks, replace=False)
    landmarks = data[landmark_idx]
    
    mds = MDS(n_components=n_components, dissimilarity='euclidean',
              normalized_stress='auto', random_state=seed, n_jobs=-1, max_iter=300)
    landmark_emb = mds.fit_transform(landmarks)
    
    D_landmarks = squareform(pdist(landmarks))
    D_landmarks_sq = D_landmarks ** 2
    
    n_l = len(landmarks)
    H = np.eye(n_l) - np.ones((n_l, n_l)) / n_l
    B = -0.5 * H @ D_landmarks_sq @ H
    
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    eigvals_safe = np.maximum(eigvals, 1e-10)
    L_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe))
    
    embedding = np.zeros((n, n_components), dtype=np.float32)
    embedding[landmark_idx] = landmark_emb
    
    non_landmark_idx = np.setdiff1d(np.arange(n), landmark_idx)
    if len(non_landmark_idx) > 0:
        D_new = np.sqrt(np.sum((data[non_landmark_idx, None, :] - landmarks[None, :, :]) ** 2, axis=2))
        D_new_sq = D_new ** 2
        mean_landmark_sq = D_landmarks_sq.mean(axis=1)
        delta = -0.5 * (D_new_sq - mean_landmark_sq[None, :])
        embedding[non_landmark_idx] = delta @ L_inv
    
    return embedding


# ============== Metrics ==============

def distance_correlation(X, Z):
    Dx, Dz = squareform(pdist(X)), squareform(pdist(Z))
    mask = np.triu(np.ones_like(Dx), k=1) > 0
    return np.corrcoef(Dx[mask], Dz[mask])[0, 1]


# ============== Method Runners ==============

def run_with_tracking(func, *args, use_gpu_mem=False, **kwargs):
    """Run function and track time + memory."""
    if use_gpu_mem and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    tracemalloc.start()
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if use_gpu_mem and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_vram = 0
    
    return result, elapsed, peak_ram / 1e6, peak_vram


def run_exact_mds(data, seed=42):
    """Sklearn exact MDS."""
    mds = MDS(n_components=2, dissimilarity='euclidean', 
              normalized_stress='auto', random_state=seed, n_jobs=-1, max_iter=300)
    return mds.fit_transform(data)


def run_cuml_mds(data, seed=42):
    """cuML GPU-accelerated MDS."""
    if not CUML_AVAILABLE:
        raise RuntimeError("cuML not available")
    mds = cuMDS(n_components=2, random_state=seed)
    emb = mds.fit_transform(data)
    if hasattr(emb, 'get'):
        emb = emb.get()
    return emb


def run_mmae(data, pca_ratio, config_base, device='cpu', epochs=150, seed=42):
    """Train MMAE and return embedding."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    n, d = data.shape
    n_components = max(2, int(d * pca_ratio))
    
    config = config_base.copy()
    config['mmae_n_components'] = n_components
    config['latent_dim'] = 2
    config['input_dim'] = d
    
    pca = PCA(n_components=n_components)
    ref_emb = pca.fit_transform(data).astype(np.float32)
    
    data_t = torch.tensor(data, dtype=torch.float32)
    ref_t = torch.tensor(ref_emb, dtype=torch.float32)
    labels_t = torch.zeros(n)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data_t, ref_t, labels_t)
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 128), shuffle=True, drop_last=True)
    
    model = build_model('mmae', config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))
    trainer = Trainer(model, optimizer, device, model_name='mmae')
    trainer.fit(loader, None, n_epochs=epochs, verbose=False)
    
    model.eval()
    with torch.no_grad():
        embedding = model.encode(data_t.to(device)).cpu().numpy()
    return embedding


# ============== Main Experiment ==============

def run_single_config(data_clean, data_noisy, labels, config_base, landmark_ratio, device, epochs, seed):
    """Run all methods on one dataset configuration."""
    results = {}
    
    # Compute number of landmarks from ratio
    n_landmarks = int(len(data_clean) * landmark_ratio)
    
    # Define methods
    methods = {
        'MDS': lambda d, s: run_exact_mds(d, s),
        'L-MDS': lambda d, s: landmark_mds(d, 2, n_landmarks, s),
        'MMAE-100': lambda d, s: run_mmae(d, 1.0, config_base, device, epochs, s),
        'MMAE-80': lambda d, s: run_mmae(d, 0.8, config_base, device, epochs, s),
    }
    
    if CUML_AVAILABLE:
        methods['cuMDS'] = lambda d, s: run_cuml_mds(d, s)
    
    for condition, data in [('clean', data_clean), ('noisy', data_noisy)]:
        results[condition] = {}
        for method_name, method_func in methods.items():
            use_gpu = method_name in ['cuMDS', 'MMAE-100', 'MMAE-80'] and device == 'cuda'
            emb, t, ram, vram = run_with_tracking(method_func, data, seed, use_gpu_mem=use_gpu)
            dcorr = distance_correlation(data, emb)
            results[condition][method_name] = {
                'emb': emb, 'time': t, 'ram': ram, 'vram': vram, 'dcorr': dcorr
            }
    
    return results


def run_experiment(sizes=[5000, 10000, 20000], noise_std=0.5, landmark_ratio=0.8, 
                   device='cpu', epochs=150, n_seeds=3, output_dir='results/mds_comparison'):
    
    print("=" * 70)
    print("MDS vs MMAE Comparison Experiment (Robust)")
    print("=" * 70)
    print(f"  Sizes: {sizes}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Noise: σ={noise_std}")
    print(f"  L-MDS landmarks: {int(landmark_ratio*100)}% of data")
    print(f"  Device: {device}")
    print(f"  cuML: {'Yes' if CUML_AVAILABLE else 'No'}")
    print("=" * 70)
    
    config_base = get_config('spheres', 'mmae')
    config_base['batch_size'] = 256
    config_base['learning_rate'] = 1e-3
    config_base['mmae_lambda'] = 1.0
    config_base['hidden_dims'] = [256, 128]
    
    # Storage for all results
    all_results = {size: [] for size in sizes}
    
    for size in sizes:
        n_samples = size // 20  # spheres dataset: 10 small + 1 big (10x)
        n_landmarks = int(size * landmark_ratio)
        print(f"\n{'='*70}")
        print(f"Dataset size: {size} points (n_samples={n_samples}, L-MDS landmarks={n_landmarks})")
        print(f"{'='*70}")
        
        for seed in range(42, 42 + n_seeds):
            print(f"\n--- Seed {seed} ---")
            
            # Generate data
            data_clean, labels = generate_spheres(n_samples=n_samples, d=100, seed=seed)
            data_noisy, _ = generate_spheres(n_samples=n_samples, d=100, seed=seed)
            np.random.seed(seed)
            data_noisy = data_noisy + noise_std * np.random.randn(*data_noisy.shape).astype(np.float32)
            data_clean, _ = normalize_features(data_clean, data_clean)
            data_noisy, _ = normalize_features(data_noisy, data_noisy)
            
            print(f"  Generated: {len(data_clean)} points, {data_clean.shape[1]}D")
            
            # Run all methods
            results = run_single_config(
                data_clean, data_noisy, labels, config_base, 
                landmark_ratio, device, epochs, seed
            )
            results['labels'] = labels
            all_results[size].append(results)
            
            # Print progress
            for method in results['clean']:
                t = results['clean'][method]['time']
                dc = results['clean'][method]['dcorr']
                print(f"    {method:<10}: {t:>6.1f}s, dcorr={dc:.3f}")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)
    
    aggregated = aggregate_results(all_results, sizes)
    print_summary(aggregated, sizes)
    
    # Create figure
    print("\nCreating figure...")
    create_figure(all_results, aggregated, sizes, noise_std, output_dir)
    
    return all_results, aggregated


def aggregate_results(all_results, sizes):
    """Aggregate results across seeds."""
    aggregated = {}
    
    # Get method names from first result
    first_result = all_results[sizes[0]][0]
    methods = list(first_result['clean'].keys())
    
    for size in sizes:
        aggregated[size] = {'clean': {}, 'noisy': {}}
        
        for condition in ['clean', 'noisy']:
            for method in methods:
                times = [r[condition][method]['time'] for r in all_results[size]]
                rams = [r[condition][method]['ram'] for r in all_results[size]]
                vrams = [r[condition][method]['vram'] for r in all_results[size]]
                dcorrs = [r[condition][method]['dcorr'] for r in all_results[size]]
                
                aggregated[size][condition][method] = {
                    'time_mean': np.mean(times), 'time_std': np.std(times),
                    'ram_mean': np.mean(rams), 'ram_std': np.std(rams),
                    'vram_mean': np.mean(vrams), 'vram_std': np.std(vrams),
                    'dcorr_mean': np.mean(dcorrs), 'dcorr_std': np.std(dcorrs),
                }
    
    return aggregated


def print_summary(aggregated, sizes):
    """Print summary table."""
    methods = list(aggregated[sizes[0]]['clean'].keys())
    
    print(f"\n{'Method':<10} {'Size':>7} {'Cond':<6} {'Time(s)':>12} {'RAM(MB)':>12} {'Dcorr':>12}")
    print("-" * 70)
    
    for size in sizes:
        for method in methods:
            for cond in ['clean', 'noisy']:
                r = aggregated[size][cond][method]
                time_str = f"{r['time_mean']:.1f}±{r['time_std']:.1f}"
                ram_str = f"{r['ram_mean']:.1f}±{r['ram_std']:.1f}"
                dcorr_str = f"{r['dcorr_mean']:.3f}±{r['dcorr_std']:.3f}"
                print(f"{method:<10} {size:>7} {cond:<6} {time_str:>12} {ram_str:>12} {dcorr_str:>12}")
        print("-" * 70)


def create_figure(all_results, aggregated, sizes, noise_std, output_dir):
    """Create publication figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use middle size for scatter plots
    mid_size = sizes[len(sizes)//2]
    scatter_results = all_results[mid_size][0]  # First seed
    labels = scatter_results['labels']
    
    n_classes = int(labels.max()) + 1
    cmap = plt.cm.Spectral
    point_colors = [cmap(labels[i] / (n_classes - 1)) for i in range(len(labels))]
    
    # Figure setup
    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 0.05, 1.0], 
                           height_ratios=[1, 1], wspace=0.25, hspace=0.35)
    
    # Scatter plots
    scatter_kw = dict(s=6, alpha=0.7, edgecolors='none')
    row_labels = [f'CLEAN (N={mid_size:,})', f'NOISY (σ={noise_std})']
    row1_methods = ['MDS', 'L-MDS', 'MMAE-100']
    row2_methods = ['MDS', 'L-MDS', 'MMAE-80']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    axes_scatter = []
    for row, (condition, methods) in enumerate([('clean', row1_methods), ('noisy', row2_methods)]):
        for col, method in enumerate(methods):
            ax = fig.add_subplot(gs[row, col])
            axes_scatter.append(ax)
            emb = scatter_results[condition][method]['emb']
            ax.scatter(emb[:, 0], emb[:, 1], c=point_colors, **scatter_kw)
            
            panel_idx = row * 3 + col
            ax.text(0.02, 0.98, panel_labels[panel_idx], transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', va='top')
            ax.set_title(method.replace('-', ' '), fontsize=11, fontweight='bold', pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='datalim')
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
    
    axes_scatter[0].annotate(row_labels[0], xy=(-0.22, 0.5), xycoords='axes fraction',
                             fontsize=10, fontweight='bold', rotation=90, va='center', ha='center')
    axes_scatter[3].annotate(row_labels[1], xy=(-0.22, 0.5), xycoords='axes fraction',
                             fontsize=10, fontweight='bold', rotation=90, va='center', ha='center')
    
    # Right panel: scaling plots
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 4], hspace=0.4)
    
    # Colors for methods
    methods_for_scaling = list(aggregated[sizes[0]]['clean'].keys())
    colors = {'MDS': '#2ecc71', 'cuMDS': '#1abc9c', 'L-MDS': '#3498db', 
              'MMAE-100': '#e74c3c', 'MMAE-80': '#9b59b6'}
    markers = {'MDS': 'o', 'cuMDS': 's', 'L-MDS': '^', 'MMAE-100': 'D', 'MMAE-80': 'v'}
    
    # (g) Time scaling
    ax_time = fig.add_subplot(gs_right[0])
    for method in methods_for_scaling:
        times = [aggregated[s]['clean'][method]['time_mean'] for s in sizes]
        stds = [aggregated[s]['clean'][method]['time_std'] for s in sizes]
        ax_time.errorbar(sizes, times, yerr=stds, marker=markers.get(method, 'o'), 
                        color=colors.get(method, 'gray'), label=method.replace('-', ' '),
                        linewidth=2, markersize=6, capsize=3)
    
    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    ax_time.set_xlabel('Dataset Size', fontsize=10)
    ax_time.set_ylabel('Time (s)', fontsize=10)
    ax_time.set_title('(g) Computation Time', fontsize=11, fontweight='bold', pad=8)
    ax_time.legend(fontsize=8, loc='upper left')
    ax_time.grid(True, alpha=0.3, linewidth=0.5)
    ax_time.set_xticks(sizes)
    ax_time.set_xticklabels([f'{s//1000}k' for s in sizes])
    
    # (h) Distance correlation at largest size
    ax_dcorr = fig.add_subplot(gs_right[1])
    largest_size = sizes[-1]
    methods_bar = methods_for_scaling
    x = np.arange(len(methods_bar))
    width = 0.35
    
    dcorr_clean = [aggregated[largest_size]['clean'][m]['dcorr_mean'] for m in methods_bar]
    dcorr_clean_std = [aggregated[largest_size]['clean'][m]['dcorr_std'] for m in methods_bar]
    dcorr_noisy = [aggregated[largest_size]['noisy'][m]['dcorr_mean'] for m in methods_bar]
    dcorr_noisy_std = [aggregated[largest_size]['noisy'][m]['dcorr_std'] for m in methods_bar]
    
    bars1 = ax_dcorr.bar(x - width/2, dcorr_clean, width, yerr=dcorr_clean_std, 
                         label='Clean', color='#27ae60', edgecolor='black', linewidth=0.5, capsize=2)
    bars2 = ax_dcorr.bar(x + width/2, dcorr_noisy, width, yerr=dcorr_noisy_std,
                         label='Noisy', color='#e67e22', edgecolor='black', linewidth=0.5, capsize=2)
    
    ax_dcorr.set_xticks(x)
    ax_dcorr.set_xticklabels([m.replace('-', '\n') for m in methods_bar], fontsize=8)
    ax_dcorr.set_ylabel('Dist. Corr.', fontsize=10)
    ax_dcorr.set_title(f'(h) Distance Correlation (N={largest_size//1000}k)', fontsize=11, fontweight='bold', pad=8)
    ax_dcorr.legend(fontsize=8, loc='lower right')
    ax_dcorr.set_ylim(0, 1.05)
    ax_dcorr.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        save_path = os.path.join(output_dir, f'mds_vs_mmae_comparison.{fmt}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MDS vs MMAE Comparison (Robust)')
    parser.add_argument('--sizes', type=int, nargs='+', default=[5000, 10000, 20000],
                       help='Dataset sizes to test')
    parser.add_argument('--noise_std', type=float, default=0.5, help='Noise std for noisy condition')
    parser.add_argument('--landmark_ratio', type=float, default=0.8, help='Ratio of data to use as landmarks for L-MDS')
    parser.add_argument('--epochs', type=int, default=150, help='MMAE training epochs')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds for averaging')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cpu/cuda)')
    parser.add_argument('--output_dir', type=str, default='results/mds_comparison', help='Output directory')
    args = parser.parse_args()
    
    run_experiment(
        sizes=args.sizes,
        noise_std=args.noise_std,
        landmark_ratio=args.landmark_ratio,
        device=args.device,
        epochs=args.epochs,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()