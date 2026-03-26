#!/usr/bin/env python
"""
Reference Dimensionality Ablation: MMAE with 1%, 5%, 10%, 20%, 50%, 80%, 100% PCA references.

Shows that optimal reference dimension is dataset-dependent.
Demonstrates MMAE's flexibility: decouples reference dimension from latent dimension.

Usage:
    python ablation_reference_dimensionality.py --datasets pbmc3k mnist fmnist cifar10 spheres
    python ablation_reference_dimensionality.py --datasets mnist --visualize_only
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import warnings
warnings.filterwarnings('ignore')

from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer
from evaluation import evaluate


DATASET_COLORS = {
    'pbmc3k': '#1f77b4',
    'mnist': '#ff7f0e',
    'fmnist': '#2ca02c',
    'cifar10': '#d62728',
    'spheres': '#9467bd',
}


def train_mmae_with_reference(dataset_name, config, reference_pct, epochs=50, device='cuda', seed=42):
    """
    Train MMAE with a specific reference dimensionality (as percentage of original).
    
    Returns: dict with evaluation metrics
    """
    try:
        config_copy = config.copy()
        config_copy['seed'] = seed
        config_copy['device'] = device
        config_copy['n_epochs'] = epochs
        
        # Load data with embeddings for this reference dimension
        input_dim = DATASET_CONFIGS[dataset_name]['input_dim']
        n_pca = max(1, min(int(input_dim * reference_pct), input_dim - 1))
        config_copy['mmae_n_components'] = n_pca
        
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
            dataset_name, config_copy, with_embeddings=True
        )
        
        # Build and train model
        model = build_model('mmae', config_copy)
        optimizer = torch.optim.Adam(model.parameters(), lr=config_copy.get('learning_rate', 1e-3))
        trainer = Trainer(model, optimizer, device=device, model_name='mmae')
        trainer.fit(train_loader, test_loader, n_epochs=epochs, verbose=False)
        
        # Evaluate
        test_data = test_dataset.data.numpy().reshape(len(test_dataset), -1)
        test_labels = test_dataset.labels.numpy()
        
        model.eval()
        with torch.no_grad():
            Z = model.encode(torch.from_numpy(test_data).float().to(device)).cpu().numpy()
        
        # Compute all available metrics (wasserstein can be slow, include if time permits)
        metrics = evaluate(test_data, Z, test_labels, ks=[10, 50, 100], compute_wasserstein=True)
        
        return {
            'reference_pct': reference_pct * 100,
            'n_pca_components': n_pca,
            'input_dim': input_dim,
            'metrics': metrics
        }
    
    except Exception as e:
        print(f"      ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def experiment_reference_dimensionality(datasets, latent_dims=[2, 64], n_runs=3, epochs=50, 
                                        output_dir='results/ablation_reference',
                                        device='cuda', base_seed=42, n_samples=5000):
    """
    Main experiment: train MMAE with different reference dimensionalities.
    
    Args:
        datasets: List of dataset names to test
        latent_dims: List of bottleneck dimensions to test (default: [2, 64])
        n_runs: Number of runs per configuration for averaging (default: 3)
        epochs: Training epochs per reference dimension per run
        output_dir: Where to save results
        device: cuda or cpu
        base_seed: Base random seed (incremented for each run)
        n_samples: Max samples per dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("ABLATION: MMAE with Varying Reference Dimensionalities")
    print(f"{'='*70}")
    print(f"Bottleneck dimensions: {latent_dims}")
    print(f"Runs per configuration: {n_runs}")
    print(f"Total models to train: {len(datasets)} × {len(latent_dims)} × {n_runs}")
    
    # Reference percentages to test
    reference_pcts = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 1.0]
    
    # Structure: all_results[dataset][latent_dim] = {metrics aggregated across runs}
    all_results = {}
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        
        for latent_dim in latent_dims:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name.upper()} | Latent Dim: {latent_dim}D")
            print(f"{'='*70}")
            
            try:
                input_dim = DATASET_CONFIGS[dataset_name]['input_dim']
                print(f"Input dimension: {input_dim}")
                print(f"Testing reference dimensions: {[int(input_dim * pct) for pct in reference_pcts]}")
                print(f"Runs: {n_runs} (will average across runs)")
                print("-" * 70)
            except Exception as e:
                print(f"Config error: {e}")
                continue
            
            # Collect results from all runs
            run_results = []  # List of dicts, one per run
            
            for run_idx in range(n_runs):
                print(f"\n  Run {run_idx + 1}/{n_runs}:")
                seed = base_seed + run_idx
                
                try:
                    config = get_config(dataset_name, 'mmae')
                    config['seed'] = seed
                    config['latent_dim'] = latent_dim
                    config['device'] = device
                    
                    if 'n_samples' in config and config['n_samples'] is not None:
                        config['n_samples'] = min(config['n_samples'], n_samples)
                    else:
                        config['n_samples'] = n_samples
                    
                except Exception as e:
                    print(f"    Config error: {e}")
                    continue
                
                run_data = {
                    'reference_pcts': [pct * 100 for pct in reference_pcts],
                    'n_pca_components': [],
                }
                
                for pct in reference_pcts:
                    print(f"    {pct*100:5.1f}% PCA reference...", end='', flush=True)
                    
                    result = train_mmae_with_reference(
                        dataset_name, config, pct, 
                        epochs=epochs, device=device, seed=seed
                    )
                    
                    if result is None:
                        print(" FAILED")
                        continue
                    
                    run_data['n_pca_components'].append(result['n_pca_components'])
                    metrics = result['metrics']
                    
                    # Dynamically collect all metrics
                    for metric_key, metric_val in metrics.items():
                        if metric_key not in run_data:
                            run_data[metric_key] = []
                        run_data[metric_key].append(metric_val)
                    
                    # Print key metrics
                    print(f" dist_corr={metrics.get('distance_correlation', 0):.4f}, "
                          f"trust={metrics.get('trustworthiness_10', 0):.4f}, "
                          f"trip={metrics.get('triplet_accuracy', 0):.4f}")
                
                run_results.append(run_data)
            
            # Aggregate results across runs (compute mean and std)
            aggregated = aggregate_runs(run_results, reference_pcts)
            all_results[dataset_name][latent_dim] = aggregated
            
            print(f"\n  Aggregated {n_runs} runs for latent_dim={latent_dim}D")
    
    # Plot results
    plot_reference_ablation(all_results, output_dir)
    
    # Save results
    save_path = os.path.join(output_dir, 'ablation_results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {save_path}")
    
    # Print summary table
    print_summary_table(all_results)
    
    return all_results


def aggregate_runs(run_results, reference_pcts):
    """
    Aggregate results across multiple runs by computing mean and std.
    
    Args:
        run_results: List of result dicts from each run
        reference_pcts: List of reference percentages
    
    Returns:
        Aggregated dict with mean and std for each metric
    """
    if not run_results:
        return {}
    
    aggregated = {
        'reference_pcts': [pct * 100 for pct in reference_pcts],
        'n_pca_components': run_results[0]['n_pca_components'],
    }
    
    # Get all metric keys from first run
    metric_keys = [k for k in run_results[0].keys() if k not in ['reference_pcts', 'n_pca_components']]
    
    # Aggregate each metric
    for metric_key in metric_keys:
        values_across_runs = []
        
        for run_result in run_results:
            if metric_key in run_result:
                values_across_runs.append(run_result[metric_key])
        
        if values_across_runs:
            # Stack values (each is a list of metric values across reference dims)
            stacked = np.array(values_across_runs)  # Shape: (n_runs, n_ref_dims)
            
            aggregated[f'{metric_key}_mean'] = stacked.mean(axis=0).tolist()
            aggregated[f'{metric_key}_std'] = stacked.std(axis=0).tolist()
    
    return aggregated


def plot_reference_ablation(results, output_dir):
    """
    Create comprehensive multi-panel figures showing all metrics across reference dimensionalities.
    Includes error bars from multiple runs.
    Structure: results[dataset][latent_dim] = {metric_mean, metric_std, ...}
    """
    datasets = list(results.keys())
    if not datasets:
        print("No results to plot")
        return
    
    # Get available latent dimensions
    first_dataset = datasets[0]
    latent_dims = sorted(list(results[first_dataset].keys()))
    
    print(f"\nPlotting results for {len(datasets)} datasets × {len(latent_dims)} latent dimensions")
    
    # ===== Key metrics across all datasets, per latent dimension =====
    key_metrics = ['distance_correlation', 'trustworthiness_10', 'continuity_10', 'triplet_accuracy']
    key_labels = ['Distance Correlation', 'Trustworthiness@10', 'Continuity@10', 'Triplet Accuracy']
    
    for latent_dim in latent_dims:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        for ax_idx, (metric, label) in enumerate(zip(key_metrics, key_labels)):
            ax = axes[ax_idx]
            
            for dataset_name in datasets:
                if latent_dim in results[dataset_name]:
                    data = results[dataset_name][latent_dim]
                    pcts = data.get('reference_pcts', [])
                    
                    metric_mean_key = f'{metric}_mean'
                    metric_std_key = f'{metric}_std'
                    
                    if metric_mean_key in data:
                        mean_vals = data[metric_mean_key]
                        std_vals = data[metric_std_key]
                        color = DATASET_COLORS.get(dataset_name, 'gray')
                        
                        # Plot with error bars
                        ax.errorbar(pcts, mean_vals, yerr=std_vals, fmt='o-', 
                                   label=dataset_name.upper(), linewidth=2,
                                   markersize=5, color=color, elinewidth=1,
                                   markeredgewidth=0.5, markeredgecolor='white', alpha=0.8)
            
            ax.axvline(x=80, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='80% (default)')
            ax.set_xlabel('Reference Dimensionality (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel(label, fontsize=10, fontweight='bold')
            ax.set_title(label, fontsize=11, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_xlim(-5, 105)
            ax.tick_params(axis='both', labelsize=9)
            
            if ax_idx == 0:
                ax.legend(fontsize=8, loc='best')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'ablation_key_metrics_latent{latent_dim}d.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # ===== Neighborhood metrics (trust/cont across k values) =====
    for latent_dim in latent_dims:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        
        for ax_idx, k in enumerate([10, 50, 100]):
            ax = axes[ax_idx]
            
            for dataset_name in datasets:
                if latent_dim in results[dataset_name]:
                    data = results[dataset_name][latent_dim]
                    pcts = data.get('reference_pcts', [])
                    
                    trust_key_mean = f'trustworthiness_{k}_mean'
                    trust_key_std = f'trustworthiness_{k}_std'
                    cont_key_mean = f'continuity_{k}_mean'
                    cont_key_std = f'continuity_{k}_std'
                    
                    if trust_key_mean in data:
                        color = DATASET_COLORS.get(dataset_name, 'gray')
                        trust_vals = data[trust_key_mean]
                        trust_std = data[trust_key_std]
                        cont_vals = data[cont_key_mean]
                        cont_std = data[cont_key_std]
                        
                        ax.errorbar(pcts, trust_vals, yerr=trust_std, fmt='o-', 
                                   label=f'{dataset_name.upper()} (Trust)',
                                   linewidth=2, markersize=5, color=color, elinewidth=1,
                                   markeredgewidth=0.5, markeredgecolor='white')
                        ax.errorbar(pcts, cont_vals, yerr=cont_std, fmt='s--',
                                   label=f'{dataset_name.upper()} (Cont)',
                                   linewidth=1.5, markersize=4, color=color, alpha=0.6,
                                   elinewidth=1, markeredgewidth=0.5, markeredgecolor='white')
            
            ax.axvline(x=80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Reference Dimensionality (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Metric Value', fontsize=10, fontweight='bold')
            ax.set_title(f'Neighborhood Metrics (k={k})', fontsize=11, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_xlim(-5, 105)
            ax.tick_params(axis='both', labelsize=9)
            
            if ax_idx == 0:
                ax.legend(fontsize=7, loc='best')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'ablation_neighborhood_metrics_latent{latent_dim}d.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # ===== kNN accuracy and density KL =====
    for latent_dim in latent_dims:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        
        # kNN accuracy
        ax = axes[0]
        for dataset_name in datasets:
            if latent_dim in results[dataset_name]:
                data = results[dataset_name][latent_dim]
                pcts = data.get('reference_pcts', [])
                
                if 'knn_accuracy_10_mean' in data:
                    color = DATASET_COLORS.get(dataset_name, 'gray')
                    values = data['knn_accuracy_10_mean']
                    std_vals = data['knn_accuracy_10_std']
                    ax.errorbar(pcts, values, yerr=std_vals, fmt='o-', label=dataset_name.upper(),
                               linewidth=2, markersize=5, color=color, elinewidth=1,
                               markeredgewidth=0.5, markeredgecolor='white')
        
        ax.axvline(x=80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Reference Dimensionality (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('kNN Accuracy (k=10)', fontsize=10, fontweight='bold')
        ax.set_title('kNN Classification', fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(-5, 105)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=8, loc='best')
        
        # Density KL at different scales
        ax = axes[1]
        for dataset_name in datasets:
            if latent_dim in results[dataset_name]:
                data = results[dataset_name][latent_dim]
                pcts = data.get('reference_pcts', [])
                
                if 'density_kl_0_1_mean' in data:
                    color = DATASET_COLORS.get(dataset_name, 'gray')
                    values = data['density_kl_0_1_mean']
                    std_vals = data['density_kl_0_1_std']
                    ax.errorbar(pcts, values, yerr=std_vals, fmt='o-', label=dataset_name.upper(),
                               linewidth=2, markersize=5, color=color, elinewidth=1,
                               markeredgewidth=0.5, markeredgecolor='white')
        
        ax.axvline(x=80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Reference Dimensionality (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Density KL (σ=0.1)', fontsize=10, fontweight='bold')
        ax.set_title('Density Preservation', fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(-5, 105)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=8, loc='best')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'ablation_classification_density_latent{latent_dim}d.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # ===== Wasserstein distances (if available) =====
    first_latent_dim = latent_dims[0]
    if 'wasserstein_H0_mean' in results[datasets[0]][first_latent_dim]:
        for latent_dim in latent_dims:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            
            for dim_idx, dim in enumerate([0, 1]):
                ax = axes[dim_idx]
                wasserstein_key_mean = f'wasserstein_H{dim}_mean'
                wasserstein_key_std = f'wasserstein_H{dim}_std'
                
                for dataset_name in datasets:
                    if latent_dim in results[dataset_name]:
                        data = results[dataset_name][latent_dim]
                        pcts = data.get('reference_pcts', [])
                        
                        if wasserstein_key_mean in data:
                            color = DATASET_COLORS.get(dataset_name, 'gray')
                            values = data[wasserstein_key_mean]
                            std_vals = data[wasserstein_key_std]
                            ax.errorbar(pcts, values, yerr=std_vals, fmt='o-', label=dataset_name.upper(),
                                       linewidth=2, markersize=5, color=color, elinewidth=1,
                                       markeredgewidth=0.5, markeredgecolor='white')
                
                ax.axvline(x=80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xlabel('Reference Dimensionality (%)', fontsize=10, fontweight='bold')
                ax.set_ylabel(f'Wasserstein Distance (H{dim})', fontsize=10, fontweight='bold')
                ax.set_title(f'Topological Preservation (H{dim})', fontsize=11, fontweight='bold', loc='left')
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.set_xlim(-5, 105)
                ax.tick_params(axis='both', labelsize=9)
                
                if dim_idx == 0:
                    ax.legend(fontsize=8, loc='best')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'ablation_wasserstein_latent{latent_dim}d.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")


def print_summary_table(results):
    """
    Print comprehensive summary table showing key metrics with mean ± std.
    Structure: results[dataset][latent_dim] = {metric_mean, metric_std}
    """
    datasets = list(results.keys())
    if not datasets:
        print("No results to summarize")
        return
    
    first_dataset = datasets[0]
    latent_dims = sorted(list(results[first_dataset].keys()))
    reference_pcts = results[first_dataset][latent_dims[0]].get('reference_pcts', [])
    
    # Key metrics to display
    key_metrics = [
        ('distance_correlation', 'Distance Correlation'),
        ('trustworthiness_10', 'Trustworthiness@10'),
        ('triplet_accuracy', 'Triplet Accuracy'),
        ('knn_accuracy_10', 'kNN Accuracy@10'),
    ]
    
    print(f"\n{'='*140}")
    print("COMPREHENSIVE METRICS SUMMARY (Mean ± Std across runs)")
    print(f"{'='*140}\n")
    
    for latent_dim in latent_dims:
        print(f"\nLATENT DIMENSION: {latent_dim}D")
        print("=" * 140)
        
        for metric_key, metric_name in key_metrics:
            print(f"\n{metric_name}:")
            print("-" * 140)
            
            # Header
            header = "Dataset".ljust(14)
            for pct in reference_pcts:
                header += f"| {pct:5.0f}% ".rjust(14)
            header += "| Optimal"
            print(header)
            print("-" * len(header))
            
            # Data rows
            for dataset_name in datasets:
                if latent_dim not in results[dataset_name]:
                    continue
                    
                data = results[dataset_name][latent_dim]
                
                metric_mean_key = f'{metric_key}_mean'
                metric_std_key = f'{metric_key}_std'
                
                if metric_mean_key not in data:
                    continue
                
                values_mean = data[metric_mean_key]
                values_std = data[metric_std_key]
                
                # Find optimal
                if 'density_kl' in metric_key or 'wasserstein' in metric_key:
                    optimal_idx = np.argmin(values_mean)
                else:
                    optimal_idx = np.argmax(values_mean)
                
                optimal_pct = reference_pcts[optimal_idx]
                optimal_val = values_mean[optimal_idx]
                optimal_std = values_std[optimal_idx]
                
                row = dataset_name.upper().ljust(14)
                for mean, std in zip(values_mean, values_std):
                    row += f"| {mean:6.4f}±{std:5.3f} "
                row += f"| {optimal_pct:5.0f}%"
                
                print(row)
            
            print()
    
    print(f"{'='*140}")
    
    # Per-dataset insights
    print("\nPER-DATASET INSIGHTS (across all bottleneck dimensions):")
    print("-" * 140)
    
    for dataset_name in datasets:
        input_dim = DATASET_CONFIGS[dataset_name].get('input_dim', 'unknown')
        print(f"\n{dataset_name.upper()} (input dim: {input_dim}):")
        
        for latent_dim in latent_dims:
            if latent_dim not in results[dataset_name]:
                continue
            
            data = results[dataset_name][latent_dim]
            
            if 'distance_correlation_mean' in data:
                dist_corr_mean = data['distance_correlation_mean']
                dist_corr_std = data['distance_correlation_std']
                
                optimal_idx = np.argmax(dist_corr_mean)
                optimal_pct = reference_pcts[optimal_idx]
                optimal_val = dist_corr_mean[optimal_idx]
                optimal_std = dist_corr_std[optimal_idx]
                original_val = dist_corr_mean[-1]
                original_std = dist_corr_std[-1]
                
                improvement = ((optimal_val - original_val) / (abs(original_val) + 1e-10)) * 100
                
                print(f"  Latent dim {latent_dim}D:")
                print(f"    Optimal reference: {optimal_pct:.0f}% → {optimal_val:.4f}±{optimal_std:.4f}")
                print(f"    100% reference:    {original_val:.4f}±{original_std:.4f} ({improvement:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='MMAE Reference Dimensionality Ablation: Train MMAE with varying reference dimensions (1%-100% PCA)'
    )
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['pbmc3k', 'mnist', 'fmnist', 'cifar10', 'spheres'],
                       help='Datasets to analyze (space-separated, default: pbmc3k mnist fmnist cifar10 spheres)')
    parser.add_argument('--output_dir', type=str, default='results/ablation_reference',
                       help='Output directory for results and figures (default: results/ablation_reference)')
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[2, 64],
                       help='Bottleneck (latent) dimensions to test (default: 2 64)')
    parser.add_argument('--n_runs', type=int, default=3,
                       help='Number of runs per configuration to average (default: 3)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per reference dimension (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Max samples per dataset (default: 5000)')
    parser.add_argument('--visualize_only', action='store_true',
                       help='Skip training, just plot existing results from output_dir')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print("MMAE REFERENCE DIMENSIONALITY ABLATION")
    print(f"{'#'*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Bottleneck dimensions: {args.latent_dims}")
    print(f"Runs per config: {args.n_runs}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"{'#'*70}\n")
    
    if args.visualize_only:
        print("Loading existing results...")
        import json
        results_path = os.path.join(args.output_dir, 'ablation_results.json')
        if not os.path.exists(results_path):
            print(f"ERROR: Results file not found at {results_path}")
            print(f"Make sure you've run the full ablation first, or check the output_dir path.")
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        plot_reference_ablation(results, args.output_dir)
        print_summary_table(results)
        print(f"\nFigures saved to: {args.output_dir}")
    else:
        results = experiment_reference_dimensionality(
            args.datasets,
            latent_dims=args.latent_dims,
            n_runs=args.n_runs,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device,
            base_seed=args.seed,
            n_samples=args.n_samples
        )
        print(f"\n{'#'*70}")
        print(f"All results saved to: {args.output_dir}")
        print(f"  Results:")
        print(f"    - ablation_results.json (complete aggregated results across runs)")
        print(f"  Figures (one per latent dimension):")
        for ld in args.latent_dims:
            print(f"    - ablation_key_metrics_latent{ld}d.png/pdf")
            print(f"    - ablation_neighborhood_metrics_latent{ld}d.png/pdf")
            print(f"    - ablation_classification_density_latent{ld}d.png/pdf")
            print(f"    - ablation_wasserstein_latent{ld}d.png/pdf")
        print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()