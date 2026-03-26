#!/usr/bin/env python
"""
Analyze Hyperparameter Search Results.

Extracts trends from trials.csv files:
- MMAE: Performance vs PCA components
- Other AEs: Performance vs batch size

Usage:
    python analyze_hyperparam_trials.py --search_dir experiments/hyperparam_search
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Key metrics to analyze
KEY_METRICS = [
    'distance_correlation',
    'triplet_accuracy', 
    'knn_accuracy_5',
    'trustworthiness_10',
    'continuity_10',
    'density_kl_0_1',
    'wasserstein_H0',
    'wasserstein_H1',
    'reconstruction_error',
]

# Metric properties: (display_name, higher_is_better)
METRIC_INFO = {
    'distance_correlation': ('Distance Correlation', True),
    'triplet_accuracy': ('Triplet Accuracy', True),
    'knn_accuracy_5': ('k-NN Accuracy (k=5)', True),
    'knn_accuracy_10': ('k-NN Accuracy (k=10)', True),
    'trustworthiness_10': ('Trustworthiness', True),
    'trustworthiness_50': ('Trustworthiness (k=50)', True),
    'continuity_10': ('Continuity', True),
    'continuity_50': ('Continuity (k=50)', True),
    'density_kl_0_1': ('Density KL', False),
    'density_kl_0_01': ('Density KL (σ=0.01)', False),
    'wasserstein_H0': ('Wasserstein H₀', False),
    'wasserstein_H1': ('Wasserstein H₁', False),
    'reconstruction_error': ('Reconstruction Error', False),
    'clustering_ari': ('Clustering ARI', True),
    'clustering_nmi': ('Clustering NMI', True),
    'silhouette_score': ('Silhouette Score', True),
}

# Colors for models
MODEL_COLORS = {
    'mmae': '#1f77b4',
    'vanilla': '#7f7f7f',
    'topoae': '#2ca02c',
    'rtdae': '#d62728',
    'geomae': '#9467bd',
    'ggae': '#8c564b',
}


def parse_folder_name(folder_name):
    """Parse model and latent dim from folder name like 'mmae_dim2' or 'rtdae_dim128'."""
    parts = folder_name.split('_dim')
    if len(parts) == 2:
        model = parts[0]
        try:
            latent_dim = int(parts[1])
        except ValueError:
            latent_dim = None
        return model, latent_dim
    return folder_name, None


def load_all_trials(search_dir):
    """Load all trials.csv files from the search directory."""
    all_data = []
    
    # Find all trials.csv files
    pattern = os.path.join(search_dir, '**/trials.csv')
    trial_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(trial_files)} trial files")
    
    for trial_file in trial_files:
        try:
            df = pd.read_csv(trial_file)
            
            # Parse path to get dataset, model, latent_dim
            path_parts = Path(trial_file).parts
            
            # Find dataset and model from path
            # Expected: .../hyperparam_search/{dataset}/results/{model}_dim{latent_dim}/trials.csv
            dataset = None
            model = None
            latent_dim = None
            
            for i, part in enumerate(path_parts):
                if part == 'results' and i > 0:
                    dataset = path_parts[i - 1]
                    if i + 1 < len(path_parts):
                        model, latent_dim = parse_folder_name(path_parts[i + 1])
            
            if dataset is None or model is None:
                # Try alternative parsing
                for i, part in enumerate(path_parts):
                    if part == 'hyperparam_search' and i + 1 < len(path_parts):
                        dataset = path_parts[i + 1]
                    if '_dim' in part:
                        model, latent_dim = parse_folder_name(part)
            
            if dataset and model:
                df['dataset'] = dataset
                df['model'] = model
                df['latent_dim'] = latent_dim
                df['source_file'] = trial_file
                all_data.append(df)
                print(f"  Loaded: {dataset}/{model}_dim{latent_dim} ({len(df)} trials)")
            else:
                print(f"  Warning: Could not parse {trial_file}")
                
        except Exception as e:
            print(f"  Error loading {trial_file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def analyze_mmae_pca_components(df, output_dir):
    """Analyze MMAE performance vs PCA components."""
    
    mmae_df = df[df['model'] == 'mmae'].copy()
    
    if len(mmae_df) == 0:
        print("No MMAE trials found")
        return
    
    if 'mmae_n_components' not in mmae_df.columns:
        print("No mmae_n_components column found")
        return
    
    print(f"\nAnalyzing MMAE: {len(mmae_df)} trials")
    
    # Get unique datasets and latent dims
    datasets = mmae_df['dataset'].unique()
    latent_dims = sorted(mmae_df['latent_dim'].dropna().unique())
    
    print(f"  Datasets: {list(datasets)}")
    print(f"  Latent dims: {list(latent_dims)}")
    
    # For each dataset, analyze PCA component effect
    for dataset in datasets:
        dataset_df = mmae_df[mmae_df['dataset'] == dataset]
        
        if len(dataset_df) < 5:
            continue
        
        # Create figure for this dataset
        metrics_to_plot = [m for m in KEY_METRICS if m in dataset_df.columns]
        n_metrics = len(metrics_to_plot)
        
        if n_metrics == 0:
            continue
        
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            display_name, higher_is_better = METRIC_INFO.get(metric, (metric, True))
            
            # Plot for each latent dim
            for latent_dim in latent_dims:
                subset = dataset_df[dataset_df['latent_dim'] == latent_dim]
                if len(subset) < 3:
                    continue
                
                # Sort by PCA components
                subset = subset.sort_values('mmae_n_components')
                
                ax.scatter(subset['mmae_n_components'], subset[metric], 
                          alpha=0.5, s=20, label=f'dim={int(latent_dim)}')
                
                # Add trend line (rolling mean)
                if len(subset) >= 5:
                    subset_sorted = subset.sort_values('mmae_n_components')
                    x = subset_sorted['mmae_n_components'].values
                    y = subset_sorted[metric].values
                    
                    # Bin and average
                    n_bins = min(10, len(subset) // 2)
                    if n_bins >= 2:
                        bins = np.linspace(x.min(), x.max(), n_bins + 1)
                        bin_centers = []
                        bin_means = []
                        for i in range(n_bins):
                            mask = (x >= bins[i]) & (x < bins[i+1])
                            if mask.sum() > 0:
                                bin_centers.append((bins[i] + bins[i+1]) / 2)
                                bin_means.append(y[mask].mean())
                        if len(bin_centers) >= 2:
                            ax.plot(bin_centers, bin_means, '-', linewidth=2)
            
            ax.set_xlabel('PCA Components')
            ax.set_ylabel(display_name)
            ax.set_title(display_name)
            
            # Add arrow indicating better direction
            direction = '↑' if higher_is_better else '↓'
            ax.annotate(f'{direction} better', xy=(0.98, 0.98), xycoords='axes fraction',
                       fontsize=8, ha='right', va='top', color='gray')
            
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
        
        # Hide unused axes
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'MMAE Performance vs PCA Components - {dataset.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'mmae_pca_analysis_{dataset}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


def analyze_batch_size_effect(df, output_dir):
    """Analyze performance vs batch size for all models."""
    
    if 'batch_size' not in df.columns:
        print("No batch_size column found")
        return
    
    print(f"\nAnalyzing batch size effect...")
    
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        if len(dataset_df) < 10:
            continue
        
        # Key metrics for batch size analysis
        metrics_to_plot = ['distance_correlation', 'triplet_accuracy', 'knn_accuracy_5', 
                          'reconstruction_error', 'train_time']
        metrics_to_plot = [m for m in metrics_to_plot if m in dataset_df.columns]
        
        if len(metrics_to_plot) == 0:
            continue
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4*len(metrics_to_plot), 4))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            display_name, higher_is_better = METRIC_INFO.get(metric, (metric, True))
            
            for model in models:
                model_df = dataset_df[dataset_df['model'] == model]
                if len(model_df) < 3:
                    continue
                
                # Aggregate by batch size
                grouped = model_df.groupby('batch_size')[metric].agg(['mean', 'std']).reset_index()
                grouped = grouped.sort_values('batch_size')
                
                color = MODEL_COLORS.get(model, 'gray')
                ax.errorbar(grouped['batch_size'], grouped['mean'], 
                           yerr=grouped['std'], fmt='o-', 
                           color=color, label=model.upper(),
                           capsize=3, markersize=5)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel(display_name)
            ax.set_title(display_name)
            
            direction = '↑' if higher_is_better else '↓'
            ax.annotate(f'{direction} better', xy=(0.98, 0.98), xycoords='axes fraction',
                       fontsize=8, ha='right', va='top', color='gray')
            
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
        
        plt.suptitle(f'Performance vs Batch Size - {dataset.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'batch_size_analysis_{dataset}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


def analyze_mmae_pca_ratio(df, output_dir):
    """
    Analyze MMAE performance vs PCA components as percentage of input dim.
    This is the key plot for the paper argument.
    """
    
    mmae_df = df[df['model'] == 'mmae'].copy()
    
    if len(mmae_df) == 0 or 'mmae_n_components' not in mmae_df.columns:
        return
    
    # We need to know input dim for each dataset
    # Estimate from max PCA components used
    input_dims = {
        'mnist': 784,
        'fmnist': 784,
        'cifar10': 3072,
        'pbmc3k': 1838,
        'paul15': 2000,
        'coil20': 16384,
        'spheres': 101,
        'earth': 3,
        'linked_tori': 100,
        'concentric_spheres': 1000,
        'klein_bottle': 4,
        'mammoth': 3,
    }
    
    # Add PCA ratio
    def get_pca_ratio(row):
        dataset = row['dataset']
        n_comp = row['mmae_n_components']
        input_dim = input_dims.get(dataset, n_comp * 2)  # Fallback
        return n_comp / input_dim * 100
    
    mmae_df['pca_ratio'] = mmae_df.apply(get_pca_ratio, axis=1)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['distance_correlation', 'triplet_accuracy', 'knn_accuracy_5', 'density_kl_0_1']
    metrics = [m for m in metrics if m in mmae_df.columns]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        display_name, higher_is_better = METRIC_INFO.get(metric, (metric, True))
        
        for dataset in mmae_df['dataset'].unique():
            subset = mmae_df[mmae_df['dataset'] == dataset]
            if len(subset) < 5:
                continue
            
            # Bin by PCA ratio
            subset = subset.copy()
            subset['pca_bin'] = pd.cut(subset['pca_ratio'], bins=10)
            grouped = subset.groupby('pca_bin')[metric].agg(['mean', 'std']).reset_index()
            
            # Get bin centers
            bin_centers = [interval.mid for interval in grouped['pca_bin']]
            
            ax.plot(bin_centers, grouped['mean'], 'o-', label=dataset.upper(), markersize=5)
        
        ax.set_xlabel('PCA Components (% of input dim)')
        ax.set_ylabel(display_name)
        ax.set_title(display_name)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, 105)
        
        direction = '↑' if higher_is_better else '↓'
        ax.annotate(f'{direction} better', xy=(0.98, 0.98), xycoords='axes fraction',
                   fontsize=8, ha='right', va='top', color='gray')
    
    plt.suptitle('MMAE Performance vs PCA Components (% of Input Dimension)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'mmae_pca_ratio_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_table(df, output_dir):
    """Create summary statistics table."""
    
    summary_rows = []
    
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
            
            if len(subset) == 0:
                continue
            
            row = {
                'dataset': dataset,
                'model': model,
                'n_trials': len(subset),
            }
            
            # Add metric statistics
            for metric in ['distance_correlation', 'triplet_accuracy', 'knn_accuracy_5']:
                if metric in subset.columns:
                    row[f'{metric}_mean'] = subset[metric].mean()
                    row[f'{metric}_std'] = subset[metric].std()
                    row[f'{metric}_max'] = subset[metric].max()
            
            # MMAE-specific: PCA component range
            if model == 'mmae' and 'mmae_n_components' in subset.columns:
                row['pca_min'] = subset['mmae_n_components'].min()
                row['pca_max'] = subset['mmae_n_components'].max()
                
                # Best PCA components (by distance correlation)
                if 'distance_correlation' in subset.columns:
                    best_idx = subset['distance_correlation'].idxmax()
                    row['best_pca'] = subset.loc[best_idx, 'mmae_n_components']
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    csv_path = os.path.join(output_dir, 'trials_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search trials')
    parser.add_argument('--search_dir', type=str, 
                       default='experiments/hyperparam_search',
                       help='Directory containing hyperparam search results')
    parser.add_argument('--output_dir', type=str,
                       default='results/hyperparam_analysis',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading trials from: {args.search_dir}")
    df = load_all_trials(args.search_dir)
    
    if len(df) == 0:
        print("No trials found!")
        return
    
    print(f"\nTotal trials loaded: {len(df)}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Models: {df['model'].unique().tolist()}")
    
    # Save combined data
    combined_path = os.path.join(args.output_dir, 'all_trials.csv')
    df.to_csv(combined_path, index=False)
    print(f"Saved combined data: {combined_path}")
    
    # Analyses
    analyze_mmae_pca_components(df, args.output_dir)
    analyze_mmae_pca_ratio(df, args.output_dir)
    analyze_batch_size_effect(df, args.output_dir)
    create_summary_table(df, args.output_dir)
    
    print(f"\nAnalysis complete. Results in: {args.output_dir}")


if __name__ == '__main__':
    main()