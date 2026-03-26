#!/usr/bin/env python
"""
Quick analysis of existing hyperparameter search results to examine
PCA component effect on Wasserstein distances.

Usage:
    # Analyze all MMAE results
    python analyze_existing_pca_wass.py --search_dir experiments/hyperparam_search
    
    # Only latent_dim=2
    python analyze_existing_pca_wass.py --search_dir experiments/hyperparam_search --latent_dim 2
    
    # Specific datasets
    python analyze_existing_pca_wass.py --search_dir experiments/hyperparam_search --datasets mnist fmnist spheres
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re


def find_mmae_trials(search_dir, datasets=None):
    """Find all MMAE trials.csv files in search directory."""
    search_dir = Path(search_dir)
    trials = []
    
    # Pattern: search_dir/dataset/results/mmae_dimX/trials.csv
    # or: search_dir/dataset_dimX/mmae/trials.csv
    # or any variation
    
    for path in search_dir.rglob('trials.csv'):
        # Check if it's in an mmae directory
        if 'mmae' in str(path).lower():
            trials.append(path)
    
    print(f"Found {len(trials)} trials.csv files with 'mmae' in path")
    
    # Also check for trials.csv that might have mmae results inside
    for path in search_dir.rglob('trials.csv'):
        if path not in trials:
            # Peek at the file to see if it has mmae_n_components column
            try:
                df = pd.read_csv(path, nrows=1)
                if 'mmae_n_components' in df.columns:
                    trials.append(path)
            except:
                pass
    
    print(f"Total {len(trials)} trials.csv files found")
    
    # Filter by dataset if specified
    if datasets:
        filtered = []
        for path in trials:
            path_lower = str(path).lower()
            for dataset in datasets:
                if dataset.lower() in path_lower:
                    filtered.append(path)
                    break
        trials = filtered
        print(f"Filtered to {len(trials)} files matching datasets: {datasets}")
    
    return trials


def extract_dataset_info(path):
    """Extract dataset name and latent dim from path."""
    path_str = str(path).lower()
    
    # Common dataset patterns - check more specific ones first!
    dataset_patterns = [
        'spheres_10000d', 'spheres_5000d', 'spheres_1000d',  # specific first
        'spheres',
        'mnist', 'fmnist', 'cifar10', 
        'paul15', 'pbmc3k', 
        'swiss_roll', 'linked_tori', 'klein_bottle'
    ]
    
    dataset = None
    for pattern in dataset_patterns:
        if pattern in path_str:
            dataset = pattern
            break
    
    # Extract latent dim from patterns like "dim2", "dim8", etc.
    latent_dim = None
    dim_match = re.search(r'dim(\d+)', path_str)
    if dim_match:
        latent_dim = int(dim_match.group(1))
    
    return dataset, latent_dim


def load_all_trials(trials_paths, latent_dim_filter=None):
    """Load all trials and organize by (dataset, latent_dim)."""
    data = defaultdict(list)
    
    for path in trials_paths:
        try:
            df = pd.read_csv(path)
            
            # Check if it has MMAE columns
            if 'mmae_n_components' not in df.columns:
                continue
            
            # Extract dataset info
            dataset, latent_dim = extract_dataset_info(path)
            
            if dataset is None:
                print(f"Warning: Could not determine dataset from {path}")
                continue
            
            if latent_dim is None:
                print(f"Warning: Could not determine latent_dim from {path}")
                continue
            
            # Filter by latent_dim if specified
            if latent_dim_filter is not None and latent_dim != latent_dim_filter:
                continue
            
            # Filter successful trials (no NaN in key columns)
            key_cols = ['mmae_n_components', 'wasserstein_H0']
            df_clean = df.dropna(subset=[c for c in key_cols if c in df.columns])
            
            if len(df_clean) == 0:
                print(f"Warning: No valid trials in {path}")
                continue
            
            # Add metadata
            df_clean['dataset'] = dataset
            df_clean['latent_dim'] = latent_dim
            df_clean['source_file'] = str(path)
            
            # Use (dataset, latent_dim) as key
            key = (dataset, latent_dim)
            data[key].append(df_clean)
            print(f"  Loaded {len(df_clean)} trials from {path.parent.name}/{path.name} "
                  f"[{dataset}, dim={latent_dim}]")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return data


def get_input_dims():
    """Known input dimensions for common datasets."""
    from config import DATASET_CONFIGS
    
    # Base dims from config
    dims = {k: v['input_dim'] for k, v in DATASET_CONFIGS.items()}
    
    # Add special cases for high-dim spheres
    dims['spheres_10000d'] = 10001
    dims['spheres_5000d'] = 5001
    dims['spheres_1000d'] = 1001
    
    return dims


def analyze_pca_wasserstein(data, output_dir=None):
    """Analyze relationship between PCA components and Wasserstein distances."""
    
    input_dims = get_input_dims()
    
    print(f"\n{'='*80}")
    print("ANALYSIS: PCA COMPONENTS vs WASSERSTEIN DISTANCES")
    print(f"{'='*80}\n")
    
    all_configs = []
    
    for (dataset, latent_dim), dfs in sorted(data.items()):
        # Combine all dataframes for this config
        df = pd.concat(dfs, ignore_index=True)
        
        if len(df) == 0:
            continue
        
        input_dim = input_dims.get(dataset)
        if input_dim is None:
            print(f"Warning: Unknown input_dim for {dataset}, skipping...")
            continue
        
        # Convert to percentage
        df['pca_pct'] = (df['mmae_n_components'] / input_dim) * 100
        
        print(f"\n{dataset.upper()} - Latent Dim {latent_dim} (input_dim={input_dim})")
        print(f"  Trials: {len(df)}")
        print(f"  PCA range: {df['mmae_n_components'].min():.0f} - {df['mmae_n_components'].max():.0f}")
        print(f"  PCA %: {df['pca_pct'].min():.1f}% - {df['pca_pct'].max():.1f}%")
        
        # Check correlations
        if 'wasserstein_H0' in df.columns:
            corr_h0 = df[['pca_pct', 'wasserstein_H0']].corr().iloc[0, 1]
            print(f"  Correlation PCA% vs Wass H0: {corr_h0:+.3f}")
            
            # Find best
            best_idx = df['wasserstein_H0'].idxmin()
            best_pca = df.loc[best_idx, 'mmae_n_components']
            best_pca_pct = df.loc[best_idx, 'pca_pct']
            best_wass = df.loc[best_idx, 'wasserstein_H0']
            print(f"  Best H0: {best_wass:.4f} at {best_pca:.0f} components ({best_pca_pct:.1f}%)")
        
        if 'wasserstein_H1' in df.columns:
            corr_h1 = df[['pca_pct', 'wasserstein_H1']].corr().iloc[0, 1]
            print(f"  Correlation PCA% vs Wass H1: {corr_h1:+.3f}")
            
            best_idx = df['wasserstein_H1'].idxmin()
            best_pca = df.loc[best_idx, 'mmae_n_components']
            best_pca_pct = df.loc[best_idx, 'pca_pct']
            best_wass = df.loc[best_idx, 'wasserstein_H1']
            print(f"  Best H1: {best_wass:.4f} at {best_pca:.0f} components ({best_pca_pct:.1f}%)")
        
        all_configs.append((dataset, latent_dim, df, input_dim))
    
    # Generate plots
    if output_dir and all_configs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("GENERATING PLOTS")
        print(f"{'='*80}")
        
        generate_plots(all_configs, output_dir)
    
    return all_configs


def generate_plots(all_configs, output_dir):
    """Generate plots for all dataset/latent_dim configs."""
    sns.set_style('whitegrid')
    
    n_configs = len(all_configs)
    
    if n_configs == 0:
        print("No configs to plot")
        return
    
    # Plot 1: H0 - all configs
    fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 4), squeeze=False)
    axes = axes.flatten()
    
    for i, (dataset, latent_dim, df, input_dim) in enumerate(all_configs):
        ax = axes[i]
        
        if 'wasserstein_H0' not in df.columns:
            ax.text(0.5, 0.5, 'No H0 data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dataset.upper()}\nLatent Dim {latent_dim}', fontweight='bold')
            continue
        
        # Scatter plot
        scatter = ax.scatter(df['pca_pct'], df['wasserstein_H0'], 
                           c=df.get('mmae_lambda', 1.0), cmap='viridis', 
                           alpha=0.6, s=50)
        
        # Trend line
        if len(df) > 2:
            z = np.polyfit(df['pca_pct'], df['wasserstein_H0'], 2)  # Quadratic
            p = np.poly1d(z)
            x_trend = np.linspace(df['pca_pct'].min(), df['pca_pct'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
        
        # Mark best
        best_idx = df['wasserstein_H0'].idxmin()
        ax.scatter(df.loc[best_idx, 'pca_pct'], df.loc[best_idx, 'wasserstein_H0'],
                  color='red', s=200, marker='*', edgecolors='black', 
                  linewidths=1.5, zorder=5, label='Best')
        
        ax.set_xlabel('PCA Components (% of input dim)', fontsize=11)
        ax.set_ylabel('Wasserstein H0', fontsize=11)
        ax.set_title(f'{dataset.upper()}\nLatent Dim {latent_dim} (d={input_dim})', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add colorbar for lambda
        if 'mmae_lambda' in df.columns and i == n_configs - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Lambda', fontsize=10)
    
    plt.tight_layout()
    h0_path = output_dir / 'existing_pca_wass_H0.png'
    plt.savefig(h0_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {h0_path}")
    plt.close()
    
    # Plot 2: H1 - all configs
    fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 4), squeeze=False)
    axes = axes.flatten()
    
    for i, (dataset, latent_dim, df, input_dim) in enumerate(all_configs):
        ax = axes[i]
        
        if 'wasserstein_H1' not in df.columns:
            ax.text(0.5, 0.5, 'No H1 data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dataset.upper()}\nLatent Dim {latent_dim}', fontweight='bold')
            continue
        
        # Scatter plot
        scatter = ax.scatter(df['pca_pct'], df['wasserstein_H1'], 
                           c=df.get('mmae_lambda', 1.0), cmap='plasma', 
                           alpha=0.6, s=50)
        
        # Trend line
        if len(df) > 2:
            z = np.polyfit(df['pca_pct'], df['wasserstein_H1'], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(df['pca_pct'].min(), df['pca_pct'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
        
        # Mark best
        best_idx = df['wasserstein_H1'].idxmin()
        ax.scatter(df.loc[best_idx, 'pca_pct'], df.loc[best_idx, 'wasserstein_H1'],
                  color='red', s=200, marker='*', edgecolors='black', 
                  linewidths=1.5, zorder=5, label='Best')
        
        ax.set_xlabel('PCA Components (% of input dim)', fontsize=11)
        ax.set_ylabel('Wasserstein H1', fontsize=11)
        ax.set_title(f'{dataset.upper()}\nLatent Dim {latent_dim} (d={input_dim})', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if 'mmae_lambda' in df.columns and i == n_configs - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Lambda', fontsize=10)
    
    plt.tight_layout()
    h1_path = output_dir / 'existing_pca_wass_H1.png'
    plt.savefig(h1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {h1_path}")
    plt.close()
    
    # Plot 3: Combined view - both H0 and H1 for each config
    fig, axes = plt.subplots(2, n_configs, figsize=(5*n_configs, 8), squeeze=False)
    
    for i, (dataset, latent_dim, df, input_dim) in enumerate(all_configs):
        # H0
        if 'wasserstein_H0' in df.columns:
            ax = axes[0, i]
            ax.scatter(df['pca_pct'], df['wasserstein_H0'], alpha=0.6, s=50)
            if len(df) > 2:
                z = np.polyfit(df['pca_pct'], df['wasserstein_H0'], 2)
                p = np.poly1d(z)
                x_trend = np.linspace(df['pca_pct'].min(), df['pca_pct'].max(), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
            best_idx = df['wasserstein_H0'].idxmin()
            ax.scatter(df.loc[best_idx, 'pca_pct'], df.loc[best_idx, 'wasserstein_H0'],
                      color='red', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
            ax.set_ylabel('Wasserstein H0', fontsize=11)
            ax.set_title(f'{dataset.upper()}\nLatent Dim {latent_dim} (d={input_dim})', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # H1
        if 'wasserstein_H1' in df.columns:
            ax = axes[1, i]
            ax.scatter(df['pca_pct'], df['wasserstein_H1'], alpha=0.6, s=50, color='darkorange')
            if len(df) > 2:
                z = np.polyfit(df['pca_pct'], df['wasserstein_H1'], 2)
                p = np.poly1d(z)
                x_trend = np.linspace(df['pca_pct'].min(), df['pca_pct'].max(), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
            best_idx = df['wasserstein_H1'].idxmin()
            ax.scatter(df.loc[best_idx, 'pca_pct'], df.loc[best_idx, 'wasserstein_H1'],
                      color='red', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
            ax.set_xlabel('PCA Components (% of input dim)', fontsize=11)
            ax.set_ylabel('Wasserstein H1', fontsize=11)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    combined_path = output_dir / 'existing_pca_wass_combined.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze existing MMAE hyperparameter search results')
    parser.add_argument('--search_dir', type=str, required=True,
                       help='Directory containing hyperparameter search results')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Filter to specific datasets (optional)')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Filter to specific latent dimension (e.g., 2, 8, 32)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: search_dir/pca_wass_analysis)')
    args = parser.parse_args()
    
    # Find trials
    print(f"Searching for MMAE trials in: {args.search_dir}")
    if args.latent_dim:
        print(f"Filtering to latent_dim={args.latent_dim}")
    trials_paths = find_mmae_trials(args.search_dir, args.datasets)
    
    if not trials_paths:
        print("No MMAE trials.csv files found!")
        return
    
    # Load all trials
    print(f"\nLoading trials...")
    data = load_all_trials(trials_paths, latent_dim_filter=args.latent_dim)
    
    if not data:
        print("No valid MMAE data found!")
        return
    
    configs = [(dataset, latent_dim) for dataset, latent_dim in sorted(data.keys())]
    print(f"\nFound data for {len(configs)} configurations:")
    for dataset, latent_dim in configs:
        print(f"  - {dataset}, latent_dim={latent_dim}")
    
    # Set output dir
    if args.output_dir is None:
        suffix = f'_dim{args.latent_dim}' if args.latent_dim else ''
        args.output_dir = Path(args.search_dir) / f'pca_wass_analysis{suffix}'
    
    # Analyze
    analyze_pca_wasserstein(data, args.output_dir)


if __name__ == '__main__':
    main()