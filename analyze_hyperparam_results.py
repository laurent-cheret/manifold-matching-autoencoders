#!/usr/bin/env python
"""
Analyze hyperparameter search results.

Visualizes relationships between hyperparameters and metrics,
identifies patterns, and compares models.

Usage:
    python analyze_hyperparam_results.py --results_dir experiments/hyperparam_search/spheres/results
    python analyze_hyperparam_results.py --results_dir experiments/hyperparam_search/mnist/results --metric knn_accuracy_5
"""

import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Metrics where lower is better
LOWER_IS_BETTER = [
    'reconstruction_error', 'wasserstein_H0', 'wasserstein_H1',
    'rmse', 'mrre_zx', 'mrre_xz',
    'density_kl_0_01', 'density_kl_0_1', 'density_kl_1_0'
]


def load_all_trials(results_dir):
    """Load all trial CSVs from hyperparameter search results."""
    results_dir = Path(results_dir)
    all_trials = []
    
    # Find all trials.csv files
    for trials_csv in results_dir.glob("**/trials.csv"):
        df = pd.read_csv(trials_csv)
        
        # Extract model and latent_dim from folder name
        folder_name = trials_csv.parent.name  # e.g., "mmae_dim2"
        parts = folder_name.rsplit("_dim", 1)
        if len(parts) == 2:
            df['model'] = parts[0]
            df['latent_dim'] = int(parts[1])
        
        all_trials.append(df)
    
    if not all_trials:
        print(f"No trials.csv found in {results_dir}")
        return None
    
    combined = pd.concat(all_trials, ignore_index=True)
    print(f"Loaded {len(combined)} trials from {len(all_trials)} experiments")
    
    return combined


def load_best_configs(results_dir):
    """Load all best_config.json files."""
    results_dir = Path(results_dir)
    best_configs = []
    
    for config_path in results_dir.glob("**/best_config.json"):
        with open(config_path) as f:
            config = json.load(f)
        config['folder'] = config_path.parent.name
        best_configs.append(config)
    
    return best_configs


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_hyperparam_vs_metric(df, hyperparam, metric, save_dir=None):
    """Scatter plot of hyperparameter vs metric, colored by model."""
    if hyperparam not in df.columns or metric not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, colors):
        mask = df['model'] == model
        subset = df[mask].dropna(subset=[hyperparam, metric])
        if len(subset) > 0:
            ax.scatter(subset[hyperparam], subset[metric], 
                      label=model, alpha=0.6, c=[color], s=50)
    
    ax.set_xlabel(hyperparam, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    
    direction = "↓" if metric in LOWER_IS_BETTER else "↑"
    ax.set_title(f"{hyperparam} vs {metric} {direction}", fontsize=14)
    
    if hyperparam in ['learning_rate', 'mmae_lambda', 'topo_lambda', 'rtd_lambda']:
        ax.set_xscale('log')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{hyperparam}_vs_{metric}.png"), dpi=150)
        plt.close()
    else:
        plt.show()


def plot_hyperparam_effect(df, hyperparam, metric, n_bins=5, save_dir=None):
    """Box plot showing effect of hyperparameter bins on metric."""
    if hyperparam not in df.columns or metric not in df.columns:
        return
    
    df_clean = df.dropna(subset=[hyperparam, metric])
    if len(df_clean) < 10:
        return
    
    # Bin the hyperparameter
    if hyperparam in ['learning_rate', 'mmae_lambda', 'topo_lambda', 'rtd_lambda']:
        df_clean['param_bin'] = pd.qcut(np.log10(df_clean[hyperparam]), n_bins, duplicates='drop')
    else:
        df_clean['param_bin'] = pd.qcut(df_clean[hyperparam], n_bins, duplicates='drop')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df_clean['model'].unique()
    
    # Create grouped box plot
    positions = []
    labels = []
    data_to_plot = []
    colors_list = []
    
    bins = sorted(df_clean['param_bin'].unique())
    width = 0.8 / len(models)
    
    color_map = dict(zip(models, plt.cm.tab10(np.linspace(0, 1, len(models)))))
    
    for i, bin_val in enumerate(bins):
        for j, model in enumerate(models):
            mask = (df_clean['param_bin'] == bin_val) & (df_clean['model'] == model)
            values = df_clean.loc[mask, metric].values
            if len(values) > 0:
                pos = i + (j - len(models)/2 + 0.5) * width
                bp = ax.boxplot([values], positions=[pos], widths=width*0.8,
                               patch_artist=True)
                bp['boxes'][0].set_facecolor(color_map[model])
                bp['boxes'][0].set_alpha(0.7)
    
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([str(b) for b in bins], rotation=45, ha='right')
    ax.set_xlabel(f"{hyperparam} (binned)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    
    direction = "↓" if metric in LOWER_IS_BETTER else "↑"
    ax.set_title(f"Effect of {hyperparam} on {metric} {direction}", fontsize=14)
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, facecolor=color_map[m], alpha=0.7) for m in models]
    ax.legend(handles, models, loc='best')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{hyperparam}_effect_on_{metric}.png"), dpi=150)
        plt.close()
    else:
        plt.show()


def plot_model_comparison(df, metric, save_dir=None):
    """Box plot comparing models on a metric across all trials."""
    if metric not in df.columns:
        return
    
    df_clean = df.dropna(subset=[metric])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(df_clean['model'].unique())
    data = [df_clean[df_clean['model'] == m][metric].values for m in models]
    
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    direction = "↓" if metric in LOWER_IS_BETTER else "↑"
    ax.set_title(f"Model Comparison: {metric} {direction}", fontsize=14)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(models)+1), means, color='red', marker='D', s=50, zorder=3, label='Mean')
    ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"model_comparison_{metric}.png"), dpi=150)
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(df, model, save_dir=None):
    """Heatmap of correlations between hyperparameters and metrics."""
    df_model = df[df['model'] == model].copy()
    
    # Identify hyperparameter columns
    hyperparam_cols = ['learning_rate', 'batch_size']
    if model == 'mmae':
        hyperparam_cols += ['mmae_n_components', 'mmae_lambda']
    elif model == 'topoae':
        hyperparam_cols += ['topo_lambda']
    elif model == 'rtdae':
        hyperparam_cols += ['rtd_lambda', 'rtd_dim', 'rtd_card']
    
    # Identify metric columns
    metric_cols = [c for c in df_model.columns if any(m in c for m in [
        'reconstruction', 'distance_correlation', 'trustworthiness', 'continuity',
        'knn_accuracy', 'density_kl', 'wasserstein', 'clustering', 'triplet'
    ]) and '_std' not in c]
    
    # Filter to existing columns
    hyperparam_cols = [c for c in hyperparam_cols if c in df_model.columns]
    metric_cols = [c for c in metric_cols if c in df_model.columns]
    
    if not hyperparam_cols or not metric_cols:
        return
    
    # Compute correlations
    corr_data = []
    for hp in hyperparam_cols:
        row = []
        for metric in metric_cols:
            valid = df_model[[hp, metric]].dropna()
            if len(valid) > 3:
                corr = valid[hp].corr(valid[metric])
            else:
                corr = np.nan
            row.append(corr)
        corr_data.append(row)
    
    corr_df = pd.DataFrame(corr_data, index=hyperparam_cols, columns=metric_cols)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title(f"Hyperparameter-Metric Correlations: {model.upper()}", fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"correlation_heatmap_{model}.png"), dpi=150)
        plt.close()
    else:
        plt.show()


def plot_pca_components_analysis(df, save_dir=None):
    """Analyze effect of PCA components for MMAE."""
    df_mmae = df[df['model'] == 'mmae'].copy()
    
    if 'mmae_n_components' not in df_mmae.columns or len(df_mmae) < 5:
        return
    
    metrics_to_plot = ['density_kl_0_1', 'trustworthiness_10', 'knn_accuracy_5', 
                       'reconstruction_error', 'clustering_ari']
    metrics_to_plot = [m for m in metrics_to_plot if m in df_mmae.columns]
    
    if not metrics_to_plot:
        return
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics_to_plot):
        valid = df_mmae[['mmae_n_components', metric]].dropna()
        
        ax.scatter(valid['mmae_n_components'], valid[metric], alpha=0.6, s=50)
        
        # Add trend line
        if len(valid) > 3:
            z = np.polyfit(valid['mmae_n_components'], valid[metric], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['mmae_n_components'].min(), 
                                valid['mmae_n_components'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
        
        direction = "↓" if metric in LOWER_IS_BETTER else "↑"
        ax.set_xlabel("PCA Components", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} {direction}", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("MMAE: Effect of PCA Components", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "mmae_pca_analysis.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_latent_dim_comparison(df, metric, save_dir=None):
    """Compare performance across latent dimensions."""
    if 'latent_dim' not in df.columns or metric not in df.columns:
        return
    
    df_clean = df.dropna(subset=[metric])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = sorted(df_clean['model'].unique())
    latent_dims = sorted(df_clean['latent_dim'].unique())
    
    x = np.arange(len(latent_dims))
    width = 0.8 / len(models)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        means = []
        stds = []
        for dim in latent_dims:
            mask = (df_clean['model'] == model) & (df_clean['latent_dim'] == dim)
            values = df_clean.loc[mask, metric]
            means.append(values.mean() if len(values) > 0 else np.nan)
            stds.append(values.std() if len(values) > 1 else 0)
        
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model, color=color, alpha=0.7, capsize=3)
    
    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(latent_dims)
    
    direction = "↓" if metric in LOWER_IS_BETTER else "↑"
    ax.set_title(f"{metric} {direction} by Latent Dimension", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"latent_dim_{metric}.png"), dpi=150)
        plt.close()
    else:
        plt.show()


# ============================================================
# STATISTICAL INSIGHTS
# ============================================================

def print_insights(df, target_metric='density_kl_0_1'):
    """Print statistical insights from hyperparameter search."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH INSIGHTS")
    print("="*80)
    
    # Best overall for each model
    print(f"\n📊 Best {target_metric} per model:")
    print("-"*60)
    
    direction = 'min' if target_metric in LOWER_IS_BETTER else 'max'
    
    for model in sorted(df['model'].unique()):
        df_model = df[df['model'] == model].dropna(subset=[target_metric])
        if len(df_model) == 0:
            continue
        
        if direction == 'min':
            best_idx = df_model[target_metric].idxmin()
        else:
            best_idx = df_model[target_metric].idxmax()
        
        best = df_model.loc[best_idx]
        print(f"\n{model.upper()}:")
        print(f"  Best {target_metric}: {best[target_metric]:.6f}")
        print(f"  Learning rate: {best.get('learning_rate', 'N/A'):.2e}" if pd.notna(best.get('learning_rate')) else "  Learning rate: N/A")
        print(f"  Batch size: {best.get('batch_size', 'N/A')}")
        
        if model == 'mmae' and 'mmae_n_components' in best:
            print(f"  PCA components: {best['mmae_n_components']:.0f}")
            print(f"  Lambda: {best.get('mmae_lambda', 'N/A'):.4f}" if pd.notna(best.get('mmae_lambda')) else "")
        elif model == 'topoae' and 'topo_lambda' in best:
            print(f"  Lambda: {best['topo_lambda']:.4f}")
        elif model == 'rtdae' and 'rtd_lambda' in best:
            print(f"  Lambda: {best['rtd_lambda']:.4f}")
            print(f"  RTD dim: {best.get('rtd_dim', 'N/A')}")
    
    # Correlation insights
    print(f"\n\n📈 Key Correlations with {target_metric}:")
    print("-"*60)
    
    for model in sorted(df['model'].unique()):
        df_model = df[df['model'] == model]
        print(f"\n{model.upper()}:")
        
        params = ['learning_rate', 'batch_size']
        if model == 'mmae':
            params += ['mmae_n_components', 'mmae_lambda']
        elif model == 'topoae':
            params += ['topo_lambda']
        elif model == 'rtdae':
            params += ['rtd_lambda', 'rtd_card']
        
        for param in params:
            if param not in df_model.columns:
                continue
            valid = df_model[[param, target_metric]].dropna()
            if len(valid) > 5:
                corr = valid[param].corr(valid[target_metric])
                if abs(corr) > 0.2:
                    sign = "↑" if corr > 0 else "↓"
                    better = "worse" if (corr > 0) == (target_metric in LOWER_IS_BETTER) else "better"
                    print(f"  {param}: r={corr:+.3f} (higher {param} → {better} {target_metric})")
    
    # Batch size insights
    print(f"\n\n🔢 Batch Size Patterns:")
    print("-"*60)
    
    for model in sorted(df['model'].unique()):
        df_model = df[df['model'] == model].dropna(subset=['batch_size', target_metric])
        if len(df_model) < 5:
            continue
        
        low_bs = df_model[df_model['batch_size'] < df_model['batch_size'].median()][target_metric]
        high_bs = df_model[df_model['batch_size'] >= df_model['batch_size'].median()][target_metric]
        
        if direction == 'min':
            low_better = low_bs.mean() < high_bs.mean()
        else:
            low_better = low_bs.mean() > high_bs.mean()
        
        preference = "smaller" if low_better else "larger"
        print(f"  {model.upper()}: Prefers {preference} batch sizes")
        print(f"    Low BS mean: {low_bs.mean():.4f}, High BS mean: {high_bs.mean():.4f}")
    
    # MMAE-specific insights
    if 'mmae' in df['model'].values:
        print(f"\n\n🎯 MMAE PCA Components Analysis:")
        print("-"*60)
        df_mmae = df[df['model'] == 'mmae'].dropna(subset=['mmae_n_components', target_metric])
        
        if len(df_mmae) > 5:
            corr = df_mmae['mmae_n_components'].corr(df_mmae[target_metric])
            print(f"  Correlation with {target_metric}: {corr:+.3f}")
            
            # Find optimal range
            if direction == 'min':
                best_10pct = df_mmae.nsmallest(max(1, len(df_mmae)//10), target_metric)
            else:
                best_10pct = df_mmae.nlargest(max(1, len(df_mmae)//10), target_metric)
            
            print(f"  Best 10% trials use PCA components: {best_10pct['mmae_n_components'].min():.0f} - {best_10pct['mmae_n_components'].max():.0f}")
            print(f"  Mean in best trials: {best_10pct['mmae_n_components'].mean():.1f}")


def generate_report(df, save_dir, target_metric='density_kl_0_1'):
    """Generate comprehensive analysis report."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating analysis report in {save_dir}...")
    
    # Print insights
    print_insights(df, target_metric)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # Model comparison
    for metric in [target_metric, 'reconstruction_error', 'trustworthiness_10', 
                   'knn_accuracy_5', 'clustering_ari']:
        if metric in df.columns:
            plot_model_comparison(df, metric, save_dir)
    
    # Hyperparameter effects
    for hp in ['learning_rate', 'batch_size']:
        plot_hyperparam_vs_metric(df, hp, target_metric, save_dir)
        plot_hyperparam_effect(df, hp, target_metric, save_dir=save_dir)
    
    # Model-specific plots
    for model in df['model'].unique():
        plot_correlation_heatmap(df, model, save_dir)
    
    # MMAE PCA analysis
    plot_pca_components_analysis(df, save_dir)
    
    # Latent dimension comparison
    if 'latent_dim' in df.columns:
        plot_latent_dim_comparison(df, target_metric, save_dir)
    
    print(f"\n✅ Report saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing hyperparam search results')
    parser.add_argument('--metric', type=str, default='density_kl_0_1',
                       help='Target metric to analyze')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: results_dir/analysis)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Only print insights, skip plot generation')
    args = parser.parse_args()
    
    # Load data
    df = load_all_trials(args.results_dir)
    if df is None:
        return
    
    print(f"\nModels: {sorted(df['model'].unique())}")
    print(f"Latent dims: {sorted(df['latent_dim'].unique()) if 'latent_dim' in df.columns else 'N/A'}")
    print(f"Total trials: {len(df)}")
    
    if args.no_plots:
        print_insights(df, args.metric)
    else:
        save_dir = args.save_dir or os.path.join(args.results_dir, 'analysis')
        generate_report(df, save_dir, args.metric)


if __name__ == '__main__':
    main()