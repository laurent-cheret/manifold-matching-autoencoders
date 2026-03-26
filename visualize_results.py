#!/usr/bin/env python
"""
Visualize bottleneck study results.

Usage:
    python visualize_results.py --results_dir results/bottleneck_study/mnist
    python visualize_results.py --results_file results/bottleneck_study/mnist/results.csv
    python visualize_results.py --results_dir results/bottleneck_study/mnist --metrics distance_correlation trustworthiness
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Base metrics to aggregate (will average across k values or sigma values)
AGGREGATED_METRICS = {
    'trustworthiness': 'trustworthiness_',
    'continuity': 'continuity_',
    'knn_accuracy': 'knn_accuracy_',
    'mrre_zx': 'mrre_zx_',
    'mrre_xz': 'mrre_xz_',
    'density_kl': 'density_kl_',
}

# Key metrics to visualize by default
DEFAULT_METRICS = [
    'reconstruction_error',
    'distance_correlation', 
    'triplet_accuracy',
    'trustworthiness',  # Will be averaged across k values
    'continuity',       # Will be averaged across k values
    'knn_accuracy',     # Will be averaged across k values
    'density_kl',       # Will be averaged across sigma values
    'wasserstein_H0',
    'wasserstein_H1',
]

# Metrics where lower is better
LOWER_IS_BETTER = ['reconstruction_error', 'wasserstein_H0', 'wasserstein_H1', 'rmse', 'mrre_zx', 'mrre_xz', 'density_kl']

# Nice display names
METRIC_NAMES = {
    'reconstruction_error': 'Reconstruction Error',
    'distance_correlation': 'Distance Correlation',
    'triplet_accuracy': 'Triplet Accuracy',
    'trustworthiness': 'Trustworthiness',
    'trustworthiness_10': 'Trustworthiness (k=10)',
    'trustworthiness_50': 'Trustworthiness (k=50)',
    'continuity': 'Continuity',
    'continuity_10': 'Continuity (k=10)',
    'continuity_50': 'Continuity (k=50)',
    'knn_accuracy': 'kNN Accuracy',
    'knn_accuracy_5': 'kNN Accuracy (k=5)',
    'knn_accuracy_10': 'kNN Accuracy (k=10)',
    'mrre_zx': 'MRRE (Z→X)',
    'mrre_xz': 'MRRE (X→Z)',
    'density_kl': 'Density KL',
    'density_kl_0_01': 'Density KL (σ=0.01)',
    'density_kl_0_1': 'Density KL (σ=0.1)',
    'density_kl_1_0': 'Density KL (σ=1.0)',
    'wasserstein_H0': 'Wasserstein H0',
    'wasserstein_H1': 'Wasserstein H1',
    'rmse': 'RMSE',
    'train_time_seconds': 'Training Time (s)',
}


def load_results(path):
    """Load results from CSV file or directory."""
    if os.path.isdir(path):
        csv_path = os.path.join(path, 'results.csv')
    else:
        csv_path = path
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def get_k_values_for_metric(df, base_name):
    """Extract k values used for a metric (e.g., trustworthiness_10, trustworthiness_50 -> [10, 50])."""
    pattern = f'^{base_name}(\\d+)$'
    k_values = []
    for col in df.columns:
        match = re.match(pattern, col)
        if match:
            k_values.append(int(match.group(1)))
    return sorted(k_values)


def aggregate_metric_across_k(df, base_metric):
    """
    Aggregate a metric across different k values (or sigma values) by averaging.
    Returns new column name, aggregated values, std values, and parameter values used.
    """
    prefix = AGGREGATED_METRICS.get(base_metric, f'{base_metric}_')
    
    # Find all columns matching this metric
    metric_cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith('_std')]
    
    if not metric_cols:
        return None, None, None, []
    
    # Extract parameter values (k or sigma)
    param_values = []
    for col in metric_cols:
        suffix = col[len(prefix):]
        # Handle both integer k values (e.g., _10) and decimal sigma (e.g., _0_1)
        if '_' in suffix:
            # Decimal format like 0_01 -> 0.01
            param_values.append(suffix.replace('_', '.'))
        else:
            # Integer format
            param_values.append(suffix)
    
    # Average across parameter values
    avg_values = df[metric_cols].mean(axis=1)
    
    # Average the stds (approximation - proper would be to combine variances)
    std_cols = [f'{c}_std' for c in metric_cols if f'{c}_std' in df.columns]
    if std_cols:
        avg_std = df[std_cols].mean(axis=1)
    else:
        avg_std = None
    
    return base_metric, avg_values, avg_std, param_values


def prepare_dataframe_with_aggregates(df):
    """Add aggregated metric columns to dataframe."""
    df = df.copy()
    k_values_used = {}
    
    for base_metric, prefix in AGGREGATED_METRICS.items():
        _, avg_values, avg_std, k_vals = aggregate_metric_across_k(df, base_metric)
        if avg_values is not None:
            df[base_metric] = avg_values
            if avg_std is not None:
                df[f'{base_metric}_std'] = avg_std
            k_values_used[base_metric] = k_vals
    
    return df, k_values_used


def get_available_metrics(df):
    """Get list of metrics available in the dataframe."""
    # Exclude non-metric columns and std columns
    exclude = ['model', 'latent_dim', 'dataset']
    metrics = [c for c in df.columns if c not in exclude and not c.endswith('_std')]
    return metrics


def plot_metric_comparison(df, metric, ax=None, show_std=True, k_values=None):
    """Plot a single metric comparison across models and latent dims."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    latent_dims = sorted(df['latent_dim'].unique())
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('latent_dim')
        
        x = model_data['latent_dim'].values
        y = model_data[metric].values
        
        # Plot line
        ax.plot(x, y, 'o-', label=model, color=colors[i], markersize=6)
        
        # Add error bars if std column exists
        std_col = f'{metric}_std'
        if show_std and std_col in df.columns:
            std = model_data[std_col].values
            ax.fill_between(x, y - std, y + std, alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Latent Dimension')
    
    # Build title with parameter values if applicable
    title = METRIC_NAMES.get(metric, metric)
    if k_values and metric in k_values:
        param_str = ', '.join(map(str, k_values[metric]))
        # Use appropriate parameter name
        if metric == 'density_kl':
            title = f'{title} (avg over σ={{{param_str}}})'
        else:
            title = f'{title} (avg over k={{{param_str}}})'
    
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Log scale for latent dim if range is large
    if max(latent_dims) / min(latent_dims) > 8:
        ax.set_xscale('log', base=2)
        ax.set_xticks(latent_dims)
        ax.set_xticklabels(latent_dims)
    
    return ax


def plot_all_metrics(df, metrics=None, save_path=None, k_values=None):
    """Create a grid of metric comparison plots."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics to plot!")
        return None
    
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        plot_metric_comparison(df, metric, ax=axes[i], k_values=k_values)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    dataset = df['dataset'].iloc[0] if 'dataset' in df.columns else 'Unknown'
    fig.suptitle(f'Bottleneck Study Results - {dataset}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_heatmap(df, metric, save_path=None, k_values=None):
    """Create heatmap of metric values (models x latent dims)."""
    pivot = df.pivot(index='model', columns='latent_dim', values=metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Choose colormap based on whether lower is better
    cmap = 'RdYlGn' if metric not in LOWER_IS_BETTER else 'RdYlGn_r'
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax)
    
    # Build title with parameter values if applicable
    title = METRIC_NAMES.get(metric, metric)
    if k_values and metric in k_values:
        param_str = ', '.join(map(str, k_values[metric]))
        if metric == 'density_kl':
            title = f'{title} (avg over σ={{{param_str}}})'
        else:
            title = f'{title} (avg over k={{{param_str}}})'
    
    ax.set_title(f'{title} by Model and Latent Dimension')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    return fig


def print_summary_table(df, metrics=None, k_values=None):
    """Print a summary table of best results."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    print("\n" + "=" * 90)
    print("SUMMARY: Best model per metric (across all latent dimensions)")
    print("=" * 90)
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        if metric in LOWER_IS_BETTER:
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        
        best_row = df.loc[best_idx]
        std_col = f'{metric}_std'
        std_val = f" ± {best_row[std_col]:.4f}" if std_col in df.columns else ""
        
        # Build metric name with parameter info
        metric_display = METRIC_NAMES.get(metric, metric)
        if k_values and metric in k_values:
            param_str = ','.join(map(str, k_values[metric]))
            if metric == 'density_kl':
                metric_display = f"{metric_display} (σ={{{param_str}}})"
            else:
                metric_display = f"{metric_display} (k={{{param_str}}})"
        
        print(f"{metric_display:45s}: {best_row['model']:20s} "
              f"(dim={int(best_row['latent_dim']):2d}) = {best_row[metric]:.4f}{std_val}")
    
    print("=" * 90)


def print_ranking_table(df, metrics=None, k_values=None):
    """Print ranking table: for each metric and latent dim, show all models ranked."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    latent_dims = sorted(df['latent_dim'].unique())
    models = sorted(df['model'].unique())
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Build metric title
        title = METRIC_NAMES.get(metric, metric)
        if k_values and metric in k_values:
            param_str = ', '.join(map(str, k_values[metric]))
            if metric == 'density_kl':
                title = f"{title} (σ={{{param_str}}})"
            else:
                title = f"{title} (k={{{param_str}}})"
        
        lower_better = metric in LOWER_IS_BETTER
        direction = "↓ lower is better" if lower_better else "↑ higher is better"
        
        print(f"\n{'='*100}")
        print(f"{title}  [{direction}]")
        print(f"{'='*100}")
        
        # Header
        header = f"{'Rank':<6}"
        for dim in latent_dims:
            header += f" {'dim='+str(dim):>18}"
        print(header)
        print("-" * 100)
        
        # For each latent dim, rank models
        rankings = {}
        for dim in latent_dims:
            dim_data = df[df['latent_dim'] == dim][['model', metric]].copy()
            dim_data = dim_data.sort_values(metric, ascending=lower_better)
            rankings[dim] = dim_data.reset_index(drop=True)
        
        # Print rows by rank
        n_models = len(models)
        for rank in range(n_models):
            row = f"#{rank+1:<5}"
            for dim in latent_dims:
                if rank < len(rankings[dim]):
                    model = rankings[dim].iloc[rank]['model']
                    value = rankings[dim].iloc[rank][metric]
                    # Truncate model name if needed
                    model_short = model[:10] if len(model) > 10 else model
                    row += f" {model_short:>10}={value:>6.3f}"
                else:
                    row += f" {'-':>18}"
            print(row)
        
        # Print winner summary
        print("-" * 100)
        winners = []
        for dim in latent_dims:
            winner = rankings[dim].iloc[0]['model']
            winners.append(f"dim={dim}: {winner}")
        print(f"Winners: {', '.join(winners)}")


def print_model_comparison_table(df, metrics=None, k_values=None, latent_dim=None):
    """Print side-by-side comparison of all models for given latent dim(s)."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    if latent_dim is not None:
        latent_dims = [latent_dim] if isinstance(latent_dim, int) else latent_dim
    else:
        latent_dims = sorted(df['latent_dim'].unique())
    
    models = sorted(df['model'].unique())
    
    for dim in latent_dims:
        print(f"\n{'='*120}")
        print(f"MODEL COMPARISON - Latent Dimension = {dim}")
        print(f"{'='*120}")
        
        df_dim = df[df['latent_dim'] == dim]
        
        # Header with model names
        header = f"{'Metric':<30}"
        for model in models:
            header += f" {model:>14}"
        header += f" {'Best':>14}"
        print(header)
        print("-" * 120)
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            # Get metric display name
            name = METRIC_NAMES.get(metric, metric)
            if len(name) > 28:
                name = name[:28] + ".."
            
            lower_better = metric in LOWER_IS_BETTER
            
            # Get values for each model
            values = {}
            for model in models:
                model_row = df_dim[df_dim['model'] == model]
                if len(model_row) > 0 and metric in model_row.columns:
                    values[model] = model_row[metric].values[0]
                else:
                    values[model] = None
            
            # Find best
            valid_values = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}
            if valid_values:
                if lower_better:
                    best_model = min(valid_values, key=valid_values.get)
                else:
                    best_model = max(valid_values, key=valid_values.get)
            else:
                best_model = None
            
            # Build row
            row = f"{name:<30}"
            for model in models:
                val = values.get(model)
                if val is not None and not pd.isna(val):
                    # Mark best with asterisk
                    marker = "*" if model == best_model else " "
                    row += f" {val:>13.4f}{marker}"
                else:
                    row += f" {'-':>14}"
            
            # Add best model name
            row += f" {best_model if best_model else '-':>14}"
            print(row)
        
        print("-" * 120)
        print("* = best for this metric")


def print_wins_summary(df, metrics=None, k_values=None):
    """Print summary of how many times each model wins across all metrics and latent dims."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    latent_dims = sorted(df['latent_dim'].unique())
    models = sorted(df['model'].unique())
    
    # Count wins per model per latent dim
    wins = {model: {dim: 0 for dim in latent_dims} for model in models}
    total_wins = {model: 0 for model in models}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        lower_better = metric in LOWER_IS_BETTER
        
        for dim in latent_dims:
            df_dim = df[df['latent_dim'] == dim]
            if lower_better:
                best_idx = df_dim[metric].idxmin()
            else:
                best_idx = df_dim[metric].idxmax()
            
            if pd.notna(best_idx):
                winner = df.loc[best_idx, 'model']
                wins[winner][dim] += 1
                total_wins[winner] += 1
    
    print(f"\n{'='*100}")
    print(f"WINS SUMMARY: Number of metrics won by each model")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Model':<20}"
    for dim in latent_dims:
        header += f" {'dim='+str(dim):>10}"
    header += f" {'TOTAL':>10}"
    print(header)
    print("-" * 100)
    
    # Sort models by total wins
    sorted_models = sorted(models, key=lambda m: total_wins[m], reverse=True)
    
    for model in sorted_models:
        row = f"{model:<20}"
        for dim in latent_dims:
            row += f" {wins[model][dim]:>10}"
        row += f" {total_wins[model]:>10}"
        print(row)
    
    print("-" * 100)
    print(f"Total metrics evaluated: {len(metrics)}")


def print_full_table(df, metrics=None, k_values=None):
    """Print full results table for all models and latent dims."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    print("\n" + "=" * 120)
    print("FULL RESULTS TABLE")
    print("=" * 120)
    
    # Print parameter info
    if k_values:
        print("Aggregated metrics (averaged across parameter values):")
        for metric, params in k_values.items():
            if metric in metrics:
                param_str = ', '.join(map(str, params))
                if metric == 'density_kl':
                    print(f"  {metric}: σ ∈ {{{param_str}}}")
                else:
                    print(f"  {metric}: k ∈ {{{param_str}}}")
        print("-" * 120)
    
    # Header
    header = f"{'Model':<20} {'Dim':>4}"
    for m in metrics:
        header += f" {m[:12]:>12}"
    print(header)
    print("-" * 120)
    
    # Sort by model then latent_dim
    df_sorted = df.sort_values(['model', 'latent_dim'])
    
    for _, row in df_sorted.iterrows():
        line = f"{row['model']:<20} {int(row['latent_dim']):>4}"
        for m in metrics:
            if m in row and pd.notna(row[m]):
                line += f" {row[m]:>12.4f}"
            else:
                line += f" {'-':>12}"
        print(line)
    
    print("=" * 120)


def print_latex_table(df, metrics=None, latent_dim=2, k_values=None):
    """Print results as LaTeX table for a specific latent dimension."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    df_dim = df[df['latent_dim'] == latent_dim]
    
    print(f"\n% LaTeX table for latent_dim={latent_dim}")
    if k_values:
        for metric, params in k_values.items():
            if metric in metrics:
                param_str = ', '.join(map(str, params))
                if metric == 'density_kl':
                    print(f"% {metric} averaged over σ ∈ {{{param_str}}}")
                else:
                    print(f"% {metric} averaged over k ∈ {{{param_str}}}")
    
    print("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    print("\\toprule")
    
    # Header
    header_parts = []
    for m in metrics:
        name = METRIC_NAMES.get(m, m)
        # Shorten for table
        name = name.replace('Reconstruction ', 'Recon. ')
        name = name.replace('Distance ', 'Dist. ')
        name = name.replace('Accuracy', 'Acc.')
        header_parts.append(name)
    
    header = "Model & " + " & ".join(header_parts) + " \\\\"
    print(header)
    print("\\midrule")
    
    # Find best values for bolding
    best_vals = {}
    for m in metrics:
        if m in df_dim.columns:
            if m in LOWER_IS_BETTER:
                best_vals[m] = df_dim[m].min()
            else:
                best_vals[m] = df_dim[m].max()
    
    # Data rows
    for _, row in df_dim.iterrows():
        values = []
        for m in metrics:
            if m in row and pd.notna(row[m]):
                val_str = f"{row[m]:.3f}"
                # Bold if best
                if m in best_vals and abs(row[m] - best_vals[m]) < 1e-6:
                    val_str = f"\\textbf{{{val_str}}}"
                values.append(val_str)
            else:
                values.append("-")
        print(f"{row['model']} & " + " & ".join(values) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description='Visualize bottleneck study results')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory containing results.csv')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to results CSV file')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       help='Specific metrics to plot (use base names like "trustworthiness" for averaged)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: same as results)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--heatmap', type=str, default=None,
                       help='Create heatmap for specific metric')
    parser.add_argument('--latex', action='store_true',
                       help='Print LaTeX table')
    parser.add_argument('--latex_dim', type=int, default=2,
                       help='Latent dimension for LaTeX table')
    parser.add_argument('--full_table', action='store_true',
                       help='Print full results table')
    parser.add_argument('--ranking', action='store_true',
                       help='Print ranking table (models ranked per metric per latent dim)')
    parser.add_argument('--compare', action='store_true',
                       help='Print side-by-side model comparison table')
    parser.add_argument('--compare_dim', type=int, nargs='+', default=None,
                       help='Latent dimensions for comparison table (default: all)')
    parser.add_argument('--wins', action='store_true',
                       help='Print wins summary (count of metrics won per model)')
    args = parser.parse_args()
    
    # Determine results path
    if args.results_file:
        results_path = args.results_file
    elif args.results_dir:
        results_path = args.results_dir
    else:
        results_path = 'results/bottleneck_study'
    
    # Load results
    df = load_results(results_path)
    print(f"Loaded {len(df)} results")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Latent dims: {sorted(df['latent_dim'].unique().tolist())}")
    
    # Add aggregated metrics
    df, k_values_used = prepare_dataframe_with_aggregates(df)
    if k_values_used:
        print(f"Aggregated metrics across parameter values:")
        for metric, params in k_values_used.items():
            param_str = ', '.join(map(str, params))
            if metric == 'density_kl':
                print(f"  {metric}: σ ∈ {{{param_str}}}")
            else:
                print(f"  {metric}: k ∈ {{{param_str}}}")
    
    # Determine save directory
    if args.save_dir:
        save_dir = args.save_dir
    elif args.results_dir:
        save_dir = args.results_dir
    elif args.results_file:
        save_dir = os.path.dirname(args.results_file)
    else:
        save_dir = '.'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get available metrics
    available_metrics = get_available_metrics(df)
    print(f"Available metrics: {available_metrics}")
    
    # Filter to requested metrics
    if args.metrics:
        metrics = [m for m in args.metrics if m in available_metrics]
    else:
        metrics = [m for m in DEFAULT_METRICS if m in available_metrics]
    
    # Print summary
    print_summary_table(df, metrics, k_values_used)
    
    # Full table
    if args.full_table:
        print_full_table(df, metrics, k_values_used)
    
    # Ranking table
    if args.ranking:
        print_ranking_table(df, metrics, k_values_used)
    
    # Model comparison table
    if args.compare:
        print_model_comparison_table(df, metrics, k_values_used, args.compare_dim)
    
    # Wins summary
    if args.wins:
        print_wins_summary(df, metrics, k_values_used)
    
    # LaTeX table
    if args.latex:
        print_latex_table(df, metrics, args.latex_dim, k_values_used)
    
    # Create plots
    plot_all_metrics(df, metrics, save_path=os.path.join(save_dir, 'metrics_comparison.png'), k_values=k_values_used)
    
    # Heatmap for specific metric
    if args.heatmap and args.heatmap in available_metrics:
        plot_heatmap(df, args.heatmap, save_path=os.path.join(save_dir, f'heatmap_{args.heatmap}.png'), k_values=k_values_used)
    
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()