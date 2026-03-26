"""
Improved plotting script for bottleneck study results.
Handles different metric scales intelligently.

Usage:
    python plot_bottleneck_results.py results/bottleneck_study/<timestamp>/raw_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

COLORS = {
    'vanilla': '#1f77b4',
    'topoae': '#ff7f0e', 
    'rtdae': '#2ca02c',
    'mmae_pca2': '#d62728',
    'mmae_pca10': '#9467bd',
    'mmae_pca50': '#8c564b',
}
MARKERS = {
    'vanilla': 'o',
    'topoae': 's',
    'rtdae': '^',
    'mmae_pca2': 'D',
    'mmae_pca10': 'v',
    'mmae_pca50': 'p',
}
LABELS = {
    'vanilla': 'Vanilla AE',
    'topoae': 'TopoAE',
    'rtdae': 'RTD-AE',
    'mmae_pca2': 'MMAE (PCA=2)',
    'mmae_pca10': 'MMAE (PCA=10)',
    'mmae_pca50': 'MMAE (PCA=50)',
}

# Metric configurations
# (metric_name, display_name, higher_is_better, scale_type, group)
# scale_type: 'normal' (0-1), 'log', 'normalize' (min-max per latent dim)
METRICS_CONFIG = {
    # Reconstruction & Time
    'reconstruction_error': ('Reconstruction Error', False, 'log', 'reconstruction'),
    'train_time_seconds': ('Training Time (s)', False, 'normal', 'time'),
    
    # Distance preservation (0-1 or close)
    'distance_correlation': ('Distance Correlation', True, 'normal', 'distance'),
    'rmse': ('RMSE', False, 'normalize', 'distance'),
    
    # Neighborhood preservation (0-1)
    'trustworthiness_10': ('Trustworthiness (k=10)', True, 'normal', 'neighborhood'),
    'trustworthiness_50': ('Trustworthiness (k=50)', True, 'normal', 'neighborhood'),
    'trustworthiness_100': ('Trustworthiness (k=100)', True, 'normal', 'neighborhood'),
    'continuity_10': ('Continuity (k=10)', True, 'normal', 'neighborhood'),
    'continuity_50': ('Continuity (k=50)', True, 'normal', 'neighborhood'),
    'continuity_100': ('Continuity (k=100)', True, 'normal', 'neighborhood'),
    
    # Ranking (0-1)
    'triplet_accuracy': ('Triplet Accuracy', True, 'normal', 'ranking'),
    'mrre_xz': ('MRRE (X→Z)', False, 'normalize', 'ranking'),
    'mrre_zx': ('MRRE (Z→X)', False, 'normalize', 'ranking'),
    
    # Classification (0-1)
    'knn_accuracy_5': ('kNN Accuracy (k=5)', True, 'normal', 'classification'),
    'knn_accuracy_10': ('kNN Accuracy (k=10)', True, 'normal', 'classification'),
    
    # Density (can vary widely)
    'density_kl_0.01': ('Density KL (σ=0.01)', False, 'log', 'density'),
    'density_kl_0.1': ('Density KL (σ=0.1)', False, 'log', 'density'),
    'density_kl_1.0': ('Density KL (σ=1.0)', False, 'log', 'density'),
    'density_kl_10.0': ('Density KL (σ=10.0)', False, 'log', 'density'),
    
    # Topology (can vary widely)
    'wasserstein_h0': ('Wasserstein H0', False, 'log', 'topology'),
    'wasserstein_h1': ('Wasserstein H1', False, 'log', 'topology'),
}


def get_model_key(row):
    """Get a consistent model key for plotting."""
    if row['model'] == 'mmae':
        return f"mmae_pca{int(row['pca_components'])}"
    return row['model']


def normalize_per_latent_dim(df, metric):
    """Normalize metric values within each latent dimension (min-max)."""
    df = df.copy()
    for latent_dim in df['latent_dim'].unique():
        mask = df['latent_dim'] == latent_dim
        values = df.loc[mask, metric]
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            df.loc[mask, f'{metric}_norm'] = (values - min_val) / (max_val - min_val)
        else:
            df.loc[mask, f'{metric}_norm'] = 0.5
    return df, f'{metric}_norm'


def plot_metric(df, metric, ax, config, show_legend=False):
    """Plot a single metric with appropriate scaling."""
    display_name, higher_is_better, scale_type, _ = config
    
    df = df.copy()
    df['model_key'] = df.apply(get_model_key, axis=1)
    
    plot_metric_name = metric
    
    # Handle normalization
    if scale_type == 'normalize':
        df, plot_metric_name = normalize_per_latent_dim(df, metric)
        display_name = f'{display_name} (normalized)'
    
    for model_key in COLORS.keys():
        subset = df[df['model_key'] == model_key].sort_values('latent_dim')
        if len(subset) == 0 or plot_metric_name not in subset.columns:
            continue
        
        values = subset[plot_metric_name].values
        
        # Skip if all NaN
        if np.all(np.isnan(values)):
            continue
        
        ax.plot(subset['latent_dim'], values, 
                color=COLORS[model_key], 
                marker=MARKERS[model_key],
                label=LABELS[model_key],
                linewidth=2,
                markersize=7,
                alpha=0.9)
    
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel(display_name)
    ax.set_title(display_name)
    ax.set_xticks(df['latent_dim'].unique())
    
    # Apply log scale if needed
    if scale_type == 'log':
        ax.set_yscale('log')
    
    # Add direction indicator
    direction = '↑' if higher_is_better else '↓'
    ax.annotate(f'{direction} better', xy=(0.98, 0.98), xycoords='axes fraction',
               fontsize=9, ha='right', va='top', color='gray',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    if show_legend:
        ax.legend(loc='best', fontsize=8)


def plot_grouped_metrics(df, save_dir=None):
    """Create plots grouped by metric type for better comparison."""
    
    df = df.copy()
    df['model_key'] = df.apply(get_model_key, axis=1)
    
    # Get available metrics
    available_metrics = {k: v for k, v in METRICS_CONFIG.items() if k in df.columns}
    
    # Group metrics
    groups = {}
    for metric, config in available_metrics.items():
        group = config[3]
        if group not in groups:
            groups[group] = []
        groups[group].append((metric, config))
    
    # === Figure 1: Normalized comparison (all metrics 0-1 or normalized) ===
    print("Creating normalized comparison plot...")
    
    # Select key metrics for normalized view
    key_metrics = [
        'distance_correlation', 'triplet_accuracy', 
        'trustworthiness_10', 'continuity_10',
        'knn_accuracy_5'
    ]
    key_metrics = [m for m in key_metrics if m in df.columns]
    
    if key_metrics:
        fig1, axes1 = plt.subplots(1, len(key_metrics), figsize=(4*len(key_metrics), 4))
        if len(key_metrics) == 1:
            axes1 = [axes1]
        
        for i, metric in enumerate(key_metrics):
            config = available_metrics[metric]
            plot_metric(df, metric, axes1[i], config, show_legend=(i == 0))
        
        # Shared legend at bottom
        handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], 
                         label=LABELS[m], linewidth=2, markersize=7)
                  for m in COLORS.keys()]
        fig1.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
                   bbox_to_anchor=(0.5, -0.08))
        
        plt.suptitle('Key Metrics Comparison (0-1 scale)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_dir:
            fig1.savefig(os.path.join(save_dir, 'key_metrics_comparison.png'), 
                        dpi=150, bbox_inches='tight')
    
    # === Figure 2: Reconstruction & Loss metrics (log scale) ===
    print("Creating reconstruction metrics plot...")
    
    recon_metrics = ['reconstruction_error']
    density_metrics = [m for m in df.columns if 'density_kl' in m]
    topo_metrics = [m for m in ['wasserstein_h0', 'wasserstein_h1'] if m in df.columns]
    
    log_metrics = recon_metrics + density_metrics + topo_metrics
    log_metrics = [m for m in log_metrics if m in df.columns]
    
    if log_metrics:
        n_cols = min(4, len(log_metrics))
        n_rows = (len(log_metrics) + n_cols - 1) // n_cols
        
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes2 = np.atleast_1d(axes2).flatten()
        
        for i, metric in enumerate(log_metrics):
            config = available_metrics.get(metric, (metric, False, 'log', 'other'))
            plot_metric(df, metric, axes2[i], config)
        
        for i in range(len(log_metrics), len(axes2)):
            axes2[i].set_visible(False)
        
        handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], 
                         label=LABELS[m], linewidth=2, markersize=7)
                  for m in COLORS.keys()]
        fig2.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))
        
        plt.suptitle('Reconstruction, Density & Topology Metrics (log scale)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_dir:
            fig2.savefig(os.path.join(save_dir, 'log_scale_metrics.png'), 
                        dpi=150, bbox_inches='tight')
    
    # === Figure 3: Neighborhood metrics ===
    print("Creating neighborhood metrics plot...")
    
    neighborhood_metrics = [m for m in df.columns if 'trustworthiness' in m or 'continuity' in m]
    
    if neighborhood_metrics:
        n_cols = min(3, len(neighborhood_metrics))
        n_rows = (len(neighborhood_metrics) + n_cols - 1) // n_cols
        
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes3 = np.atleast_1d(axes3).flatten()
        
        for i, metric in enumerate(neighborhood_metrics):
            config = available_metrics.get(metric, (metric, True, 'normal', 'neighborhood'))
            plot_metric(df, metric, axes3[i], config)
        
        for i in range(len(neighborhood_metrics), len(axes3)):
            axes3[i].set_visible(False)
        
        handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], 
                         label=LABELS[m], linewidth=2, markersize=7)
                  for m in COLORS.keys()]
        fig3.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))
        
        plt.suptitle('Neighborhood Preservation Metrics', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_dir:
            fig3.savefig(os.path.join(save_dir, 'neighborhood_metrics.png'), 
                        dpi=150, bbox_inches='tight')
    
    # === Figure 4: Relative performance (normalized per latent dim) ===
    print("Creating relative performance plot...")
    
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
    axes4 = axes4.flatten()
    
    relative_metrics = ['reconstruction_error', 'distance_correlation', 'triplet_accuracy',
                       'trustworthiness_10', 'continuity_10', 'knn_accuracy_5']
    relative_metrics = [m for m in relative_metrics if m in df.columns]
    
    for i, metric in enumerate(relative_metrics):
        if i >= len(axes4):
            break
        
        ax = axes4[i]
        original_config = available_metrics.get(metric, (metric, True, 'normal', 'other'))
        display_name, higher_is_better, _, group = original_config
        
        # Normalize within each latent dim for fair comparison
        df_norm, norm_metric = normalize_per_latent_dim(df, metric)
        
        for model_key in COLORS.keys():
            subset = df_norm[df_norm['model_key'] == model_key].sort_values('latent_dim')
            if len(subset) == 0:
                continue
            
            values = subset[norm_metric].values
            if not higher_is_better:
                values = 1 - values  # Flip so higher is always better
            
            ax.plot(subset['latent_dim'], values,
                   color=COLORS[model_key], marker=MARKERS[model_key],
                   label=LABELS[model_key], linewidth=2, markersize=7)
        
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Relative Performance')
        ax.set_title(f'{display_name}\n(normalized, higher=better)')
        ax.set_xticks(df['latent_dim'].unique())
        ax.set_ylim(-0.05, 1.05)
    
    for i in range(len(relative_metrics), len(axes4)):
        axes4[i].set_visible(False)
    
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], 
                     label=LABELS[m], linewidth=2, markersize=7)
              for m in COLORS.keys()]
    fig4.legend(handles=handles, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Relative Performance (normalized per latent dim, higher=better)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        fig4.savefig(os.path.join(save_dir, 'relative_performance.png'), 
                    dpi=150, bbox_inches='tight')
    
    # === Figure 5: MMAE PCA comparison ===
    print("Creating MMAE comparison plot...")
    
    mmae_df = df[df['model'] == 'mmae'].copy()
    
    if len(mmae_df) > 0:
        mmae_metrics = ['reconstruction_error', 'distance_correlation', 'triplet_accuracy',
                       'trustworthiness_10', 'continuity_10', 'knn_accuracy_5']
        mmae_metrics = [m for m in mmae_metrics if m in mmae_df.columns]
        
        fig5, axes5 = plt.subplots(2, 3, figsize=(14, 9))
        axes5 = axes5.flatten()
        
        pca_colors = {2: '#d62728', 10: '#9467bd', 50: '#8c564b'}
        pca_markers = {2: 'D', 10: 'v', 50: 'p'}
        
        for i, metric in enumerate(mmae_metrics):
            if i >= len(axes5):
                break
            ax = axes5[i]
            config = available_metrics.get(metric, (metric, True, 'normal', 'other'))
            display_name, higher_is_better, scale_type, _ = config
            
            for pca in [2, 10, 50]:
                subset = mmae_df[mmae_df['pca_components'] == pca].sort_values('latent_dim')
                if len(subset) == 0:
                    continue
                ax.plot(subset['latent_dim'], subset[metric],
                       marker=pca_markers[pca], label=f'PCA={pca}', 
                       color=pca_colors[pca], linewidth=2, markersize=8)
            
            ax.set_xlabel('Latent Dimension')
            ax.set_ylabel(display_name)
            ax.set_title(display_name)
            ax.legend(fontsize=9)
            ax.set_xticks(mmae_df['latent_dim'].unique())
            
            if scale_type == 'log':
                ax.set_yscale('log')
            
            direction = '↑' if higher_is_better else '↓'
            ax.annotate(f'{direction} better', xy=(0.98, 0.98), xycoords='axes fraction',
                       fontsize=9, ha='right', va='top', color='gray',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        for i in range(len(mmae_metrics), len(axes5)):
            axes5[i].set_visible(False)
        
        plt.suptitle('MMAE: Effect of PCA Components', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_dir:
            fig5.savefig(os.path.join(save_dir, 'mmae_pca_analysis.png'), 
                        dpi=150, bbox_inches='tight')
    
    # === Figure 6: Training time ===
    print("Creating training time plot...")
    
    if 'train_time_seconds' in df.columns:
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        
        for model_key in COLORS.keys():
            subset = df[df['model_key'] == model_key].sort_values('latent_dim')
            if len(subset) == 0:
                continue
            ax6.plot(subset['latent_dim'], subset['train_time_seconds'],
                    color=COLORS[model_key], marker=MARKERS[model_key],
                    label=LABELS[model_key], linewidth=2, markersize=8)
        
        ax6.set_xlabel('Latent Dimension')
        ax6.set_ylabel('Training Time (seconds)')
        ax6.set_title('Training Time vs Latent Dimension')
        ax6.set_xticks(df['latent_dim'].unique())
        ax6.legend(loc='best')
        ax6.annotate('↓ better', xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=10, ha='right', va='top', color='gray')
        
        plt.tight_layout()
        
        if save_dir:
            fig6.savefig(os.path.join(save_dir, 'training_time.png'), 
                        dpi=150, bbox_inches='tight')
    
    # === Figure 7: Bar comparison at each latent dim ===
    print("Creating bar comparison plot...")
    
    latent_dims = sorted(df['latent_dim'].unique())
    metric_for_bars = 'triplet_accuracy' if 'triplet_accuracy' in df.columns else 'distance_correlation'
    
    fig7, axes7 = plt.subplots(1, len(latent_dims), figsize=(4*len(latent_dims), 5))
    if len(latent_dims) == 1:
        axes7 = [axes7]
    
    for i, latent_dim in enumerate(latent_dims):
        ax = axes7[i]
        subset = df[df['latent_dim'] == latent_dim].sort_values(metric_for_bars, ascending=False)
        
        colors = [COLORS.get(m, 'gray') for m in subset['model_key']]
        bars = ax.bar(range(len(subset)), subset[metric_for_bars], color=colors, alpha=0.8)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, subset[metric_for_bars])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels([LABELS.get(m, m) for m in subset['model_key']], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_title(f'd = {latent_dim}')
        ax.set_ylabel(metric_for_bars.replace('_', ' ').title())
        ax.set_ylim(0, min(1.15, subset[metric_for_bars].max() * 1.15))
    
    plt.suptitle(f'Model Comparison: {metric_for_bars.replace("_", " ").title()}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        fig7.savefig(os.path.join(save_dir, 'bar_comparison.png'), 
                    dpi=150, bbox_inches='tight')
    
    # === Figure 8: Overall ranking ===
    print("Creating overall ranking plot...")
    
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    
    # Compute average rank
    rank_metrics = ['reconstruction_error', 'distance_correlation', 'triplet_accuracy',
                   'trustworthiness_10', 'continuity_10', 'knn_accuracy_5']
    rank_metrics = [m for m in rank_metrics if m in df.columns]
    
    higher_is_better_dict = {
        'reconstruction_error': False,
        'distance_correlation': True,
        'triplet_accuracy': True,
        'trustworthiness_10': True,
        'continuity_10': True,
        'knn_accuracy_5': True,
    }
    
    model_keys = list(COLORS.keys())
    model_ranks = {m: [] for m in model_keys}
    
    for latent_dim in latent_dims:
        for metric in rank_metrics:
            subset = df[df['latent_dim'] == latent_dim].copy()
            if len(subset) == 0 or metric not in subset.columns:
                continue
            
            ascending = not higher_is_better_dict.get(metric, True)
            subset['rank'] = subset[metric].rank(ascending=ascending)
            
            for model_key in model_keys:
                model_subset = subset[subset['model_key'] == model_key]
                if len(model_subset) > 0:
                    model_ranks[model_key].append(model_subset['rank'].values[0])
    
    avg_ranks = {m: np.mean(ranks) if ranks else np.nan for m, ranks in model_ranks.items()}
    
    # Sort by rank
    sorted_models = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
    sorted_ranks = [avg_ranks[m] for m in sorted_models]
    sorted_colors = [COLORS[m] for m in sorted_models]
    sorted_labels = [LABELS[m] for m in sorted_models]
    
    bars = ax8.barh(range(len(sorted_models)), sorted_ranks, color=sorted_colors, alpha=0.8)
    ax8.set_yticks(range(len(sorted_models)))
    ax8.set_yticklabels(sorted_labels)
    ax8.set_xlabel('Average Rank (lower is better)')
    ax8.set_title('Overall Model Ranking\n(averaged across all metrics and latent dimensions)')
    ax8.invert_yaxis()
    
    for i, (bar, rank) in enumerate(zip(bars, sorted_ranks)):
        ax8.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        fig8.savefig(os.path.join(save_dir, 'overall_ranking.png'), 
                    dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print(f"\nAll plots saved to: {save_dir}")


def print_summary_table(df):
    """Print a formatted summary table."""
    df = df.copy()
    df['model_key'] = df.apply(get_model_key, axis=1)
    
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    
    # Select columns
    cols = ['model_key', 'latent_dim']
    metric_cols = ['reconstruction_error', 'distance_correlation', 'triplet_accuracy',
                  'trustworthiness_10', 'continuity_10', 'knn_accuracy_5', 'train_time_seconds']
    cols.extend([c for c in metric_cols if c in df.columns])
    
    # Format and print
    display_df = df[cols].copy()
    for col in metric_cols:
        if col in display_df.columns:
            if col == 'train_time_seconds':
                display_df[col] = display_df[col].apply(lambda x: f'{x:.1f}' if pd.notna(x) else '-')
            else:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
    
    # Rename columns for display
    col_names = {
        'model_key': 'Model',
        'latent_dim': 'Dim',
        'reconstruction_error': 'Recon↓',
        'distance_correlation': 'DistCorr↑',
        'triplet_accuracy': 'Triplet↑',
        'trustworthiness_10': 'Trust@10↑',
        'continuity_10': 'Cont@10↑',
        'knn_accuracy_5': 'kNN@5↑',
        'train_time_seconds': 'Time(s)↓'
    }
    display_df = display_df.rename(columns=col_names)
    
    print(display_df.to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_bottleneck_results.py <path_to_raw_results.csv>")
        print("\nLooking for most recent results...")
        
        results_dir = 'results/bottleneck_study'
        if os.path.exists(results_dir):
            subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            if subdirs:
                latest = sorted(subdirs)[-1]
                csv_path = os.path.join(results_dir, latest, 'raw_results.csv')
                if os.path.exists(csv_path):
                    print(f"Found: {csv_path}")
                else:
                    print("No raw_results.csv found")
                    sys.exit(1)
            else:
                print("No result directories found")
                sys.exit(1)
        else:
            print(f"Results directory not found: {results_dir}")
            sys.exit(1)
    else:
        csv_path = sys.argv[1]
    
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} results")
    print(f"Models: {df['model'].unique()}")
    print(f"Latent dims: {sorted(df['latent_dim'].unique())}")
    print(f"Available metrics: {[c for c in df.columns if c not in ['model', 'latent_dim', 'pca_components', 'display_name', 'model_key']]}")
    
    print_summary_table(df)
    
    save_dir = os.path.dirname(csv_path)
    plot_grouped_metrics(df, save_dir=save_dir)