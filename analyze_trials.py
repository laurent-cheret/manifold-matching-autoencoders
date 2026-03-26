#!/usr/bin/env python
"""
Quick analysis of hyperparameter search results.
Focuses on understanding how PCA components affect different metrics.

Usage:
    python analyze_trials.py --results_dir experiments/hyperparam_search/spheres/results/mmae_dim2
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_trials(csv_path, output_dir=None):
    """Analyze trials.csv and show PCA component effects."""
    df = pd.read_csv(csv_path)
    
    if 'mmae_n_components' not in df.columns:
        print("No mmae_n_components column found!")
        return
    
    # Filter out failed trials
    df = df[~df.isnull().any(axis=1)]
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH ANALYSIS")
    print(f"{'='*80}")
    print(f"Total trials: {len(df)}")
    print(f"PCA components range: [{df['mmae_n_components'].min():.0f}, {df['mmae_n_components'].max():.0f}]")
    
    # Metrics to analyze - prioritize wasserstein_H0
    metrics = [
        'wasserstein_H0', 'wasserstein_H1',
        'density_kl_0_1', 'reconstruction_error', 
        'trustworthiness_10', 'continuity_10',
        'distance_correlation', 'triplet_accuracy'
    ]
    metrics = [m for m in metrics if m in df.columns]
    
    # Compute correlation with PCA components
    print(f"\n{'='*80}")
    print("CORRELATION WITH PCA COMPONENTS")
    print(f"{'='*80}")
    correlations = {}
    for metric in metrics:
        corr = df[['mmae_n_components', metric]].corr().iloc[0, 1]
        correlations[metric] = corr
        direction = "↓ better" if metric in ['reconstruction_error', 'wasserstein_H0', 'wasserstein_H1', 'density_kl_0_1'] else "↑ better"
        print(f"{metric:30s}: {corr:+.3f}  {direction}")
    
    # Show best trials for different PCA ranges
    print(f"\n{'='*80}")
    print("BEST TRIALS BY PCA RANGE")
    print(f"{'='*80}")
    
    pca_min, pca_max = df['mmae_n_components'].min(), df['mmae_n_components'].max()
    ranges = [
        (pca_min, pca_min + (pca_max - pca_min) * 0.33, "Low"),
        (pca_min + (pca_max - pca_min) * 0.33, pca_min + (pca_max - pca_min) * 0.67, "Mid"),
        (pca_min + (pca_max - pca_min) * 0.67, pca_max, "High")
    ]
    
    for low, high, label in ranges:
        subset = df[(df['mmae_n_components'] >= low) & (df['mmae_n_components'] <= high)]
        if len(subset) > 0:
            best_idx = subset['density_kl_0_1'].idxmin()
            best = subset.loc[best_idx]
            print(f"\n{label} PCA [{low:.0f}, {high:.0f}]:")
            print(f"  Best by density_kl: trial={best['trial']:.0f}, n_comp={best['mmae_n_components']:.0f}")
            for m in ['density_kl_0_1', 'wasserstein_H0', 'reconstruction_error', 'distance_correlation']:
                if m in best:
                    print(f"    {m}: {best[m]:.4f}")
            
            # Also show best by wasserstein_H0 in this range
            if 'wasserstein_H0' in subset.columns:
                best_wass_idx = subset['wasserstein_H0'].idxmin()
                best_wass = subset.loc[best_wass_idx]
                if best_wass_idx != best_idx:
                    print(f"  Best by wasserstein_H0: trial={best_wass['trial']:.0f}, n_comp={best_wass['mmae_n_components']:.0f}")
                    print(f"    wasserstein_H0: {best_wass['wasserstein_H0']:.4f}")
                    print(f"    density_kl_0_1: {best_wass['density_kl_0_1']:.4f}")
    
    # Find overall best by different metrics
    print(f"\n{'='*80}")
    print(f"OVERALL BEST TRIALS")
    print(f"{'='*80}")
    
    # Best by density_kl
    best_idx = df['density_kl_0_1'].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest by density_kl_0_1:")
    print(f"  Trial: {best['trial']:.0f}, PCA: {best['mmae_n_components']:.0f}, Lambda: {best['mmae_lambda']:.4f}")
    print(f"  density_kl_0_1: {best['density_kl_0_1']:.4f}")
    if 'wasserstein_H0' in best:
        print(f"  wasserstein_H0: {best['wasserstein_H0']:.4f}")
    
    # Best by wasserstein_H0
    if 'wasserstein_H0' in df.columns:
        best_wass_idx = df['wasserstein_H0'].idxmin()
        best_wass = df.loc[best_wass_idx]
        print(f"\nBest by wasserstein_H0:")
        print(f"  Trial: {best_wass['trial']:.0f}, PCA: {best_wass['mmae_n_components']:.0f}, Lambda: {best_wass['mmae_lambda']:.4f}")
        print(f"  wasserstein_H0: {best_wass['wasserstein_H0']:.4f}")
        print(f"  density_kl_0_1: {best_wass['density_kl_0_1']:.4f}")
        
        if best_wass_idx != best_idx:
            print(f"\n  ⚠️  Different trials optimize different metrics!")
            print(f"     PCA difference: {abs(best_wass['mmae_n_components'] - best['mmae_n_components']):.0f} components")
    
    # Create plots
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot 1: Key metrics vs PCA components (larger, focused plot)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        key_metrics = ['wasserstein_H0', 'density_kl_0_1', 'reconstruction_error', 'distance_correlation']
        key_metrics = [m for m in key_metrics if m in df.columns]
        
        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            scatter = ax.scatter(df['mmae_n_components'], df[metric], 
                               c=df['mmae_lambda'], cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('PCA Components', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} (r={correlations[metric]:+.2f})', fontsize=12, fontweight='bold')
            
            # Add trend line
            z = np.polyfit(df['mmae_n_components'], df[metric], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['mmae_n_components'].min(), df['mmae_n_components'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
            
            # Mark best point
            if metric in ['reconstruction_error', 'wasserstein_H0', 'density_kl_0_1']:
                best_val_idx = df[metric].idxmin()
            else:
                best_val_idx = df[metric].idxmax()
            ax.scatter(df.loc[best_val_idx, 'mmae_n_components'], 
                      df.loc[best_val_idx, metric], 
                      color='red', s=200, marker='*', edgecolors='black', linewidths=1.5,
                      label='Best', zorder=5)
            ax.legend()
            
            # Colorbar for lambda
            if i == len(key_metrics) - 1:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Lambda', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'key_metrics_vs_pca.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir / 'key_metrics_vs_pca.png'}")
        plt.close()
        
        # Plot 2: All metrics vs PCA components
        n_metrics = len(metrics)
        n_rows = (n_metrics + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.scatter(df['mmae_n_components'], df[metric], alpha=0.6)
            ax.set_xlabel('PCA Components')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} (r={correlations[metric]:+.2f})')
            
            # Add trend line
            z = np.polyfit(df['mmae_n_components'], df[metric], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['mmae_n_components'].min(), df['mmae_n_components'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5)
        
        # Hide extra subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'all_metrics_vs_pca.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'all_metrics_vs_pca.png'}")
        plt.close()
        
        # Plot 3: Wasserstein H0 vs H1 colored by PCA
        if 'wasserstein_H0' in df.columns and 'wasserstein_H1' in df.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['wasserstein_H0'], df['wasserstein_H1'], 
                                c=df['mmae_n_components'], cmap='RdYlGn', 
                                s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
            plt.xlabel('Wasserstein H0', fontsize=12)
            plt.ylabel('Wasserstein H1', fontsize=12)
            plt.title('Topology Preservation: H0 vs H1 (colored by PCA components)', fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label('PCA Components', fontsize=11)
            
            # Mark best H0 point
            best_h0_idx = df['wasserstein_H0'].idxmin()
            plt.scatter(df.loc[best_h0_idx, 'wasserstein_H0'], 
                       df.loc[best_h0_idx, 'wasserstein_H1'],
                       color='red', s=300, marker='*', edgecolors='black', linewidths=2,
                       label=f'Best H0 (PCA={df.loc[best_h0_idx, "mmae_n_components"]:.0f})', zorder=5)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'wasserstein_h0_vs_h1.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir / 'wasserstein_h0_vs_h1.png'}")
            plt.close()
    
    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    
    # Find metrics with strong PCA correlation
    strong_corr = [(m, c) for m, c in correlations.items() if abs(c) > 0.3]
    
    if strong_corr:
        print(f"\nMetrics strongly affected by PCA components (|r| > 0.3):")
        for m, c in sorted(strong_corr, key=lambda x: abs(x[1]), reverse=True):
            direction = "→ Higher PCA = worse" if c > 0 else "→ Higher PCA = better"
            print(f"  {m}: {c:+.3f}  {direction}")
    
    # Special analysis for wasserstein_H0
    if 'wasserstein_H0' in correlations:
        wass_corr = correlations['wasserstein_H0']
        print(f"\n📊 Wasserstein H0 correlation: {wass_corr:+.3f}")
        if wass_corr > 0.3:
            print("   → Topology preservation DEGRADES with more PCA components")
            print("   → Consider using wasserstein_H0 as optimization target")
        elif wass_corr < -0.3:
            print("   → Topology preservation IMPROVES with more PCA components")
            print("   → Current density_kl_0_1 target may be fine")
        else:
            print("   → Wasserstein H0 is relatively stable across PCA range")
    
    # Check if objectives conflict
    if 'wasserstein_H0' in df.columns:
        best_density_idx = df['density_kl_0_1'].idxmin()
        best_wass_idx = df['wasserstein_H0'].idxmin()
        
        if best_density_idx != best_wass_idx:
            pca_diff = abs(df.loc[best_wass_idx, 'mmae_n_components'] - 
                          df.loc[best_density_idx, 'mmae_n_components'])
            print(f"\n⚠️  CONFLICTING OBJECTIVES DETECTED!")
            print(f"   Best density_kl_0_1 uses PCA={df.loc[best_density_idx, 'mmae_n_components']:.0f}")
            print(f"   Best wasserstein_H0 uses PCA={df.loc[best_wass_idx, 'mmae_n_components']:.0f}")
            print(f"   Difference: {pca_diff:.0f} components")
            if pca_diff > 5:
                print(f"\n   💡 Recommendation: Switch to wasserstein_H0 as optimization target")
                print(f"      for better topology preservation")
    
    return df, correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing trials.csv')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    args = parser.parse_args()
    
    csv_path = Path(args.results_dir) / 'trials.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    output_dir = Path(args.results_dir) if not args.no_plots else None
    analyze_trials(csv_path, output_dir)


if __name__ == '__main__':
    main()