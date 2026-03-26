"""
Aggregate bottleneck study results from multiple experiment runs.

Averages metrics across all timestamped folders in bottleneck_study directory.
Creates comprehensive tables with averaged metrics including:
- Reconstruction error
- Distance correlation  
- Trustworthiness (averaged across k=10,50,100)
- Continuity (averaged across k=10,50,100)
- MRRE (averaged xz and zx)
- kNN accuracy (averaged across k=5,10)
- Triplet accuracy
- Training time

Usage:
    python aggregate_bottleneck_results.py --base-dir /content/drive/MyDrive/TOPO_COMPARE/results/bottleneck_study
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse


def load_all_results(base_dir):
    """Load raw_results.csv from all timestamped folders."""
    base_path = Path(base_dir)
    all_dfs = []
    
    if not base_path.exists():
        print(f"Error: Directory not found: {base_dir}")
        return None
    
    # Find all timestamped folders
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"No folders found in {base_dir}")
        return None
    
    print(f"Found {len(folders)} folders:")
    for folder in sorted(folders):
        csv_path = folder / 'raw_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            print(f"  ✓ {folder.name}: {len(df)} rows")
        else:
            print(f"  ✗ {folder.name}: no raw_results.csv")
    
    if not all_dfs:
        print("No raw_results.csv files found!")
        return None
    
    return pd.concat(all_dfs, ignore_index=True)


def aggregate_metrics(df):
    """
    Aggregate metrics by averaging:
    - Trustworthiness across k=10,50,100
    - Continuity across k=10,50,100  
    - MRRE across xz and zx
    - kNN accuracy across k=5,10
    """
    result_rows = []
    
    # Group by model, latent_dim, AND pca_components to keep MMAE variants separate
    group_cols = ['model', 'latent_dim']
    if 'pca_components' in df.columns:
        group_cols.append('pca_components')
    
    for group_key, group in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 3:
            model, latent_dim, pca_comp = group_key
        else:
            model, latent_dim = group_key
            pca_comp = None
        
        row = {
            'model': model,
            'latent_dim': latent_dim,
        }
        
        # Get model display info
        if 'display_name' in group.columns:
            row['display_name'] = group['display_name'].iloc[0]
        if pca_comp is not None and pd.notna(pca_comp):
            row['pca_components'] = pca_comp
        
        # Single-value metrics - just average
        single_metrics = [
            'reconstruction_error',
            'distance_correlation', 
            'triplet_accuracy',
            'train_time_seconds',
            'wasserstein_H0',  # Note: capital H in CSV
            'wasserstein_H1',
        ]
        
        for metric in single_metrics:
            if metric in group.columns:
                row[metric] = group[metric].mean()
                row[f'{metric}_std'] = group[metric].std()
        
        # Trustworthiness - average across different k values (trust_k10, trust_k50, trust_k100)
        trust_cols = [c for c in group.columns if c.startswith('trust_k')]
        if trust_cols:
            trust_values = group[trust_cols].values.flatten()
            trust_values = trust_values[~np.isnan(trust_values)]
            if len(trust_values) > 0:
                row['trustworthiness_avg'] = trust_values.mean()
                row['trustworthiness_std'] = trust_values.std()
        
        # Continuity - average across different k values (cont_k10, cont_k50, cont_k100)
        cont_cols = [c for c in group.columns if c.startswith('cont_k')]
        if cont_cols:
            cont_values = group[cont_cols].values.flatten()
            cont_values = cont_values[~np.isnan(cont_values)]
            if len(cont_values) > 0:
                row['continuity_avg'] = cont_values.mean()
                row['continuity_std'] = cont_values.std()
        
        # MRRE - average xz and zx across all k values
        mrre_cols = [c for c in group.columns if c.startswith('mrre_')]
        if mrre_cols:
            mrre_values = group[mrre_cols].values.flatten()
            mrre_values = mrre_values[~np.isnan(mrre_values)]
            if len(mrre_values) > 0:
                row['mrre_avg'] = mrre_values.mean()
                row['mrre_std'] = mrre_values.std()
        
        # kNN accuracy - average across different k values (knn_k5, knn_k10)
        knn_cols = [c for c in group.columns if c.startswith('knn_k') and not c.endswith('_std')]
        if knn_cols:
            knn_values = group[knn_cols].values.flatten()
            knn_values = knn_values[~np.isnan(knn_values)]
            if len(knn_values) > 0:
                row['knn_accuracy_avg'] = knn_values.mean()
                row['knn_accuracy_std'] = knn_values.std()
        
        # Density KL - average across different sigma values
        density_cols = [c for c in group.columns if c.startswith('density_kl_')]
        if density_cols:
            density_values = group[density_cols].values.flatten()
            density_values = density_values[~np.isnan(density_values)]
            if len(density_values) > 0:
                row['density_kl_avg'] = density_values.mean()
                row['density_kl_std'] = density_values.std()
        
        # Count number of runs
        row['n_runs'] = len(group)
        
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)


def print_comprehensive_table(df):
    """Print comprehensive results table."""
    
    # Define metrics to display
    metrics = [
        ('reconstruction_error', 'Recon Error ↓', '.4f'),
        ('distance_correlation', 'Dist Corr ↑', '.4f'),
        ('trustworthiness_avg', 'Trust (avg) ↑', '.4f'),
        ('continuity_avg', 'Cont (avg) ↑', '.4f'),
        ('mrre_avg', 'MRRE (avg) ↓', '.4f'),
        ('knn_accuracy_avg', 'kNN (avg) ↑', '.4f'),
        ('triplet_accuracy', 'Triplet ↑', '.4f'),
        ('density_kl_avg', 'Dens KL (avg) ↓', '.4f'),
        ('wasserstein_H0', 'Wass H0 ↓', '.4f'),
        ('wasserstein_H1', 'Wass H1 ↓', '.4f'),
        ('train_time_seconds', 'Time (s) ↓', '.1f'),
        ('n_runs', 'N', '.0f'),
    ]
    
    print("\n" + "="*150)
    print("AGGREGATED BOTTLENECK STUDY RESULTS")
    print("="*150)
    
    for latent_dim in sorted(df['latent_dim'].unique()):
        print(f"\n{'─'*150}")
        print(f"LATENT DIM = {latent_dim}")
        print(f"{'─'*150}")
        
        subset = df[df['latent_dim'] == latent_dim].copy()
        
        # Sort by model, then by pca_components for consistent display
        if 'pca_components' in subset.columns:
            subset['sort_key'] = subset.apply(
                lambda x: (x['model'], x.get('pca_components', 0) if pd.notna(x.get('pca_components')) else 0), 
                axis=1
            )
        else:
            subset['sort_key'] = subset['model']
        subset = subset.sort_values('sort_key')
        
        # Build display table
        display_data = []
        for _, row in subset.iterrows():
            model_name = row.get('display_name', row['model'])
            if 'pca_components' in row and pd.notna(row['pca_components']):
                model_name = f"{model_name} (PCA={int(row['pca_components'])})"
            
            row_data = [model_name]
            
            for metric_col, _, fmt in metrics:
                if metric_col in row and pd.notna(row[metric_col]):
                    val = row[metric_col]
                    # Add std if available
                    std_col = f'{metric_col}_std'
                    if std_col in row and pd.notna(row[std_col]) and metric_col != 'n_runs':
                        row_data.append(f"{val:{fmt}} ± {row[std_col]:.4f}")
                    else:
                        row_data.append(f"{val:{fmt}}")
                else:
                    row_data.append("-")
            
            display_data.append(row_data)
        
        # Print header
        header = ["Model"] + [m[1] for m in metrics]
        col_widths = [max(len(h), 25) for h in header]
        
        # Update widths based on data
        for row_data in display_data:
            for i, val in enumerate(row_data):
                col_widths[i] = max(col_widths[i], len(str(val)))
        
        # Print table
        header_str = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
        print(header_str)
        print("─" * len(header_str))
        
        for row_data in display_data:
            row_str = " | ".join(str(v).ljust(w) for v, w in zip(row_data, col_widths))
            print(row_str)
    
    print("\n" + "="*150)
    print("Legend: ↑ = higher is better, ↓ = lower is better")
    print("Averaged metrics: Trust (k=10,50,100), Cont (k=10,50,100), MRRE (xz,zx), kNN (k=5,10), Dens KL (σ=0.01,0.1,1.0,10.0)")
    print("="*150)


def save_latex_table(df, output_path):
    """Save results as LaTeX table."""
    
    metrics = [
        ('reconstruction_error', 'Recon ↓'),
        ('distance_correlation', 'Dist Corr ↑'),
        ('trustworthiness_avg', 'Trust ↑'),
        ('continuity_avg', 'Cont ↑'),
        ('mrre_avg', 'MRRE ↓'),
        ('knn_accuracy_avg', 'kNN ↑'),
        ('triplet_accuracy', 'Triplet ↑'),
        ('train_time_seconds', 'Time ↓'),
    ]
    
    with open(output_path, 'w') as f:
        f.write("% Aggregated Bottleneck Study Results\n")
        f.write("% Generated from multiple experiment runs\n\n")
        
        for latent_dim in sorted(df['latent_dim'].unique()):
            f.write(f"\n% Latent Dimension: {latent_dim}\n")
            subset = df[df['latent_dim'] == latent_dim].copy()
            
            # Sort by model, then pca_components
            if 'pca_components' in subset.columns:
                subset['sort_key'] = subset.apply(
                    lambda x: (x['model'], x.get('pca_components', 0) if pd.notna(x.get('pca_components')) else 0), 
                    axis=1
                )
            else:
                subset['sort_key'] = subset['model']
            subset = subset.sort_values('sort_key')
            
            # Header
            header = "Model & " + " & ".join([m[1] for m in metrics]) + " \\\\\n"
            f.write(header)
            f.write("\\hline\n")
            
            # Rows
            for _, row in subset.iterrows():
                model_name = row.get('display_name', row['model'])
                if 'pca_components' in row and pd.notna(row['pca_components']):
                    model_name = f"{model_name} (PCA={int(row['pca_components'])})"
                
                values = [model_name]
                for metric_col, _ in metrics:
                    if metric_col in row and pd.notna(row[metric_col]):
                        val = row[metric_col]
                        std_col = f'{metric_col}_std'
                        if std_col in row and pd.notna(row[std_col]):
                            if metric_col == 'train_time_seconds':
                                values.append(f"{val:.1f} $\\pm$ {row[std_col]:.1f}")
                            else:
                                values.append(f"{val:.4f} $\\pm$ {row[std_col]:.4f}")
                        else:
                            if metric_col == 'train_time_seconds':
                                values.append(f"{val:.1f}")
                            else:
                                values.append(f"{val:.4f}")
                    else:
                        values.append("-")
                
                f.write(" & ".join(values) + " \\\\\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Aggregate bottleneck study results')
    parser.add_argument('--base-dir', type=str, required=True,
                      help='Base directory containing timestamped result folders')
    parser.add_argument('--output-csv', type=str, default='aggregated_results.csv',
                      help='Output CSV file name')
    parser.add_argument('--output-latex', type=str, default='aggregated_results_latex.txt',
                      help='Output LaTeX table file name')
    args = parser.parse_args()
    
    # Load all results
    print(f"Loading results from: {args.base_dir}\n")
    all_results = load_all_results(args.base_dir)
    
    if all_results is None or len(all_results) == 0:
        print("No results to aggregate!")
        return 1
    
    print(f"\nTotal rows loaded: {len(all_results)}")
    print(f"Unique models: {all_results['model'].unique()}")
    print(f"Unique latent dims: {sorted(all_results['latent_dim'].unique())}")
    if 'pca_components' in all_results.columns:
        print(f"MMAE PCA components: {sorted(all_results[all_results['model']=='mmae']['pca_components'].dropna().unique())}")
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = aggregate_metrics(all_results)
    
    print(f"Aggregated to {len(aggregated)} unique configurations")
    
    # Print comprehensive table
    print_comprehensive_table(aggregated)
    
    # Save outputs
    output_dir = Path(args.base_dir)
    csv_path = output_dir / args.output_csv
    latex_path = output_dir / args.output_latex
    
    aggregated.to_csv(csv_path, index=False)
    print(f"\n✓ Saved aggregated CSV to: {csv_path}")
    
    save_latex_table(aggregated, latex_path)
    print(f"✓ Saved LaTeX table to: {latex_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())