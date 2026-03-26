#!/usr/bin/env python
"""
Generate LaTeX tables from final evaluation results.

Usage:
    # Generate tables for latent dim 2
    python generate_latex_tables.py --results results/final/spheres/final_results.csv --latent_dim 2 --dataset Spheres
    
    # Generate only compact table (no std)
    python generate_latex_tables.py --results results/final/mnist/final_results.csv --latent_dim 2 --dataset MNIST --compact_only
    
    # Save to file
    python generate_latex_tables.py --results results/final/pbmc3k/final_results.csv --latent_dim 2 --dataset PBMC3k --output tables.tex
"""

import argparse
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Metrics where lower is better
LOWER_IS_BETTER = [
    'reconstruction_error', 
    'train_time',
    'wasserstein_H0', 'wasserstein_H1',
    'density_kl_0_01', 'density_kl_0_1', 'density_kl_1_0',
]

# Model display names and order
MODEL_NAMES = {
    'vanilla': 'Vanilla AE',
    'mmae': 'MMAE (Ours)',
    'topoae': 'TopoAE',
    'rtdae': 'RTD-AE',
    'geomae': 'GeomAE',
    'ggae': 'GGAE',
}

# Order: baseline first, then ours, then competitors
MODEL_ORDER = ['vanilla', 'mmae', 'topoae', 'rtdae', 'geomae', 'ggae']

# Metrics to display with their abbreviations and directions
METRICS_CONFIG = [
    ('reconstruction_error', 'Rec', 'down'),
    ('train_time', 'Time', 'down'),
    ('distance_correlation', 'DC', 'up'),
    ('triplet_accuracy', 'TA', 'up'),
    ('knn_accuracy_10', 'kNN', 'up'),
    ('density_kl_0_01', 'KL$_{.01}$', 'down'),
    ('density_kl_0_1', 'KL$_{.1}$', 'down'),
    ('density_kl_1_0', 'KL$_{1}$', 'down'),
    ('wasserstein_H0', '$W_0$', 'down'),
    ('wasserstein_H1', '$W_1$', 'down'),
]


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def format_value_compact(val, metric_name='', precision=2):
    """Format value without std (compact version)."""
    if pd.isna(val):
        return '-'
    
    # Special formatting for time (show as integer seconds)
    if 'time' in metric_name.lower():
        return f"{val:.0f}"
    
    # Large Wasserstein values (>10)
    if abs(val) > 10:
        return f"{val:.1f}"
    
    # Small values
    if abs(val) < 0.01:
        return f"{val:.3f}"
    
    return f"{val:.{precision}f}"


def format_value_with_std(val, std=None, metric_name='', precision=2):
    """Format value with std."""
    if pd.isna(val):
        return '-'
    
    # Special formatting for time
    if 'time' in metric_name.lower():
        if std is not None and not pd.isna(std) and std > 0:
            return f"{val:.0f}$\\pm${std:.0f}"
        return f"{val:.0f}"
    
    # Large values (>10)
    if abs(val) > 10:
        if std is not None and not pd.isna(std) and std > 0:
            return f"{val:.1f}$\\pm${std:.1f}"
        return f"{val:.1f}"
    
    # Small values
    if abs(val) < 0.01:
        if std is not None and not pd.isna(std) and std > 0:
            return f"{val:.3f}$\\pm${std:.3f}"
        return f"{val:.3f}"
    
    if std is not None and not pd.isna(std) and std > 0:
        return f"{val:.{precision}f}$\\pm${std:.{precision}f}"
    
    return f"{val:.{precision}f}"


def make_bold(text):
    """Wrap text in bold."""
    return f"\\textbf{{{text}}}"


def get_arrow(direction):
    """Get LaTeX arrow for direction."""
    if direction == 'up':
        return '$\\uparrow$'
    elif direction == 'down':
        return '$\\downarrow$'
    return ''


# ============================================================
# TABLE GENERATION
# ============================================================

def generate_compact_table(df, dataset_name, latent_dim):
    """Generate compact table with means only (no std)."""
    
    # Filter to specified latent dim
    df_dim = df[df['latent_dim'] == latent_dim].copy()
    
    if len(df_dim) == 0:
        return f"% No results for latent_dim={latent_dim}\n"
    
    # Find best values for each metric
    best_vals = {}
    for metric, abbrev, direction in METRICS_CONFIG:
        if metric in df_dim.columns:
            vals = df_dim[metric].dropna()
            if len(vals) > 0:
                if metric in LOWER_IS_BETTER:
                    best_vals[metric] = vals.min()
                else:
                    best_vals[metric] = vals.max()
    
    # Build header
    header_parts = ['Method']
    for metric, abbrev, direction in METRICS_CONFIG:
        arrow = get_arrow(direction)
        header_parts.append(f"{abbrev}{arrow}")
    
    n_cols = len(header_parts)
    col_spec = 'l' + 'c' * (n_cols - 1)
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{dataset_name} ($d={latent_dim}$): Structure Preservation Metrics}}")
    lines.append(f"\\label{{tab:{dataset_name.lower().replace(' ', '_')}_dim{latent_dim}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")
    
    # Add rows for each model
    for model in MODEL_ORDER:
        model_data = df_dim[df_dim['model'] == model]
        if len(model_data) == 0:
            continue
        
        row = model_data.iloc[0]
        display_name = MODEL_NAMES.get(model, model)
        
        row_parts = [display_name]
        for metric, abbrev, direction in METRICS_CONFIG:
            if metric not in df_dim.columns or pd.isna(row.get(metric)):
                row_parts.append("-")
                continue
            
            val = row[metric]
            formatted = format_value_compact(val, metric_name=metric)
            
            # Bold if best
            is_best = metric in best_vals and not pd.isna(val) and abs(val - best_vals[metric]) < 1e-6
            if is_best:
                formatted = make_bold(formatted)
            
            row_parts.append(formatted)
        
        lines.append(" & ".join(row_parts) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_full_table(df, dataset_name, latent_dim):
    """Generate full table with means ± std."""
    
    # Filter to specified latent dim
    df_dim = df[df['latent_dim'] == latent_dim].copy()
    
    if len(df_dim) == 0:
        return f"% No results for latent_dim={latent_dim}\n"
    
    # Find best values for each metric
    best_vals = {}
    for metric, abbrev, direction in METRICS_CONFIG:
        if metric in df_dim.columns:
            vals = df_dim[metric].dropna()
            if len(vals) > 0:
                if metric in LOWER_IS_BETTER:
                    best_vals[metric] = vals.min()
                else:
                    best_vals[metric] = vals.max()
    
    # Build header
    header_parts = ['Method']
    for metric, abbrev, direction in METRICS_CONFIG:
        arrow = get_arrow(direction)
        header_parts.append(f"{abbrev}{arrow}")
    
    n_cols = len(header_parts)
    col_spec = 'l' + 'c' * (n_cols - 1)
    
    lines = []
    lines.append("\\begin{table*}[htbp]")  # Use table* for full width
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{dataset_name} ($d={latent_dim}$): Structure Preservation Metrics (mean $\\pm$ std)}}")
    lines.append(f"\\label{{tab:{dataset_name.lower().replace(' ', '_')}_dim{latent_dim}_full}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")
    
    # Add rows for each model
    for model in MODEL_ORDER:
        model_data = df_dim[df_dim['model'] == model]
        if len(model_data) == 0:
            continue
        
        row = model_data.iloc[0]
        display_name = MODEL_NAMES.get(model, model)
        
        row_parts = [display_name]
        for metric, abbrev, direction in METRICS_CONFIG:
            if metric not in df_dim.columns or pd.isna(row.get(metric)):
                row_parts.append("-")
                continue
            
            val = row[metric]
            std = row.get(f'{metric}_std')
            formatted = format_value_with_std(val, std, metric_name=metric)
            
            # Bold if best
            is_best = metric in best_vals and not pd.isna(val) and abs(val - best_vals[metric]) < 1e-6
            if is_best:
                formatted = make_bold(formatted)
            
            row_parts.append(formatted)
        
        lines.append(" & ".join(row_parts) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_ranking_summary(df, dataset_name, latent_dim):
    """Generate a ranking summary showing which method wins on each metric."""
    
    df_dim = df[df['latent_dim'] == latent_dim].copy()
    
    if len(df_dim) == 0:
        return ""
    
    lines = []
    lines.append(f"% ===== RANKING SUMMARY: {dataset_name} (d={latent_dim}) =====")
    
    for metric, abbrev, direction in METRICS_CONFIG:
        if metric not in df_dim.columns:
            continue
        
        vals = df_dim[['model', metric]].dropna()
        if len(vals) == 0:
            continue
        
        if metric in LOWER_IS_BETTER:
            best_idx = vals[metric].idxmin()
        else:
            best_idx = vals[metric].idxmax()
        
        best_model = vals.loc[best_idx, 'model']
        best_val = vals.loc[best_idx, metric]
        
        display_name = MODEL_NAMES.get(best_model, best_model)
        lines.append(f"% {abbrev}: {display_name} ({best_val:.3f})")
    
    lines.append("")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from evaluation results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results CSV')
    parser.add_argument('--latent_dim', type=int, required=True,
                       help='Latent dimensionality to generate table for')
    parser.add_argument('--dataset', type=str, default='Dataset',
                       help='Dataset name for caption (e.g., Spheres, MNIST, PBMC3k)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: print to stdout)')
    parser.add_argument('--compact_only', action='store_true',
                       help='Generate only compact table (no std)')
    parser.add_argument('--full_only', action='store_true',
                       help='Generate only full table (with std)')
    args = parser.parse_args()
    
    # Load results
    df = pd.read_csv(args.results)
    
    print(f"Loaded {len(df)} results from {args.results}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Latent dims available: {sorted(df['latent_dim'].unique().tolist())}")
    print(f"Generating table for latent_dim={args.latent_dim}")
    
    # Check if requested latent_dim exists
    if args.latent_dim not in df['latent_dim'].values:
        print(f"ERROR: latent_dim={args.latent_dim} not found in results!")
        print(f"Available: {sorted(df['latent_dim'].unique().tolist())}")
        return
    
    output_lines = []
    
    # Add ranking summary as comments
    output_lines.append(generate_ranking_summary(df, args.dataset, args.latent_dim))
    
    # Generate tables based on flags
    if args.compact_only:
        output_lines.append("% ========== COMPACT TABLE (means only) ==========\n")
        output_lines.append(generate_compact_table(df, args.dataset, args.latent_dim))
    elif args.full_only:
        output_lines.append("% ========== FULL TABLE (with std) ==========\n")
        output_lines.append(generate_full_table(df, args.dataset, args.latent_dim))
    else:
        # Generate both
        output_lines.append("% ========== COMPACT TABLE (means only) ==========\n")
        output_lines.append(generate_compact_table(df, args.dataset, args.latent_dim))
        output_lines.append("\n\n")
        output_lines.append("% ========== FULL TABLE (with std) ==========\n")
        output_lines.append(generate_full_table(df, args.dataset, args.latent_dim))
    
    output_lines.append("\n")
    output = "".join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + "="*60)
        print("GENERATED LATEX:")
        print("="*60)
        print(output)


if __name__ == '__main__':
    main()