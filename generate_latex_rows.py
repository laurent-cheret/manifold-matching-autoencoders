#!/usr/bin/env python
"""
Generate LaTeX table rows from final evaluation CSVs.

Usage:
    # Synthetic datasets (2D latent)
    python generate_latex_rows.py --mode synthetic --results results.csv --latent_dim 2
    
    # Real-world datasets (high-D latent)
    python generate_latex_rows.py --mode realworld --results results.csv --latent_dim 64
    
    # Multiple files at once
    python generate_latex_rows.py --mode synthetic --results spheres.csv tori.csv swiss.csv --latent_dim 2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ORDER = ['vanilla', 'mmae', 'topoae', 'rtdae', 'geomae', 'ggae', 'spae']

MODEL_NAMES = {
    'vanilla': 'Vanilla AE',
    'mmae': 'MMAE (Ours)',
    'topoae': 'TopoAE',
    'rtdae': 'RTD-AE',
    'geomae': 'GeomAE',
    'ggae': 'GGAE',
    'spae': 'SPAE',
}

# Synthetic table metrics (2D visualization)
SYNTHETIC_METRICS = [
    # (csv_column, display_name, lower_is_better)
    ('distance_correlation', 'DC', False),
    ('triplet_accuracy', 'TA', False),
    ('density_kl_0_1', 'KL$_{0.1}$', True),
]

# Real-world table metrics (high-D representation)
REALWORLD_METRICS = [
    ('reconstruction_error', 'Rec', True),
    ('distance_correlation', 'DC', False),
    ('triplet_accuracy', 'TA', False),
    ('density_kl_0_1', 'KL$_{0.1}$', True),
    ('trustworthiness_avg', 'Trust', False),  # averaged across k
    ('continuity_avg', 'Cont', False),        # averaged across k
    ('wasserstein_H0', 'W$_0$', True),
]

# Columns to average for trust/cont
TRUST_COLS = ['trustworthiness_5', 'trustworthiness_10', 'trustworthiness_50', 'trustworthiness_100']
CONT_COLS = ['continuity_5', 'continuity_10', 'continuity_50', 'continuity_100']


def preprocess_dataframe(df):
    """Compute averaged trustworthiness and continuity across k values."""
    df = df.copy()
    
    # Average trustworthiness
    trust_available = [c for c in TRUST_COLS if c in df.columns]
    if trust_available:
        df['trustworthiness_avg'] = df[trust_available].mean(axis=1)
    
    # Average continuity
    cont_available = [c for c in CONT_COLS if c in df.columns]
    if cont_available:
        df['continuity_avg'] = df[cont_available].mean(axis=1)
    
    return df


# =============================================================================
# FORMATTING
# =============================================================================

def format_value(val, lower_is_better=False, precision=2):
    """Format a single value."""
    if pd.isna(val):
        return '-'
    
    if abs(val) >= 100:
        return f"{val:.1f}"
    elif abs(val) >= 10:
        return f"{val:.2f}"
    elif abs(val) < 0.01:
        return f"{val:.3f}"
    else:
        return f"{val:.{precision}f}"


def find_best_values(df, metrics):
    """Find best value for each metric across all models."""
    best = {}
    for col, name, lower_is_better in metrics:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                best[col] = vals.min() if lower_is_better else vals.max()
    return best


def is_best(val, best_val, tol=1e-6):
    """Check if value is the best (within tolerance)."""
    if pd.isna(val) or best_val is None:
        return False
    return abs(val - best_val) < tol


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_dataset_rows(df, dataset_name, latent_dim, metrics, bold_best=True):
    """Generate LaTeX rows for a single dataset."""
    df_dim = df[df['latent_dim'] == latent_dim].copy()
    
    if len(df_dim) == 0:
        print(f"% WARNING: No results for latent_dim={latent_dim}")
        return []
    
    # Find best values
    best = find_best_values(df_dim, metrics) if bold_best else {}
    
    lines = []
    n_models = sum(1 for m in MODEL_ORDER if m in df_dim['model'].values)
    
    for i, model in enumerate(MODEL_ORDER):
        model_data = df_dim[df_dim['model'] == model]
        if len(model_data) == 0:
            continue
        
        row = model_data.iloc[0]
        
        # First column: dataset name (only for first model) or empty
        if i == 0:
            first_col = f"\\multirow{{{n_models}}}{{*}}{{{dataset_name}}}"
        else:
            first_col = ""
        
        # Second column: model name
        model_display = MODEL_NAMES.get(model, model)
        
        # Metric columns
        metric_vals = []
        for col, name, lower_is_better in metrics:
            if col not in df_dim.columns:
                metric_vals.append('-')
                continue
            
            val = row.get(col)
            formatted = format_value(val, lower_is_better)
            
            # Bold if best
            if bold_best and col in best and is_best(val, best[col]):
                formatted = f"\\textbf{{{formatted}}}"
            
            metric_vals.append(formatted)
        
        # Build row
        row_str = f"{first_col} & {model_display} & " + " & ".join(metric_vals) + " \\\\"
        lines.append(row_str)
    
    return lines


def generate_header(metrics, mode='synthetic'):
    """Generate table header."""
    header_parts = ['Dataset', 'Method']
    for col, name, lower_is_better in metrics:
        arrow = '$\\downarrow$' if lower_is_better else '$\\uparrow$'
        header_parts.append(f"{name}{arrow}")
    return " & ".join(header_parts) + " \\\\"


def generate_table_skeleton(metrics, mode='synthetic'):
    """Generate full table skeleton."""
    n_cols = len(metrics) + 2  # Dataset + Method + metrics
    col_spec = 'll' + 'c' * len(metrics)
    
    if mode == 'synthetic':
        caption = "Structure preservation on synthetic datasets (2D latent). Best per dataset in \\textbf{bold}."
        label = "tab:synthetic_results"
    else:
        caption = "Representation quality on real-world datasets. Best per dataset in \\textbf{bold}."
        label = "tab:realworld_results"
    
    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        generate_header(metrics, mode),
        "\\midrule",
        "% === PASTE DATASET ROWS HERE ===",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX table rows')
    parser.add_argument('--mode', type=str, required=True, choices=['synthetic', 'realworld'],
                        help='Table type: synthetic (2D) or realworld (high-D)')
    parser.add_argument('--results', type=str, nargs='+', default=None,
                        help='Path(s) to results CSV file(s)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names (in order matching --results). Auto-detected if not provided.')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent dimension to extract')
    parser.add_argument('--no_bold', action='store_true',
                        help='Disable bolding best values')
    parser.add_argument('--skeleton_only', action='store_true',
                        help='Print only table skeleton (no data needed)')
    args = parser.parse_args()
    
    # Select metrics based on mode
    metrics = SYNTHETIC_METRICS if args.mode == 'synthetic' else REALWORLD_METRICS
    
    # Print skeleton only
    if args.skeleton_only:
        print("\n% " + "="*60)
        print(f"% TABLE SKELETON ({args.mode.upper()})")
        print("% " + "="*60)
        print(generate_table_skeleton(metrics, args.mode))
        return
    
    # Validate required args for data processing
    if args.results is None:
        parser.error("--results is required unless using --skeleton_only")
    if args.latent_dim is None:
        parser.error("--latent_dim is required unless using --skeleton_only")
    
    # Process each results file
    all_rows = []
    
    for i, results_path in enumerate(args.results):
        if not Path(results_path).exists():
            print(f"% WARNING: File not found: {results_path}")
            continue
        
        df = pd.read_csv(results_path)
        
        # Preprocess to compute averaged metrics (only needed for realworld)
        if args.mode == 'realworld':
            df = preprocess_dataframe(df)
        
        # Determine dataset name
        if args.datasets and i < len(args.datasets):
            dataset_name = args.datasets[i]
        else:
            # Try to extract from filename
            dataset_name = Path(results_path).stem.replace('final_results_', '').split('_')[0].title()
        
        print(f"\n% === {dataset_name.upper()} ===")
        print(f"% Source: {results_path}")
        print(f"% Latent dim: {args.latent_dim}")
        print(f"% Models found: {df['model'].unique().tolist()}")
        
        rows = generate_dataset_rows(
            df, dataset_name, args.latent_dim, metrics, 
            bold_best=not args.no_bold
        )
        
        if rows:
            print("\\midrule" if all_rows else "")
            for row in rows:
                print(row)
            all_rows.extend(rows)
        else:
            print(f"% No results for latent_dim={args.latent_dim}")
    
    # Summary
    print(f"\n% Total rows generated: {len(all_rows)}")


if __name__ == '__main__':
    main()