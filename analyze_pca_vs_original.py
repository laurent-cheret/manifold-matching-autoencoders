#!/usr/bin/env python
"""
PCA vs Original Distances Analysis.

Demonstrates that for high-dimensional real-world data, PCA-reduced distances
are MORE meaningful than original space distances due to:
1. Distance contrast loss (curse of dimensionality)
2. Noise filtering
3. Better preservation of intrinsic structure

Usage:
    python analyze_pca_vs_original.py
    python analyze_pca_vs_original.py --datasets mnist pbmc3k
    python analyze_pca_vs_original.py --n_samples 1000
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from config import get_config, DATASET_CONFIGS
from data import load_data


# =============================================================================
# DATASETS
# =============================================================================

SYNTHETIC_DATASETS = [
    'spheres', 'concentric_spheres', 'linked_tori', 
    'mammoth', 'earth'
]

REAL_DATASETS = [
    'mnist', 'fmnist', 'cifar10', 'coil20', 'pbmc3k', 'paul15'
]

ALL_DATASETS = SYNTHETIC_DATASETS + REAL_DATASETS

# Colors for plotting
COLORS = {
    # Synthetic - blues/greens
    'spheres': '#1f77b4',
    'concentric_spheres': '#17becf',
    'linked_tori': '#2ca02c',
    'klein_bottle': '#98df8a',
    'mammoth': '#7f7f7f',
    'earth': '#8c564b',
    # Real - oranges/reds/purples
    'mnist': '#ff7f0e',
    'fmnist': '#ffbb78',
    'cifar10': '#d62728',
    'coil20': '#e377c2',
    'pbmc3k': '#9467bd',
    'paul15': '#c5b0d5',
}


# =============================================================================
# METRICS
# =============================================================================

def distance_matrix(X):
    """Compute pairwise Euclidean distance matrix."""
    return squareform(pdist(X, metric='euclidean'))


def distance_contrast(D):
    """
    Compute distance contrast: std(distances) / mean(distances).
    Higher = more discriminative distances (better).
    In high dimensions, this approaches 0 (curse of dimensionality).
    """
    d = D[np.triu_indices_from(D, k=1)]
    return d.std() / (d.mean() + 1e-10)


def distance_correlation(D1, D2):
    """Pearson correlation between pairwise distances."""
    d1 = D1[np.triu_indices_from(D1, k=1)]
    d2 = D2[np.triu_indices_from(D2, k=1)]
    return pearsonr(d1, d2)[0]


def knn_accuracy(X, labels, k=5, cv=5):
    """k-NN classification accuracy using cross-validation."""
    # Handle continuous labels by discretizing
    unique_labels = np.unique(labels)
    if len(unique_labels) > 100:  # Likely continuous
        # Discretize into 10 bins
        labels = pd.qcut(labels, q=10, labels=False, duplicates='drop')
    
    labels = labels.astype(int)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return np.nan, np.nan
    
    # Ensure enough samples per class for CV
    min_per_class = min(np.bincount(labels))
    if min_per_class < cv:
        cv = max(2, min_per_class)
    
    knn = KNeighborsClassifier(n_neighbors=min(k, len(X) - 1))
    try:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(knn, X, labels, cv=skf, scoring='accuracy')
        return scores.mean(), scores.std()
    except:
        return np.nan, np.nan


def label_triplet_accuracy(D, labels, n_triplets=5000, seed=42):
    """
    Triplet accuracy using class labels as ground truth.
    Same-class points should be closer than different-class points.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    
    # Discretize continuous labels
    unique_labels = np.unique(labels)
    if len(unique_labels) > 50:
        labels = pd.qcut(labels, q=10, labels=False, duplicates='drop')
    
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return np.nan
    
    correct = 0
    total = 0
    
    for _ in range(n_triplets):
        anchor_idx = rng.randint(n)
        anchor_label = labels[anchor_idx]
        
        same_class = np.where(labels == anchor_label)[0]
        same_class = same_class[same_class != anchor_idx]
        if len(same_class) == 0:
            continue
        pos_idx = rng.choice(same_class)
        
        diff_class = np.where(labels != anchor_label)[0]
        if len(diff_class) == 0:
            continue
        neg_idx = rng.choice(diff_class)
        
        if D[anchor_idx, pos_idx] < D[anchor_idx, neg_idx]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else np.nan


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_flat(dataset_name, n_samples=2000, seed=42):
    """Load dataset and return flattened data with labels."""
    config = get_config(dataset_name)
    config['seed'] = seed
    config['arch_type'] = 'mlp'  # Force flat
    
    # Set sample limit
    if 'n_samples' in config and config['n_samples'] is not None:
        config['n_samples'] = min(config['n_samples'], n_samples)
    else:
        config['n_samples'] = n_samples
    
    try:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
            dataset_name, config, with_embeddings=False
        )
    except Exception as e:
        print(f"    Error loading {dataset_name}: {e}")
        return None, None, None
    
    X = train_dataset.data.numpy()
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    labels = train_dataset.labels.numpy()
    
    # Subsample if still too large
    if len(X) > n_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
    
    input_dim = X.shape[1]
    
    return X, labels, input_dim


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_dataset(dataset_name, X, labels, input_dim):
    """
    Analyze a single dataset: compare original vs PCA distances.
    """
    results = {
        'dataset': dataset_name,
        'n_samples': len(X),
        'input_dim': input_dim,
        'is_synthetic': dataset_name in SYNTHETIC_DATASETS,
    }
    
    # Original space metrics
    D_orig = distance_matrix(X)
    results['contrast_original'] = distance_contrast(D_orig)
    
    knn_orig, knn_std = knn_accuracy(X, labels)
    results['knn_original'] = knn_orig
    results['knn_original_std'] = knn_std
    
    results['triplet_original'] = label_triplet_accuracy(D_orig, labels)
    
    # PCA at different fractions
    pca_fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    
    # Max PCA components is min(n_samples, n_features) - 1
    max_components = min(len(X), input_dim) - 1
    
    best_knn = knn_orig if not np.isnan(knn_orig) else 0
    best_knn_frac = 1.0
    best_triplet = results['triplet_original'] if not np.isnan(results['triplet_original']) else 0
    best_triplet_frac = 1.0
    
    for frac in pca_fractions:
        n_comp = max(2, int(frac * input_dim))
        n_comp = min(n_comp, max_components)
        
        if n_comp < 2:
            continue
        
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X)
        D_pca = distance_matrix(X_pca)
        
        # Store metrics
        frac_str = f'{int(frac*100):02d}'
        results[f'contrast_pca{frac_str}'] = distance_contrast(D_pca)
        results[f'dist_corr_pca{frac_str}'] = distance_correlation(D_orig, D_pca)
        
        knn_pca, _ = knn_accuracy(X_pca, labels)
        results[f'knn_pca{frac_str}'] = knn_pca
        
        triplet_pca = label_triplet_accuracy(D_pca, labels)
        results[f'triplet_pca{frac_str}'] = triplet_pca
        
        # Track best
        if not np.isnan(knn_pca) and knn_pca > best_knn:
            best_knn = knn_pca
            best_knn_frac = frac
        if not np.isnan(triplet_pca) and triplet_pca > best_triplet:
            best_triplet = triplet_pca
            best_triplet_frac = frac
    
    results['best_knn'] = best_knn
    results['best_knn_frac'] = best_knn_frac
    results['knn_improvement'] = best_knn - (knn_orig if not np.isnan(knn_orig) else 0)
    
    results['best_triplet'] = best_triplet
    results['best_triplet_frac'] = best_triplet_frac
    results['triplet_improvement'] = best_triplet - (results['triplet_original'] if not np.isnan(results['triplet_original']) else 0)
    
    # Concentration improvement (best PCA vs original)
    n_comp_10 = max(2, int(0.10 * input_dim))
    n_comp_10 = min(n_comp_10, max_components)
    pca_10 = PCA(n_components=n_comp_10, random_state=42)
    X_pca_10 = pca_10.fit_transform(X)
    D_pca_10 = distance_matrix(X_pca_10)
    conc_pca_10 = distance_contrast(D_pca_10)
    results['contrast_improvement'] = (conc_pca_10 - results['contrast_original']) / (results['contrast_original'] + 1e-10)
    
    return results


def run_analysis(datasets, n_samples=2000, output_dir='results/pca_vs_original'):
    """Run analysis across all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    print(f"\n{'='*70}")
    print(f"PCA vs ORIGINAL DISTANCES ANALYSIS")
    print(f"{'='*70}")
    print(f"Datasets: {datasets}")
    print(f"Samples per dataset: {n_samples}")
    print(f"{'='*70}\n")
    
    for dataset_name in datasets:
        print(f"\n--- {dataset_name.upper()} ---")
        
        X, labels, input_dim = load_dataset_flat(dataset_name, n_samples)
        
        if X is None:
            print(f"  Skipping {dataset_name}")
            continue
        
        print(f"  Shape: {X.shape}, Labels: {len(np.unique(labels))} unique")
        
        results = analyze_dataset(dataset_name, X, labels, input_dim)
        all_results.append(results)
        
        # Print summary
        print(f"  Original: contrast={results['contrast_original']:.4f}, "
              f"kNN={results['knn_original']:.4f}, "
              f"triplet={results['triplet_original']:.4f}")
        
        if results['best_knn_frac'] < 1.0:
            print(f"  Best kNN at PCA {results['best_knn_frac']*100:.0f}%: "
                  f"{results['best_knn']:.4f} (+{results['knn_improvement']:.4f})")
        
        if results['contrast_improvement'] > 0.1:
            print(f"  Concentration improved by {results['contrast_improvement']*100:.1f}% with PCA")
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'pca_vs_original_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def create_summary_figure(df, output_dir):
    """Create publication-quality summary figure."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Separate synthetic vs real
    synthetic = df[df['is_synthetic'] == True]
    real = df[df['is_synthetic'] == False]
    
    # Panel A: Distance Concentration
    ax = axes[0]
    
    x_pos = []
    colors = []
    labels_plot = []
    conc_orig = []
    conc_pca = []
    
    for i, row in df.iterrows():
        x_pos.append(i)
        colors.append(COLORS.get(row['dataset'], 'gray'))
        labels_plot.append(row['dataset'].upper())
        conc_orig.append(row['contrast_original'])
        # Use PCA 10%
        conc_pca.append(row.get('contrast_pca10', row['contrast_original']))
    
    width = 0.35
    x = np.arange(len(df))
    
    bars1 = ax.bar(x - width/2, conc_orig, width, label='Original', color='lightgray', edgecolor='black')
    bars2 = ax.bar(x + width/2, conc_pca, width, label='PCA 10%', color='steelblue', edgecolor='black')
    
    ax.set_ylabel('Distance Contrast (std/mean)', fontweight='bold')
    ax.set_title('A. Distance Contrast\n(Higher = Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Add separator between synthetic and real
    if len(synthetic) > 0 and len(real) > 0:
        ax.axvline(x=len(synthetic) - 0.5, color='red', linestyle='--', alpha=0.5)
    
    # Panel B: k-NN Accuracy
    ax = axes[1]
    
    knn_orig = df['knn_original'].fillna(0).values
    knn_best = df['best_knn'].fillna(0).values
    
    bars1 = ax.bar(x - width/2, knn_orig, width, label='Original', color='lightgray', edgecolor='black')
    bars2 = ax.bar(x + width/2, knn_best, width, label='Best PCA', color='forestgreen', edgecolor='black')
    
    ax.set_ylabel('k-NN Accuracy', fontweight='bold')
    ax.set_title('B. Classification Accuracy\n(Higher = Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)
    
    if len(synthetic) > 0 and len(real) > 0:
        ax.axvline(x=len(synthetic) - 0.5, color='red', linestyle='--', alpha=0.5)
    
    # Panel C: Triplet Accuracy
    ax = axes[2]
    
    triplet_orig = df['triplet_original'].fillna(0.5).values
    triplet_best = df['best_triplet'].fillna(0.5).values
    
    bars1 = ax.bar(x - width/2, triplet_orig, width, label='Original', color='lightgray', edgecolor='black')
    bars2 = ax.bar(x + width/2, triplet_best, width, label='Best PCA', color='darkorange', edgecolor='black')
    
    ax.set_ylabel('Triplet Accuracy', fontweight='bold')
    ax.set_title('C. Label-Based Triplet Acc.\n(Higher = Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    
    if len(synthetic) > 0 and len(real) > 0:
        ax.axvline(x=len(synthetic) - 0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, 'pca_vs_original_summary.png')
    pdf_path = os.path.join(output_dir, 'pca_vs_original_summary.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def create_contrast_curve(df, output_dir):
    """Create figure showing distance contrast vs dimensionality."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot each dataset
    for _, row in df.iterrows():
        name = row['dataset']
        dim = row['input_dim']
        n_samples = row['n_samples']
        is_synth = row['is_synthetic']
        
        # Max possible PCA components
        max_comp = min(n_samples, dim) - 1
        
        # Collect contrast values with actual PCA dimensions used
        fracs = [1.0, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05]
        concs = []
        actual_dims = []
        
        for f in fracs:
            if f == 1.0:
                key = 'contrast_original'
                actual_dim = dim
            else:
                key = f'contrast_pca{int(f*100):02d}'
                # Actual PCA components used (matching analyze_dataset logic)
                actual_dim = max(2, int(f * dim))
                actual_dim = min(actual_dim, max_comp)
            
            if key in row and not pd.isna(row[key]):
                concs.append(row[key])
                actual_dims.append(actual_dim)
        
        if len(concs) == 0:
            continue
        
        style = '-' if is_synth else '--'
        marker = 'o' if is_synth else 's'
        color = COLORS.get(name, 'gray')
        
        ax.plot(actual_dims, concs, style, marker=marker, color=color, 
                label=name.upper(), markersize=6, linewidth=2)
    
    ax.set_xlabel('Number of Dimensions (PCA Components)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Distance Contrast (std/mean)', fontsize=11, fontweight='bold')
    ax.set_title('Distance Contrast vs Dimensionality\n(Solid=Synthetic, Dashed=Real)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'contrast_vs_dim.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(df):
    """Print a summary table for the paper."""
    
    print("\n" + "="*80)
    print("SUMMARY: Why PCA Distances are Better for High-D Data")
    print("="*80)
    
    print("\n--- SYNTHETIC DATASETS (Low ambient dimension) ---")
    synth = df[df['is_synthetic'] == True]
    for _, row in synth.iterrows():
        knn_diff = row['knn_improvement']
        symbol = "≈" if abs(knn_diff) < 0.02 else ("+" if knn_diff > 0 else "-")
        print(f"  {row['dataset']:20s} (dim={row['input_dim']:5d}): "
              f"kNN orig={row['knn_original']:.3f}, best={row['best_knn']:.3f} "
              f"[{symbol}] at {row['best_knn_frac']*100:.0f}%")
    
    print("\n--- REAL DATASETS (High ambient dimension) ---")
    real = df[df['is_synthetic'] == False]
    for _, row in real.iterrows():
        knn_diff = row['knn_improvement']
        symbol = "≈" if abs(knn_diff) < 0.02 else ("+" if knn_diff > 0 else "-")
        conc_imp = row['contrast_improvement'] * 100
        print(f"  {row['dataset']:20s} (dim={row['input_dim']:5d}): "
              f"kNN orig={row['knn_original']:.3f}, best={row['best_knn']:.3f} "
              f"[{symbol}] at {row['best_knn_frac']*100:.0f}%, "
              f"contrast +{conc_imp:.0f}%")
    
    print("\n--- KEY FINDINGS ---")
    
    synth_improvement = synth['knn_improvement'].mean()
    real_improvement = real['knn_improvement'].mean() if len(real) > 0 else 0
    
    print(f"  Synthetic: Avg kNN improvement = {synth_improvement:+.4f}")
    print(f"  Real:      Avg kNN improvement = {real_improvement:+.4f}")
    
    if len(real) > 0:
        real_conc_imp = real['contrast_improvement'].mean() * 100
        print(f"  Real:      Avg contrast improvement = +{real_conc_imp:.1f}%")
    
    print("\n" + "="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PCA vs Original Distances Analysis')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Datasets to analyze (default: all)')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Samples per dataset')
    parser.add_argument('--output_dir', type=str, default='results/pca_vs_original',
                       help='Output directory')
    args = parser.parse_args()
    
    datasets = args.datasets if args.datasets else ALL_DATASETS
    
    # Run analysis
    df = run_analysis(datasets, n_samples=args.n_samples, output_dir=args.output_dir)
    
    if len(df) > 0:
        # Create figures
        print("\nGenerating figures...")
        create_summary_figure(df, args.output_dir)
        create_contrast_curve(df, args.output_dir)
        
        # Print summary
        print_summary_table(df)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()