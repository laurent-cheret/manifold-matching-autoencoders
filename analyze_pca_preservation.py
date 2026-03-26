#!/usr/bin/env python
"""
PCA Distance Preservation Analysis.

Shows how PCA preserves data structure as we increase the number of components.
This justifies MMAE's use of PCA as a reference target.

Usage:
    python analyze_pca_preservation.py --datasets spheres mnist fmnist cifar10 pbmc3k
    python analyze_pca_preservation.py --datasets spheres mnist --n_samples 2000
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from config import get_config, DATASET_CONFIGS
from data import load_data


# Component percentages to evaluate
COMPONENT_PERCENTAGES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 1.0]

# Colors for datasets
DATASET_COLORS = {
    'spheres': '#1f77b4',
    'mnist': '#ff7f0e', 
    'fmnist': '#2ca02c',
    'cifar10': '#d62728',
    'pbmc3k': '#9467bd',
    'paul15': '#17becf',
    'swiss_roll': '#8c564b',
    'concentric_spheres': '#e377c2',
    'tree_clusters': '#7f7f7f',
}

DATASET_MARKERS = {
    'spheres': 'o',
    'mnist': 's',
    'fmnist': '^',
    'cifar10': 'D',
    'pbmc3k': 'v',
    'paul15': 'P',
    'swiss_roll': 'p',
    'concentric_spheres': 'h',
    'tree_clusters': '*',
}


def distance_matrix(X):
    """Compute pairwise distance matrix."""
    return squareform(pdist(X))


def distance_correlation(X, Z):
    """Pearson correlation between pairwise distances."""
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    mask = np.triu(np.ones_like(Dx), k=1) > 0
    return pearsonr(Dx[mask], Dz[mask])[0]


def distance_spearman(X, Z):
    """Spearman correlation between pairwise distances."""
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    mask = np.triu(np.ones_like(Dx), k=1) > 0
    return spearmanr(Dx[mask], Dz[mask])[0]


def neighbors_and_ranks(D, k):
    """Get k-nearest neighbors and ranks."""
    idx = np.argsort(D, axis=-1, kind='stable')
    return idx[:, 1:k+1], idx.argsort(axis=-1, kind='stable')


def trustworthiness(X, Z, k=10):
    """Trustworthiness metric."""
    n = X.shape[0]
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    Nx, Rx = neighbors_and_ranks(Dx, k)
    Nz, _ = neighbors_and_ranks(Dz, k)
    
    result = 0.0
    for i in range(n):
        for j in np.setdiff1d(Nz[i], Nx[i]):
            result += Rx[i, j] - k
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result


def continuity(X, Z, k=10):
    """Continuity metric."""
    return trustworthiness(Z, X, k)


def triplet_accuracy(X, Z, n_triplets=10000, seed=42):
    """Triplet distance ranking accuracy."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    
    if n < 3:
        return 0.0
    
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    
    n_triplets = min(n_triplets, n * (n-1) * (n-2) // 6)
    
    correct = 0
    for _ in range(n_triplets):
        i, j, k = rng.choice(n, 3, replace=False)
        x_order = Dx[i, j] < Dx[i, k]
        z_order = Dz[i, j] < Dz[i, k]
        if x_order == z_order:
            correct += 1
    
    return correct / n_triplets


def density_kl_divergence(X, Z, sigma=0.1):
    """KL divergence between density estimates."""
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    
    Dx = Dx / (Dx.max() + 1e-10)
    Dz = Dz / (Dz.max() + 1e-10)
    
    density_x = np.sum(np.exp(-(Dx ** 2) / sigma), axis=-1)
    density_x = density_x / (density_x.sum() + 1e-10)
    
    density_z = np.sum(np.exp(-(Dz ** 2) / sigma), axis=-1)
    density_z = density_z / (density_z.sum() + 1e-10)
    
    eps = 1e-10
    kl = np.sum(density_x * (np.log(density_x + eps) - np.log(density_z + eps)))
    
    return kl


def compute_persistence_diagrams(X, max_dim=1):
    """Compute persistence diagrams using gudhi."""
    try:
        import gudhi as gd
        D = distance_matrix(X)
        rips = gd.RipsComplex(distance_matrix=D, max_edge_length=np.inf)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        st.compute_persistence()
        
        diagrams = {}
        for dim in range(max_dim + 1):
            intervals = st.persistence_intervals_in_dimension(dim)
            finite = intervals[np.isfinite(intervals[:, 1])] if len(intervals) > 0 else np.array([]).reshape(0, 2)
            diagrams[dim] = finite
        return diagrams
    except ImportError:
        return None


def wasserstein_distance(dgm1, dgm2, order=1):
    """Compute Wasserstein distance between persistence diagrams."""
    try:
        import gudhi.wasserstein as gw
        return gw.wasserstein_distance(dgm1, dgm2, order=order)
    except:
        try:
            import gudhi.hera as hera
            return hera.wasserstein_distance(dgm1, dgm2, internal_p=order)
        except:
            return None


def evaluate_pca_preservation(X, n_components, seed=42):
    """Evaluate how well PCA with n_components preserves structure."""
    # Fit PCA
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    
    results = {}
    
    # Distance metrics
    results['distance_correlation'] = distance_correlation(X, X_pca)
    results['distance_spearman'] = distance_spearman(X, X_pca)
    
    # Neighborhood metrics - average across multiple k values
    ks = [5, 10, 50, 100]
    valid_ks = [k for k in ks if k < X.shape[0] - 1]
    
    if valid_ks:
        trust_values = [trustworthiness(X, X_pca, k=k) for k in valid_ks]
        cont_values = [continuity(X, X_pca, k=k) for k in valid_ks]
        results['trustworthiness'] = np.mean(trust_values)
        results['continuity'] = np.mean(cont_values)
    
    # Triplet accuracy
    results['triplet_accuracy'] = triplet_accuracy(X, X_pca, seed=seed)
    
    # Density KL - average across multiple sigma values
    sigmas = [0.01, 0.1, 1.0]
    kl_values = [density_kl_divergence(X, X_pca, sigma=s) for s in sigmas]
    results['density_kl'] = np.mean(kl_values)
    
    # Variance explained
    results['variance_explained'] = np.sum(pca.explained_variance_ratio_)
    
    # Wasserstein - use subsampling for larger datasets
    wass_subsample = 500
    wass_n_runs = 5
    
    if X.shape[0] > wass_subsample:
        # Subsample for Wasserstein computation
        w0_values = []
        w1_values = []
        rng = np.random.RandomState(seed)
        
        for run in range(wass_n_runs):
            idx = rng.choice(X.shape[0], wass_subsample, replace=False)
            X_sub = X[idx]
            X_pca_sub = X_pca[idx]
            
            # Normalize for comparable scales
            X_dists = distance_matrix(X_sub)
            Z_dists = distance_matrix(X_pca_sub)
            X_norm = X_sub / (np.percentile(X_dists, 90) + 1e-10)
            Z_norm = X_pca_sub / (np.percentile(Z_dists, 90) + 1e-10)
            
            dgm_x = compute_persistence_diagrams(X_norm, max_dim=1)
            dgm_z = compute_persistence_diagrams(Z_norm, max_dim=1)
            
            if dgm_x is not None and dgm_z is not None:
                w0 = wasserstein_distance(dgm_x[0], dgm_z[0])
                w1 = wasserstein_distance(dgm_x[1], dgm_z[1])
                if w0 is not None:
                    w0_values.append(w0)
                if w1 is not None:
                    w1_values.append(w1)
        
        if w0_values:
            results['wasserstein_H0'] = np.mean(w0_values)
        if w1_values:
            results['wasserstein_H1'] = np.mean(w1_values)
    else:
        # Small dataset - compute directly
        X_dists = distance_matrix(X)
        Z_dists = distance_matrix(X_pca)
        X_norm = X / (np.percentile(X_dists, 90) + 1e-10)
        Z_norm = X_pca / (np.percentile(Z_dists, 90) + 1e-10)
        
        dgm_x = compute_persistence_diagrams(X_norm, max_dim=1)
        dgm_z = compute_persistence_diagrams(Z_norm, max_dim=1)
        
        if dgm_x is not None and dgm_z is not None:
            w0 = wasserstein_distance(dgm_x[0], dgm_z[0])
            w1 = wasserstein_distance(dgm_x[1], dgm_z[1])
            if w0 is not None:
                results['wasserstein_H0'] = w0
            if w1 is not None:
                results['wasserstein_H1'] = w1
    
    return results


def load_dataset_for_analysis(dataset_name, n_samples=2000, seed=42):
    """Load dataset and return flattened data."""
    config = get_config(dataset_name)
    config['seed'] = seed
    
    # Limit samples for efficiency - handle None case
    if 'n_samples' in config and config['n_samples'] is not None:
        config['n_samples'] = min(config['n_samples'], n_samples)
    else:
        config['n_samples'] = n_samples
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        dataset_name, config, with_embeddings=False
    )
    
    # Get data
    X = train_dataset.data.numpy()
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    
    # Subsample if still too large
    if len(X) > n_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
    
    return X


def run_pca_analysis(datasets, n_samples=2000, output_dir='results/pca_analysis', seed=42):
    """Run PCA preservation analysis across datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            X = load_dataset_for_analysis(dataset_name, n_samples=n_samples, seed=seed)
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
            continue
        
        input_dim = X.shape[1]
        print(f"  Shape: {X.shape}, Input dim: {input_dim}")
        
        dataset_results = {pct: {} for pct in COMPONENT_PERCENTAGES}
        
        for pct in COMPONENT_PERCENTAGES:
            n_components = max(1, int(input_dim * pct))
            # Cap at actual dimensions
            n_components = min(n_components, min(X.shape[0], X.shape[1]))
            
            print(f"  PCA {pct*100:.0f}% ({n_components} components)...", end=' ')
            
            results = evaluate_pca_preservation(X, n_components, seed=seed)
            dataset_results[pct] = results
            dataset_results[pct]['n_components'] = n_components
            
            print(f"dist_corr={results['distance_correlation']:.4f}, trust={results['trustworthiness']:.4f}")
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def normalize_metric(values, higher_is_better=True):
    """Normalize metric values to [0, 1] range."""
    values = np.array(values)
    v_min, v_max = values.min(), values.max()
    
    if v_max - v_min < 1e-10:
        return np.ones_like(values)
    
    normalized = (values - v_min) / (v_max - v_min)
    
    if not higher_is_better:
        normalized = 1 - normalized
    
    return normalized


def plot_results(all_results, output_dir, normalize=True):
    """Create plots for all metrics."""
    
    # Metrics to plot
    metrics_config = {
        'distance_correlation': {'higher_is_better': True, 'title': 'Distance Correlation'},
        'distance_spearman': {'higher_is_better': True, 'title': 'Distance Spearman'},
        'trustworthiness': {'higher_is_better': True, 'title': 'Trustworthiness (avg)'},
        'continuity': {'higher_is_better': True, 'title': 'Continuity (avg)'},
        'triplet_accuracy': {'higher_is_better': True, 'title': 'Triplet Accuracy'},
        'density_kl': {'higher_is_better': False, 'title': 'Density KL (avg)'},
        'variance_explained': {'higher_is_better': True, 'title': 'Variance Explained'},
        'wasserstein_H0': {'higher_is_better': False, 'title': 'Wasserstein H₀'},
        'wasserstein_H1': {'higher_is_better': False, 'title': 'Wasserstein H₁'},
    }
    
    percentages = [p * 100 for p in COMPONENT_PERCENTAGES]
    
    for metric, config in metrics_config.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        has_data = False
        
        for dataset_name, dataset_results in all_results.items():
            values = []
            valid_pcts = []
            
            for pct in COMPONENT_PERCENTAGES:
                if metric in dataset_results[pct]:
                    values.append(dataset_results[pct][metric])
                    valid_pcts.append(pct * 100)
            
            if not values:
                continue
            
            has_data = True
            
            if normalize:
                plot_values = normalize_metric(values, config['higher_is_better']) * 100
                ylabel = 'Normalized Score (%)'
            else:
                plot_values = values
                ylabel = config['title']
            
            color = DATASET_COLORS.get(dataset_name, 'gray')
            marker = DATASET_MARKERS.get(dataset_name, 'o')
            
            ax.plot(valid_pcts, plot_values, 
                   color=color, marker=marker, markersize=8, linewidth=2,
                   label=dataset_name.upper())
        
        if not has_data:
            plt.close()
            continue
        
        ax.set_xlabel('PCA Components (% of original dimensionality)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{config["title"]} vs PCA Components', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        
        if normalize:
            ax.set_ylim(-5, 105)
        
        plt.tight_layout()
        
        suffix = '_normalized' if normalize else '_absolute'
        save_path = os.path.join(output_dir, f'pca_{metric}{suffix}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # Create combined plot with key metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    key_metrics = ['distance_correlation', 'trustworthiness', 'continuity', 
                   'triplet_accuracy', 'density_kl', 'variance_explained']
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 3, idx % 3]
        config = metrics_config[metric]
        
        for dataset_name, dataset_results in all_results.items():
            values = []
            valid_pcts = []
            
            for pct in COMPONENT_PERCENTAGES:
                if metric in dataset_results[pct]:
                    values.append(dataset_results[pct][metric])
                    valid_pcts.append(pct * 100)
            
            if not values:
                continue
            
            if normalize:
                plot_values = normalize_metric(values, config['higher_is_better']) * 100
            else:
                plot_values = values
            
            color = DATASET_COLORS.get(dataset_name, 'gray')
            marker = DATASET_MARKERS.get(dataset_name, 'o')
            
            ax.plot(valid_pcts, plot_values,
                   color=color, marker=marker, markersize=6, linewidth=1.5,
                   label=dataset_name.upper())
        
        ax.set_xlabel('PCA Components (%)')
        ax.set_ylabel('Score (%)' if normalize else config['title'])
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        if normalize:
            ax.set_ylim(-5, 105)
    
    # Single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(all_results), 
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    suffix = '_normalized' if normalize else '_absolute'
    save_path = os.path.join(output_dir, f'pca_combined{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_results_csv(all_results, output_dir):
    """Save results to CSV."""
    import pandas as pd
    
    rows = []
    for dataset_name, dataset_results in all_results.items():
        for pct, metrics in dataset_results.items():
            row = {
                'dataset': dataset_name,
                'pct_components': pct * 100,
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'pca_preservation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PCA distance preservation')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['spheres', 'mnist', 'fmnist', 'pbmc3k', 'paul15'],
                       help='Datasets to analyze')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Max samples per dataset (default: 2000)')
    parser.add_argument('--output_dir', type=str, default='results/pca_analysis',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Plot absolute values instead of normalized')
    parser.add_argument('--replot', type=str, default=None,
                       help='Path to CSV file to replot without recomputing')
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"PCA DISTANCE PRESERVATION ANALYSIS")
    print(f"{'#'*60}")
    
    if args.replot:
        # Load from CSV and replot
        print(f"Replotting from: {args.replot}")
        all_results = load_results_from_csv(args.replot)
        output_dir = os.path.dirname(args.replot) or args.output_dir
    else:
        print(f"Datasets: {args.datasets}")
        print(f"Samples: {args.n_samples}")
        print(f"Output: {args.output_dir}")
        print(f"{'#'*60}\n")
        
        # Run analysis
        all_results = run_pca_analysis(
            datasets=args.datasets,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            seed=args.seed
        )
        output_dir = args.output_dir
        
        # Save CSV
        save_results_csv(all_results, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_results(all_results, output_dir, normalize=not args.no_normalize)
    
    # Also create absolute value plots
    if not args.no_normalize:
        plot_results(all_results, output_dir, normalize=False)
    
    # Create 3-panel paper figure
    print(f"\nGenerating 3-panel paper figure...")
    plot_three_panel_figure(all_results, output_dir, normalize=not args.no_normalize)
    
    print(f"\n{'#'*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'#'*60}")
    print(f"Results saved to: {output_dir}")


def load_results_from_csv(csv_path):
    """Load results from CSV file for replotting."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    all_results = {}
    for dataset_name in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset_name]
        dataset_results = {}
        
        for _, row in dataset_df.iterrows():
            pct = row['pct_components'] / 100  # Convert back to fraction
            metrics = row.to_dict()
            # Remove non-metric columns
            for col in ['dataset', 'pct_components']:
                metrics.pop(col, None)
            dataset_results[pct] = metrics
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def plot_three_panel_figure(all_results, output_dir, normalize=True):
    """
    Create 3-panel figure for paper (single column, ICML format):
    1. Distance Preservation (distance_correlation, triplet_accuracy)
    2. Neighborhood Preservation (trustworthiness, continuity)
    3. Topological Preservation (wasserstein_H0, wasserstein_H1)
    """
    
    # Define metric groups
    metric_groups = {
        'Distance': {
            'metrics': ['distance_correlation', 'triplet_accuracy'],
            'higher_is_better': True
        },
        'Neighborhood': {
            'metrics': ['trustworthiness', 'continuity'],
            'higher_is_better': True
        },
        'Topology': {
            'metrics': ['wasserstein_H0', 'wasserstein_H1'],
            'higher_is_better': False
        }
    }
    
    percentages = [p * 100 for p in COMPONENT_PERCENTAGES]
    
    # Horizontal layout: 1 row, 3 columns - fits single column width
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    
    for idx, (group_name, group_config) in enumerate(metric_groups.items()):
        ax = axes[idx]
        metrics = group_config['metrics']
        higher_is_better = group_config['higher_is_better']
        
        for dataset_name, dataset_results in all_results.items():
            # Compute average across metrics in group
            avg_values = []
            valid_pcts = []
            
            for pct in COMPONENT_PERCENTAGES:
                metric_values = []
                for metric in metrics:
                    if metric in dataset_results[pct]:
                        val = dataset_results[pct][metric]
                        if val is not None and not np.isnan(val):
                            metric_values.append(val)
                
                if metric_values:
                    avg_values.append(np.mean(metric_values))
                    valid_pcts.append(pct * 100)
            
            if not avg_values:
                continue
            
            if normalize:
                # Normalize to percentage
                plot_values = normalize_metric(avg_values, higher_is_better) * 100
            else:
                plot_values = avg_values
            
            color = DATASET_COLORS.get(dataset_name, 'gray')
            marker = DATASET_MARKERS.get(dataset_name, 'o')
            
            ax.plot(valid_pcts, plot_values,
                   color=color, marker=marker, markersize=5, linewidth=2,
                   label=dataset_name.upper(), markeredgewidth=0)
        
        ax.set_xlabel('PCA (%)', fontsize=9, fontweight='bold')
        
        # Y label only on first plot
        if idx == 0:
            if normalize:
                ax.set_ylabel('Score (%)', fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('Score', fontsize=9, fontweight='bold')
        
        if normalize:
            ax.set_ylim(-5, 105)
        
        ax.set_title(group_name, fontsize=10, fontweight='bold', pad=3)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        ax.set_xlim(0, 105)
        
        # Larger, bolder tick labels
        ax.tick_params(axis='both', which='major', labelsize=8, width=1.2, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Thicker spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    # Legend at bottom, spread across full width
    handles, labels = axes[0].get_legend_handles_labels()
    n_datasets = len(handles)
    fig.legend(handles, labels, loc='lower center', ncol=n_datasets,
               fontsize=8, bbox_to_anchor=(0.5, -0.08),
               frameon=True, fancybox=False, edgecolor='black',
               handlelength=1.5, columnspacing=1, handletextpad=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.3)
    
    suffix = '_normalized' if normalize else '_absolute'
    
    # Save PNG
    save_path = os.path.join(output_dir, f'pca_three_panel{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    # Save PDF for paper
    save_path_pdf = os.path.join(output_dir, f'pca_three_panel{suffix}.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_pdf}")


if __name__ == '__main__':
    main()