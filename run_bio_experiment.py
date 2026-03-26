#!/usr/bin/env python
"""
Improved biological validation experiments with publication-quality figures.

Features:
- Fixed GGAE dataset bug
- 2D and 3D latent space visualizations
- Trajectory arrows showing differentiation paths
- Interactive 3D Plotly HTML files
- Clean publication-ready layouts

Usage:
    python run_bio_experiment_v2.py --dataset paul15 --all_models --latent_dim 2
    python run_bio_experiment_v2.py --dataset paul15 --all_models --latent_dim 3
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("Warning: scanpy not installed. Install with: pip install scanpy")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, kendalltau
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from config import get_config, DATASET_CONFIGS
from models import build_model
from training import Trainer
from evaluation import evaluate


# =============================================================================
# PUBLICATION STYLE SETTINGS
# =============================================================================

PAPER_STYLE = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Lineage colors - colorblind friendly, more saturated
LINEAGE_COLORS = {
    'Erythroid': '#E31A1C',      # Red
    'Megakaryocyte': '#FF7F00',  # Orange
    'Granulocyte': '#33A02C',    # Green
    'Monocyte': '#1F78B4',       # Blue
    'Basophil': '#6A3D9A',       # Purple
    'Lymphoid': '#B2DF8A',       # Light green
    'Other': '#999999',          # Gray
}

# Plotly colors (same palette)
LINEAGE_COLORS_PLOTLY = {
    'Erythroid': '#E31A1C',
    'Megakaryocyte': '#FF7F00',
    'Granulocyte': '#33A02C',
    'Monocyte': '#1F78B4',
    'Basophil': '#6A3D9A',
    'Lymphoid': '#B2DF8A',
    'Other': '#999999',
}

# Methods to compare (focused set)
METHOD_ORDER = ['mmae', 'topoae', 'rtdae', 'vanilla']

METHOD_NAMES = {
    'pca': 'PCA',
    'vanilla': 'Vanilla AE',
    'mmae': 'MMAE (Ours)',
    'topoae': 'TopoAE',
    'rtdae': 'RTD-AE',
    'geomae': 'GeomAE',
    'ggae': 'GGAE',
}

METHOD_COLORS = {
    'pca': '#17becf',
    'vanilla': '#7f7f7f',
    'mmae': '#E31A1C',
    'topoae': '#1F78B4',
    'rtdae': '#33A02C',
    'geomae': '#FF7F00',
    'ggae': '#6A3D9A',
}

# Paul15 trajectory definitions
PAUL15_TRAJECTORIES = {
    'Erythroid': {
        'path': ['7MEP', '1Ery', '2Ery', '3Ery', '4Ery', '5Ery', '6Ery'],
        'color': LINEAGE_COLORS['Erythroid'],
    },
    'Megakaryocyte': {
        'path': ['7MEP', '8Mk'],
        'color': LINEAGE_COLORS['Megakaryocyte'],
    },
    'Granulocyte': {
        'path': ['9GMP', '10GMP', '16Neu', '17Neu', '18Eos'],
        'color': LINEAGE_COLORS['Granulocyte'],
    },
    'Monocyte': {
        'path': ['9GMP', '10GMP', '11DC', '14Mo', '15Mo'],
        'color': LINEAGE_COLORS['Monocyte'],
    },
    'Basophil': {
        'path': ['9GMP', '10GMP', '12Baso', '13Baso'],
        'color': LINEAGE_COLORS['Basophil'],
    },
}


# =============================================================================
# 3D ARROW CLASS FOR MATPLOTLIB
# =============================================================================

class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_paul15():
    """Load Paul15 hematopoiesis dataset."""
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy required. Install: pip install scanpy")
    
    print("Loading Paul15 dataset...")
    adata = sc.datasets.paul15()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    data = adata.X
    if hasattr(data, 'toarray'):
        data = data.toarray()
    
    # STRING labels for pseudotime/visualization (e.g., '7MEP', '1Ery')
    labels = adata.obs['paul15_clusters'].values.astype(str)
    
    # NUMERIC label_ids for clustering metrics (e.g., 0, 1, 2, ...)
    unique_labels = np.unique(labels)
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    label_ids = np.array([label_to_id[l] for l in labels])
    
    # Original cell_type_order - linear ordering for pseudotime correlation
    # This ordering reflects differentiation progression
    cell_type_order = {
        '1Ery': 0, '2Ery': 1, '3Ery': 2, '4Ery': 3, '5Ery': 4, '6Ery': 5,
        '7MEP': 6, '8Mk': 7, '9GMP': 8, '10GMP': 9, '11DC': 10,
        '12Baso': 11, '13Baso': 12, '14Mo': 13, '15Mo': 14,
        '16Neu': 15, '17Neu': 16, '18Eos': 17, '19Lymph': 18,
    }
    
    # Lineage mapping for coloring
    lineage_map = {}
    for ct in unique_labels:
        if 'Ery' in ct:
            lineage_map[ct] = 'Erythroid'
        elif 'MEP' in ct or 'Mk' in ct:
            lineage_map[ct] = 'Megakaryocyte'
        elif 'GMP' in ct or 'Neu' in ct or 'Eos' in ct:
            lineage_map[ct] = 'Granulocyte'
        elif 'Mo' in ct or 'DC' in ct:
            lineage_map[ct] = 'Monocyte'
        elif 'Baso' in ct:
            lineage_map[ct] = 'Basophil'
        elif 'Lymph' in ct:
            lineage_map[ct] = 'Lymphoid'
        else:
            lineage_map[ct] = 'Other'
    
    lineages = np.array([lineage_map[l] for l in labels])
    
    print(f"  {data.shape[0]} cells, {data.shape[1]} genes, {len(unique_labels)} cell types")
    
    return data, labels, label_ids, lineages, cell_type_order


# =============================================================================
# FIXED DATASET CLASS
# =============================================================================

class BioDataset(torch.utils.data.Dataset):
    """Dataset wrapper for biological data - FIXED to always return 3 values."""
    
    def __init__(self, data, labels, pca_embeddings=None, return_indices=False):
        self.data = torch.FloatTensor(data)
        self.labels = labels
        self.pca_embeddings = pca_embeddings
        self.return_indices = return_indices
        if pca_embeddings is not None:
            self.pca_embeddings = torch.FloatTensor(pca_embeddings)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Always return 3 values: (data, embedding_or_placeholder, index)
        if self.pca_embeddings is not None:
            return self.data[idx], self.pca_embeddings[idx], idx
        else:
            # Return data as placeholder for embedding when no PCA
            return self.data[idx], self.data[idx], idx


# =============================================================================
# METRICS
# =============================================================================

def compute_geodesic_distances(data, k=15, n_samples=1000):
    """Compute geodesic distances using KNN graph shortest paths."""
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        subset = data[indices]
    else:
        indices = np.arange(len(data))
        subset = data
    
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(subset)
    distances, neighbors = nn.kneighbors(subset)
    
    n = len(subset)
    row_indices = np.repeat(np.arange(n), k)
    col_indices = neighbors.flatten()
    weights = distances.flatten()
    
    graph = csr_matrix((weights, (row_indices, col_indices)), shape=(n, n))
    graph = graph + graph.T
    graph.data /= 2
    
    geodesic_dists = shortest_path(graph, directed=False)
    max_finite = np.max(geodesic_dists[np.isfinite(geodesic_dists)])
    geodesic_dists[~np.isfinite(geodesic_dists)] = max_finite * 2
    
    return geodesic_dists, indices


def compute_geodesic_correlation(data, latents, k=15, n_samples=1000):
    """Compute correlation between geodesic and latent distances."""
    geodesic_dists, indices = compute_geodesic_distances(data, k, n_samples)
    latents_subset = latents[indices]
    
    latent_dists = np.sqrt(np.sum(
        (latents_subset[:, np.newaxis, :] - latents_subset[np.newaxis, :, :]) ** 2, 
        axis=-1
    ))
    
    triu_idx = np.triu_indices(len(indices), k=1)
    geo_flat = geodesic_dists[triu_idx]
    lat_flat = latent_dists[triu_idx]
    
    corr, _ = spearmanr(geo_flat, lat_flat)
    return corr


def compute_pseudotime(latents, labels, root_types=None):
    """Compute pseudotime as distance from root cell centroid.
    
    Args:
        latents: (n_cells, latent_dim) array
        labels: (n_cells,) array of STRING labels like '7MEP', '1Ery'
        root_types: list of root cell type strings
    
    Returns:
        pseudotime: (n_cells,) normalized distances from root
    """
    if root_types is None:
        root_types = ['7MEP', '9GMP']
    
    # labels should be strings to match root_types
    root_mask = np.isin(labels, root_types)
    
    if root_mask.sum() == 0:
        print(f"  Warning: No root cells found. Labels sample: {labels[:5]}")
        # Fallback: use centroid of all cells
        root_centroid = latents.mean(axis=0)
    else:
        root_centroid = latents[root_mask].mean(axis=0)
    
    distances = np.sqrt(np.sum((latents - root_centroid) ** 2, axis=1))
    pseudotime = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
    return pseudotime


def evaluate_pseudotime(pseudotime, labels, cell_type_order):
    """Compute pseudotime correlation with known ordering.
    
    Args:
        pseudotime: (n_cells,) computed pseudotime values
        labels: (n_cells,) STRING labels like '7MEP', '1Ery'
        cell_type_order: dict mapping string label -> expected order int
    
    Returns:
        spearman, kendall correlation coefficients
    """
    # Get expected order for each cell based on its string label
    expected_order = np.array([cell_type_order.get(str(l), -1) for l in labels])
    valid_mask = expected_order >= 0
    
    n_valid = valid_mask.sum()
    if n_valid < 10:
        print(f"  Warning: Only {n_valid} cells with valid ordering")
        return np.nan, np.nan
    
    spearman, _ = spearmanr(pseudotime[valid_mask], expected_order[valid_mask])
    kendall, _ = kendalltau(pseudotime[valid_mask], expected_order[valid_mask])
    return spearman, kendall


def compute_auto_bandwidth(data, n_samples=1000):
    """Compute automatic bandwidth for GGAE."""
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        subset = data[indices]
    else:
        subset = data
    
    diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    mask = sq_dists > 0
    
    if mask.sum() > 0:
        return float(np.median(sq_dists[mask]))
    return 1.0


# =============================================================================
# HYPERPARAMETER LOADING
# =============================================================================

def load_best_config(hyperparam_dir, model_name, latent_dim=2):
    """Load best hyperparameters from search results."""
    config_path = os.path.join(hyperparam_dir, f'{model_name}_dim{latent_dim}', 'best_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            best_config = json.load(f)
        print(f"  Loaded config for {model_name}")
        return best_config['hyperparameters']
    return None


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(model_name, data, labels, label_ids, lineages, cell_type_order,
                config_overrides=None, best_hyperparams=None, device='cuda'):
    """Train a model and return results."""
    
    print(f"\n{'='*60}")
    print(f"Training {METHOD_NAMES.get(model_name, model_name)}")
    print(f"{'='*60}")
    
    config = {
        'input_dim': data.shape[1],
        'latent_dim': config_overrides.get('latent_dim', 2),
        'hidden_dims': [512, 256, 128],
        'n_epochs': config_overrides.get('n_epochs', 100),
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
    }
    
    # Apply best hyperparameters
    if best_hyperparams:
        if 'learning_rate' in best_hyperparams:
            config['learning_rate'] = best_hyperparams['learning_rate']
        if 'batch_size' in best_hyperparams:
            config['batch_size'] = int(best_hyperparams['batch_size'])
        
        if model_name == 'mmae':
            if 'mmae_n_components' in best_hyperparams:
                config['mmae_n_components'] = int(best_hyperparams['mmae_n_components'])
            if 'mmae_lambda' in best_hyperparams:
                config['mmae_weight'] = best_hyperparams['mmae_lambda']
        elif model_name == 'topoae':
            if 'topo_lambda' in best_hyperparams:
                config['topoae_weight'] = best_hyperparams['topo_lambda']
        elif model_name == 'rtdae':
            if 'rtd_lambda' in best_hyperparams:
                config['rtd_weight'] = best_hyperparams['rtd_lambda']
        elif model_name == 'ggae':
            if 'gg_lambda' in best_hyperparams:
                config['ggae_lambda'] = best_hyperparams['gg_lambda']
            if 'gg_bandwidth' in best_hyperparams:
                config['ggae_bandwidth'] = best_hyperparams['gg_bandwidth']
    
    # Model-specific defaults
    if model_name == 'mmae' and 'mmae_n_components' not in config:
        config['mmae_n_components'] = 500
        config['mmae_weight'] = 1.0
    elif model_name == 'topoae' and 'topoae_weight' not in config:
        config['topoae_weight'] = 1.0
    elif model_name == 'rtdae' and 'rtd_weight' not in config:
        config['rtd_weight'] = 1.0
    elif model_name == 'ggae':
        if 'ggae_lambda' not in config:
            config['ggae_lambda'] = 1.0
        if 'ggae_bandwidth' not in config:
            config['ggae_bandwidth'] = compute_auto_bandwidth(data)
    
    # Prepare PCA embeddings for MMAE
    pca_embeddings = None
    if model_name == 'mmae':
        n_comp = config.get('mmae_n_components', 500)
        pca = PCA(n_components=min(n_comp, data.shape[1], data.shape[0]))
        pca_embeddings = pca.fit_transform(data)
        print(f"  PCA: {data.shape[1]} → {pca_embeddings.shape[1]} ({pca.explained_variance_ratio_.sum()*100:.1f}% var)")
    
    # Create dataset (FIXED: always returns 3 values)
    dataset = BioDataset(data, labels, pca_embeddings, return_indices=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Build model
    model = build_model(model_name, config)
    
    # GGAE: precompute kernel
    if model_name == 'ggae':
        print("  Precomputing GGAE kernel...")
        data_tensor = torch.FloatTensor(data).to(device)
        model.precompute_kernel(data_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    trainer = Trainer(model, optimizer, device, model_name=model_name)
    
    start_time = time.time()
    trainer.fit(loader, loader, n_epochs=config['n_epochs'], verbose=False)
    train_time = time.time() - start_time
    print(f"  Time: {train_time:.1f}s")
    
    # Extract latents
    model.eval()
    with torch.no_grad():
        latents = model.encode(torch.FloatTensor(data).to(device)).cpu().numpy()
    
    # Compute metrics
    metrics = evaluate(data, latents, label_ids)
    metrics['train_time'] = train_time
    
    # Geodesic correlation
    print("  Computing geodesic correlation...")
    metrics['geodesic_correlation'] = compute_geodesic_correlation(data, latents)
    
    # Pseudotime
    pseudotime = compute_pseudotime(latents, labels)
    if cell_type_order:
        spearman, kendall = evaluate_pseudotime(pseudotime, labels, cell_type_order)
        metrics['pseudotime_spearman'] = spearman
        metrics['pseudotime_kendall'] = kendall
    
    print(f"  DC={metrics['distance_correlation']:.3f}, Geo={metrics['geodesic_correlation']:.3f}, PT={metrics.get('pseudotime_spearman', 0):.3f}")
    
    # Clustering
    n_clusters = len(np.unique(label_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(latents)
    metrics['ARI'] = adjusted_rand_score(label_ids, pred)
    metrics['NMI'] = normalized_mutual_info_score(label_ids, pred)
    
    return {
        'latents': latents,
        'labels': labels,
        'lineages': lineages,
        'pseudotime': pseudotime,
        'metrics': metrics,
        'config': config,
    }


# =============================================================================
# 2D VISUALIZATION WITH TRAJECTORY ARROWS
# =============================================================================

def compute_centroids(latents, labels):
    """Compute centroids for each cell type.
    
    Args:
        latents: (n_cells, latent_dim) array
        labels: (n_cells,) array of STRING labels like '7MEP', '1Ery'
    
    Returns:
        dict mapping label string -> centroid array
    """
    unique_labels = np.unique(labels)
    centroids = {}
    for l in unique_labels:
        mask = labels == l
        if mask.sum() > 0:
            centroids[str(l)] = latents[mask].mean(axis=0)
    return centroids


def draw_trajectory_arrows_2d(ax, centroids, trajectories, arrow_scale=1.0):
    """Draw trajectory arrows on 2D plot."""
    for traj_name, traj_info in trajectories.items():
        path = traj_info['path']
        color = traj_info['color']
        
        for i in range(len(path) - 1):
            start_label = path[i]
            end_label = path[i + 1]
            
            if start_label in centroids and end_label in centroids:
                start = centroids[start_label]
                end = centroids[end_label]
                
                # Shorten arrow slightly to not overlap with points
                direction = end - start
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    start_adj = start + 0.1 * direction * length
                    end_adj = end - 0.1 * direction * length
                    
                    ax.annotate('', xy=end_adj, xytext=start_adj,
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             lw=2.5 * arrow_scale, alpha=0.8))


def create_2d_comparison_figure(results_dict, save_path=None):
    """Create publication-quality 2D comparison figure."""
    plt.rcParams.update(PAPER_STYLE)
    
    methods = [m for m in METHOD_ORDER if m in results_dict]
    n_methods = len(methods)
    
    if n_methods == 0:
        print("No results to plot!")
        return
    
    # Figure: one row of latent spaces + metrics table below
    fig = plt.figure(figsize=(4 * n_methods, 5.5))
    gs = GridSpec(2, n_methods, height_ratios=[1, 0.15], hspace=0.25, wspace=0.1)
    
    # Compute global limits
    all_latents = []
    for method in methods:
        latents = results_dict[method]['latents']
        latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-8)
        all_latents.append(latents_norm)
    
    all_concat = np.vstack(all_latents)
    lim = max(abs(all_concat.min()), abs(all_concat.max())) * 1.15
    
    for i, method in enumerate(methods):
        res = results_dict[method]
        latents_norm = all_latents[i]
        lineages = res['lineages']
        labels = res['labels']
        m = res['metrics']
        
        ax = fig.add_subplot(gs[0, i])
        
        # Plot cells by lineage
        for lin in LINEAGE_COLORS.keys():
            mask = lineages == lin
            if mask.sum() > 0:
                ax.scatter(latents_norm[mask, 0], latents_norm[mask, 1],
                          c=LINEAGE_COLORS[lin], s=8, alpha=0.6,
                          rasterized=True, label=lin)
        
        # Draw trajectory arrows
        centroids = compute_centroids(latents_norm, labels)
        draw_trajectory_arrows_2d(ax, centroids, PAUL15_TRAJECTORIES, arrow_scale=0.8)
        
        # Mark progenitors
        for prog in ['7MEP', '9GMP']:
            if prog in centroids:
                c = centroids[prog]
                ax.scatter([c[0]], [c[1]], c='white', s=150, edgecolors='black', 
                          linewidths=2, zorder=10)
                ax.annotate(prog, (c[0], c[1]), fontsize=8, fontweight='bold',
                           ha='center', va='bottom', xytext=(0, 8),
                           textcoords='offset points')
        
        # Title with key metrics
        dc = m.get('distance_correlation', 0)
        geo = m.get('geodesic_correlation', 0)
        pt = m.get('pseudotime_spearman', 0) or 0
        train_time = m.get('train_time', 0)
        
        title = f"{METHOD_NAMES.get(method, method)}"
        ax.set_title(title, fontsize=14, fontweight='bold' if method == 'mmae' else 'normal')
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Add metrics as text below plot
        metrics_text = f"DC={dc:.2f}  Geo={geo:.2f}  PT={pt:.2f}  {train_time:.0f}s"
        ax.text(0.5, -0.08, metrics_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add legend only to first plot
        if i == 0:
            handles = [mpatches.Patch(color=c, label=l) 
                      for l, c in LINEAGE_COLORS.items() if l != 'Other']
            ax.legend(handles=handles, loc='upper left', fontsize=8, 
                     frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# 3D VISUALIZATION (STATIC)
# =============================================================================

def draw_trajectory_arrows_3d(ax, centroids, trajectories):
    """Draw trajectory arrows on 3D plot."""
    for traj_name, traj_info in trajectories.items():
        path = traj_info['path']
        color = traj_info['color']
        
        for i in range(len(path) - 1):
            start_label = path[i]
            end_label = path[i + 1]
            
            if start_label in centroids and end_label in centroids:
                start = centroids[start_label]
                end = centroids[end_label]
                
                # Draw line (3D arrows are tricky, use line + endpoint marker)
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                       color=color, linewidth=2.5, alpha=0.8)
                
                # Add arrow head as cone approximation
                direction = end - start
                length = np.linalg.norm(direction)
                if length > 0:
                    ax.scatter([end[0]], [end[1]], [end[2]], 
                             c=color, s=50, marker='>', alpha=0.8)


def create_3d_comparison_figure(results_dict, save_path=None, elev=25, azim=45):
    """Create publication-quality 3D comparison figure."""
    plt.rcParams.update(PAPER_STYLE)
    
    methods = [m for m in METHOD_ORDER if m in results_dict]
    n_methods = len(methods)
    
    # Check if 3D
    sample_latent = list(results_dict.values())[0]['latents']
    if sample_latent.shape[1] != 3:
        print(f"Skipping 3D plot: latent dim is {sample_latent.shape[1]}")
        return
    
    fig = plt.figure(figsize=(5 * n_methods, 5))
    
    for i, method in enumerate(methods):
        res = results_dict[method]
        latents = res['latents']
        lineages = res['lineages']
        labels = res['labels']
        m = res['metrics']
        
        latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-8)
        
        ax = fig.add_subplot(1, n_methods, i + 1, projection='3d')
        
        # Plot cells by lineage
        for lin in LINEAGE_COLORS.keys():
            mask = lineages == lin
            if mask.sum() > 0:
                ax.scatter(latents_norm[mask, 0], latents_norm[mask, 1], latents_norm[mask, 2],
                          c=LINEAGE_COLORS[lin], s=8, alpha=0.5, label=lin, rasterized=True)
        
        # Draw trajectories
        centroids = compute_centroids(latents_norm, labels)
        draw_trajectory_arrows_3d(ax, centroids, PAUL15_TRAJECTORIES)
        
        # Mark progenitors
        for prog in ['7MEP', '9GMP']:
            if prog in centroids:
                c = centroids[prog]
                ax.scatter([c[0]], [c[1]], [c[2]], c='white', s=150, 
                          edgecolors='black', linewidths=2, zorder=10)
                ax.text(c[0], c[1], c[2], f'  {prog}', fontsize=9, fontweight='bold')
        
        dc = m.get('distance_correlation', 0)
        geo = m.get('geodesic_correlation', 0)
        pt = m.get('pseudotime_spearman', 0) or 0
        train_time = m.get('train_time', 0)
        
        title = f"{METHOD_NAMES.get(method, method)}\nDC={dc:.2f} Geo={geo:.2f} PT={pt:.2f}"
        ax.set_title(title, fontsize=12, fontweight='bold' if method == 'mmae' else 'normal')
        
        ax.set_xlabel('Z1')
        ax.set_ylabel('Z2')
        ax.set_zlabel('Z3')
        ax.view_init(elev=elev, azim=azim)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=8, frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# INTERACTIVE 3D (PLOTLY)
# =============================================================================

def create_interactive_3d_single(res, method, save_path):
    """Create interactive Plotly 3D plot for single method."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive plot")
        return
    
    latents = res['latents']
    lineages = res['lineages']
    labels = res['labels']
    m = res['metrics']
    
    if latents.shape[1] != 3:
        print(f"Skipping: latent dim is {latents.shape[1]}")
        return
    
    latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-8)
    
    fig = go.Figure()
    
    # Add traces for each lineage
    for lin in LINEAGE_COLORS_PLOTLY.keys():
        mask = lineages == lin
        if mask.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=latents_norm[mask, 0],
                y=latents_norm[mask, 1],
                z=latents_norm[mask, 2],
                mode='markers',
                marker=dict(size=3, color=LINEAGE_COLORS_PLOTLY[lin], opacity=0.6),
                name=lin,
                text=labels[mask],
                hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}'
            ))
    
    # Add trajectory lines
    centroids = compute_centroids(latents_norm, labels)
    
    for traj_name, traj_info in PAUL15_TRAJECTORIES.items():
        path = traj_info['path']
        color = traj_info['color']
        
        x_coords, y_coords, z_coords = [], [], []
        for label in path:
            if label in centroids:
                c = centroids[label]
                x_coords.append(c[0])
                y_coords.append(c[1])
                z_coords.append(c[2])
        
        if len(x_coords) > 1:
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines+markers',
                line=dict(color=color, width=5),
                marker=dict(size=8, color=color),
                name=f'{traj_name} trajectory',
                showlegend=False
            ))
    
    # Mark progenitors
    for prog in ['7MEP', '9GMP']:
        if prog in centroids:
            c = centroids[prog]
            fig.add_trace(go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]],
                mode='markers+text',
                marker=dict(size=12, color='white', line=dict(color='black', width=2)),
                text=[prog],
                textposition='top center',
                name=prog,
                showlegend=False
            ))
    
    dc = m.get('distance_correlation', 0)
    geo = m.get('geodesic_correlation', 0)
    pt = m.get('pseudotime_spearman', 0) or 0
    
    fig.update_layout(
        title=dict(
            text=f"{METHOD_NAMES.get(method, method)}<br>DC={dc:.3f} | Geo={geo:.3f} | PT={pt:.3f}",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='Z1',
            yaxis_title='Z2',
            zaxis_title='Z3',
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=800,
        height=700
    )
    
    fig.write_html(save_path)
    print(f"Saved interactive: {save_path}")


def create_interactive_3d_comparison(results_dict, save_path):
    """Create interactive Plotly comparison with all methods."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available")
        return
    
    methods = [m for m in METHOD_ORDER if m in results_dict]
    n_methods = len(methods)
    
    # Check 3D
    sample_latent = list(results_dict.values())[0]['latents']
    if sample_latent.shape[1] != 3:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=n_methods,
        specs=[[{'type': 'scatter3d'}] * n_methods],
        subplot_titles=[METHOD_NAMES.get(m, m) for m in methods],
        horizontal_spacing=0.02
    )
    
    for i, method in enumerate(methods):
        res = results_dict[method]
        latents = res['latents']
        lineages = res['lineages']
        labels = res['labels']
        
        latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-8)
        
        # Add points for each lineage
        for j, lin in enumerate(LINEAGE_COLORS_PLOTLY.keys()):
            mask = lineages == lin
            if mask.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=latents_norm[mask, 0],
                    y=latents_norm[mask, 1],
                    z=latents_norm[mask, 2],
                    mode='markers',
                    marker=dict(size=2, color=LINEAGE_COLORS_PLOTLY[lin], opacity=0.5),
                    name=lin,
                    legendgroup=lin,
                    showlegend=(i == 0),
                    text=labels[mask],
                ), row=1, col=i+1)
        
        # Add trajectories
        centroids = compute_centroids(latents_norm, labels)
        
        for traj_name, traj_info in PAUL15_TRAJECTORIES.items():
            path = traj_info['path']
            color = traj_info['color']
            
            x_coords, y_coords, z_coords = [], [], []
            for label in path:
                if label in centroids:
                    c = centroids[label]
                    x_coords.append(c[0])
                    y_coords.append(c[1])
                    z_coords.append(c[2])
            
            if len(x_coords) > 1:
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='lines',
                    line=dict(color=color, width=4),
                    showlegend=False
                ), row=1, col=i+1)
    
    fig.update_layout(
        title="3D Latent Space Comparison - Paul15 Hematopoiesis",
        height=600,
        width=400 * n_methods,
        showlegend=True
    )
    
    fig.write_html(save_path)
    print(f"Saved interactive comparison: {save_path}")


# =============================================================================
# COMBINED 2D vs 3D FIGURE
# =============================================================================

def create_2d_vs_3d_figure(results_2d, results_3d, save_path=None):
    """Create side-by-side 2D vs 3D comparison for MMAE."""
    plt.rcParams.update(PAPER_STYLE)
    
    if 'mmae' not in results_2d or 'mmae' not in results_3d:
        print("MMAE results required for 2D vs 3D comparison")
        return
    
    fig = plt.figure(figsize=(12, 5))
    
    # 2D plot
    ax1 = fig.add_subplot(1, 2, 1)
    res_2d = results_2d['mmae']
    latents_2d = res_2d['latents']
    latents_2d_norm = (latents_2d - latents_2d.mean(0)) / (latents_2d.std(0) + 1e-8)
    lineages = res_2d['lineages']
    labels = res_2d['labels']
    m_2d = res_2d['metrics']
    
    for lin in LINEAGE_COLORS.keys():
        mask = lineages == lin
        if mask.sum() > 0:
            ax1.scatter(latents_2d_norm[mask, 0], latents_2d_norm[mask, 1],
                       c=LINEAGE_COLORS[lin], s=10, alpha=0.6, label=lin, rasterized=True)
    
    centroids_2d = compute_centroids(latents_2d_norm, labels)
    draw_trajectory_arrows_2d(ax1, centroids_2d, PAUL15_TRAJECTORIES)
    
    dc_2d = m_2d.get('distance_correlation', 0)
    pt_2d = m_2d.get('pseudotime_spearman', 0) or 0
    ax1.set_title(f"MMAE 2D\nDC={dc_2d:.3f}  PT={pt_2d:.3f}", fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 3D plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    res_3d = results_3d['mmae']
    latents_3d = res_3d['latents']
    latents_3d_norm = (latents_3d - latents_3d.mean(0)) / (latents_3d.std(0) + 1e-8)
    lineages_3d = res_3d['lineages']
    labels_3d = res_3d['labels']
    m_3d = res_3d['metrics']
    
    for lin in LINEAGE_COLORS.keys():
        mask = lineages_3d == lin
        if mask.sum() > 0:
            ax2.scatter(latents_3d_norm[mask, 0], latents_3d_norm[mask, 1], latents_3d_norm[mask, 2],
                       c=LINEAGE_COLORS[lin], s=10, alpha=0.5, rasterized=True)
    
    centroids_3d = compute_centroids(latents_3d_norm, labels_3d)
    draw_trajectory_arrows_3d(ax2, centroids_3d, PAUL15_TRAJECTORIES)
    
    dc_3d = m_3d.get('distance_correlation', 0)
    pt_3d = m_3d.get('pseudotime_spearman', 0) or 0
    ax2.set_title(f"MMAE 3D\nDC={dc_3d:.3f}  PT={pt_3d:.3f}", fontsize=14, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    ax2.set_xlabel('Z1')
    ax2.set_ylabel('Z2')
    ax2.set_zlabel('Z3')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# METRICS TABLE
# =============================================================================

def create_metrics_table(results_dict, save_path=None):
    """Create clean LaTeX metrics table."""
    methods = [m for m in METHOD_ORDER if m in results_dict]
    
    # Find best values
    metrics_keys = ['distance_correlation', 'geodesic_correlation', 'pseudotime_spearman', 'train_time']
    best = {}
    for key in metrics_keys:
        vals = [(m, results_dict[m]['metrics'].get(key, np.nan)) for m in methods]
        vals = [(m, v) for m, v in vals if not np.isnan(v)]
        if vals:
            if key == 'train_time':
                best[key] = min(vals, key=lambda x: x[1])[0]
            else:
                best[key] = max(vals, key=lambda x: x[1])[0]
    
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Dist. Corr $\\uparrow$ & Geo. Corr $\\uparrow$ & PT Corr $\\uparrow$ & Time (s) $\\downarrow$ \\\\",
        "\\midrule"
    ]
    
    for method in methods:
        m = results_dict[method]['metrics']
        row = [METHOD_NAMES.get(method, method)]
        
        for key in metrics_keys:
            val = m.get(key, np.nan)
            if np.isnan(val):
                row.append("-")
            else:
                if key == 'train_time':
                    val_str = f"{val:.1f}"
                else:
                    val_str = f"{val:.3f}"
                
                if best.get(key) == method:
                    val_str = f"\\textbf{{{val_str}}}"
                row.append(val_str)
        
        lines.append(" & ".join(row) + " \\\\")
    
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    
    table = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
        print(f"Saved: {save_path}")
    
    return table


def print_summary(results_dict):
    """Print summary table to console."""
    methods = [m for m in METHOD_ORDER if m in results_dict]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<15} {'DC':>10} {'Geo DC':>10} {'PT':>10} {'Time':>10}")
    print("-" * 80)
    
    for method in methods:
        m = results_dict[method]['metrics']
        dc = m.get('distance_correlation', 0)
        geo = m.get('geodesic_correlation', 0)
        pt = m.get('pseudotime_spearman', 0) or 0
        t = m.get('train_time', 0)
        
        print(f"{METHOD_NAMES.get(method, method):<15} {dc:>10.3f} {geo:>10.3f} {pt:>10.3f} {t:>9.1f}s")
    
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improved Bio Experiments')
    parser.add_argument('--dataset', type=str, default='paul15', choices=['paul15'])
    parser.add_argument('--model', type=str, default='mmae',
                        choices=['vanilla', 'mmae', 'topoae', 'rtdae', 'geomae', 'ggae'])
    parser.add_argument('--all_models', action='store_true')
    parser.add_argument('--hyperparam_dir', type=str, default=None)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/bio_experiments_v2')
    parser.add_argument('--run_both_dims', action='store_true', 
                        help='Run both 2D and 3D for comparison')
    args = parser.parse_args()
    
    # Load data
    data, labels, label_ids, lineages, cell_type_order = load_paul15()
    
    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Models to run (focused set)
    if args.all_models:
        models_to_run = ['vanilla', 'mmae', 'topoae', 'rtdae']
    else:
        models_to_run = [args.model]
    
    def run_experiment(latent_dim):
        """Run experiment for given latent dimension."""
        config_overrides = {
            'latent_dim': latent_dim,
            'n_epochs': args.epochs,
        }
        
        results = {}
        
        for model_name in models_to_run:
            best_hyperparams = None
            if args.hyperparam_dir:
                best_hyperparams = load_best_config(args.hyperparam_dir, model_name, latent_dim)
            
            try:
                res = train_model(
                    model_name, data, labels, label_ids, lineages, cell_type_order,
                    config_overrides, best_hyperparams, args.device
                )
                results[model_name] = res
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    if args.run_both_dims:
        # Run both 2D and 3D
        print("\n" + "=" * 60)
        print("RUNNING 2D EXPERIMENTS")
        print("=" * 60)
        results_2d = run_experiment(2)
        
        print("\n" + "=" * 60)
        print("RUNNING 3D EXPERIMENTS")
        print("=" * 60)
        results_3d = run_experiment(3)
        
        # Create visualizations
        create_2d_comparison_figure(results_2d, os.path.join(save_dir, 'comparison_2d.png'))
        create_3d_comparison_figure(results_3d, os.path.join(save_dir, 'comparison_3d.png'))
        create_2d_vs_3d_figure(results_2d, results_3d, os.path.join(save_dir, 'mmae_2d_vs_3d.png'))
        
        # Interactive 3D
        os.makedirs(os.path.join(save_dir, 'interactive'), exist_ok=True)
        for method in results_3d:
            create_interactive_3d_single(
                results_3d[method], method,
                os.path.join(save_dir, 'interactive', f'{method}_3d.html')
            )
        create_interactive_3d_comparison(
            results_3d, os.path.join(save_dir, 'interactive', 'comparison_3d.html')
        )
        
        # Tables
        print("\n2D Results:")
        print_summary(results_2d)
        create_metrics_table(results_2d, os.path.join(save_dir, 'metrics_2d.tex'))
        
        print("\n3D Results:")
        print_summary(results_3d)
        create_metrics_table(results_3d, os.path.join(save_dir, 'metrics_3d.tex'))
        
    else:
        # Single dimension
        results = run_experiment(args.latent_dim)
        
        if args.latent_dim == 2:
            create_2d_comparison_figure(results, os.path.join(save_dir, 'comparison_2d.png'))
        else:
            create_3d_comparison_figure(results, os.path.join(save_dir, 'comparison_3d.png'))
            
            # Interactive
            os.makedirs(os.path.join(save_dir, 'interactive'), exist_ok=True)
            for method in results:
                create_interactive_3d_single(
                    results[method], method,
                    os.path.join(save_dir, 'interactive', f'{method}_3d.html')
                )
            create_interactive_3d_comparison(
                results, os.path.join(save_dir, 'interactive', 'comparison_3d.html')
            )
        
        print_summary(results)
        create_metrics_table(results, os.path.join(save_dir, f'metrics_{args.latent_dim}d.tex'))
    
    # Save raw results
    all_metrics = {}
    if args.run_both_dims:
        all_metrics['2d'] = {m: {k: float(v) if isinstance(v, (float, np.floating)) else v 
                                 for k, v in r['metrics'].items()} 
                            for m, r in results_2d.items()}
        all_metrics['3d'] = {m: {k: float(v) if isinstance(v, (float, np.floating)) else v 
                                 for k, v in r['metrics'].items()} 
                            for m, r in results_3d.items()}
    else:
        all_metrics[f'{args.latent_dim}d'] = {
            m: {k: float(v) if isinstance(v, (float, np.floating)) else v 
                for k, v in r['metrics'].items()} 
            for m, r in results.items()
        }
    
    with open(os.path.join(save_dir, 'all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()