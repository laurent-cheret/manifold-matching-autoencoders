#!/usr/bin/env python
"""
PCA Component Ablation: Compare MMAE at different PCA values against competitors.

Competitors use their best hyperparameters, MMAE varies PCA components.

Usage:
    python run_pca_comparison.py --dataset paul15 \
        --hyperparam_dir results/hyperparam_search/paul15 \
        --pca_values 20 50 100 200 500
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
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

from models import build_model
from training import Trainer
from evaluation import evaluate


# =============================================================================
# STYLE
# =============================================================================

PAPER_STYLE = {
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

LINEAGE_COLORS = {
    'Erythroid': '#d62728',
    'Megakaryocyte': '#ff7f0e',
    'Granulocyte': '#2ca02c',
    'Monocyte': '#1f77b4',
    'Basophil': '#9467bd',
    'Lymphoid': '#bcbd22',
}

COMPETITOR_COLORS = {
    'vanilla': '#7f7f7f',
    'topoae': '#1f77b4',
    'rtdae': '#2ca02c',
    'geomae': '#ff7f0e',
    'ggae': '#9467bd',
}

COMPETITOR_NAMES = {
    'vanilla': 'Vanilla',
    'topoae': 'TopoAE',
    'rtdae': 'RTD-AE',
    'geomae': 'GeomAE',
    'ggae': 'GGAE',
}


# =============================================================================
# DATA
# =============================================================================

def load_paul15():
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy required")
    
    print("Loading Paul15...")
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    data = adata.X
    if hasattr(data, 'toarray'):
        data = data.toarray()
    
    labels = adata.obs['paul15_clusters'].values.astype(str)
    unique_labels = np.unique(labels)
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    label_ids = np.array([label_to_id[l] for l in labels])
    
    cell_type_order = {
        '1Ery': 0, '2Ery': 1, '3Ery': 2, '4Ery': 3, '5Ery': 4, '6Ery': 5,
        '7MEP': 6, '8Mk': 7, '9GMP': 8, '10GMP': 9, '11DC': 10,
        '12Baso': 11, '13Baso': 12, '14Mo': 13, '15Mo': 14,
        '16Neu': 15, '17Neu': 16, '18Eos': 17, '19Lymph': 18,
    }
    
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
    
    lineages = np.array([lineage_map.get(l, 'Other') for l in labels])
    
    print(f"  {data.shape[0]} cells, {data.shape[1]} genes")
    return data, labels, label_ids, lineages, cell_type_order


# =============================================================================
# UTILS
# =============================================================================

def load_best_config(hyperparam_dir, model_name, latent_dim=2):
    config_path = os.path.join(hyperparam_dir, f'{model_name}_dim{latent_dim}', 'best_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)['hyperparameters']
    return None


def compute_pseudotime(latents, labels, root_types=None):
    if root_types is None:
        root_types = ['7MEP', '9GMP', '10GMP']
    root_mask = np.isin(labels, root_types)
    if root_mask.sum() == 0:
        root_idx = 0
    else:
        root_indices = np.where(root_mask)[0]
        root_idx = root_indices[len(root_indices) // 2]
    distances = np.sqrt(np.sum((latents - latents[root_idx]) ** 2, axis=1))
    return (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)


def evaluate_pseudotime(pseudotime, labels, cell_type_order):
    expected_order = np.array([cell_type_order.get(l, -1) for l in labels])
    valid_mask = expected_order >= 0
    if valid_mask.sum() < 10:
        return np.nan
    spearman, _ = spearmanr(pseudotime[valid_mask], expected_order[valid_mask])
    return spearman


def evaluate_branch_separation(latents, lineages):
    unique_lineages = np.unique(lineages)
    centroids = {lin: latents[lineages == lin].mean(axis=0) for lin in unique_lineages}
    
    inter_distances = []
    for i, lin1 in enumerate(unique_lineages):
        for lin2 in unique_lineages[i+1:]:
            inter_distances.append(np.linalg.norm(centroids[lin1] - centroids[lin2]))
    
    intra_spreads = []
    for lin in unique_lineages:
        mask = lineages == lin
        if mask.sum() > 1:
            intra_spreads.append(np.mean(np.linalg.norm(latents[mask] - centroids[lin], axis=1)))
    
    return np.mean(inter_distances) / (np.mean(intra_spreads) + 1e-8)


class BioDataset(torch.utils.data.Dataset):
    def __init__(self, data, pca_embeddings=None, return_indices=False):
        self.data = torch.FloatTensor(data)
        self.pca_embeddings = torch.FloatTensor(pca_embeddings) if pca_embeddings is not None else None
        self.return_indices = return_indices
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.pca_embeddings is not None:
            return self.data[idx], self.pca_embeddings[idx], idx
        elif self.return_indices:
            return self.data[idx], idx, idx
        return self.data[idx], idx


def compute_auto_bandwidth(data, n_samples=1000):
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        subset = data[indices]
    else:
        subset = data
    diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    mask = sq_dists > 0
    return float(np.median(sq_dists[mask])) if mask.sum() > 0 else 1.0


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model_name, data, labels, label_ids, lineages, cell_type_order,
                config, device='cuda'):
    print(f"\n  Training {model_name}...")
    
    # Prepare data
    pca_embeddings = None
    if model_name == 'mmae':
        from sklearn.decomposition import PCA
        n_comp = config.get('mmae_n_components', 100)
        pca = PCA(n_components=min(n_comp, data.shape[1], data.shape[0]))
        pca_embeddings = pca.fit_transform(data)
        print(f"    PCA: {data.shape[1]} → {pca_embeddings.shape[1]} ({pca.explained_variance_ratio_.sum()*100:.1f}% var)")
    
    return_indices = model_name == 'ggae'
    dataset = BioDataset(data, pca_embeddings, return_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    model = build_model(model_name, config)
    
    if model_name == 'ggae':
        data_tensor = torch.FloatTensor(data).to(device)
        model.precompute_kernel(data_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config.get('weight_decay', 1e-5))
    trainer = Trainer(model, optimizer, device, model_name=model_name)
    
    start_time = time.time()
    trainer.fit(loader, loader, n_epochs=config['n_epochs'], verbose=False)
    train_time = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        latents = model.encode(torch.FloatTensor(data).to(device)).cpu().numpy()
    
    metrics = evaluate(data, latents, label_ids)
    metrics['train_time'] = train_time
    
    pseudotime = compute_pseudotime(latents, labels)
    metrics['pseudotime_spearman'] = evaluate_pseudotime(pseudotime, labels, cell_type_order)
    metrics['branch_separation'] = evaluate_branch_separation(latents, lineages)
    
    n_clusters = len(np.unique(label_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(latents)
    metrics['ARI'] = adjusted_rand_score(label_ids, pred)
    
    print(f"    DC={metrics['distance_correlation']:.3f}, PT={metrics['pseudotime_spearman']:.3f}, "
          f"Branch={metrics['branch_separation']:.2f}, Time={train_time:.1f}s")
    
    return {'latents': latents, 'lineages': lineages, 'pseudotime': pseudotime, 'metrics': metrics}


# =============================================================================
# FIGURE
# =============================================================================

def create_comparison_figure(mmae_results, competitor_results, pca_values, save_path):
    """
    Create figure with:
    - Row 1: MMAE at different PCA values (latent spaces)
    - Row 2: Competitors (latent spaces)
    - Row 3: Metrics comparison
    """
    plt.rcParams.update(PAPER_STYLE)
    
    n_pca = len(pca_values)
    n_comp = len(competitor_results)
    n_cols = max(n_pca, n_comp)
    
    fig = plt.figure(figsize=(2.5 * n_cols, 8))
    gs = GridSpec(3, n_cols, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.1)
    
    # Collect all latents for consistent limits
    all_latents = []
    for pca in pca_values:
        if pca in mmae_results:
            l = mmae_results[pca]['latents']
            all_latents.append((l - l.mean(0)) / (l.std(0) + 1e-8))
    for name, res in competitor_results.items():
        l = res['latents']
        all_latents.append((l - l.mean(0)) / (l.std(0) + 1e-8))
    
    all_concat = np.vstack(all_latents)
    lim = max(abs(all_concat.min()), abs(all_concat.max())) * 1.1
    
    # Row 1: MMAE variants
    for i, pca in enumerate(pca_values):
        if pca not in mmae_results:
            continue
        res = mmae_results[pca]
        latents = res['latents']
        latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-8)
        lineages = res['lineages']
        m = res['metrics']
        
        ax = fig.add_subplot(gs[0, i])
        for lin in LINEAGE_COLORS.keys():
            mask = lineages == lin
            if mask.sum() > 0:
                ax.scatter(latents_norm[mask, 0], latents_norm[mask, 1],
                          c=LINEAGE_COLORS[lin], s=5, alpha=0.6, rasterized=True)
        
        dc = m['distance_correlation']
        pt = m['pseudotime_spearman']
        br = m['branch_separation']
        ax.set_title(f"MMAE-{pca}\nDC={dc:.2f} PT={pt:.2f}", fontsize=10, fontweight='bold')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        if i == 0:
            ax.set_ylabel('MMAE (varying PCA)', fontsize=11, fontweight='bold')
    
    # Row 2: Competitors
    comp_names = list(competitor_results.keys())
    for i, name in enumerate(comp_names):
        if i >= n_cols:
            break
        res = competitor_results[name]
        latents = res['latents']
        latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-8)
        lineages = res['lineages']
        m = res['metrics']
        
        ax = fig.add_subplot(gs[1, i])
        for lin in LINEAGE_COLORS.keys():
            mask = lineages == lin
            if mask.sum() > 0:
                ax.scatter(latents_norm[mask, 0], latents_norm[mask, 1],
                          c=LINEAGE_COLORS[lin], s=5, alpha=0.6, rasterized=True)
        
        dc = m['distance_correlation']
        pt = m['pseudotime_spearman']
        if np.isnan(pt):
            pt = 0
        ax.set_title(f"{COMPETITOR_NAMES.get(name, name)}\nDC={dc:.2f} PT={pt:.2f}", fontsize=10)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        if i == 0:
            ax.set_ylabel('Competitors', fontsize=11, fontweight='bold')
    
    # Legend
    lineage_handles = [mpatches.Patch(color=c, label=l) for l, c in LINEAGE_COLORS.items()]
    fig.legend(handles=lineage_handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=6, fontsize=8, frameon=False)
    
    # Row 3: Metrics bar chart
    ax_bar = fig.add_subplot(gs[2, :])
    
    # Prepare data
    all_methods = [f"MMAE-{p}" for p in pca_values] + [COMPETITOR_NAMES.get(n, n) for n in comp_names]
    all_colors = ['#d62728'] * len(pca_values) + [COMPETITOR_COLORS.get(n, '#999') for n in comp_names]
    
    metrics_to_plot = ['distance_correlation', 'pseudotime_spearman', 'branch_separation']
    metric_labels = ['Distance Corr.', 'Pseudotime', 'Branch Sep.']
    
    x = np.arange(len(all_methods))
    width = 0.25
    
    for j, (key, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = []
        for pca in pca_values:
            if pca in mmae_results:
                v = mmae_results[pca]['metrics'].get(key, 0)
                if key == 'branch_separation':
                    v = min(v / 5.0, 1.0)
                values.append(v if not np.isnan(v) else 0)
        for name in comp_names:
            v = competitor_results[name]['metrics'].get(key, 0)
            if key == 'branch_separation':
                v = min(v / 5.0, 1.0)
            values.append(v if not np.isnan(v) else 0)
        
        offset = (j - 1) * width
        bars = ax_bar.bar(x + offset, values, width, label=label, alpha=0.8)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
    ax_bar.set_ylabel('Score', fontsize=11)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend(loc='upper right', fontsize=9, ncol=3)
    ax_bar.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add vertical line separating MMAE from competitors
    ax_bar.axvline(x=len(pca_values) - 0.5, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_tradeoff_figure(mmae_results, competitor_results, pca_values, save_path):
    """Create figure showing DC vs PT tradeoff with PCA as parameter."""
    plt.rcParams.update(PAPER_STYLE)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # MMAE trajectory
    mmae_dc = [mmae_results[p]['metrics']['distance_correlation'] for p in pca_values if p in mmae_results]
    mmae_pt = [mmae_results[p]['metrics']['pseudotime_spearman'] for p in pca_values if p in mmae_results]
    valid_pca = [p for p in pca_values if p in mmae_results]
    
    ax.plot(mmae_dc, mmae_pt, 'o-', color='#d62728', linewidth=2, markersize=10, label='MMAE', zorder=10)
    
    # Annotate PCA values
    for i, pca in enumerate(valid_pca):
        ax.annotate(f'{pca}', (mmae_dc[i], mmae_pt[i]), textcoords="offset points",
                   xytext=(8, 5), fontsize=9, color='#d62728')
    
    # Competitors as individual points
    for name, res in competitor_results.items():
        m = res['metrics']
        dc = m['distance_correlation']
        pt = m['pseudotime_spearman']
        if np.isnan(pt):
            pt = 0
        ax.scatter([dc], [pt], s=120, c=COMPETITOR_COLORS.get(name, '#999'),
                  marker='s', label=COMPETITOR_NAMES.get(name, name), zorder=5)
    
    ax.set_xlabel('Distance Correlation ↑', fontsize=12)
    ax.set_ylabel('Pseudotime Correlation ↑', fontsize=12)
    ax.set_title('Topology-Biology Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Ideal corner annotation
    ax.annotate('← Better overall', xy=(0.85, 0.75), fontsize=10, color='gray',
               xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='paul15')
    parser.add_argument('--hyperparam_dir', type=str, required=True)
    parser.add_argument('--pca_values', type=int, nargs='+', default=[20, 50, 100, 200, 500])
    parser.add_argument('--competitors', type=str, nargs='+', 
                        default=['vanilla', 'topoae', 'rtdae', 'geomae'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/pca_comparison')
    args = parser.parse_args()
    
    # Load data
    data, labels, label_ids, lineages, cell_type_order = load_paul15()
    
    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    base_config = {
        'input_dim': data.shape[1],
        'latent_dim': 2,
        'hidden_dims': [512, 256, 128],
        'n_epochs': args.epochs,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
    }
    
    # Load MMAE best config (for lr, batch_size, lambda - not PCA)
    mmae_best = load_best_config(args.hyperparam_dir, 'mmae') or {}
    
    # Train MMAE at different PCA values
    print("\n" + "="*60)
    print("MMAE PCA ABLATION")
    print("="*60)
    
    mmae_results = {}
    for pca in args.pca_values:
        if pca > data.shape[1]:
            continue
        print(f"\nPCA = {pca}")
        
        config = base_config.copy()
        config['mmae_n_components'] = pca
        config['mmae_weight'] = mmae_best.get('mmae_lambda', 1.0)
        if 'learning_rate' in mmae_best:
            config['learning_rate'] = mmae_best['learning_rate']
        if 'batch_size' in mmae_best:
            config['batch_size'] = int(mmae_best['batch_size'])
        
        mmae_results[pca] = train_model('mmae', data, labels, label_ids, lineages,
                                         cell_type_order, config, args.device)
    
    # Train competitors with their best configs
    print("\n" + "="*60)
    print("COMPETITORS (Best Configs)")
    print("="*60)
    
    competitor_results = {}
    for name in args.competitors:
        best = load_best_config(args.hyperparam_dir, name) or {}
        print(f"\n{COMPETITOR_NAMES.get(name, name)}: {best}")
        
        config = base_config.copy()
        if 'learning_rate' in best:
            config['learning_rate'] = best['learning_rate']
        if 'batch_size' in best:
            config['batch_size'] = int(best['batch_size'])
        
        # Model-specific params
        if name == 'topoae':
            config['topoae_weight'] = best.get('topo_lambda', 1.0)
        elif name == 'rtdae':
            config['rtd_weight'] = best.get('rtd_lambda', 1.0)
        elif name == 'geomae':
            config['geomae_weight'] = best.get('geom_lambda', 1.0)
        elif name == 'ggae':
            config['ggae_lambda'] = best.get('gg_lambda', 1.0)
            config['ggae_bandwidth'] = best.get('gg_bandwidth', compute_auto_bandwidth(data))
        
        try:
            competitor_results[name] = train_model(name, data, labels, label_ids, lineages,
                                                    cell_type_order, config, args.device)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Create figures
    create_comparison_figure(mmae_results, competitor_results, args.pca_values,
                            os.path.join(save_dir, 'pca_comparison.png'))
    
    create_tradeoff_figure(mmae_results, competitor_results, args.pca_values,
                          os.path.join(save_dir, 'tradeoff.png'))
    
    # Save metrics
    all_metrics = {
        'mmae': {str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                         for kk, vv in v['metrics'].items()}
                 for k, v in mmae_results.items()},
        'competitors': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                           for kk, vv in v['metrics'].items()}
                       for k, v in competitor_results.items()}
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<15} {'DC':>8} {'PT':>8} {'Branch':>8} {'Time':>8}")
    print("-"*80)
    
    for pca in args.pca_values:
        if pca not in mmae_results:
            continue
        m = mmae_results[pca]['metrics']
        print(f"{'MMAE-'+str(pca):<15} {m['distance_correlation']:>8.3f} "
              f"{m['pseudotime_spearman']:>8.3f} {m['branch_separation']:>8.2f} "
              f"{m['train_time']:>7.1f}s")
    
    print("-"*80)
    
    for name in args.competitors:
        if name not in competitor_results:
            continue
        m = competitor_results[name]['metrics']
        pt = m['pseudotime_spearman']
        if np.isnan(pt):
            pt = 0
        print(f"{COMPETITOR_NAMES.get(name, name):<15} {m['distance_correlation']:>8.3f} "
              f"{pt:>8.3f} {m['branch_separation']:>8.2f} {m['train_time']:>7.1f}s")
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()