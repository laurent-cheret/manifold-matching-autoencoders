#!/usr/bin/env python
"""
Pseudotime Neighborhood Consistency (PNC) metric for trajectory preservation.

For each cell, finds k-NN in latent space and measures std of pseudotime among neighbors.
Lower = better (neighbors are developmentally similar).

Usage:
    python compute_trajectory_metric.py
    python compute_trajectory_metric.py --k 10 20 50
    python compute_trajectory_metric.py --best_configs_dir experiments/hyperparam_search/paul15/results
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Project imports
from config import get_config
from models import build_model
from training import Trainer, get_latents
from data.base import normalize_features, compute_pca_embeddings, create_dataloaders


def load_best_config(best_configs_dir, model_name, latent_dim=2):
    """Load best hyperparameters from hyperparameter search results."""
    if best_configs_dir is None:
        return None
    config_path = os.path.join(best_configs_dir, f'{model_name}_dim{latent_dim}', 'best_config.json')
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            return json.load(f).get('hyperparameters', {})
    except:
        return None


def apply_best_config(config, best_params, model_name):
    """Apply best hyperparameters to config."""
    if best_params is None:
        return config
    if 'learning_rate' in best_params:
        config['learning_rate'] = best_params['learning_rate']
    if 'batch_size' in best_params:
        config['batch_size'] = int(best_params['batch_size'])
    
    if model_name == 'mmae':
        if 'mmae_lambda' in best_params: config['mmae_lambda'] = best_params['mmae_lambda']
        if 'mmae_n_components' in best_params: config['mmae_n_components'] = int(best_params['mmae_n_components'])
    elif model_name == 'topoae':
        if 'topo_lambda' in best_params: config['topo_lambda'] = best_params['topo_lambda']
    elif model_name == 'rtdae':
        if 'rtd_lambda' in best_params: config['rtd_lambda'] = best_params['rtd_lambda']
        if 'rtd_dim' in best_params: config['rtd_dim'] = int(best_params['rtd_dim'])
        if 'rtd_card' in best_params: config['rtd_card'] = int(best_params['rtd_card'])
    elif model_name == 'geomae':
        if 'geom_lambda' in best_params: config['geom_lambda'] = best_params['geom_lambda']
    elif model_name == 'ggae':
        if 'gg_lambda' in best_params: config['ggae_lambda'] = best_params['gg_lambda']
        if 'gg_bandwidth' in best_params: config['ggae_bandwidth'] = best_params['gg_bandwidth']
    return config


def load_paul15_with_pseudotime(config, with_embeddings=False, return_indices=False):
    """Load Paul15 and compute diffusion pseudotime."""
    import scanpy as sc
    from sklearn.model_selection import train_test_split
    
    seed = config.get("seed", 42)
    
    adata = sc.datasets.paul15()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.diffmap(adata)
    
    cluster_labels = adata.obs['paul15_clusters'].cat.codes.values
    dc1 = adata.obsm['X_diffmap'][:, 0]
    cluster_means = [dc1[cluster_labels == c].mean() for c in np.unique(cluster_labels)]
    root_cluster = np.unique(cluster_labels)[np.argmin(cluster_means)]
    root_cell = np.where(cluster_labels == root_cluster)[0][np.argmin(dc1[cluster_labels == root_cluster])]
    
    adata.uns['iroot'] = root_cell
    sc.tl.dpt(adata)
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X = X.astype(np.float32)
    pseudotime = np.nan_to_num(adata.obs['dpt_pseudotime'].values, nan=1.0).astype(np.float32)
    
    train_data, test_data, train_pt, test_pt = train_test_split(
        X, pseudotime, test_size=config.get("val_size", 0.15), random_state=seed
    )
    train_data, test_data = normalize_features(train_data, test_data)
    
    train_emb, test_emb = None, None
    if with_embeddings:
        train_emb, test_emb = compute_pca_embeddings(train_data, test_data, config.get('mmae_n_components', 50))
    
    return create_dataloaders(
        train_data, test_data, train_pt, test_pt,
        batch_size=config.get('batch_size', 256),
        train_emb=train_emb, test_emb=test_emb, return_indices=return_indices
    )


def pseudotime_neighborhood_consistency(latents, pseudotime, k=10):
    """
    Compute Pseudotime Neighborhood Consistency (PNC).
    
    For each cell, find k-NN in latent space, compute std of pseudotime among neighbors.
    Lower = better (neighbors are developmentally similar).
    
    Returns:
        mean_std: Average pseudotime std across all cells
        all_stds: Per-cell pseudotime std values
    """
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 because cell is its own neighbor
    nn.fit(latents)
    _, indices = nn.kneighbors(latents)
    
    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1:]
    
    # Compute std of pseudotime for each cell's neighbors
    all_stds = []
    for i in range(len(latents)):
        neighbor_pt = pseudotime[neighbor_indices[i]]
        all_stds.append(np.std(neighbor_pt))
    
    all_stds = np.array(all_stds)
    return np.mean(all_stds), all_stds


def train_and_get_latents(model_name, config, train_loader, test_loader, device, epochs):
    """Train model and return full dataset latents."""
    model = build_model(model_name, config)
    
    if model_name == 'ggae':
        train_data = train_loader.dataset.data.view(len(train_loader.dataset), -1)
        model.precompute_kernel(train_data.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    trainer = Trainer(model, optimizer, device, model_name=model_name)
    trainer.fit(train_loader, test_loader, n_epochs=epochs, verbose=False)
    
    latents_train, pt_train = get_latents(model, train_loader, device)
    latents_test, pt_test = get_latents(model, test_loader, device)
    
    latents = np.concatenate([latents_train, latents_test], axis=0)
    pseudotime = np.concatenate([pt_train, pt_test], axis=0)
    
    return latents, pseudotime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20, 50], help='k values for k-NN')
    parser.add_argument('--pca_components', type=int, default=50)
    parser.add_argument('--best_configs_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/paul15_trajectory')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    MODELS = ['vanilla', 'mmae', 'topoae', 'rtdae', 'geomae', 'ggae']
    NAMES = {'vanilla': 'Vanilla AE', 'mmae': 'MMAE (Ours)', 'topoae': 'TopoAE',
             'rtdae': 'RTD-AE', 'geomae': 'GeomAE', 'ggae': 'GGAE'}
    
    base_config = {
        'input_dim': 2000, 'latent_dim': args.latent_dim, 'hidden_dims': [512, 256, 128],
        'arch_type': 'mlp', 'batch_size': args.batch_size, 'learning_rate': 1e-3,
        'seed': args.seed, 'val_size': 0.15, 'n_top_genes': 2000,
        'mmae_lambda': 1.0, 'mmae_n_components': args.pca_components,
        'topo_lambda': 1.0, 'rtd_lambda': 1.0, 'rtd_dim': 1, 'rtd_card': 50,
        'geom_lambda': 1.0, 'ggae_lambda': 1.0, 'ggae_bandwidth': None,
    }
    
    # Store results
    results = {model: {} for model in MODELS}
    
    print("="*70)
    print("PSEUDOTIME NEIGHBORHOOD CONSISTENCY (PNC)")
    print("Lower = better (neighbors are developmentally similar)")
    print("="*70)
    
    for model_name in MODELS:
        print(f"\n>>> Training {NAMES[model_name]}...")
        
        config = base_config.copy()
        best_params = load_best_config(args.best_configs_dir, model_name, args.latent_dim)
        config = apply_best_config(config, best_params, model_name)
        
        is_mmae = model_name == 'mmae'
        is_ggae = model_name == 'ggae'
        
        try:
            train_loader, test_loader, train_ds, _ = load_paul15_with_pseudotime(
                config, with_embeddings=is_mmae, return_indices=is_ggae
            )
            
            if is_ggae and config.get('ggae_bandwidth') is None:
                train_data = train_ds.data.view(len(train_ds), -1)
                idx = torch.randperm(len(train_data))[:min(1000, len(train_data))]
                dist_sq = torch.cdist(train_data[idx], train_data[idx]).pow(2)
                config['ggae_bandwidth'] = dist_sq[dist_sq > 0].median().item()
            
            latents, pseudotime = train_and_get_latents(
                model_name, config, train_loader, test_loader, args.device, args.epochs
            )
            
            # Compute PNC for each k
            for k in args.k:
                pnc, _ = pseudotime_neighborhood_consistency(latents, pseudotime, k=k)
                results[model_name][f'PNC_k{k}'] = pnc
            
            results[model_name]['latents'] = latents
            results[model_name]['pseudotime'] = pseudotime
            
        except Exception as e:
            print(f"    Failed: {e}")
            for k in args.k:
                results[model_name][f'PNC_k{k}'] = np.nan
    
    # === Print Results Table ===
    print("\n" + "="*70)
    print("RESULTS: Pseudotime Neighborhood Consistency (PNC)")
    print("="*70)
    
    # Build table
    header = ['Method'] + [f'PNC (k={k})↓' for k in args.k]
    print(f"{'Method':<15}" + "".join([f"{'PNC(k='+str(k)+')↓':>12}" for k in args.k]))
    print("-"*70)
    
    table_data = []
    for model_name in MODELS:
        row = [NAMES[model_name]]
        for k in args.k:
            val = results[model_name].get(f'PNC_k{k}', np.nan)
            row.append(val)
        table_data.append(row)
        print(f"{NAMES[model_name]:<15}" + "".join([f"{val:>12.4f}" if not np.isnan(val) else f"{'N/A':>12}" for val in row[1:]]))
    
    # Save to CSV
    df = pd.DataFrame(table_data, columns=header)
    df.to_csv(os.path.join(args.output_dir, 'pnc_results.csv'), index=False)
    print(f"\nResults saved to {args.output_dir}/pnc_results.csv")
    
    # === Bar Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(MODELS))
    width = 0.8 / len(args.k)
    
    for i, k in enumerate(args.k):
        values = [results[m].get(f'PNC_k{k}', np.nan) for m in MODELS]
        offset = (i - len(args.k)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f'k={k}')
    
    ax.set_ylabel('PNC (lower = better)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Pseudotime Neighborhood Consistency on Paul15', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m] for m in MODELS], rotation=15, ha='right')
    ax.legend(title='Neighbors')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pnc_comparison.png'), dpi=150, facecolor='white')
    plt.savefig(os.path.join(args.output_dir, 'pnc_comparison.pdf'), facecolor='white')
    print(f"Plot saved to {args.output_dir}/pnc_comparison.png")
    
    plt.show()


if __name__ == '__main__':
    main()