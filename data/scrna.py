"""scRNA-seq and single-cell datasets for demonstrating topology preservation benefits."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings, split_train_val_test


# =============================================================================
# PBMC3k - Peripheral Blood Mononuclear Cells (scRNA-seq)
# =============================================================================

def load_pbmc3k_data(n_top_genes=2000, seed=42):
    """Load and preprocess PBMC3k dataset from scanpy."""
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required: pip install scanpy")
    
    adata = sc.datasets.pbmc3k_processed()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    
    labels = adata.obs['louvain'].cat.codes.values.astype(np.float32)
    cell_types = adata.obs['louvain'].cat.categories.tolist()
    
    print(f"PBMC3k: {X.shape[0]} cells, {X.shape[1]} genes")
    print(f"Cell types ({len(cell_types)}): {cell_types}")
    print(f"Cells per type: {np.bincount(labels.astype(int))}")
    
    return X, labels, cell_types


@register_dataset("pbmc3k")
def load_pbmc3k(config, with_embeddings=False, return_indices=False):
    """Load PBMC3k scRNA-seq dataset."""
    seed = config.get("seed", 42)
    
    data, labels, cell_types = load_pbmc3k_data(seed=seed)
    
    n_samples = config.get('n_samples', None)
    if n_samples is not None and n_samples < len(data):
        np.random.seed(seed)
        idx = np.random.choice(len(data), n_samples, replace=False)
        data = data[idx]
        labels = labels[idx]
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        data, labels, seed=seed
    )

    train_data = train_data.astype(np.float32)
    val_data = val_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", config.get("input_dim"))
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_data, test_data, n_components, seed=seed, val_data=val_data
        )
        print(f"Computed PCA embeddings with {n_components} components")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )


# =============================================================================
# Paul15 - Bone Marrow Hematopoiesis (scRNA-seq, trajectory structure)
# =============================================================================

def load_paul15_data(n_top_genes=2000, seed=42):
    """Load Paul et al. 2015 bone marrow differentiation dataset."""
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required: pip install scanpy")
    
    adata = sc.datasets.paul15()
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    
    labels = adata.obs['paul15_clusters'].cat.codes.values.astype(np.float32)
    cell_types = adata.obs['paul15_clusters'].cat.categories.tolist()
    
    print(f"Paul15: {X.shape[0]} cells, {X.shape[1]} genes")
    print(f"Cell types ({len(cell_types)}): {cell_types[:5]}... (showing first 5)")
    print(f"Cells per type: {np.bincount(labels.astype(int))}")
    
    return X, labels, cell_types


@register_dataset("paul15")
def load_paul15(config, with_embeddings=False, return_indices=False):
    """Load Paul15 hematopoiesis dataset."""
    seed = config.get("seed", 42)
    n_top_genes = config.get("n_top_genes", 2000)
    
    data, labels, cell_types = load_paul15_data(n_top_genes=n_top_genes, seed=seed)
    
    n_samples = config.get('n_samples', None)
    if n_samples is not None and n_samples < len(data):
        np.random.seed(seed)
        idx = np.random.choice(len(data), n_samples, replace=False)
        data = data[idx]
        labels = labels[idx]
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        data, labels, seed=seed
    )

    train_data, val_data, test_data = normalize_features(train_data, test_data, val_data=val_data)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", config.get("input_dim"))
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_data, test_data, n_components, seed=seed, val_data=val_data
        )
        print(f"Computed PCA embeddings with {n_components} components")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )


# =============================================================================
# Levine32 - Mass Cytometry / CyTOF (protein markers, large scale)
# =============================================================================

def load_levine32_data(seed=42, max_cells=50000):
    """Load Levine et al. 2015 CyTOF dataset (32 protein markers)."""
    import os
    import urllib.request
    
    cache_dir = os.path.expanduser("~/.cache/mmae_data")
    os.makedirs(cache_dir, exist_ok=True)
    
    data_path = os.path.join(cache_dir, "levine32.npz")
    
    if not os.path.exists(data_path):
        print("Downloading Levine32 dataset...")
        
        X, labels = None, None
        
        try:
            import scprep
            print("  Loading via scprep...")
            X, labels = scprep.io.load_csv(
                scprep.io.download.download_url(
                    "https://ndownloader.figshare.com/files/10034578",
                    "levine32.csv"
                ),
                cell_axis='row',
                gene_names=True
            )
            if hasattr(X, 'values'):
                X = X.values
            if X.shape[1] == 33:
                labels = X[:, -1]
                X = X[:, :-1]
        except Exception as e:
            print(f"  scprep failed: {e}")
        
        if X is None:
            import pandas as pd
            
            figshare_url = "https://ndownloader.figshare.com/files/10034578"
            csv_path = os.path.join(cache_dir, "levine32.csv")
            
            print(f"  Downloading from Figshare...")
            try:
                urllib.request.urlretrieve(figshare_url, csv_path)
                df = pd.read_csv(csv_path)
                
                if 'label' in df.columns:
                    labels = df['label'].values
                    X = df.drop('label', axis=1).values
                elif 'population' in df.columns:
                    labels = df['population'].values  
                    X = df.drop('population', axis=1).values
                else:
                    labels = df.iloc[:, -1].values
                    X = df.iloc[:, :-1].values
                
                os.remove(csv_path)
            except Exception as e:
                print(f"  Figshare failed: {e}")
                raise RuntimeError(
                    "Could not download Levine32. Options:\n"
                    "1. pip install scprep\n"
                    "2. Manually download from: https://flowrepository.org/id/FR-FCM-ZZPH"
                )
        
        cofactor = 5
        X = np.arcsinh(X / cofactor).astype(np.float32)
        labels = labels.astype(np.float32)
        
        np.savez(data_path, X=X, labels=labels)
        print(f"  Saved to {data_path}")
    else:
        print("  Loading cached data...")
        data = np.load(data_path)
        X = data['X']
        labels = data['labels']
    
    valid_mask = ~np.isnan(labels) & (labels >= 0)
    X = X[valid_mask]
    labels = labels[valid_mask]
    
    label_map = {old: new for new, old in enumerate(np.unique(labels))}
    labels = np.array([label_map[l] for l in labels], dtype=np.float32)
    
    if len(X) > max_cells:
        np.random.seed(seed)
        idx = np.random.choice(len(X), max_cells, replace=False)
        X = X[idx]
        labels = labels[idx]
    
    print(f"Levine32: {X.shape[0]} cells, {X.shape[1]} protein markers")
    print(f"Cell populations: {len(np.unique(labels))}")
    print(f"Cells per population: {np.bincount(labels.astype(int))}")
    
    return X, labels
