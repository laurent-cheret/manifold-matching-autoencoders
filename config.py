"""Configuration system with dataset and model registries."""

# Base training config
BASE_CONFIG = {
    "batch_size": 256,
    "n_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "device": "cuda",
    "seed": 42,
    "val_size": 0.15,
    "k_neighbors": [10, 50, 100],
}

# Dataset-specific configs
# Using DeepAE-style MLP (1000-500-250) for image datasets to match TopoAE/RTD-AE papers
DATASET_CONFIGS = {
    "spheres": {
        "input_dim": 101,
        "n_samples": 500,
        "d": 100,
        "n_spheres": 11,
        "r": 5,
        "latent_dim": 2,
        "hidden_dims": [64, 64],
        "arch_type": "mlp",
        "input_shape": None,
    },
    "mnist": {
        "input_dim": 784,
        "latent_dim": 2,
        "hidden_dims": [1000, 500, 250],
        "arch_type": "mlp",
        "input_shape": (1, 28, 28),
        "n_samples": 20000,
    },
    "fmnist": {
        "input_dim": 784,
        "latent_dim": 2,
        "hidden_dims": [1000, 500, 250],
        "arch_type": "mlp",
        "input_shape": (1, 28, 28),
        "n_samples": 20000,
    },
    "cifar10": {
        "input_dim": 3072,
        "latent_dim": 2,
        "hidden_dims": [1000, 500, 250],
        "arch_type": "conv", #"mlp",
        "input_shape": (3, 32, 32),
        "n_samples": None,
    },
    "swiss_roll": {
        "input_dim": 100,
        "latent_dim": 2,
        "hidden_dims": [256, 128, 64],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": 30000,
        "n_classes": 10,
    },
    "concentric_spheres": {
        "input_dim": 1000,
        "latent_dim": 2,
        "hidden_dims": [512, 256, 128],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples_per_shell": 500,
        "n_shells": 5,
        "d": 1000,
    },
    "tree_clusters": {
        "input_dim": 100,
        "latent_dim": 2,
        "hidden_dims": [256, 128, 64],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples_per_cluster": 500,
        "n_levels": 3,
        "branch_factor": 2,
        "d": 100,
    },
    "linked_tori": {
        "input_dim": 100,
        "latent_dim": 2,
        "hidden_dims": [256, 128, 64],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": 5000,
        "d": 100,
        "R": 3.0,  # Major radius
        "r": 1.0,  # Minor radius
        "noise": 0.02,
        "verify": True,  # Verify linking number on construction
    },
    "klein_bottle": {
        "input_dim": 4,
        "latent_dim": 2,
        "hidden_dims": [64, 32],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": 10000,
        "noise": 0.05,
        "mmae_n_components": 4,  # Full dimensionality since it's already 4D
    },
    "mammoth": {
        "input_dim": 3,
        "latent_dim": 2,
        "hidden_dims": [64, 32],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": 50000,
        "mmae_n_components": 3,
    },
    "branching_tree": {
        "input_dim": 100,
        "latent_dim": 2,
        "hidden_dims": [256, 128, 64],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": 10000,
        "d": 100,
        "noise": 0.08,
    },
    "coil20": {
        "input_dim": 16384,
        "latent_dim": 2,
        "hidden_dims": [1000, 500, 250],
        "arch_type": "conv",
        "input_shape": (1, 128, 128),
        "n_samples": None,
    },
    "earth": {
        "input_dim": 3,
        "latent_dim": 2,
        "hidden_dims": [64, 32],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": None,
        "mmae_n_components": 3,
    },
    


    # ==========================================================================
    # Single-cell biological datasets
    # ==========================================================================
    "pbmc3k": {
        # Peripheral blood mononuclear cells (scRNA-seq)
        # ~2638 cells, 1838 genes, 8 cell types
        "input_dim": 1838,
        "latent_dim": 2,
        "hidden_dims": [1000, 500, 250],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": None,  # Use all cells
    },
    "paul15": {
        # Bone marrow hematopoiesis (scRNA-seq) - TRAJECTORY structure
        # ~2700 cells, 2000 HVGs, 19 cell types along differentiation
        "input_dim": 2000,
        "latent_dim": 2,
        "hidden_dims": [1000, 500, 250],
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": None,  # Use all cells
        "n_top_genes": 2000,
    },
    "levine32": {
        # Mass cytometry / CyTOF (PROTEIN markers) - different modality
        # ~100k cells, 32 protein markers, 14 cell populations
        "input_dim": 32,
        "latent_dim": 2,
        "hidden_dims": [250, 100],  # Smaller network for 32D input
        "arch_type": "mlp",
        "input_shape": None,
        "n_samples": 10000,  # Subsample for experiments
        "max_cells": 50000,
    },
}

# Model-specific configs
MODEL_CONFIGS = {
    "vanilla": {},
    "topoae": {
        "topo_lambda": 1.0,
        "match_edges": "symmetric",
    },
    "rtdae": {
        "rtd_lambda": 1.0,
        "rtd_dim": 1,
        "rtd_card": 50,
        "rtd_engine": "ripser",
        "rtd_is_sym": True,
        "rtd_lp": 1.0,
    },
    "geomae": {
        "geom_lambda": 100.0,
    },
    "ggae": {
        "ggae_lambda": 0.45780,
        "ggae_bandwidth": 53.40,
    },
    "mmae": {
        "mmae_lambda": 1.0,
        "mmae_n_components": 80,
    },
    "spae": {
        'spae_lambda': 1.0,
        'spae_variant': 'r2',  # or 'r1' or 'both'
    },
    "mmae_recon": {
        "mmae_lambda": 1.0,
        "mmae_n_components": 80,
        "n_interp_steps": 5,
    },
    "mmae_rank": {
        "mmae_lambda": 1.0,
        "mmae_n_components": 80,
        "n_triplets": 256,
        "use_triplet_rank": True,  # False for soft rank correlation
    },
    "mmae_local": {
        "mmae_lambda": 1.0,
        "mmae_n_components": 80,
        "k_neighbors": 10,
    },
    "mmae_knn": {
        "mmae_lambda": 1.0,
        "mmae_k": 15,  # try 10-30
        "mmae_knn_loss": "mse",  # or "ratio" or "stress"
    }, 
}


def get_config(dataset_name, model_name=None):
    """Build config for a specific dataset and optionally a model."""
    config = BASE_CONFIG.copy()
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config.update(DATASET_CONFIGS[dataset_name])
    config["dataset_name"] = dataset_name
    
    if model_name:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        config.update(MODEL_CONFIGS[model_name])
        config["model_name"] = model_name
    
    return config