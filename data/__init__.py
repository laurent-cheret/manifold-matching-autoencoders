"""Data module - imports register all datasets."""

from .base import get_dataset, list_datasets, BaseDataset, compute_pca_embeddings, set_global_seed

# Import to trigger registration
from . import spheres
from . import mnist
from . import cifar
from . import scrna
from . import swiss_roll
from . import concentric_spheres
from . import tree_clusters
from . import linked_tori
from . import klein_bottle
from . import mammoth
from . import branching_tree
from . import coil20
from . import earth

def load_data(dataset_name, config, with_embeddings=False, return_indices=False):
    """Load dataset by name.
    
    Args:
        dataset_name: name of dataset
        config: config dict
        with_embeddings: if True, compute PCA embeddings for MMAE
        return_indices: if True, return sample indices with each batch (for GGAE)
    """
    loader_fn = get_dataset(dataset_name)
    return loader_fn(config, with_embeddings=with_embeddings, return_indices=return_indices)