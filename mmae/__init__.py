"""Manifold-Matching Autoencoders.

A small library for training autoencoders with an optional manifold-matching
regularizer that aligns pairwise latent distances with a reference embedding
(PCA, UMAP, or t-SNE) of the input data.
"""

from .model import Autoencoder, MLPEncoder, MLPDecoder
from .reference import compute_reference
from .data import load_dataset, list_datasets
from .train import train_run

__all__ = [
    "Autoencoder",
    "MLPEncoder",
    "MLPDecoder",
    "compute_reference",
    "load_dataset",
    "list_datasets",
    "train_run",
]
