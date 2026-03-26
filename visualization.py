"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt


def plot_latent(latent, labels, title="", ax=None, s=10, alpha=0.7):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if latent.shape[1] == 2:
        ax.scatter(latent[:, 0], latent[:, 1], c=labels, cmap='Spectral', s=s, alpha=alpha)
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def plot_comparison(results, labels, figsize=(15, 5)):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (name, latent) in zip(axes, results.items()):
        plot_latent(latent, labels, title=name, ax=ax)
    plt.tight_layout()
    return fig


def plot_history(histories, metric='total_loss'):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, h in histories.items():
        if f'train_{metric}' in h:
            ax.plot(h[f'train_{metric}'], label=f'{name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend()
    return fig


def plot_metrics(metrics_dict):
    import pandas as pd
    df = pd.DataFrame(metrics_dict).T
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    plt.tight_layout()
    return fig


def plot_reconstructions(originals, reconstructions, n_samples=10, figsize=(20, 4)):
    """Plot original vs reconstructed images."""
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        # Original
        img = originals[i]
        if img.shape[0] in [1, 3]:  # Channel first
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstruction
        rec = reconstructions[i]
        if rec.shape[0] in [1, 3]:
            rec = rec.transpose(1, 2, 0)
        if rec.shape[-1] == 1:
            rec = rec.squeeze(-1)
        axes[1, i].imshow(rec, cmap='gray' if rec.ndim == 2 else None)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    return fig