"""Plotting utilities: 2D / 3D latent scatter and per-epoch GIFs."""

from __future__ import annotations

import io
import os

import matplotlib.pyplot as plt
import numpy as np


def _scatter_2d(ax, z, y, title=""):
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="Spectral", s=8,
                    alpha=0.9, edgecolors="black", linewidths=0.1)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return sc


def _scatter_3d(ax, z, y, title=""):
    sc = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=y, cmap="Spectral",
                    s=6, alpha=0.85)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title(title)
    return sc


def plot_latents(z, y, save_path=None, title=""):
    """Plot a single 2D or 3D scatter of latent codes."""
    if z.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = _scatter_2d(ax, z, y, title)
    elif z.shape[1] >= 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = _scatter_3d(ax, z[:, :3], y, title)
    else:
        raise ValueError(f"latent dim must be >= 2, got {z.shape[1]}")
    fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
        return save_path
    return fig


def make_latent_gif(snapshots, save_path: str, fps: int = 8, title_prefix: str = ""):
    """Build a GIF showing the latent space evolving across epochs.

    Args:
        snapshots: list of (epoch, z, y) tuples returned by `train_run`.
        save_path: output path ending in `.gif`.
        fps: animation frame rate.
        title_prefix: prepended to each frame title (e.g. "MNIST · MMAE").
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "make_latent_gif requires `imageio`. Install with: pip install imageio"
        ) from e

    if not snapshots:
        raise ValueError("No snapshots to render. Run train_run with snapshot_every > 0.")

    last_z = snapshots[-1][1]
    is_3d = last_z.shape[1] >= 3
    if is_3d:
        last_z = last_z[:, :3]
    pad = 0.05 * (last_z.max(axis=0) - last_z.min(axis=0) + 1e-8)
    lims = list(zip(last_z.min(axis=0) - pad, last_z.max(axis=0) + pad))

    frames = []
    for epoch, z, y in snapshots:
        if is_3d:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            _scatter_3d(ax, z[:, :3], y, f"{title_prefix} epoch {epoch}".strip())
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            _scatter_2d(ax, z, y, f"{title_prefix} epoch {epoch}".strip())
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    return save_path
