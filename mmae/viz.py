"""Plotting utilities: 2D / 3D latent scatter and per-epoch GIFs.

Key design choice: marker size, alpha, and color handling adapt to dataset
size and label type. With 17k MNIST points the default matplotlib scatter
overplots into a colored blob; the helpers below shrink markers, lower alpha,
strip edge strokes, and use a discrete palette + legend for class labels.
"""

from __future__ import annotations

import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# --------------------------------------------------------------------------- #
# Adaptive style helpers                                                       #
# --------------------------------------------------------------------------- #

def _auto_marker_size(n: int) -> float:
    """Shrink markers as the number of points grows."""
    if n < 1000:
        return 14
    if n < 5000:
        return 6
    if n < 15000:
        return 3
    if n < 50000:
        return 1.5
    return 0.8


def _auto_alpha(n: int) -> float:
    """Lower alpha to combat overplotting at high density."""
    if n < 1000:
        return 0.9
    if n < 5000:
        return 0.7
    if n < 15000:
        return 0.55
    if n < 50000:
        return 0.4
    return 0.3


def _auto_edge(n: int) -> float:
    """Edge strokes only help at very low N; otherwise they dominate the dot."""
    return 0.15 if n < 1000 else 0.0


def _is_categorical(y: np.ndarray) -> bool:
    """Treat as categorical when y is integer-like with <= 20 distinct values."""
    if y.dtype.kind in ("U", "S", "O"):
        return True
    if y.dtype.kind in ("i", "u"):
        return len(np.unique(y)) <= 20
    # Float labels: only categorical if they're whole numbers and few unique.
    if y.dtype.kind == "f":
        if not np.allclose(y, np.round(y)):
            return False
        return len(np.unique(y)) <= 20
    return False


def _categorical_palette(n_classes: int):
    """Pick a perceptually-distinct discrete palette."""
    if n_classes <= 10:
        return plt.get_cmap("tab10").colors[:n_classes]
    if n_classes <= 20:
        return plt.get_cmap("tab20").colors[:n_classes]
    # Fall back to evenly-spaced Spectral entries.
    return plt.get_cmap("Spectral")(np.linspace(0, 1, n_classes))


# --------------------------------------------------------------------------- #
# Core scatter primitives                                                      #
# --------------------------------------------------------------------------- #

def _scatter_2d(ax, z, y, title="", legend=True):
    n = len(z)
    s = _auto_marker_size(n)
    alpha = _auto_alpha(n)
    lw = _auto_edge(n)

    if _is_categorical(y):
        classes = np.unique(y).astype(int)
        palette = _categorical_palette(len(classes))
        for i, c in enumerate(classes):
            mask = y == c
            ax.scatter(
                z[mask, 0], z[mask, 1],
                c=[palette[i]], s=s, alpha=alpha, linewidths=lw,
                edgecolors="black" if lw > 0 else "none",
                label=str(c), rasterized=n > 5000,
            )
        if legend:
            leg = ax.legend(
                title="class", loc="best", fontsize=7, title_fontsize=8,
                markerscale=max(1.0, 8.0 / max(s, 0.5)),
                framealpha=0.85, handletextpad=0.4, borderpad=0.3,
            )
        sc = None
    else:
        sc = ax.scatter(
            z[:, 0], z[:, 1], c=y, cmap="Spectral",
            s=s, alpha=alpha, linewidths=lw,
            edgecolors="black" if lw > 0 else "none",
            rasterized=n > 5000,
        )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linewidth=0.5)
    return sc


def _scatter_3d(ax, z, y, title=""):
    n = len(z)
    s = _auto_marker_size(n)
    alpha = _auto_alpha(n)

    if _is_categorical(y):
        classes = np.unique(y).astype(int)
        palette = _categorical_palette(len(classes))
        for i, c in enumerate(classes):
            mask = y == c
            ax.scatter(
                z[mask, 0], z[mask, 1], z[mask, 2],
                c=[palette[i]], s=s, alpha=alpha, label=str(c),
                depthshade=False,
            )
        ax.legend(title="class", fontsize=7, title_fontsize=8,
                  loc="upper left", framealpha=0.85)
        sc = None
    else:
        sc = ax.scatter(
            z[:, 0], z[:, 1], z[:, 2],
            c=y, cmap="Spectral", s=s, alpha=alpha, depthshade=False,
        )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title(title)
    return sc


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def plot_latents(z, y, save_path=None, title="", figsize=None, dpi=180):
    """Plot a single 2D or 3D scatter of latent codes."""
    z = np.asarray(z)
    y = np.asarray(y)

    if z.shape[1] == 2:
        fig, ax = plt.subplots(figsize=figsize or (8, 8))
        sc = _scatter_2d(ax, z, y, title)
        if sc is not None:
            fig.colorbar(sc, ax=ax, shrink=0.7)
    elif z.shape[1] >= 3:
        fig = plt.figure(figsize=figsize or (9, 8))
        ax = fig.add_subplot(111, projection="3d")
        sc = _scatter_3d(ax, z[:, :3], y, title)
        if sc is not None:
            fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    else:
        raise ValueError(f"latent dim must be >= 2, got {z.shape[1]}")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


def make_latent_gif(snapshots, save_path: str, fps: int = 8, title_prefix: str = "",
                    figsize=(7, 7), dpi=120):
    """Build a GIF showing the latent space evolving across epochs."""
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
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            _scatter_3d(ax, z[:, :3], y, f"{title_prefix} epoch {epoch}".strip())
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        else:
            fig, ax = plt.subplots(figsize=figsize)
            _scatter_2d(ax, z, y, f"{title_prefix} epoch {epoch}".strip(),
                        legend=(epoch == snapshots[-1][0]))
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    return save_path
