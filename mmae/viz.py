"""Plotting utilities: 2D / 3D latent scatter and per-epoch GIFs.

Marker size, alpha, and color handling adapt to dataset size and label type.
The two public functions (`plot_latents`, `make_latent_gif`) accept an
optional `reference` array (2D or 3D) — when given, the figure has two panels
side-by-side: the static reference embedding on the left, the autoencoder's
latent on the right.
"""

from __future__ import annotations

import io
import os

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Adaptive style helpers                                                       #
# --------------------------------------------------------------------------- #

def _auto_marker_size(n: int) -> float:
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
    return 0.15 if n < 1000 else 0.0


def _is_categorical(y: np.ndarray) -> bool:
    if y.dtype.kind in ("U", "S", "O"):
        return True
    if y.dtype.kind in ("i", "u"):
        return len(np.unique(y)) <= 20
    if y.dtype.kind == "f":
        if not np.allclose(y, np.round(y)):
            return False
        return len(np.unique(y)) <= 20
    return False


def _categorical_palette(n_classes: int):
    if n_classes <= 10:
        return plt.get_cmap("tab10").colors[:n_classes]
    if n_classes <= 20:
        return plt.get_cmap("tab20").colors[:n_classes]
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
            ax.legend(
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


def _scatter_3d(ax, z, y, title="", legend=True):
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
        if legend:
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
# Layout helpers                                                               #
# --------------------------------------------------------------------------- #

def _add_panel(fig, gridspec_idx, dim):
    """Add a subplot of the right kind (2D or 3D) at the given gridspec index."""
    if dim >= 3:
        return fig.add_subplot(gridspec_idx, projection="3d")
    return fig.add_subplot(gridspec_idx)


def _compute_lims(z, pad_frac=0.05):
    """Padded per-axis min/max for stable camera framing."""
    z = np.asarray(z)
    pad = pad_frac * (z.max(axis=0) - z.min(axis=0) + 1e-8)
    return list(zip(z.min(axis=0) - pad, z.max(axis=0) + pad))


def _apply_lims(ax, lims, is_3d):
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    if is_3d:
        ax.set_zlim(lims[2])


def _render_panel(ax, z, y, title, lims=None, legend=True):
    """Dispatch 2D vs 3D rendering and apply optional fixed limits."""
    is_3d = z.shape[1] >= 3
    if is_3d:
        sc = _scatter_3d(ax, z[:, :3], y, title, legend=legend)
    else:
        sc = _scatter_2d(ax, z, y, title, legend=legend)
    if lims is not None:
        _apply_lims(ax, lims, is_3d)
    return sc


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def plot_latents(z, y, save_path=None, title="",
                 reference=None, reference_y=None, reference_title="reference",
                 figsize=None, dpi=180):
    """Plot the latent scatter, optionally side-by-side with the reference.

    If `reference` is a 2D or 3D array (N, dim), the figure has two panels:
    left = static reference embedding, right = the autoencoder's latent.
    """
    z = np.asarray(z)
    y = np.asarray(y)
    has_ref = reference is not None

    if has_ref:
        reference = np.asarray(reference)
        if reference_y is None:
            reference_y = y
        else:
            reference_y = np.asarray(reference_y)

        ref_dim = reference.shape[1]
        lat_dim = z.shape[1]
        # Roomier figure when one of the panels is 3D.
        any_3d = ref_dim >= 3 or lat_dim >= 3
        fig = plt.figure(figsize=figsize or (15 if any_3d else 13, 6.5))
        ax_ref = _add_panel(fig, 121, ref_dim)
        ax_lat = _add_panel(fig, 122, lat_dim)
        _render_panel(ax_ref, reference, reference_y,
                      title=reference_title, legend=False)
        _render_panel(ax_lat, z, y, title=title, legend=True)
    else:
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
                    reference=None, reference_y=None, reference_title="reference",
                    figsize=None, dpi=120):
    """Build a GIF of the latent space evolving across epochs.

    If `reference` is provided (2D or 3D array), each frame has the static
    reference on the left and the evolving latent on the right.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "make_latent_gif requires `imageio`. Install with: pip install imageio"
        ) from e

    if not snapshots:
        raise ValueError("No snapshots to render. Run train_run with snapshot_every > 0.")

    last_epoch = snapshots[-1][0]
    last_z = snapshots[-1][1]
    lat_dim = last_z.shape[1]
    lat_lims = _compute_lims(last_z[:, :3] if lat_dim >= 3 else last_z)

    has_ref = reference is not None
    if has_ref:
        reference = np.asarray(reference)
        if reference_y is None:
            reference_y = snapshots[-1][2]
        else:
            reference_y = np.asarray(reference_y)
        ref_dim = reference.shape[1]
        ref_lims = _compute_lims(reference[:, :3] if ref_dim >= 3 else reference)
        any_3d = ref_dim >= 3 or lat_dim >= 3
        default_size = (13 if any_3d else 11, 5.5)
    else:
        default_size = (7, 7)
    fig_size = figsize or default_size

    frames = []
    for epoch, z, y in snapshots:
        is_last = (epoch == last_epoch)
        if has_ref:
            fig = plt.figure(figsize=fig_size)
            ax_ref = _add_panel(fig, 121, ref_dim)
            ax_lat = _add_panel(fig, 122, lat_dim)
            _render_panel(ax_ref, reference, reference_y,
                          title=reference_title, lims=ref_lims, legend=False)
            _render_panel(
                ax_lat, z, y,
                title=f"{title_prefix} epoch {epoch}".strip(),
                lims=lat_lims, legend=is_last,
            )
        else:
            if lat_dim >= 3:
                fig = plt.figure(figsize=fig_size)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=fig_size)
            _render_panel(
                ax, z, y,
                title=f"{title_prefix} epoch {epoch}".strip(),
                lims=lat_lims, legend=is_last,
            )

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    return save_path
