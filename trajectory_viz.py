"""
Trajectory visualization for Paul15 hematopoiesis data.
Shows differentiation directions with high-contrast paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke

# Known differentiation paths in Paul15
PAUL15_PATHS = {
    'Erythroid': ['7MEP', '1Ery', '2Ery', '3Ery', '4Ery', '5Ery', '6Ery'],
    'Megakaryocyte': ['7MEP', '8Mk'],
    'Granulocyte': ['9GMP', '10GMP', '16Neu', '17Neu', '18Eos'],
    'Monocyte': ['9GMP', '10GMP', '11DC', '14Mo', '15Mo'],
    'Basophil': ['9GMP', '10GMP', '12Baso', '13Baso'],
}

# Colors for cell scatter points
LINEAGE_COLORS = {
    'Erythroid': '#d62728',
    'Megakaryocyte': '#ff7f0e',
    'Granulocyte': '#2ca02c',
    'Monocyte': '#1f77b4',
    'Basophil': '#9467bd',
}

# High-contrast colors for trajectory LINES (different from point colors)
TRAJECTORY_COLORS = {
    'Erythroid': '#8B0000',      # dark red
    'Megakaryocyte': '#FF4500',  # orange-red
    'Granulocyte': '#006400',    # dark green
    'Monocyte': '#00008B',       # dark blue
    'Basophil': '#4B0082',       # indigo
}

# Markers for trajectory endpoints
TRAJECTORY_MARKERS = {
    'Erythroid': 's',      # square
    'Megakaryocyte': 'D',  # diamond
    'Granulocyte': '^',    # triangle up
    'Monocyte': 'v',       # triangle down
    'Basophil': 'p',       # pentagon
}

METHOD_NAMES = {
    'mmae': 'MMAE (Ours)', 'vanilla': 'Vanilla AE', 'topoae': 'TopoAE',
    'rtdae': 'RTD-AE', 'geomae': 'GeomAE', 'ggae': 'GGAE'
}


def compute_centroids(latents, labels):
    """Compute centroid for each cell type."""
    centroids = {}
    for ct in np.unique(labels):
        mask = labels == ct
        if mask.sum() > 0:
            centroids[ct] = latents[mask].mean(axis=0)
    return centroids


def draw_paul15_trajectories(ax, latents, labels, linewidth=2.5, show_intermediate=True):
    """Draw trajectories with high contrast - black lines with white outline."""
    centroids = compute_centroids(latents, labels)
    
    white_outline = [withStroke(linewidth=linewidth+2, foreground='white')]
    
    for lineage, path in PAUL15_PATHS.items():
        # Get all centroids for this path
        path_points = []
        path_labels = []
        for ct in path:
            if ct in centroids:
                path_points.append(centroids[ct])
                path_labels.append(ct)
        
        if len(path_points) < 2:
            continue
            
        path_points = np.array(path_points)
        color = TRAJECTORY_COLORS[lineage]
        marker = TRAJECTORY_MARKERS[lineage]
        
        # Draw line with white outline for visibility
        ax.plot(path_points[:, 0], path_points[:, 1], 
               color=color, linewidth=linewidth, zorder=6,
               path_effects=white_outline, solid_capstyle='round')
        
        # Arrow at the end
        if len(path_points) >= 2:
            dx = path_points[-1, 0] - path_points[-2, 0]
            dy = path_points[-1, 1] - path_points[-2, 1]
            ax.annotate('', xy=path_points[-1], 
                       xytext=(path_points[-1, 0] - dx*0.15, path_points[-1, 1] - dy*0.15),
                       arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                      mutation_scale=15),
                       zorder=7)
        
        # Show intermediate points as small numbered circles
        if show_intermediate and len(path_points) > 2:
            for j, (pt, lbl) in enumerate(zip(path_points[1:-1], path_labels[1:-1])):
                ax.scatter([pt[0]], [pt[1]], s=120, c='white', edgecolors=color, 
                          linewidth=2, zorder=8, marker='o')
                ax.text(pt[0], pt[1], str(j+1), fontsize=7, ha='center', va='center',
                       fontweight='bold', color=color, zorder=9)
        
        # Terminal cell marker (larger, distinct shape)
        ax.scatter([path_points[-1, 0]], [path_points[-1, 1]], 
                  s=200, c=color, edgecolors='white', linewidth=2,
                  marker=marker, zorder=8, label=lineage)
    
    # Mark progenitor cells (white circles with black edge)
    for prog in ['7MEP', '9GMP']:
        if prog in centroids:
            ax.scatter([centroids[prog][0]], [centroids[prog][1]], 
                      s=250, c='white', edgecolors='black', linewidth=3,
                      zorder=10, marker='o')
            ax.annotate(prog, centroids[prog], fontsize=10, ha='center', va='center',
                       fontweight='bold', zorder=11)


def create_trajectory_comparison(results_dict, save_path=None, figsize=None):
    """Create figure comparing methods with trajectory overlays."""
    methods = list(results_dict.keys())
    if 'mmae' in methods:
        methods.remove('mmae')
        methods = ['mmae'] + methods
    
    n = len(methods)
    if figsize is None:
        figsize = (5 * n, 5.5)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    # Normalize latents
    all_latents_norm = []
    for m in methods:
        lat = results_dict[m]['latents']
        all_latents_norm.append((lat - lat.mean(0)) / (lat.std(0) + 1e-8))
    
    lim = max(abs(np.vstack(all_latents_norm).min()), 
              abs(np.vstack(all_latents_norm).max())) * 1.1
    
    for i, method in enumerate(methods):
        ax = axes[i]
        res = results_dict[method]
        lat_norm = all_latents_norm[i]
        labels = res['labels']
        lineages = res.get('lineages')
        m = res['metrics']
        
        # Plot cells (faded background)
        if lineages is not None:
            for lin, color in LINEAGE_COLORS.items():
                mask = lineages == lin
                if mask.sum() > 0:
                    ax.scatter(lat_norm[mask, 0], lat_norm[mask, 1],
                              c=color, s=10, alpha=0.3, rasterized=True)
        
        # Draw trajectories
        draw_paul15_trajectories(ax, lat_norm, labels, linewidth=2, show_intermediate=False)
        
        dc = m.get('distance_correlation', 0)
        pt = m.get('pseudotime_spearman', 0)
        if np.isnan(pt):
            pt = 0
        
        ax.set_title(f'{METHOD_NAMES.get(method, method)}\nDC={dc:.2f}, PT={pt:.2f}',
                    fontsize=12, fontweight='bold' if method == 'mmae' else 'normal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Legend with trajectory markers
    handles = []
    for lin in PAUL15_PATHS.keys():
        handles.append(plt.Line2D([0], [0], marker=TRAJECTORY_MARKERS[lin], color='w',
                                  markerfacecolor=TRAJECTORY_COLORS[lin], markersize=10,
                                  markeredgecolor='white', markeredgewidth=1, label=lin))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=5, fontsize=9, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
        plt.close()
    
    return fig


def create_individual_trajectory_plots(results_dict, save_dir):
    """Create separate trajectory plot for each method with full detail."""
    os.makedirs(save_dir, exist_ok=True)
    
    for method, res in results_dict.items():
        lat = res['latents']
        lat_norm = (lat - lat.mean(0)) / (lat.std(0) + 1e-8)
        labels = res['labels']
        lineages = res.get('lineages')
        m = res['metrics']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        lim = max(abs(lat_norm.min()), abs(lat_norm.max())) * 1.1
        
        # Plot cells (faded)
        if lineages is not None:
            for lin, color in LINEAGE_COLORS.items():
                mask = lineages == lin
                if mask.sum() > 0:
                    ax.scatter(lat_norm[mask, 0], lat_norm[mask, 1],
                              c=color, s=20, alpha=0.35, rasterized=True)
        
        # Draw trajectories with intermediate markers
        draw_paul15_trajectories(ax, lat_norm, labels, linewidth=3, show_intermediate=True)
        
        dc = m.get('distance_correlation', 0)
        pt = m.get('pseudotime_spearman', 0)
        if np.isnan(pt):
            pt = 0
        
        title = METHOD_NAMES.get(method, method)
        ax.set_title(f'{title}\nDC={dc:.2f}, PT={pt:.2f}', fontsize=16, fontweight='bold')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Legend
        handles = []
        for lin in PAUL15_PATHS.keys():
            handles.append(plt.Line2D([0], [0], marker=TRAJECTORY_MARKERS[lin], color='w',
                                      markerfacecolor=TRAJECTORY_COLORS[lin], markersize=12,
                                      markeredgecolor='white', markeredgewidth=1.5, label=lin))
        ax.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        fpath = os.path.join(save_dir, f'trajectory_{method}.png')
        plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(fpath.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f'Saved: {fpath}')
        plt.close()