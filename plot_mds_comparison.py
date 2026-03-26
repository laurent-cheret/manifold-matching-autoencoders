#!/usr/bin/env python
"""
Generate publication figure for MDS vs MMAE comparison.

Usage:
    python plot_mds_comparison.py --results_file results/mds_comparison/results.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication settings
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12


def plot_embedding(ax, embedding, labels, title, dist_corr, time_s):
    """Plot 2D embedding with metrics."""
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap='Spectral', s=25, alpha=0.7, 
        edgecolors='black', linewidths=0.3
    )
    
    ax.set_xlabel('$z_1$', fontsize=16)
    ax.set_ylabel('$z_2$', fontsize=16)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Add metrics text box
    textstr = f'$\\rho$ = {dist_corr:.3f}\n{time_s:.1f}s'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    return scatter


def plot_scaling(ax_time, ax_mem, scaling_results):
    """Plot scaling analysis (time and memory)."""
    sample_sizes = sorted([int(k) for k in scaling_results.keys()])
    
    methods = ['classical_mds', 'landmark_mds', 'mmae']
    colors = {'classical_mds': '#e74c3c', 'landmark_mds': '#3498db', 'mmae': '#2ecc71'}
    labels_map = {'classical_mds': 'Classical MDS', 'landmark_mds': 'Landmark MDS', 'mmae': 'MMAE'}
    markers = {'classical_mds': 'o', 'landmark_mds': 's', 'mmae': '^'}
    
    # Time plot
    for method in methods:
        sizes = []
        times = []
        for n in sample_sizes:
            if method in scaling_results[str(n)]:
                sizes.append(n)
                times.append(scaling_results[str(n)][method]['time_seconds'])
        
        if sizes:
            ax_time.plot(sizes, times, marker=markers[method], color=colors[method],
                        linewidth=2.5, markersize=8, label=labels_map[method])
    
    ax_time.set_xlabel('Sample Size', fontsize=16)
    ax_time.set_ylabel('Time (s)', fontsize=16)
    ax_time.set_title('Computational Time', fontsize=16, fontweight='bold')
    ax_time.set_yscale('log')
    ax_time.grid(True, alpha=0.3, which='both')
    ax_time.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    
    # Memory plot
    for method in methods:
        sizes = []
        mems = []
        for n in sample_sizes:
            if method in scaling_results[str(n)]:
                sizes.append(n)
                mems.append(scaling_results[str(n)][method]['memory_mb'])
        
        if sizes:
            ax_mem.plot(sizes, mems, marker=markers[method], color=colors[method],
                       linewidth=2.5, markersize=8, label=labels_map[method])
    
    ax_mem.set_xlabel('Sample Size', fontsize=16)
    ax_mem.set_ylabel('Memory (MB)', fontsize=16)
    ax_mem.set_title('Peak Memory', fontsize=16, fontweight='bold')
    ax_mem.grid(True, alpha=0.3)


def plot_noise_robustness(ax, noise_results):
    """Plot quality vs PCA percentage."""
    pca_percentages = sorted([int(k) for k in noise_results['mmae'].keys()])
    mmae_corrs = [noise_results['mmae'][str(p)]['distance_correlation'] for p in pca_percentages]
    lmds_corr = noise_results['landmark_mds']['distance_correlation']
    
    # MMAE line
    ax.plot(pca_percentages, mmae_corrs, marker='^', color='#2ecc71',
           linewidth=3, markersize=10, label='MMAE', zorder=3)
    
    # LMDS flat line
    ax.axhline(y=lmds_corr, color='#3498db', linestyle='--', linewidth=2.5,
              label='Landmark MDS', zorder=2)
    
    ax.set_xlabel('PCA Variance (%)', fontsize=16)
    ax.set_ylabel('Distance Correlation $\\rho$', fontsize=16)
    ax.set_title('Noise Robustness', fontsize=16, fontweight='bold')
    ax.set_xlim([pca_percentages[-1] - 5, pca_percentages[0] + 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=13)
    
    # Annotate improvement
    improvement = (mmae_corrs[-1] - lmds_corr) / lmds_corr * 100
    textstr = f'+{improvement:.0f}% improvement\nwith PCA denoising'
    props = dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='mds_comparison_figure.pdf')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Create figure (landscape, 2 rows)
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.06, right=0.98, top=0.94, bottom=0.08)
    
    # ========== ROW 1: CLEAN DATA ==========
    ax_mds = fig.add_subplot(gs[0, 0])
    ax_lmds = fig.add_subplot(gs[0, 1])
    ax_mmae = fig.add_subplot(gs[0, 2])
    ax_scaling_time = fig.add_subplot(gs[0, 3])
    
    # Plot embeddings
    clean = results['clean_data']
    
    plot_embedding(ax_mds, np.array(clean['classical_mds']['embedding']),
                  np.array(clean['classical_mds']['labels']),
                  '(a) Classical MDS',
                  clean['classical_mds']['distance_correlation'],
                  clean['classical_mds']['time_seconds'])
    
    plot_embedding(ax_lmds, np.array(clean['landmark_mds']['embedding']),
                  np.array(clean['landmark_mds']['labels']),
                  '(b) Landmark MDS',
                  clean['landmark_mds']['distance_correlation'],
                  clean['landmark_mds']['time_seconds'])
    
    cbar = plot_embedding(ax_mmae, np.array(clean['mmae']['embedding']),
                         np.array(clean['mmae']['labels']),
                         '(c) MMAE (100%)',
                         clean['mmae']['distance_correlation'],
                         clean['mmae']['time_seconds'])
    
    # Scaling plot (time only, memory will be in separate axis)
    ax_scaling_mem = ax_scaling_time.twinx()
    plot_scaling(ax_scaling_time, ax_scaling_mem, results['scaling'])
    ax_scaling_time.set_title('(d) Efficiency Comparison', fontsize=16, fontweight='bold')
    
    # ========== ROW 2: NOISE ROBUSTNESS ==========
    ax_lmds_noisy = fig.add_subplot(gs[1, 0])
    ax_mmae_100 = fig.add_subplot(gs[1, 1])
    ax_mmae_80 = fig.add_subplot(gs[1, 2])
    ax_noise_quality = fig.add_subplot(gs[1, 3])
    
    # Plot noisy embeddings
    noise = results['noise_robustness']
    
    plot_embedding(ax_lmds_noisy, np.array(noise['landmark_mds']['embedding']),
                  np.array(noise['landmark_mds']['labels']),
                  '(e) Landmark MDS (noisy)',
                  noise['landmark_mds']['distance_correlation'],
                  noise['landmark_mds']['time_seconds'])
    
    plot_embedding(ax_mmae_100, np.array(noise['mmae']['100']['embedding']),
                  np.array(noise['mmae']['100']['labels']),
                  '(f) MMAE 100% (noisy)',
                  noise['mmae']['100']['distance_correlation'],
                  noise['mmae']['100']['time_seconds'])
    
    plot_embedding(ax_mmae_80, np.array(noise['mmae']['80']['embedding']),
                  np.array(noise['mmae']['80']['labels']),
                  '(g) MMAE 80% PCA',
                  noise['mmae']['80']['distance_correlation'],
                  noise['mmae']['80']['time_seconds'])
    
    # Noise robustness plot
    plot_noise_robustness(ax_noise_quality, noise)
    ax_noise_quality.set_title('(h) Quality vs Denoising', fontsize=16, fontweight='bold')
    
    # Add colorbar (shared for all)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = plt.colorbar(cbar, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Sphere Label', fontsize=14)
    
    # Save
    plt.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(args.output_file.replace('.pdf', '.png'), dpi=args.dpi, bbox_inches='tight')
    
    print(f"\nFigure saved to {args.output_file}")


if __name__ == '__main__':
    main()