#!/usr/bin/env python
"""
Plot Topology Sufficiency results.

Reads results.csv from the experiment and generates the key figure:
topology metrics (Wasserstein H0, H1) and distance correlation vs latent dim,
showing convergence of MM-Reg to topological methods as d grows.

Usage:
    python plot_topology_sufficiency.py results/topology_sufficiency/spheres/results.csv
    python plot_topology_sufficiency.py results.csv --output figures/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ---- Style ----
COLORS = {
    "vanilla":  "#888888",
    "topoae":   "#ff7f0e",
    "rtdae":    "#2ca02c",
    "mmae":     "#d62728",
    "geomae":   "#9467bd",
    "ggae":     "#17becf",
    "spae":     "#e377c2",
}
MARKERS = {
    "vanilla": "o", "topoae": "s", "rtdae": "^",
    "mmae": "D", "geomae": "v", "ggae": "p", "spae": "*",
}
LABELS = {
    "vanilla": "Vanilla AE", "topoae": "TopoAE", "rtdae": "RTD-AE",
    "mmae": "MM-Reg", "geomae": "GeomAE", "ggae": "GGAE", "spae": "SPAE",
}

# Metrics to plot: (column, display_name, lower_is_better)
METRICS = [
    ("wasserstein_H0",       "Wasserstein $H_0$ ↓",     True),
    ("wasserstein_H1",       "Wasserstein $H_1$ ↓",     True),
    ("distance_correlation", "Distance Correlation ↑",   False),
    ("triplet_accuracy",     "Triplet Accuracy ↑",       False),
    ("reconstruction_error", "Reconstruction Error ↓",   True),
    ("train_time_seconds",   "Training Time (s) ↓",      True),
]


def style_for(model):
    base = model.split("_pca")[0] if "_pca" in model else model
    return (
        COLORS.get(base, "#333333"),
        MARKERS.get(base, "o"),
        LABELS.get(base, model) if base == model else model.replace("_", " "),
    )


def plot_results(df, output_dir=None):
    """Main figure: metrics vs latent dimension."""
    models = df["model"].unique()
    latent_dims = sorted(df["latent_dim"].unique())

    # Filter to metrics that actually exist in the data
    available = [(col, name, lower) for col, name, lower in METRICS if col in df.columns]
    n = len(available)
    if n == 0:
        print("No plottable metrics found in results.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (col, display, lower_is_better) in zip(axes, available):
        for model in models:
            sub = df[df["model"] == model].sort_values("latent_dim")
            if col not in sub.columns or sub[col].isna().all():
                continue
            color, marker, label = style_for(model)
            std_col = f"{col}_std"
            y = sub[col].values
            x = sub["latent_dim"].values
            ax.plot(x, y, color=color, marker=marker, label=label,
                    linewidth=1.8, markersize=6, alpha=0.9)
            if std_col in sub.columns:
                yerr = sub[std_col].values
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)

        ax.set_xlabel("Latent dimension $d$")
        ax.set_ylabel(display)
        ax.set_xscale("log", base=2)
        ax.set_xticks(latent_dims)
        ax.set_xticklabels([str(d) for d in latent_dims])
        if lower_is_better and col.startswith("wasserstein"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Shared legend below
    handles, labels_seen = [], []
    for model in models:
        color, marker, label = style_for(model)
        if label not in labels_seen:
            handles.append(Line2D([0], [0], color=color, marker=marker,
                                  linewidth=1.8, markersize=6, label=label))
            labels_seen.append(label)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else ""
    fig.suptitle(f"Topology Sufficiency — {dataset}", fontsize=13, y=1.02)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"topology_sufficiency_{dataset}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved to {path}")
        path_pdf = path.replace(".png", ".pdf")
        fig.savefig(path_pdf, bbox_inches="tight")
        print(f"Saved to {path_pdf}")
    else:
        plt.show()


def plot_convergence_gap(df, output_dir=None):
    """
    Focused figure: gap between mmae and topoae on Wasserstein H0/H1
    as a function of latent dim. Shows the "convergence" predicted by theory.
    """
    if "topoae" not in df["model"].values or "mmae" not in df["model"].values:
        print("Need both topoae and mmae in results for convergence gap plot.")
        return

    topo = df[df["model"] == "topoae"].set_index("latent_dim")
    mmae = df[df["model"] == "mmae"].set_index("latent_dim")
    dims = sorted(set(topo.index) & set(mmae.index))

    wass_cols = [c for c in ["wasserstein_H0", "wasserstein_H1"] if c in df.columns]
    if not wass_cols:
        print("No Wasserstein columns found.")
        return

    fig, axes = plt.subplots(1, len(wass_cols), figsize=(5 * len(wass_cols), 4))
    if len(wass_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, wass_cols):
        gap = []
        for d in dims:
            t_val = topo.loc[d, col] if not pd.isna(topo.loc[d, col]) else np.nan
            m_val = mmae.loc[d, col] if not pd.isna(mmae.loc[d, col]) else np.nan
            gap.append(m_val - t_val)

        ax.bar(range(len(dims)), gap, color="#d62728", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Latent dimension $d$")
        ax.set_ylabel(f"MM-Reg − TopoAE ({col.replace('wasserstein_', 'W ')})")
        ax.set_title(col.replace("wasserstein_", "Wasserstein "))
        ax.grid(True, axis="y", alpha=0.3)

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else ""
    fig.suptitle(f"Topology Gap: MM-Reg vs TopoAE — {dataset}\n(→ 0 means MM-Reg matches TopoAE topologically)",
                 fontsize=11, y=1.05)
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"convergence_gap_{dataset}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved to {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot topology sufficiency results")
    parser.add_argument("csv", type=str, help="Path to results.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures (default: show)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows: {df['model'].nunique()} models, "
          f"dims={sorted(df['latent_dim'].unique())}")

    plot_results(df, args.output)
    plot_convergence_gap(df, args.output)


if __name__ == "__main__":
    main()