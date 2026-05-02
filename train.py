#!/usr/bin/env python
"""Train a Manifold-Matching Autoencoder.

Examples
--------
    # MMAE with PCA reference (defaults)
    python train.py --dataset mnist --epochs 50

    # Vanilla AE baseline (regularizer turned off)
    python train.py --dataset mnist --regularizer none --epochs 50

    # MMAE with 2D UMAP reference
    python train.py --dataset mnist --reference umap --ref_dim 2

    # MMAE with 2D t-SNE reference
    python train.py --dataset spheres --reference tsne --ref_dim 2

    # Save a per-epoch GIF of the latent space (mammoth defaults to 50k pts)
    python train.py --dataset mammoth --epochs 80 --gif --snapshot_every 1

    # Use the full mammoth dataset (~1M points; needs GPU)
    python train.py --dataset mammoth --n_samples 0 --epochs 30
"""

import argparse
import json
import os
import time

from mmae import train_run, list_datasets
from mmae.viz import plot_latents, make_latent_gif


def parse_args():
    p = argparse.ArgumentParser(description="Train a Manifold-Matching Autoencoder")
    p.add_argument("--dataset", choices=list_datasets(), default="mnist")
    p.add_argument("--regularizer", choices=["mmae", "none"], default="mmae",
                   help="'mmae' = with manifold-matching loss, 'none' = vanilla AE")
    p.add_argument("--reference", choices=["pca", "umap", "tsne"], default="pca",
                   help="Reference embedding for the manifold-matching loss")
    p.add_argument("--ref_dim", type=int, default=2,
                   help="Dimensionality of the reference embedding")
    p.add_argument("--lam", type=float, default=1.0,
                   help="Weight on the manifold-matching term (lambda)")

    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Cap training-pool size (mnist/fmnist/mammoth). "
                        "Pass 0 to use the full dataset.")

    p.add_argument("--output_dir", default="results")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--gif", action="store_true",
                   help="After training, render an animated GIF of the latent space")
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="Capture latent embedding every N epochs (>0 required for --gif)")
    p.add_argument("--gif_fps", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    if args.gif and args.snapshot_every <= 0:
        args.snapshot_every = 1

    if args.n_samples == 0:
        args.n_samples = None

    start = time.time()
    result = train_run(
        dataset=args.dataset,
        regularizer=args.regularizer,
        reference=args.reference,
        ref_dim=args.ref_dim,
        lam=args.lam,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        snapshot_every=args.snapshot_every,
        n_samples=args.n_samples,
    )
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    if args.no_save:
        return

    run_name = f"{args.dataset}_{args.regularizer}"
    if args.regularizer == "mmae":
        run_name += f"_{args.reference}{args.ref_dim}_lam{args.lam}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(result.history, f, indent=2)

    final_z = result.snapshots[-1][1] if result.snapshots else None
    if final_z is None:
        from mmae.train import _encode_dataset
        final_z = _encode_dataset(result.model, result.bundle.X_test, result.device)
        final_y = result.bundle.y_test
    else:
        final_y = result.snapshots[-1][2]

    title = f"{args.dataset} - {args.regularizer}"
    if args.regularizer == "mmae":
        title += f" - {args.reference}{args.ref_dim}"

    plot_path = plot_latents(
        final_z, final_y,
        save_path=os.path.join(out_dir, "latent.png"),
        title=title,
    )
    print(f"Saved final latent plot to {plot_path}")

    if args.gif:
        gif_path = os.path.join(out_dir, "latent_evolution.gif")
        make_latent_gif(
            result.snapshots, gif_path, fps=args.gif_fps,
            title_prefix=f"{args.dataset} - {args.regularizer}",
        )
        print(f"Saved latent-evolution GIF to {gif_path}")


if __name__ == "__main__":
    main()
