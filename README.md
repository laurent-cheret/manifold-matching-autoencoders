# Manifold-Matching Autoencoders

Train an autoencoder whose latent space preserves the **pairwise distance
structure** of a reference embedding (PCA, UMAP, or t-SNE) of the input data.

The whole regularizer fits in ten lines:

```python
def manifold_matching_loss(z, ref):
    z_dist = torch.cdist(z, z)
    r_dist = torch.cdist(ref, ref)
    z_norm = (z_dist - z_dist.mean()) / (z_dist.std() + 1e-8)
    r_norm = (r_dist - r_dist.mean()) / (r_dist.std() + 1e-8)
    return ((z_norm - r_norm) ** 2).mean()

# Total loss = MSE reconstruction + λ * manifold_matching_loss(z, reference)
```

The reference is computed once on the training set; each sample carries its
reference coordinates through the dataloader, and the regularizer matches
batches of latent codes to batches of reference codes.

---

## Install

```bash
pip install -r requirements.txt
```

That's it. Datasets are auto-downloaded on first use into `./data_cache/`.

---

## Quick start

```bash
# MMAE with PCA reference (defaults: ref_dim=2, lambda=1.0)
python train.py --dataset mnist --epochs 50

# Vanilla autoencoder baseline (regularizer disabled)
python train.py --dataset mnist --regularizer none --epochs 50

# MMAE with 2D UMAP reference
python train.py --dataset mnist --reference umap --ref_dim 2

# MMAE with 2D t-SNE reference
python train.py --dataset spheres --reference tsne --ref_dim 2

# Render a GIF of the latent space evolving each epoch
python train.py --dataset mammoth --epochs 80 --gif --snapshot_every 1
```

Each run writes to `results/<dataset>_<regularizer>[_<reference><ref_dim>...]/`
with `latent.png`, `history.json`, and (if `--gif`) `latent_evolution.gif`.

---

## Datasets

| Name      | Source                                  | Shape           | Notes                    |
| --------- | --------------------------------------- | --------------- | ------------------------ |
| `mnist`   | torchvision (auto-downloads)            | 784             | Full 60k train; cap with `--n_samples N` |
| `fmnist`  | torchvision (auto-downloads)            | 784             | Full 60k train; cap with `--n_samples N` |
| `spheres` | synthetic (TopoAE-style nested spheres) | 101             | 11 spheres in 100+1 dims |
| `mammoth` | Smithsonian 3D scan (auto-downloads)    | 3               | ~1M pts; default subsample 50k (use `--n_samples 0` for full set) |

---

## Try it from a notebook (Colab-friendly)

Open `examples/colab_quickstart.ipynb`. It clones the repo, installs deps,
trains MMAE on MNIST or Mammoth, and renders the per-epoch latent-space GIF
inline.

---

## Use it as a library

```python
from mmae import train_run
from mmae.viz import plot_latents, make_latent_gif

result = train_run(
    dataset="mnist", regularizer="mmae", reference="pca", ref_dim=50,
    latent_dim=2, epochs=30, snapshot_every=1,
)
plot_latents(result.snapshots[-1][1], result.snapshots[-1][2], save_path="latent.png")
make_latent_gif(result.snapshots, save_path="latent.gif", fps=8)
```

---

## What's in the box

```
manifold-matching-autoencoders/
├── train.py                        # CLI entrypoint
├── requirements.txt
├── README.md
├── examples/
│   └── colab_quickstart.ipynb      # one-click Colab demo with GIF
└── mmae/
    ├── __init__.py
    ├── model.py                    # MLPEncoder, MLPDecoder, Autoencoder
    ├── reference.py                # PCA / UMAP / t-SNE reference computation
    ├── data.py                     # mnist, fmnist, spheres, mammoth
    ├── train.py                    # training loop with snapshot callback
    └── viz.py                      # 2D/3D scatter + per-epoch GIF
```

---

## Citation

If you use this code please cite the paper:

> Cheret, L., Létourneau, V., Nejadgholi, I., Drummond, C., Al Osman, H., &
> Fraser, M. (2026). *Manifold-Matching Autoencoders*. arXiv:2603.16568.
