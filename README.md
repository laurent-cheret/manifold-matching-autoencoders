# TOPO_COMPARE

Official code for **"Manifold Matching via Autoencoders"** (ICML submission).

Benchmarks topology-preserving autoencoders across synthetic, image, and biological datasets. Implements MMAE alongside TopoAE, RTD-AE, GeomAE, GGAE, and a vanilla baseline.

---

## Setup

```bash
pip install -r requirements.txt
```

Datasets are downloaded automatically on first run into `data/raw/` (gitignored).

---

## Quick Start — Reproduce All Results

### Option A: Skip HPO (use precomputed configs, ~2–3 h on GPU)

```bash
bash run_pipeline.sh --skip-hpo
```

### Option B: Full pipeline including HPO (~12–20 h on GPU)

```bash
bash run_pipeline.sh
```

Results land in `results/final/` and LaTeX tables in `results/latex/`.

---

## Step-by-Step

### 1. Hyperparameter Search

```bash
# Single model + dataset (2D)
python hyperparam_search_optuna.py \
    --model mmae --dataset mnist \
    --latent_dims 2 16 32 64 \
    --opt_metric density_kl_0_1 \
    --n_trials 50 --epochs 50

# Grid search mode
python hyperparam_search_optuna.py \
    --model topoae --dataset spheres \
    --latent_dims 2 --sampler grid
```

Best configs are saved automatically to `configs/best_hparams/<dataset>/<dim>D_best_config_<model>.json`.

### 2. Final Evaluation

```bash
python run_final_evaluation.py \
    --dataset mnist \
    --best_configs_dir configs/best_hparams/mnist \
    --output_dir results/final/mnist_dim2 \
    --latent_dim 2 --n_seeds 5
```

### 3. Single-Run Experiment

```bash
python run_experiment.py \
    --dataset spheres --model mmae \
    --latent_dim 2 --epochs 100
```

---

## Project Structure

```
TOPO_COMPARE/
├── config.py                   # Dataset & model configuration registry
├── training.py                 # Trainer class
├── evaluation.py               # Evaluation metrics (trustworthiness, Wasserstein, ...)
├── models/                     # Model implementations
│   ├── base.py                 # MLP encoder/decoder, model registry
│   ├── mmae.py                 # Manifold Matching AE (proposed)
│   ├── topo_ae.py              # TopoAE (Moor et al., ICML 2020)
│   ├── rtd_ae.py               # RTD-AE (Trofimov et al., ICLR 2023)
│   ├── geom_ae.py              # GeomAE (Nazari et al., ICML 2023)
│   ├── ggae.py                 # GGAE (Lim et al., ICML 2024)
│   ├── vanilla_ae.py           # Vanilla AE (reconstruction only)
│   └── ...
├── data/                       # Data loaders (auto-download)
│   ├── base.py                 # Splits, normalisation, PCA embeddings
│   ├── mnist.py / cifar.py     # Image datasets (Torchvision)
│   ├── scrna.py                # scRNA-seq (Scanpy: PBMC3k, Paul15)
│   └── ...                     # Synthetic datasets
├── configs/
│   └── best_hparams/           # Precomputed best configs (per dataset / dim)
│       └── <dataset>/
│           └── <dim>D_best_config_<model>.json
├── run_experiment.py           # Single experiment runner
├── run_final_evaluation.py     # Multi-seed final evaluation
├── hyperparam_search_optuna.py # Bayesian HPO (Optuna)
├── generate_latex_table.py     # Paper table generation
└── run_pipeline.sh             # Master reproducibility script
```

---

## Data Splits

All datasets use a strict **70 / 15 / 15** train / validation / test split:

| Split | Purpose |
|-------|---------|
| Train | Model weight optimisation |
| Validation | Hyperparameter selection (HPO) |
| Test | Final reported metrics only — never touched during training or HPO |

For MNIST, FMNIST, and CIFAR-10 the official Torchvision test set is used as the held-out test set.

---

## Reproducibility

All experiments use `seed=42` by default. Full determinism is enforced via:

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Pass `--seed <N>` to any script to override.

---

## Hardware

| Setup | HPO (~50 trials × 50 epochs) | Final eval (5 seeds × 150 epochs) |
|-------|-------------------------------|-----------------------------------|
| A100 / V100 | ~12 h | ~2 h |
| RTX 3090 | ~20 h | ~3 h |
| CPU only | Not recommended | ~8 h |
