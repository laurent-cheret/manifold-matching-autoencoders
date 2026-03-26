#!/usr/bin/env bash
# =============================================================================
# TOPO_COMPARE — Master Pipeline Script
# Reproduces all tables and figures from the ICML submission.
#
# Usage:
#   bash run_pipeline.sh              # Full pipeline (HPO + final eval)
#   bash run_pipeline.sh --skip-hpo  # Skip HPO, use precomputed configs
#
# Hardware expectations (rough wall-clock times):
#   With GPU (A100/V100):  ~12 h for full HPO, ~2 h for eval only
#   With GPU (RTX 3090):   ~20 h for full HPO, ~3 h for eval only
#   CPU only:              Not recommended for HPO; eval ~8 h
# =============================================================================

set -euo pipefail

SKIP_HPO=false
for arg in "$@"; do
  [[ "$arg" == "--skip-hpo" ]] && SKIP_HPO=true
done

SEED=42
DEVICE=cuda
HPO_TRIALS=50
HPO_EPOCHS=50
EVAL_EPOCHS=150
LATENT_DIMS="2 16 32 64"

SYNTHETIC_DATASETS="spheres swiss_roll concentric_spheres linked_tori branching_tree"
REAL_DATASETS="mnist fmnist cifar10"
BIO_DATASETS="pbmc3k paul15"
ALL_DATASETS="$SYNTHETIC_DATASETS $REAL_DATASETS $BIO_DATASETS"

MODELS="vanilla topoae rtdae mmae"

echo "============================================================"
echo "TOPO_COMPARE Pipeline"
echo "  Seed:        $SEED"
echo "  Device:      $DEVICE"
echo "  Skip HPO:    $SKIP_HPO"
echo "============================================================"

# ------------------------------------------------------------------
# Step 1: Hyperparameter optimisation (skip if --skip-hpo)
# ------------------------------------------------------------------
if [ "$SKIP_HPO" = false ]; then
  echo ""
  echo ">>> Step 1: Hyperparameter optimisation"
  for DATASET in $ALL_DATASETS; do
    for MODEL in $MODELS; do
      echo "  HPO: $MODEL on $DATASET (dims: $LATENT_DIMS)"
      python hyperparam_search_optuna.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --latent_dims $LATENT_DIMS \
        --metric density_kl_0_1 \
        --n_trials $HPO_TRIALS \
        --epochs $HPO_EPOCHS \
        --device "$DEVICE" \
        --seed $SEED \
        --results_dir "results/hypersearch/${MODEL}_${DATASET}"
    done
  done
else
  echo ""
  echo ">>> Step 1: SKIPPED (using precomputed configs in configs/best_hparams/)"
fi

# ------------------------------------------------------------------
# Step 2: Final evaluation (Table 1 — synthetic datasets)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 2: Final evaluation — synthetic datasets"
for DATASET in $SYNTHETIC_DATASETS; do
  echo "  Evaluating on $DATASET"
  python run_final_evaluation.py \
    --dataset "$DATASET" \
    --best_configs_dir "configs/best_hparams/$DATASET" \
    --output_dir "results/final/$DATASET" \
    --latent_dim 2 \
    --n_seeds 5 \
    --device "$DEVICE" \
    --seed $SEED
done

# ------------------------------------------------------------------
# Step 3: Final evaluation (Table 2 — real-world datasets, multi-dim)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 3: Final evaluation — real-world datasets"
for DATASET in $REAL_DATASETS; do
  for DIM in 2 16 32 64; do
    echo "  Evaluating on $DATASET (dim=$DIM)"
    python run_final_evaluation.py \
      --dataset "$DATASET" \
      --best_configs_dir "configs/best_hparams/$DATASET" \
      --output_dir "results/final/${DATASET}_dim${DIM}" \
      --latent_dim $DIM \
      --n_seeds 3 \
      --device "$DEVICE" \
      --seed $SEED
  done
done

# ------------------------------------------------------------------
# Step 4: Biological datasets
# ------------------------------------------------------------------
echo ""
echo ">>> Step 4: Biological datasets"
for DATASET in $BIO_DATASETS; do
  echo "  Evaluating on $DATASET"
  python run_final_evaluation.py \
    --dataset "$DATASET" \
    --best_configs_dir "configs/best_hparams/$DATASET" \
    --output_dir "results/final/$DATASET" \
    --latent_dim 2 \
    --n_seeds 3 \
    --device "$DEVICE" \
    --seed $SEED
done

# ------------------------------------------------------------------
# Step 5: Generate LaTeX tables
# ------------------------------------------------------------------
echo ""
echo ">>> Step 5: Generating LaTeX tables"
python generate_latex_table.py --results_dir results/final --output_dir results/latex

echo ""
echo "============================================================"
echo "Pipeline complete. Results in results/final/, tables in results/latex/"
echo "============================================================"
