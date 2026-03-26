#!/usr/bin/env python
"""
Hyperparameter search using Optuna (Bayesian optimization).

Usage:
    # MMAE search optimizing density_kl_0_1
    python hyperparam_search_optuna.py --model mmae --dataset mnist --latent_dim 2 --metric density_kl_0_1 --n_trials 50

    # With pruning (early stopping of bad trials)
    python hyperparam_search_optuna.py --model mmae --dataset mnist --latent_dim 2 --n_trials 100 --pruning

    # Grid search mode (exhaustive)
    python hyperparam_search_optuna.py --model mmae --dataset mnist --latent_dim 2 --sampler grid
"""

import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import torch

try:
    import optuna
    from optuna.samplers import TPESampler, GridSampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

# Import from project
from config import get_config, DATASET_CONFIGS
from data import load_data, set_global_seed
from models import build_model
from training import Trainer
from evaluation import evaluate


# Metrics where lower is better
LOWER_IS_BETTER = [
    'reconstruction_error', 'wasserstein_H0', 'wasserstein_H1', 
    'rmse', 'mrre_zx', 'mrre_xz', 
    'density_kl_0_01', 'density_kl_0_1', 'density_kl_1_0'
]


class HyperparameterSearch:
    """Optuna-based hyperparameter search."""
    
    def __init__(
        self,
        model_name,
        dataset_name,
        latent_dim,
        target_metric,
        epochs,
        device,
        seed=42,
        batch_sizes=None,
        lr_min=1e-4,
        lr_max=1e-2
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.latent_dim = latent_dim
        self.target_metric = target_metric
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.batch_sizes = batch_sizes or [16, 32, 64, 128, 256]
        self.lr_min = lr_min
        self.lr_max = lr_max
        
        # Precompute bounds
        self.input_dim = DATASET_CONFIGS[dataset_name]['input_dim']
        self.max_pca = int(self.input_dim * 0.8)
        
        # Direction
        self.direction = 'minimize' if target_metric in LOWER_IS_BETTER else 'maximize'
        
        # Cache dataset to avoid reloading
        self._data_cache = {}
    
    def _get_data(self, batch_size, n_components=None):
        """Get data loaders, caching when possible."""
        cache_key = (batch_size, n_components)
        
        if cache_key not in self._data_cache:
            config = get_config(self.dataset_name, self.model_name)
            config['batch_size'] = batch_size
            config['latent_dim'] = self.latent_dim
            
            if n_components is not None:
                config['mmae_n_components'] = n_components
            
            needs_embeddings = self.model_name == 'mmae'
            data = load_data(self.dataset_name, config, with_embeddings=needs_embeddings)
            # data is (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
            self._data_cache[cache_key] = data
        
        return self._data_cache[cache_key]
    
    def suggest_hyperparameters(self, trial):
        """Suggest hyperparameters using Optuna trial."""
        params = {}
        
        # Common hyperparameters
        params['learning_rate'] = trial.suggest_float('learning_rate', self.lr_min, self.lr_max, log=True)
        params['batch_size'] = trial.suggest_int('batch_size', 16, 512, log=True)
        
        # Model-specific hyperparameters
        if self.model_name == 'mmae':
            params['mmae_n_components'] = trial.suggest_int('mmae_n_components', 2, self.max_pca)
            params['mmae_lambda'] = trial.suggest_float('mmae_lambda', 0.1, 10.0, log=True)
        
        elif self.model_name == 'topoae':
            params['topo_lambda'] = trial.suggest_float('topo_lambda', 0.01, 10.0, log=True)
        
        elif self.model_name == 'rtdae':
            params['rtd_lambda'] = trial.suggest_float('rtd_lambda', 0.01, 10.0, log=True)
            params['rtd_dim'] = trial.suggest_categorical('rtd_dim', [0, 1])
            params['rtd_card'] = trial.suggest_int('rtd_card', 20, 100)
        
        return params
    
    def objective(self, trial):
        """Optuna objective function."""
        # Get hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        # Build config
        config = get_config(self.dataset_name, self.model_name)
        config['latent_dim'] = self.latent_dim
        config['device'] = self.device
        config['seed'] = self.seed + trial.number
        config['learning_rate'] = params['learning_rate']
        config['batch_size'] = int(params['batch_size'])
        
        # Model-specific params
        if self.model_name == 'mmae':
            config['mmae_n_components'] = int(params['mmae_n_components'])
            config['mmae_lambda'] = params['mmae_lambda']
        elif self.model_name == 'topoae':
            config['topo_lambda'] = params['topo_lambda']
        elif self.model_name == 'rtdae':
            config['rtd_lambda'] = params['rtd_lambda']
            config['rtd_dim'] = int(params['rtd_dim'])
            config['rtd_card'] = int(params['rtd_card'])
        
        # Load data — use val set for HPO metric, not test set
        n_components = params.get('mmae_n_components')
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = self._get_data(
            config['batch_size'], n_components
        )
        
        # Create model
        model = build_model(self.model_name, config)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Set seed for this trial for reproducibility
        set_global_seed(self.seed + trial.number)

        # Train — use val_loader for monitoring, not test_loader
        trainer = Trainer(model, optimizer, device=self.device, model_name=self.model_name)
        start_time = time.time()
        history = trainer.fit(train_loader, val_loader, n_epochs=self.epochs, verbose=False)
        train_time = time.time() - start_time

        # Evaluate HPO metric on validation set (test set is never touched during search)
        val_data = val_dataset.data.numpy()
        val_labels = val_dataset.labels.numpy()

        model.eval()
        with torch.no_grad():
            val_tensor = torch.from_numpy(val_data).float().to(self.device)
            embeddings = model.encode(val_tensor).cpu().numpy()

        # Pass opt_metric for dynamic routing — skips unneeded expensive metrics
        metrics = evaluate(val_data, embeddings, val_labels, ks=[10, 50],
                           compute_wasserstein=(self.target_metric in {'wasserstein_H0', 'wasserstein_H1'}),
                           opt_metric=self.target_metric)
        
        # Store all metrics as user attributes
        for k, v in metrics.items():
            trial.set_user_attr(k, float(v) if isinstance(v, (np.floating, float)) else v)
        trial.set_user_attr('train_time', train_time)
        
        # Return target metric
        score = metrics.get(self.target_metric, float('inf') if self.direction == 'minimize' else float('-inf'))
        return float(score)
    
    def run(self, n_trials, sampler_type='tpe', results_dir=None, pruning=False):
        """Run the hyperparameter search."""
        # Create sampler
        if sampler_type == 'tpe':
            sampler = TPESampler(seed=self.seed)
        elif sampler_type == 'random':
            sampler = RandomSampler(seed=self.seed)
        elif sampler_type == 'grid':
            # Define grid for grid search
            search_space = self._get_grid_search_space()
            sampler = GridSampler(search_space)
        else:
            sampler = TPESampler(seed=self.seed)
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=f'{self.model_name}_{self.dataset_name}_dim{self.latent_dim}'
        )
        
        # Optimize
        print(f"\n{'='*80}")
        print(f"Hyperparameter Search: {self.model_name.upper()} on {self.dataset_name}")
        print(f"{'='*80}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Target metric: {self.target_metric} ({self.direction})")
        print(f"Sampler: {sampler_type}")
        print(f"Number of trials: {n_trials}")
        print(f"Epochs per trial: {self.epochs}")
        print(f"{'='*80}\n")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Results
        print(f"\n{'='*80}")
        print(f"SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Best trial: {study.best_trial.number + 1}")
        print(f"Best {self.target_metric}: {study.best_value:.6f}")
        print(f"Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        
        # Save results
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

            # Save trials dataframe
            df = study.trials_dataframe()
            df.to_csv(os.path.join(results_dir, 'search_results.csv'), index=False)

            # Save best config in structured path: configs/best_hparams/<dataset>/<dim>D_best_config.json
            best_config = {
                'model': self.model_name,
                'dataset': self.dataset_name,
                'latent_dim': self.latent_dim,
                'target_metric': self.target_metric,
                'best_score': float(study.best_value),
                'best_trial': study.best_trial.number + 1,
                'hyperparameters': study.best_params,
                'all_metrics': study.best_trial.user_attrs
            }

            with open(os.path.join(results_dir, 'best_config.json'), 'w') as f:
                json.dump(best_config, f, indent=2)

            # Also write to canonical path for run_final_evaluation.py
            canonical_dir = os.path.join(
                'configs', 'best_hparams', self.dataset_name
            )
            os.makedirs(canonical_dir, exist_ok=True)
            canonical_path = os.path.join(canonical_dir, f'{self.latent_dim}D_best_config_{self.model_name}.json')
            with open(canonical_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"  Canonical config saved to: {canonical_path}")

            print(f"\nResults saved to: {results_dir}")
        
        return study
    
    def _get_grid_search_space(self):
        """Define grid for grid search."""
        # Generate log-spaced learning rates
        lr_values = np.logspace(np.log10(self.lr_min), np.log10(self.lr_max), 5).tolist()
        
        space = {
            'learning_rate': lr_values,
            'batch_size': self.batch_sizes,
        }
        
        if self.model_name == 'mmae':
            # Sample PCA components at specific percentages
            pca_values = [2] + [int(self.input_dim * p) for p in [0.1, 0.3, 0.5, 0.8]]
            space['mmae_n_components'] = pca_values
            space['mmae_lambda'] = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        elif self.model_name == 'topoae':
            space['topo_lambda'] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        elif self.model_name == 'rtdae':
            space['rtd_lambda'] = [0.1, 0.5, 1.0, 2.0, 5.0]
            space['rtd_dim'] = [0, 1]
            space['rtd_card'] = [30, 50, 80]
        
        return space


def visualize_search_results(results_dir):
    """Visualize hyperparameter search results."""
    import matplotlib.pyplot as plt
    
    # Load results
    df = pd.read_csv(os.path.join(results_dir, 'search_results.csv'))
    
    with open(os.path.join(results_dir, 'best_config.json')) as f:
        best_config = json.load(f)
    
    target_metric = best_config['target_metric']
    
    # Plot optimization history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Optimization history
    ax = axes[0, 0]
    values = df['value'].values
    best_so_far = np.minimum.accumulate(values) if target_metric in LOWER_IS_BETTER else np.maximum.accumulate(values)
    ax.plot(values, 'o', alpha=0.5, label='Trial value')
    ax.plot(best_so_far, '-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel(target_metric)
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter importance (if enough trials)
    ax = axes[0, 1]
    param_cols = [c for c in df.columns if c.startswith('params_')]
    if len(param_cols) > 0 and len(df) > 10:
        correlations = []
        for col in param_cols:
            if df[col].dtype in [np.float64, np.int64]:
                corr = df[col].corr(df['value'])
                correlations.append((col.replace('params_', ''), abs(corr)))
        
        if correlations:
            correlations.sort(key=lambda x: x[1], reverse=True)
            names, values = zip(*correlations)
            ax.barh(names, values)
            ax.set_xlabel('|Correlation| with target')
            ax.set_title('Parameter Importance')
    else:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
        ax.set_title('Parameter Importance')
    
    # 3. Learning rate vs score
    ax = axes[1, 0]
    if 'params_learning_rate' in df.columns:
        ax.scatter(df['params_learning_rate'], df['value'], alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(target_metric)
        ax.set_title('Learning Rate vs Score')
        ax.grid(True, alpha=0.3)
    
    # 4. Model-specific parameter
    ax = axes[1, 1]
    if 'params_mmae_n_components' in df.columns:
        ax.scatter(df['params_mmae_n_components'], df['value'], alpha=0.6)
        ax.set_xlabel('PCA Components')
        ax.set_ylabel(target_metric)
        ax.set_title('PCA Components vs Score')
        ax.grid(True, alpha=0.3)
    elif 'params_topo_lambda' in df.columns:
        ax.scatter(df['params_topo_lambda'], df['value'], alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('Topo Lambda')
        ax.set_ylabel(target_metric)
        ax.set_title('Topo Lambda vs Score')
        ax.grid(True, alpha=0.3)
    elif 'params_rtd_lambda' in df.columns:
        ax.scatter(df['params_rtd_lambda'], df['value'], alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('RTD Lambda')
        ax.set_ylabel(target_metric)
        ax.set_title('RTD Lambda vs Score')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'search_visualization.png'), dpi=150)
    print(f"Saved visualization to {os.path.join(results_dir, 'search_visualization.png')}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search using Optuna')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to optimize')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to use')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Single latent dimension (use --latent_dims for multiple)')
    parser.add_argument('--latent_dims', type=int, nargs='+', default=None,
                       help='List of latent dimensions to search over (e.g. --latent_dims 2 16 32 64)')
    parser.add_argument('--metric', '--opt_metric', dest='metric', type=str, default='density_kl_0_1',
                       help='Metric to optimize (default: density_kl_0_1)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials (default: 50)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per trial (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--sampler', type=str, default='tpe',
                       choices=['tpe', 'random', 'grid'],
                       help='Sampling strategy (default: tpe)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 64, 128, 256],
                       help='Batch sizes to search over (default: 16 32 64 128 256)')
    parser.add_argument('--lr_min', type=float, default=1e-4,
                       help='Minimum learning rate (default: 1e-4)')
    parser.add_argument('--lr_max', type=float, default=1e-2,
                       help='Maximum learning rate (default: 1e-2)')
    parser.add_argument('--pruning', action='store_true',
                       help='Enable pruning (early stopping of bad trials)')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Visualize existing results from directory')
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required. Install with: pip install optuna")
        return
    
    # Visualization mode
    if args.visualize:
        visualize_search_results(args.visualize)
        return
    
    # Resolve which latent dims to sweep
    if args.latent_dims is not None:
        dims_to_run = args.latent_dims
    elif args.latent_dim is not None:
        dims_to_run = [args.latent_dim]
    else:
        print("Error: provide --latent_dim or --latent_dims")
        return

    for latent_dim in dims_to_run:
        print(f"\n{'='*80}")
        print(f"LATENT DIM: {latent_dim}")
        print(f"{'='*80}")

        # Setup results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_results_dir = args.results_dir or \
            f'results/hypersearch/{args.model}_{args.dataset}_dim{latent_dim}_{timestamp}'

        os.makedirs(run_results_dir, exist_ok=True)

        # Save search config
        search_config = {**vars(args), 'latent_dim_run': latent_dim}
        with open(os.path.join(run_results_dir, 'search_config.json'), 'w') as f:
            json.dump(search_config, f, indent=2)

        # Run search
        search = HyperparameterSearch(
            model_name=args.model,
            dataset_name=args.dataset,
            latent_dim=latent_dim,
            target_metric=args.metric,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
            batch_sizes=args.batch_sizes,
            lr_min=args.lr_min,
            lr_max=args.lr_max
        )

        study = search.run(
            n_trials=args.n_trials,
            sampler_type=args.sampler,
            results_dir=run_results_dir,
            pruning=args.pruning
        )

        visualize_search_results(run_results_dir)


if __name__ == '__main__':
    main()