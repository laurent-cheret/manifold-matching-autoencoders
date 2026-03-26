#!/usr/bin/env python
"""
Hyperparameter search with config file support and checkpointing.

Usage:
    # Run all models and latent dims from config
    python run_hyperparam_search.py --config experiments/hyperparam_search/mnist/config.json
    
    # Run specific model only
    python run_hyperparam_search.py --config experiments/hyperparam_search/mnist/config.json --model mmae
    
    # Run specific latent dim only
    python run_hyperparam_search.py --config experiments/hyperparam_search/mnist/config.json --latent_dim 2
    
    # Resume interrupted search (automatic - just run same command again)
    python run_hyperparam_search.py --config experiments/hyperparam_search/mnist/config.json
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
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

# Import from project
from config import get_config, DATASET_CONFIGS
from data import load_data
from models import build_model
from training import Trainer
from evaluation import evaluate


# Metrics where lower is better
LOWER_IS_BETTER = [
    'reconstruction_error', 'wasserstein_H0', 'wasserstein_H1', 
    'rmse', 'mrre_zx', 'mrre_xz', 
    'density_kl_0_01', 'density_kl_0_1', 'density_kl_1_0'
]


def load_search_config(config_path):
    """Load search configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_output_dir(base_dir, model_name, latent_dim):
    """Get output directory for a specific model and latent dim."""
    return os.path.join(base_dir, f'{model_name}_dim{latent_dim}')


def load_checkpoint(output_dir):
    """Load checkpoint if exists, returns (completed_trials, best_score, best_params, best_trial)."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"  Resuming from trial {checkpoint['completed_trials'] + 1}")
        return checkpoint
    return None


def save_checkpoint(output_dir, completed_trials, best_score, best_params, best_trial):
    """Save checkpoint for resuming."""
    checkpoint = {
        'completed_trials': completed_trials,
        'best_score': best_score,
        'best_params': best_params,
        'best_trial': best_trial,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'checkpoint.json'), 'w') as f:
        json.dump(checkpoint, f, indent=2)


def save_trial_result(output_dir, trial_data):
    """Append trial result to CSV (creates if doesn't exist)."""
    csv_path = os.path.join(output_dir, 'trials.csv')
    df_new = pd.DataFrame([trial_data])
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(csv_path, index=False)


def save_best_config(output_dir, model_name, dataset_name, latent_dim, target_metric, best_score, best_params, all_metrics=None):
    """Save best configuration found."""
    best_config = {
        'model': model_name,
        'dataset': dataset_name,
        'latent_dim': latent_dim,
        'target_metric': target_metric,
        'best_score': float(best_score),
        'hyperparameters': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in best_params.items()},
        'timestamp': datetime.now().isoformat()
    }
    if all_metrics:
        best_config['all_metrics'] = all_metrics
    
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)


class ConfigBasedSearch:
    """Hyperparameter search based on config file."""
    
    def __init__(self, search_config, model_name, latent_dim, output_dir, device='cuda', seed=42):
        self.search_config = search_config
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.device = device
        self.seed = seed
        
        self.dataset_name = search_config['dataset']
        self.common = search_config['common']
        self.model_config = search_config['models'].get(model_name, {})
        
        # Get input dim for PCA bounds
        self.input_dim = DATASET_CONFIGS[self.dataset_name]['input_dim']
        
        # Target metric
        self.target_metric = self.common.get('target_metric', 'density_kl_0_1')
        self.direction = 'minimize' if self.target_metric in LOWER_IS_BETTER else 'maximize'
        
        # Data cache
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
            self._data_cache[cache_key] = data
        
        return self._data_cache[cache_key]
    
    def sample_hyperparameters(self, rng):
        """Sample hyperparameters based on config."""
        params = {}
        common = self.common
        model_cfg = self.model_config
        
        # Common params (can be overridden per model)
        lr_min = model_cfg.get('lr_min', common.get('lr_min', 1e-4))
        lr_max = model_cfg.get('lr_max', common.get('lr_max', 1e-2))
        params['learning_rate'] = float(np.exp(rng.uniform(np.log(lr_min), np.log(lr_max))))
        
        # Batch size - check model-specific first, then common
        bs_min = model_cfg.get('batch_size_min', common.get('batch_size_min', 16))
        bs_max = model_cfg.get('batch_size_max', common.get('batch_size_max', 256))
        params['batch_size'] = int(rng.integers(bs_min, bs_max + 1))
        
        # Model-specific params
        if self.model_name == 'mmae':
            pca_min = model_cfg.get('mmae_pca_min', 2)
            pca_max_pct = model_cfg.get('mmae_pca_max_percent', 0.8)
            pca_max = int(self.input_dim * pca_max_pct)
            params['mmae_n_components'] = int(rng.integers(pca_min, pca_max + 1))
            
            lam_min = model_cfg.get('mmae_lambda_min', 0.1)
            lam_max = model_cfg.get('mmae_lambda_max', 10.0)
            params['mmae_lambda'] = float(np.exp(rng.uniform(np.log(lam_min), np.log(lam_max))))
        
        elif self.model_name == 'topoae':
            lam_min = model_cfg.get('topo_lambda_min', 0.01)
            lam_max = model_cfg.get('topo_lambda_max', 10.0)
            params['topo_lambda'] = float(np.exp(rng.uniform(np.log(lam_min), np.log(lam_max))))
        
        elif self.model_name == 'rtdae':
            lam_min = model_cfg.get('rtd_lambda_min', 0.01)
            lam_max = model_cfg.get('rtd_lambda_max', 10.0)
            params['rtd_lambda'] = float(np.exp(rng.uniform(np.log(lam_min), np.log(lam_max))))
            
            dim_choices = model_cfg.get('rtd_dim_choices', [0, 1])
            params['rtd_dim'] = int(rng.choice(dim_choices))
            
            card_min = model_cfg.get('rtd_card_min', 20)
            card_max = model_cfg.get('rtd_card_max', 100)
            params['rtd_card'] = int(rng.integers(card_min, card_max + 1))
        
        return params
    
    def run_trial(self, params, trial_seed):
        """Run a single trial with given hyperparameters."""
        # Build config
        config = get_config(self.dataset_name, self.model_name)
        config['latent_dim'] = self.latent_dim
        config['device'] = self.device
        config['seed'] = trial_seed
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
        
        # Load data
        n_components = params.get('mmae_n_components')
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = self._get_data(
            config['batch_size'], n_components
        )
        
        # Create model and optimizer
        model = build_model(self.model_name, config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Train
        trainer = Trainer(model, optimizer, device=self.device, model_name=self.model_name)
        epochs = self.common.get('epochs', 50)
        start_time = time.time()
        history = trainer.fit(train_loader, val_loader, n_epochs=epochs, verbose=False)
        train_time = time.time() - start_time

        # Evaluate on val set for HPO metric computation
        test_data = val_dataset.data.numpy()
        test_labels = val_dataset.labels.numpy()

        model.eval()
        with torch.no_grad():
            test_tensor = torch.from_numpy(test_data).float().to(self.device)
            embeddings = model.encode(test_tensor).cpu().numpy()
            reconstructions = model.decode(model.encode(test_tensor)).cpu().numpy()

        # Compute reconstruction error
        reconstruction_error = float(np.mean((test_data - reconstructions) ** 2))

        metrics = evaluate(test_data, embeddings, test_labels, ks=[10, 50])
        metrics['reconstruction_error'] = reconstruction_error
        metrics['train_time'] = train_time
        
        return metrics
    
    def run(self):
        """Run the hyperparameter search."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # n_trials = self.common.get('n_trials', 50)
        n_trials = self.model_config.get('n_trials', self.common.get('n_trials', 50))
        # Check for existing checkpoint
        checkpoint = load_checkpoint(self.output_dir)
        if checkpoint:
            start_trial = checkpoint['completed_trials']
            best_score = checkpoint['best_score']
            best_params = checkpoint['best_params']
            best_trial = checkpoint['best_trial']
        else:
            start_trial = 0
            best_score = float('inf') if self.direction == 'minimize' else float('-inf')
            best_params = None
            best_trial = None
        
        if start_trial >= n_trials:
            print(f"  All {n_trials} trials already completed. Skipping.")
            return best_params, best_score
        
        rng = np.random.default_rng(self.seed)
        # Advance RNG to correct position if resuming
        for _ in range(start_trial):
            self.sample_hyperparameters(rng)
        
        print(f"\n{'='*80}")
        print(f"Hyperparameter Search: {self.model_name.upper()} on {self.dataset_name} (dim={self.latent_dim})")
        print(f"{'='*80}")
        print(f"Target metric: {self.target_metric} ({self.direction})")
        
        # Show batch size range (per-model or common)
        bs_min = self.model_config.get('batch_size_min', self.common.get('batch_size_min', 16))
        bs_max = self.model_config.get('batch_size_max', self.common.get('batch_size_max', 256))
        lr_min = self.model_config.get('lr_min', self.common.get('lr_min', 1e-4))
        lr_max = self.model_config.get('lr_max', self.common.get('lr_max', 1e-2))
        print(f"Batch size: [{bs_min}, {bs_max}], LR: [{lr_min:.0e}, {lr_max:.0e}]")
        
        print(f"Trials: {start_trial}/{n_trials} completed, running {n_trials - start_trial} more")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")
        
        best_metrics = None
        
        for trial in range(start_trial, n_trials):
            params = self.sample_hyperparameters(rng)
            
            print(f"Trial {trial + 1}/{n_trials}")
            print(f"  Params: {params}")
            
            try:
                metrics = self.run_trial(params, self.seed + trial)
                score = metrics.get(self.target_metric, float('inf') if self.direction == 'minimize' else float('-inf'))
                
                # Check if best
                is_better = (self.direction == 'minimize' and score < best_score) or \
                           (self.direction == 'maximize' and score > best_score)
                
                if is_better:
                    best_score = score
                    best_params = params.copy()
                    best_trial = trial + 1
                    best_metrics = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}
                    print(f"  *** New best! {self.target_metric}={score:.6f} ***")
                else:
                    print(f"  {self.target_metric}={score:.6f} (best={best_score:.6f})")
                
                # Save trial result
                trial_data = {
                    'trial': trial + 1,
                    **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in params.items()},
                    **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
                    'is_best': is_better
                }
                save_trial_result(self.output_dir, trial_data)
                
                # Save checkpoint
                save_checkpoint(self.output_dir, trial + 1, best_score, best_params, best_trial)
                
                # Save best config
                if best_params:
                    save_best_config(
                        self.output_dir, self.model_name, self.dataset_name,
                        self.latent_dim, self.target_metric, best_score, best_params, best_metrics
                    )
                
            except Exception as e:
                print(f"  Trial failed: {e}")
                trial_data = {
                    'trial': trial + 1,
                    **params,
                    'error': str(e)
                }
                save_trial_result(self.output_dir, trial_data)
                save_checkpoint(self.output_dir, trial + 1, best_score, best_params, best_trial)
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"SEARCH COMPLETE: {self.model_name} on {self.dataset_name} (dim={self.latent_dim})")
        print(f"{'='*80}")
        print(f"Best trial: {best_trial}")
        print(f"Best {self.target_metric}: {best_score:.6f}")
        print(f"Best hyperparameters:")
        if best_params:
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        print(f"Results saved to: {self.output_dir}")
        
        return best_params, best_score


def run_all_searches(config_path, output_base_dir, model_filter=None, latent_dim_filter=None, device='cuda', seed=42):
    """Run searches for all enabled models and latent dims."""
    search_config = load_search_config(config_path)
    
    dataset_name = search_config['dataset']
    latent_dims = search_config['common'].get('latent_dims', [2])
    
    # Filter latent dims if specified
    if latent_dim_filter is not None:
        latent_dims = [d for d in latent_dims if d == latent_dim_filter]
    
    # Get enabled models
    models = []
    for model_name, model_cfg in search_config['models'].items():
        if model_cfg.get('enabled', False):
            if model_filter is None or model_name == model_filter:
                models.append(model_name)
    
    print(f"\n{'#'*80}")
    print(f"HYPERPARAMETER SEARCH CAMPAIGN")
    print(f"{'#'*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Models: {models}")
    print(f"Latent dims: {latent_dims}")
    print(f"Output: {output_base_dir}")
    print(f"{'#'*80}\n")
    
    results_summary = []
    
    for latent_dim in latent_dims:
        for model_name in models:
            output_dir = get_output_dir(output_base_dir, model_name, latent_dim)
            
            search = ConfigBasedSearch(
                search_config=search_config,
                model_name=model_name,
                latent_dim=latent_dim,
                output_dir=output_dir,
                device=device,
                seed=seed
            )
            
            best_params, best_score = search.run()
            
            results_summary.append({
                'model': model_name,
                'latent_dim': latent_dim,
                'best_score': best_score,
                'output_dir': output_dir
            })
    
    # Save overall summary
    summary_path = os.path.join(output_base_dir, 'search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'config_path': config_path,
            'results': results_summary,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n{'#'*80}")
    print(f"ALL SEARCHES COMPLETE")
    print(f"{'#'*80}")
    print(f"Summary saved to: {summary_path}")
    for r in results_summary:
        print(f"  {r['model']} dim={r['latent_dim']}: best_score={r['best_score']:.6f}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search from config file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to search config JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same folder as config)')
    parser.add_argument('--model', type=str, default=None,
                       choices=['vanilla', 'mmae', 'topoae', 'rtdae'],
                       help='Run only this model (default: all enabled)')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Run only this latent dim (default: all from config)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        config_dir = os.path.dirname(args.config)
        args.output_dir = os.path.join(config_dir, 'results')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run searches
    run_all_searches(
        config_path=args.config,
        output_base_dir=args.output_dir,
        model_filter=args.model,
        latent_dim_filter=args.latent_dim,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()