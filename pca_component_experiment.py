#!/usr/bin/env python
"""
Experiment: PCA Component Selection for MMAE

Test different combinations of PCA components as reference.
Usage:
    python pca_component_experiment.py --dataset mammoth --components 0 1
    python pca_component_experiment.py --dataset mammoth --components 0 2
    python pca_component_experiment.py --dataset mammoth --components 1 2
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import from your project
from config import get_config
from data import load_data
from data.base import normalize_features, create_dataloaders
from models import build_model
from training import Trainer, get_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mammoth')
    parser.add_argument('--components', type=int, nargs='+', default=[0, 1],
                       help='Which PCA components to use, e.g., --components 0 2')
    parser.add_argument('--mmae_lambda', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    latent_dim = len(args.components)
    comp_str = '_'.join(map(str, args.components))
    print(f"=== PCA Components {args.components} (latent_dim={latent_dim}) ===")

    # Load raw data (without embeddings, we'll compute our own)
    config = get_config(args.dataset, 'mmae')
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        args.dataset, config, with_embeddings=False
    )
    
    # Extract raw data from datasets
    train_data = train_dataset.data.numpy()
    test_data = test_dataset.data.numpy()
    train_labels = train_dataset.labels.numpy()
    test_labels = test_dataset.labels.numpy()

    # Compute PCA and select components
    train_flat = train_data.reshape(train_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)
    
    max_comp = max(args.components) + 1
    pca = PCA(n_components=max_comp)
    train_pca_all = pca.fit_transform(train_flat)
    test_pca_all = pca.transform(test_flat)
    
    # Select only specified components
    train_emb = train_pca_all[:, args.components].astype(np.float32)
    test_emb = test_pca_all[:, args.components].astype(np.float32)
    
    print(f"PCA variance explained: {pca.explained_variance_ratio_}")
    print(f"Selected components variance: {pca.explained_variance_ratio_[args.components]}")

    # Create new dataloaders with our custom embeddings
    train_loader, test_loader, _, _ = create_dataloaders(
        train_data, test_data, train_labels, test_labels,
        batch_size=64, train_emb=train_emb, test_emb=test_emb
    )

    # Build and train
    config['latent_dim'] = latent_dim
    config['mmae_lambda'] = args.mmae_lambda
    model = build_model('mmae', config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, args.device, model_name='mmae')
    trainer.fit(train_loader, test_loader, n_epochs=args.epochs, verbose=True)

    # Plot
    latents, labels = get_latents(model, test_loader, args.device)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reference PCA
    axes[0].scatter(test_emb[:, 0], test_emb[:, 1], c=test_labels, cmap='Spectral', s=10, alpha=0.7)
    axes[0].set_title(f'Reference: PCA[{args.components}]')
    axes[0].axis('equal')
    
    # Learned latent
    axes[1].scatter(latents[:, 0], latents[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.7)
    axes[1].set_title(f'MMAE Latent (trained on PCA[{args.components}])')
    axes[1].axis('equal')
    
    plt.suptitle(f'MMAE with PCA components {args.components}')
    plt.tight_layout()
    plt.savefig(f'pca_components_{comp_str}.png', dpi=150)
    plt.show()
    print(f"Saved: pca_components_{comp_str}.png")


if __name__ == '__main__':
    main()