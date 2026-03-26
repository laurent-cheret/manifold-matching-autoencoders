"""Diagnostic script to analyze MMAE variants on Swiss roll."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll

# Adjust imports based on your project structure
# from models import build_model
# from data import get_dataset

def generate_swiss_roll(n_samples=2000, noise=0.05):
    """Generate swiss roll with color labels."""
    X, t = make_swiss_roll(n_samples, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    return X, t

def simple_encoder(input_dim, hidden_dims, latent_dim):
    """Simple MLP encoder."""
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.extend([torch.nn.Linear(prev_dim, h), torch.nn.ReLU()])
        prev_dim = h
    layers.append(torch.nn.Linear(prev_dim, latent_dim))
    return torch.nn.Sequential(*layers)

def simple_decoder(latent_dim, hidden_dims, output_dim):
    """Simple MLP decoder."""
    layers = []
    prev_dim = latent_dim
    for h in reversed(hidden_dims):
        layers.extend([torch.nn.Linear(prev_dim, h), torch.nn.ReLU()])
        prev_dim = h
    layers.append(torch.nn.Linear(prev_dim, output_dim))
    return torch.nn.Sequential(*layers)


class DiagnosticMMAE(torch.nn.Module):
    """MMAE with diagnostic output."""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, loss_type='mse'):
        super().__init__()
        self.encoder = simple_encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = simple_decoder(latent_dim, hidden_dims, input_dim)
        self.loss_type = loss_type
        
    def forward(self, x, ref_emb=None):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        
        rec_loss = torch.nn.functional.mse_loss(x_rec, x)
        
        if ref_emb is None:
            return rec_loss, z, {'recon': rec_loss.item(), 'dist': 0.0}
        
        # Distance matrices
        z_dist = torch.cdist(z, z, p=2)
        ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
        
        if self.loss_type == 'mse':
            dist_loss = ((z_dist - ref_dist) ** 2).mean()
        elif self.loss_type == 'correlation':
            mask = torch.triu(torch.ones_like(z_dist), diagonal=1).bool()
            z_flat = z_dist[mask]
            ref_flat = ref_dist[mask]
            z_c = z_flat - z_flat.mean()
            ref_c = ref_flat - ref_flat.mean()
            corr = (z_c * ref_c).sum() / (torch.sqrt((z_c**2).sum() * (ref_c**2).sum()) + 1e-8)
            dist_loss = 1 - corr
        elif self.loss_type == 'rank':
            # Triplet-based
            B = z.shape[0]
            n_triplets = 256
            anchors = torch.randint(0, B, (n_triplets,))
            others1 = torch.randint(0, B, (n_triplets,))
            others2 = torch.randint(0, B, (n_triplets,))
            
            ref_d1 = (ref_emb[anchors] - ref_emb[others1]).norm(dim=1)
            ref_d2 = (ref_emb[anchors] - ref_emb[others2]).norm(dim=1)
            lat_d1 = (z[anchors] - z[others1]).norm(dim=1)
            lat_d2 = (z[anchors] - z[others2]).norm(dim=1)
            
            margin = 0.1
            ref_order = (ref_d1 < ref_d2).float()
            lat_diff = lat_d1 - lat_d2
            loss1 = ref_order * torch.relu(lat_diff + margin)
            loss2 = (1 - ref_order) * torch.relu(-lat_diff + margin)
            dist_loss = (loss1 + loss2).mean()
        
        return rec_loss + dist_loss, z, {
            'recon': rec_loss.item(), 
            'dist': dist_loss.item(),
            'z_dist_mean': z_dist.mean().item(),
            'ref_dist_mean': ref_dist.mean().item(),
            'z_dist_std': z_dist.std().item(),
            'ref_dist_std': ref_dist.std().item(),
        }


def train_and_diagnose(model, X, ref_emb, n_epochs=100, lr=1e-3, lam=1.0):
    """Train with detailed diagnostics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, z, info = model(X, ref_emb)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | recon: {info['recon']:.4f} | dist: {info['dist']:.4f} | "
                  f"z_dist: {info.get('z_dist_mean', 0):.2f}±{info.get('z_dist_std', 0):.2f} | "
                  f"ref_dist: {info.get('ref_dist_mean', 0):.2f}±{info.get('ref_dist_std', 0):.2f}")
        
        history.append(info)
    
    # Final embedding
    with torch.no_grad():
        _, z_final, _ = model(X, ref_emb)
    
    return z_final.numpy(), history


def main():
    print("=" * 60)
    print("MMAE Variants Diagnostic on Swiss Roll")
    print("=" * 60)
    
    # Generate data
    X, t = generate_swiss_roll(n_samples=1500, noise=0.05)
    print(f"\nData shape: {X.shape}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    
    # PCA reference
    pca = PCA(n_components=2)
    ref_emb = torch.tensor(pca.fit_transform(X.numpy()), dtype=torch.float32)
    print(f"PCA ref shape: {ref_emb.shape}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Check reference distances
    ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    print(f"Reference distances: mean={ref_dist.mean():.2f}, std={ref_dist.std():.2f}")
    
    # Input distances
    input_dist = torch.cdist(X, X, p=2)
    print(f"Input distances: mean={input_dist.mean():.2f}, std={input_dist.std():.2f}")
    
    # Train different variants
    variants = ['mse', 'correlation', 'rank']
    results = {}
    
    config = {
        'input_dim': 3,
        'hidden_dims': [64, 32],
        'latent_dim': 2,
    }
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training: {variant.upper()}")
        print('='*60)
        
        model = DiagnosticMMAE(
            config['input_dim'], 
            config['hidden_dims'], 
            config['latent_dim'],
            loss_type=variant
        )
        
        z, history = train_and_diagnose(model, X, ref_emb, n_epochs=200)
        results[variant] = z
    
    # Also train vanilla AE (no ref_emb)
    print(f"\n{'='*60}")
    print("Training: VANILLA (no distance loss)")
    print('='*60)
    model = DiagnosticMMAE(config['input_dim'], config['hidden_dims'], config['latent_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(200):
        optimizer.zero_grad()
        loss, z, info = model(X, None)  # No ref!
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | recon: {info['recon']:.4f}")
    with torch.no_grad():
        _, z_vanilla, _ = model(X, None)
    results['vanilla'] = z_vanilla.numpy()
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Latent spaces
    for idx, (name, z) in enumerate(results.items()):
        ax = axes[0, idx] if idx < 3 else axes[1, idx-3]
        sc = ax.scatter(z[:, 0], z[:, 1], c=t.numpy(), cmap='Spectral', s=5, alpha=0.7)
        ax.set_title(f'{name.upper()} latent space')
        ax.set_aspect('equal')
    
    # Row 2: Additional diagnostics
    # Original swiss roll
    ax = axes[1, 0]
    ax.scatter(X[:, 0], X[:, 2], c=t.numpy(), cmap='Spectral', s=5, alpha=0.7)
    ax.set_title('Original Swiss Roll (X-Z view)')
    
    # PCA reference
    ax = axes[1, 1]
    ax.scatter(ref_emb[:, 0], ref_emb[:, 1], c=t.numpy(), cmap='Spectral', s=5, alpha=0.7)
    ax.set_title('PCA Reference (target distances)')
    
    # Correlation analysis
    ax = axes[1, 2]
    for name, z in results.items():
        if name == 'vanilla':
            continue
        z_dist = np.linalg.norm(z[:, None] - z[None, :], axis=2)
        ref_dist_np = ref_dist.numpy()
        mask = np.triu(np.ones_like(z_dist), k=1).astype(bool)
        corr = np.corrcoef(z_dist[mask], ref_dist_np[mask])[0, 1]
        ax.bar(name, corr)
    ax.set_ylabel('Distance correlation with PCA ref')
    ax.set_title('How well did we match PCA distances?')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mmae_diagnostic.png', dpi=150)
    print(f"\nSaved visualization to mmae_diagnostic.png")
    
    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("If all variants look similar AND similar to PCA reference,")
    print("then the loss IS working - we're successfully matching PCA distances.")
    print("")
    print("The problem: PCA doesn't unfold! It's the REFERENCE that's wrong,")
    print("not the loss function.")
    print("")
    print("PCA of swiss roll gives a compressed/overlapping view,")
    print("not an unfolded view. So matching PCA distances = getting compressed view.")


if __name__ == '__main__':
    main()