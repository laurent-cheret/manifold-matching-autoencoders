"""Test MMAE-KNN on Swiss Roll - does it unfold?"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll


def simple_encoder(input_dim, hidden_dims, latent_dim):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.extend([torch.nn.Linear(prev_dim, h), torch.nn.ReLU()])
        prev_dim = h
    layers.append(torch.nn.Linear(prev_dim, latent_dim))
    return torch.nn.Sequential(*layers)


def simple_decoder(latent_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = latent_dim
    for h in reversed(hidden_dims):
        layers.extend([torch.nn.Linear(prev_dim, h), torch.nn.ReLU()])
        prev_dim = h
    layers.append(torch.nn.Linear(prev_dim, output_dim))
    return torch.nn.Sequential(*layers)


def knn_distance_loss(z, x, k=15):
    """Only preserve distances to k-nearest neighbors."""
    B = z.shape[0]
    
    x_dist = torch.cdist(x, x, p=2)
    z_dist = torch.cdist(z, z, p=2)
    
    x_dist_no_self = x_dist.clone()
    x_dist_no_self.fill_diagonal_(float('inf'))
    
    _, knn_idx = x_dist_no_self.topk(k, largest=False, dim=1)
    
    mask = torch.zeros(B, B, device=z.device, dtype=torch.bool)
    row_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(-1, k)
    mask[row_idx, knn_idx] = True
    mask = mask | mask.T
    
    x_masked = x_dist[mask]
    z_masked = z_dist[mask]
    
    x_norm = x_masked / (x_masked.max() + 1e-8)
    z_norm = z_masked / (z_masked.max() + 1e-8)
    
    return ((z_norm - x_norm) ** 2).mean()


class MMAEKNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, k=15, lam=1.0):
        super().__init__()
        self.encoder = simple_encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = simple_decoder(latent_dim, hidden_dims, input_dim)
        self.k = k
        self.lam = lam
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        
        rec_loss = torch.nn.functional.mse_loss(x_rec, x)
        knn_loss = knn_distance_loss(z, x, self.k)
        
        return rec_loss + self.lam * knn_loss, z, {
            'recon': rec_loss.item(),
            'knn': knn_loss.item()
        }


def main():
    print("Testing MMAE-KNN on Swiss Roll")
    print("=" * 50)
    
    # Generate swiss roll
    X, t = make_swiss_roll(n_samples=2000, noise=0.1)
    X = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    
    print(f"Data: {X.shape}")
    
    # Test different k values
    k_values = [5, 10, 15, 30, 50]
    results = {}
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        model = MMAEKNN(
            input_dim=3,
            hidden_dims=[128, 64],
            latent_dim=2,
            k=k,
            lam=1.0
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(300):
            optimizer.zero_grad()
            loss, z, info = model(X)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: recon={info['recon']:.4f}, knn={info['knn']:.4f}")
        
        with torch.no_grad():
            _, z_final, _ = model(X)
        
        results[k] = z_final.numpy()
    
    # Visualize
    n_plots = len(k_values) + 2
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Original
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 2], c=t.numpy(), cmap='Spectral', s=5, alpha=0.7)
    ax.set_title('Original (X-Z view)')
    ax.set_aspect('equal')
    
    # Ground truth unfolding (t vs y)
    ax = axes[1]
    ax.scatter(t.numpy(), X[:, 1].numpy(), c=t.numpy(), cmap='Spectral', s=5, alpha=0.7)
    ax.set_title('Ground Truth Unfolding (t vs y)')
    ax.set_aspect('equal')
    
    # k-NN results
    for idx, (k, z) in enumerate(results.items()):
        ax = axes[idx + 2]
        ax.scatter(z[:, 0], z[:, 1], c=t.numpy(), cmap='Spectral', s=5, alpha=0.7)
        ax.set_title(f'MMAE-KNN k={k}')
        ax.set_aspect('equal')
    
    # Hide extra axes
    for idx in range(len(k_values) + 2, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('mmae_knn_test.png', dpi=150)
    print(f"\nSaved to mmae_knn_test.png")
    
    # Quantitative: check if colors are separated (unfolded = smooth gradient)
    print("\n" + "=" * 50)
    print("Unfolding quality (lower = more unfolded):")
    print("=" * 50)
    
    for k, z in results.items():
        # Check if nearby points in latent have similar t values
        z_dist = np.linalg.norm(z[:, None] - z[None, :], axis=2)
        np.fill_diagonal(z_dist, np.inf)
        
        # For each point, get t values of 10 nearest latent neighbors
        knn_idx = np.argsort(z_dist, axis=1)[:, :10]
        t_np = t.numpy()
        
        # Variance of t among neighbors (low = smooth = unfolded)
        neighbor_t_var = np.mean([np.var(t_np[idx]) for idx in knn_idx])
        print(f"  k={k}: neighbor t-variance = {neighbor_t_var:.4f}")


if __name__ == '__main__':
    main()