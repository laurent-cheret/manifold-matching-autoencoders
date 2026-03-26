"""MMAE-KNN: Local-only distance preservation.

Key insight: Only preserve distances to k-nearest neighbors in INPUT space.
Everything else is free to rearrange → allows unfolding.

No PCA reference needed. No graph Laplacian. Just masked distance matching.
"""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder


def knn_distance_loss(z, x, k=15):
    """
    Only preserve distances to k-nearest neighbors in input space.
    
    Args:
        z: Latent embeddings (B, latent_dim)
        x: Input data (B, input_dim) - used to find neighbors
        k: Number of neighbors to preserve
    
    Returns:
        Loss scalar
    """
    B = z.shape[0]
    
    if B <= k:
        # Fall back to full distance matching for tiny batches
        z_dist = torch.cdist(z, z, p=2)
        x_dist = torch.cdist(x, x, p=2)
        return ((z_dist - x_dist) ** 2).mean()
    
    # Input distances (find neighbors in original space)
    x_dist = torch.cdist(x, x, p=2)  # (B, B)
    
    # Latent distances
    z_dist = torch.cdist(z, z, p=2)  # (B, B)
    
    # Find k-nearest neighbors for each point (excluding self)
    # Add large value to diagonal to exclude self
    x_dist_no_self = x_dist.clone()
    x_dist_no_self.fill_diagonal_(float('inf'))
    
    _, knn_idx = x_dist_no_self.topk(k, largest=False, dim=1)  # (B, k)
    
    # Create mask for k-NN pairs
    mask = torch.zeros(B, B, device=z.device, dtype=torch.bool)
    row_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(-1, k)
    mask[row_idx, knn_idx] = True
    
    # Make symmetric (if i is neighbor of j, also consider j neighbor of i)
    mask = mask | mask.T
    
    # Normalize distances to be scale-invariant
    x_dist_masked = x_dist[mask]
    z_dist_masked = z_dist[mask]
    
    # Normalize by max to make scale-invariant
    x_norm = x_dist_masked / (x_dist_masked.max() + 1e-8)
    z_norm = z_dist_masked / (z_dist_masked.max() + 1e-8)
    
    return ((z_norm - x_norm) ** 2).mean()


def knn_ratio_loss(z, x, k=15):
    """
    Preserve distance RATIOS among k-nearest neighbors.
    More flexible than absolute distances.
    """
    B = z.shape[0]
    
    if B <= k:
        return torch.tensor(0.0, device=z.device)
    
    x_dist = torch.cdist(x, x, p=2)
    z_dist = torch.cdist(z, z, p=2)
    
    x_dist_no_self = x_dist.clone()
    x_dist_no_self.fill_diagonal_(float('inf'))
    
    _, knn_idx = x_dist_no_self.topk(k, largest=False, dim=1)
    
    # For each point, get distances to its k neighbors
    row_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(-1, k)
    
    x_knn_dist = x_dist[row_idx, knn_idx]  # (B, k)
    z_knn_dist = z_dist[row_idx, knn_idx]  # (B, k)
    
    # Normalize each point's neighbor distances (preserve ratios)
    x_knn_norm = x_knn_dist / (x_knn_dist.sum(dim=1, keepdim=True) + 1e-8)
    z_knn_norm = z_knn_dist / (z_knn_dist.sum(dim=1, keepdim=True) + 1e-8)
    
    return ((z_knn_norm - x_knn_norm) ** 2).mean()


def knn_stress_loss(z, x, k=15):
    """
    Kruskal stress but only on k-NN distances.
    Finds optimal scaling then measures fit.
    """
    B = z.shape[0]
    
    if B <= k:
        return torch.tensor(0.0, device=z.device)
    
    x_dist = torch.cdist(x, x, p=2)
    z_dist = torch.cdist(z, z, p=2)
    
    x_dist_no_self = x_dist.clone()
    x_dist_no_self.fill_diagonal_(float('inf'))
    
    _, knn_idx = x_dist_no_self.topk(k, largest=False, dim=1)
    
    # Create mask
    mask = torch.zeros(B, B, device=z.device, dtype=torch.bool)
    row_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(-1, k)
    mask[row_idx, knn_idx] = True
    mask = mask | mask.T
    
    x_masked = x_dist[mask]
    z_masked = z_dist[mask]
    
    # Optimal scale: min_α ||αz - x||²  →  α = (z·x)/(z·z)
    scale = (z_masked * x_masked).sum() / (z_masked ** 2).sum().clamp(min=1e-8)
    z_scaled = z_masked * scale
    
    # Normalized stress
    stress = ((z_scaled - x_masked) ** 2).sum() / (x_masked ** 2).sum().clamp(min=1e-8)
    
    return stress


@register_model('mmae_knn')
class MMAEKNN(nn.Module):
    """MMAE with k-NN only distance preservation.
    
    No PCA reference needed - uses input distances directly,
    but only for k-nearest neighbors. Distant points are free
    to rearrange, enabling manifold unfolding.
    
    Args:
        config: Dictionary with:
            - input_dim, latent_dim, hidden_dims: Architecture
            - mmae_lambda: Weight for distance loss (default 1.0)
            - mmae_k: Number of neighbors (default 15)
            - mmae_knn_loss: 'mse', 'ratio', or 'stress' (default 'mse')
    """
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get("mmae_lambda", 1.0)
        self.k = config.get("mmae_k", 15)
        self.loss_type = config.get("mmae_knn_loss", "mse")
        self.recon_loss = nn.MSELoss()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, ref_emb=None):
        """
        Forward pass.
        
        Args:
            x: Input data (B, input_dim)
            ref_emb: IGNORED - we use input distances directly
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        # Flatten x for distance computation
        x_flat = x.view(x.shape[0], -1)
        
        if self.loss_type == 'mse':
            dist_loss = knn_distance_loss(z, x_flat, self.k)
        elif self.loss_type == 'ratio':
            dist_loss = knn_ratio_loss(z, x_flat, self.k)
        elif self.loss_type == 'stress':
            dist_loss = knn_stress_loss(z, x_flat, self.k)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_loss = rec_loss + self.lam * dist_loss
        
        return total_loss, {
            "recon_loss": rec_loss.item(),
            "knn_dist_loss": dist_loss.item()
        }