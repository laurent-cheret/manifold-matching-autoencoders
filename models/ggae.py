"""Graph Geometry-Preserving Autoencoder (Lim et al., ICML 2024).

Corrected implementation that precomputes kernel matrix over full dataset
and slices it per batch, as described in the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from .base import register_model, get_encoder, get_decoder


def compute_full_kernel(X, bandwidth):
    """Compute kernel matrix over full dataset.
    
    Args:
        X: Full dataset (N, d)
        bandwidth: Kernel bandwidth
        
    Returns:
        K: Kernel matrix (N, N)
    """
    # Pairwise squared distances
    X_sq = (X ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    dist_sq = X_sq + X_sq.T - 2 * X @ X.T  # (N, N)
    dist_sq = torch.clamp(dist_sq, min=0)
    
    # Gaussian kernel
    K = torch.exp(-dist_sq / bandwidth)
    return K


def compute_laplacian_from_kernel(K, bandwidth):
    """Compute normalized Laplacian from kernel matrix.
    
    Args:
        K: Kernel matrix (B, B) - can be a slice of full kernel
        bandwidth: Kernel bandwidth (for normalization constant)
        
    Returns:
        L: Normalized Laplacian (B, B)
    """
    N = K.shape[0]
    c = 1/4
    
    # Normalized Laplacian
    d_i = K.sum(dim=1)  # (N,)
    D_inv = torch.diag(1.0 / (d_i + 1e-8))  # (N, N)
    K_tilde = D_inv @ K @ D_inv
    D_tilde_inv = torch.diag(1.0 / (K_tilde.sum(dim=1) + 1e-8))
    L = (D_tilde_inv @ K_tilde - torch.eye(N, device=K.device)) / (c * bandwidth)
    
    return L


def compute_JGinvJT(L, z):
    """Compute the JGinvJT matrix for distortion measurement.
    
    Args:
        L: Graph Laplacian (N, N)
        z: Latent points (N, n)
        
    Returns:
        H_tilde: JGinvJT matrix (N, n, n)
    """
    N = L.shape[0]
    n = z.shape[-1]
    
    # Expand z for outer products
    catY1 = z.unsqueeze(-1).expand(N, n, n)  # (N, n, n)
    catY2 = z.unsqueeze(-2).expand(N, n, n)  # (N, n, n)
    
    # Term 1: L @ (y ⊗ y)
    term1 = catY1 * catY2  # (N, n, n)
    term1 = (L @ term1.view(N, n*n)).view(N, n, n)
    
    # Term 2 and 3
    LY = L @ z  # (N, n)
    catLY2 = LY.unsqueeze(-2).expand(N, n, n)
    term2 = catY1 * catLY2
    
    catLY1 = LY.unsqueeze(-1).expand(N, n, n)
    term3 = catY2 * catLY1
    
    H_tilde = 0.5 * (term1 - term2 - term3)
    
    return H_tilde


def relaxed_distortion_measure(H):
    """Compute relaxed distortion measure.
    
    The full formula is: E[Tr(H²) - 2Tr(H) + n]
    We omit +n since it's constant and doesn't affect gradients.
    
    Args:
        H: JGinvJT matrix (N, n, n)
        
    Returns:
        Scalar distortion measure
    """
    # Tr(H) for each point
    TrH = H.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)
    
    # Tr(H²) for each point  
    H2 = H @ H
    TrH2 = H2.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)
    
    # Add n to make loss non-negative (n = latent dim)
    n = H.shape[-1]
    
    return (TrH2 - 2 * TrH + n).mean()


class GGAELoss(nn.Module):
    """Graph geometry loss with precomputed kernel."""
    
    def __init__(self, bandwidth=50.0):
        super().__init__()
        self.bandwidth = bandwidth
        self.K_full = None  # Will store precomputed kernel
        
    def precompute_kernel(self, X_full):
        """Precompute kernel matrix for full dataset.
        
        Call this once before training with the full training data.
        
        Args:
            X_full: Full training dataset (N, d), flattened
        """
        self.K_full = compute_full_kernel(X_full, self.bandwidth)
        print(f"[GGAE] Precomputed kernel matrix: {self.K_full.shape}")
        print(f"[GGAE] Kernel stats: min={self.K_full.min():.4f}, max={self.K_full.max():.4f}, mean={self.K_full.mean():.4f}")
        
    def forward(self, z, indices):
        """Compute graph geometry loss for a batch.
        
        Args:
            z: Latent codes for batch (B, n)
            indices: Indices of batch samples in full dataset (B,)
            
        Returns:
            Distortion loss (scalar)
        """
        if self.K_full is None:
            raise RuntimeError("Must call precompute_kernel() before training!")
        
        # Slice kernel matrix for this batch
        K_batch = self.K_full[indices][:, indices]  # (B, B)
        
        # Compute Laplacian from sliced kernel
        L = compute_laplacian_from_kernel(K_batch, self.bandwidth)
        
        # Compute distortion
        H = compute_JGinvJT(L, z)
        loss = relaxed_distortion_measure(H)
        
        return loss


@register_model("ggae")
class GGAE(nn.Module):
    """Graph Geometry-Preserving Autoencoder.
    
    NOTE: This model requires special handling:
    1. Call model.precompute_kernel(X_train_flat) before training
    2. DataLoader must provide sample indices (see training notes)
    
    Args:
        config: Dictionary with:
            - input_dim: Input dimension
            - latent_dim: Latent dimension (default 2)
            - hidden_dims: Hidden layer dimensions
            - ggae_lambda: Weight for geometry loss (default 1.0)
            - ggae_bandwidth: Bandwidth for Gaussian kernel (default 50.0)
    """
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get('ggae_lambda', 1.0)
        bandwidth = config.get('ggae_bandwidth', 50.0)
        self.recon_loss_fn = nn.MSELoss()
        self.ggae_loss = GGAELoss(bandwidth=bandwidth)
        self._kernel_precomputed = False
        
    def precompute_kernel(self, X_full):
        """Precompute kernel matrix. Call before training.
        
        Args:
            X_full: Full training data (N, d) flattened, as tensor
        """
        self.ggae_loss.precompute_kernel(X_full)
        self._kernel_precomputed = True
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, indices=None):
        """Forward pass.
        
        Args:
            x: Input batch (B, ...)
            indices: Sample indices in full dataset (B,). Required for GGAE loss.
            
        Returns:
            total_loss, loss_components dict
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        
        # Reconstruction loss
        rec_loss = self.recon_loss_fn(x_rec, x)
        
        # Graph geometry loss (only if kernel is precomputed and indices provided)
        if self._kernel_precomputed and indices is not None:
            gg_loss = self.ggae_loss(z, indices)
        else:
            # Fallback: no regularization (essentially vanilla AE)
            gg_loss = torch.tensor(0.0, device=x.device)
            if self._kernel_precomputed and indices is None:
                pass  # Silently skip during evaluation
        
        total_loss = rec_loss + self.lam * gg_loss
        
        return total_loss, {
            'recon_loss': rec_loss.item(),
            'gg_loss': gg_loss.item() if torch.is_tensor(gg_loss) else gg_loss
        }