"""Structure-Preserving Autoencoder (SPAE)."""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder


def spae_r1_loss(z, x_flat, eps=1e-8):
    """Point-wise scaling: mean of per-point log-ratio variances.
    
    Ensures d_E(h(x_i), h(x_j)) ≈ c_i * d_I(x_i, x_j) for each point i.
    Good for local/intra-cluster structure.
    """
    z_dist = torch.cdist(z, z, p=2)
    x_dist = torch.cdist(x_flat, x_flat, p=2)
    
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    
    log_ratios = torch.log(z_dist / (x_dist + eps) + eps)
    
    # Variance per row (excluding diagonal)
    variances = torch.zeros(n, device=z.device)
    for i in range(n):
        row = log_ratios[i, mask[i]]
        variances[i] = torch.var(row)
    
    return variances.mean()


def spae_r2_loss(z, x_flat, eps=1e-8):
    """Global scaling: variance of all log-ratios.
    
    Ensures d_E(h(x_i), h(x_j)) ≈ c * d_I(x_i, x_j) for single global c.
    Good for global/inter-cluster structure.
    """
    z_dist = torch.cdist(z, z, p=2)
    x_dist = torch.cdist(x_flat, x_flat, p=2)
    
    # Upper triangle only (i < j)
    mask = torch.triu(torch.ones_like(z_dist, dtype=torch.bool), diagonal=1)
    
    ratios = z_dist[mask] / (x_dist[mask] + eps)
    return torch.var(torch.log(ratios + eps))


@register_model('spae')
class SPAE(nn.Module):
    """Structure-Preserving Autoencoder.
    
    Regularizes latent space so distances are linearly scaled versions
    of input space distances. Supports R1 (local) and R2 (global) variants.
    """
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get("spae_lambda", 1.0)
        self.variant = config.get("spae_variant", "r2")  # 'r1', 'r2', or 'both'
        self.recon_loss = nn.MSELoss()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, ref_emb=None):
        """
        Forward pass.
        
        Args:
            x: input data
            ref_emb: ignored (for API compatibility with MMAE)
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        # Flatten input for distance computation
        x_flat = x.view(x.size(0), -1)
        
        if self.variant == "r1":
            struct_loss = spae_r1_loss(z, x_flat)
        elif self.variant == "r2":
            struct_loss = spae_r2_loss(z, x_flat)
        else:  # 'both'
            struct_loss = spae_r1_loss(z, x_flat) + spae_r2_loss(z, x_flat)
        
        total_loss = rec_loss + self.lam * struct_loss
        
        return total_loss, {
            "recon_loss": rec_loss.item(),
            "struct_loss": struct_loss.item()
        }