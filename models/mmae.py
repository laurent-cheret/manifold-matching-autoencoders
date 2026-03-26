"""Manifold Matching Autoencoder (MMAE)."""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder


# def distance_preserving_loss(z, ref_emb):
#     """
#     Compute distance preservation loss between latent and reference embeddings.
    
#     L = sum_{i<j} (||z_i - z_j||_2 - ||r_i - r_j||_2)^2
#     """
#     z_dist = torch.cdist(z, z, p=2)
#     ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
#     # Only use upper triangle (i < j)
#     mask = torch.triu(torch.ones_like(z_dist), diagonal=1).bool()
#     loss = ((z_dist[mask] - ref_dist[mask]) ** 2).mean()
#     return loss
def stress_loss(z, ref_emb):
    """Kruskal stress - finds optimal scaling then measures fit.
    
    Preserves distance RATIOS, not just ranking.
    If ref has 5:1 ratio, latent must also have ~5:1 ratio.
    """
    z_dist = torch.cdist(z, z, p=2)
    ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
    # Optimal scale factor (least squares): min_α ||αZ - R||²
    # Solution: α = (Z·R) / (Z·Z)
    scale = (z_dist * ref_dist).sum() / (z_dist ** 2).sum().clamp(min=1e-8)
    z_scaled = z_dist * scale
    
    # Normalized stress
    stress = ((z_scaled - ref_dist) ** 2).sum() / (ref_dist ** 2).sum().clamp(min=1e-8)
    return stress
def distance_preserving_loss(z, ref_emb):
    """Distance preservation loss with normalization for scale invariance."""
    z_dist = torch.cdist(z, z, p=2)
    ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
    # Normalize to make loss scale-invariant
    z_dist_norm = (z_dist - z_dist.mean()) / (z_dist.std() + 1e-8)
    ref_dist_norm = (ref_dist - ref_dist.mean()) / (ref_dist.std() + 1e-8)
    # return ((z_dist - ref_dist) ** 2).mean()
    return ((z_dist_norm - ref_dist_norm) ** 2).mean()

def correlation_loss(z, ref_emb):
    z_dist = torch.cdist(z, z, p=2)
    ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
    # Extract unique pairs (upper triangle)
    mask = torch.triu(torch.ones_like(z_dist), diagonal=1).bool()
    z_flat = z_dist[mask]      # [n*(n-1)/2] distances
    ref_flat = ref_dist[mask]
    
    # Center both (remove mean)
    z_c = z_flat - z_flat.mean()
    ref_c = ref_flat - ref_flat.mean()
    
    # Pearson correlation = covariance / (std_z * std_ref)
    #                     = Σ(z_c * ref_c) / sqrt(Σz_c² * Σref_c²)
    corr = (z_c * ref_c).sum() / (
        torch.sqrt((z_c**2).sum() * (ref_c**2).sum()) + 1e-8
    )
    
    # corr = 1.0: perfect preservation
    # corr = 0.0: no relationship
    # corr < 0.0: inverted (close→far, far→close)
    
    return 1 - corr  # Loss: minimize to maximize correlation


# def distance_preserving_loss(z, ref_emb, eps=1e-8):
#     """Distance preservation loss with max normalization to preserve distance ratios."""
#     z_dist = torch.cdist(z, z, p=2)
#     ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
#     # Max normalization preserves distance ratios
#     z_dist_norm = z_dist / (z_dist.max() + eps)
#     ref_dist_norm = ref_dist / (ref_dist.max() + eps)
    
#     mask = torch.triu(torch.ones_like(z_dist), diagonal=1).bool()
#     return ((z_dist_norm[mask] - ref_dist_norm[mask]) ** 2).mean()
    
@register_model('mmae')
class MMAE(nn.Module):
    """Manifold Matching Autoencoder.
    
    Uses PCA embeddings as reference for distance preservation regularization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get("mmae_lambda", 1.0)
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
            ref_emb: reference PCA embeddings (B, n_components), required for training
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        if ref_emb is not None:
            dist_loss = distance_preserving_loss(z, ref_emb) #stress_loss(z, ref_emb) #correlation_loss(z,ref_emb) #  # #  # # # # # 
            total_loss = rec_loss + self.lam * dist_loss
            return total_loss, {
                "recon_loss": rec_loss.item(),
                "dist_loss": dist_loss.item()
            }
        else:
            return rec_loss, {"recon_loss": rec_loss.item()}