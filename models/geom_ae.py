"""Geometric Autoencoder (Nazari et al., ICML 2023).

Regularizes the autoencoder to have uniform local scaling by minimizing
the variance of log determinants of the pullback metric (Jacobian^T @ Jacobian).
"""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder

try:
    from torch.func import jacfwd, vmap
except ImportError:
    from functorch import jacfwd, vmap


def batch_jacobian(f, x):
    """Compute Jacobian of f for a batch of inputs.
    
    Args:
        f: Function mapping (D,) -> (N,)
        x: Batch of inputs (B, D)
    
    Returns:
        Jacobians (B, N, D)
    """
    if x.ndim == 1:
        return jacfwd(f)(x)
    return vmap(jacfwd(f))(x)


class GeometricLoss(nn.Module):
    """Geometric regularizer based on pullback metric determinants."""
    
    def __init__(self, decoder, latent_dim=2):
        super().__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim
    
    def forward(self, z):
        # Store original training state and switch to eval
        was_training = self.decoder.training
        self.decoder.eval()
        
        # Disable gradient tracking for batchnorm running stats
        for module in self.decoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False
        
        try:
            def immersion(z_single):
                out = self.decoder(z_single.unsqueeze(0))
                return out.view(-1)
            
            J = batch_jacobian(immersion, z)
            G = torch.bmm(J.transpose(1, 2), J)
            
            eps = 1e-6
            G = G + eps * torch.eye(self.latent_dim, device=z.device).unsqueeze(0)
            
            log_dets = torch.logdet(G)
            valid_mask = torch.isfinite(log_dets)
            if not valid_mask.any():
                return torch.tensor(0.0, device=z.device, requires_grad=True)
            
            return torch.var(log_dets[valid_mask])
        finally:
            # Restore original state
            for module in self.decoder.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.track_running_stats = True
            if was_training:
                self.decoder.train()


@register_model("geomae")
class GeomAE(nn.Module):
    """Geometric Autoencoder.
    
    Combines reconstruction loss with geometric regularizer that encourages
    uniform local scaling of the decoder mapping.
    
    Args:
        config: Dictionary with:
            - input_dim: Input dimension
            - latent_dim: Latent dimension (default 2)
            - hidden_dims: Hidden layer dimensions
            - geom_lambda: Weight for geometric loss (default 1.0)
    """
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.latent_dim = config.get('latent_dim', 2)
        self.lam = config.get('geom_lambda', 1.0)
        self.recon_loss = nn.MSELoss()
        self.geom_loss = GeometricLoss(self.decoder, self.latent_dim)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, *args):
        """Forward pass with combined losses.
        
        Args:
            x: Input batch
            
        Returns:
            total_loss, loss_components dict
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        
        # Reconstruction loss
        rec_loss = self.recon_loss(x_rec, x)
        
        # Geometric loss (variance of log determinants)
        geom_loss = self.geom_loss(z)
        
        # Combined loss
        total_loss = rec_loss + self.lam * geom_loss
        return total_loss, {
            'recon_loss': rec_loss.item(),
            'geom_loss': geom_loss.item()
        }