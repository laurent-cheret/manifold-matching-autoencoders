"""Vanilla Autoencoder."""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder


@register_model('vanilla')
class VanillaAE(nn.Module):
    """Vanilla Autoencoder with reconstruction loss only."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.recon_loss = nn.MSELoss()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, *args):
        z = self.encode(x)
        x_recon = self.decode(z)
        loss = self.recon_loss(x_recon, x)
        return loss, {'recon_loss': loss.item()}