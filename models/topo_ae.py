"""Topological Autoencoder (Moor et al., ICML 2020)."""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder
from .topology import PersistentHomologyCalculation


class TopoSignature(nn.Module):
    def __init__(self, match_edges="symmetric"):
        super().__init__()
        self.match_edges = match_edges
        self.ph = PersistentHomologyCalculation()
    
    def forward(self, d1, d2):
        p1, _ = self.ph(d1.detach().cpu().numpy())
        p2, _ = self.ph(d2.detach().cpu().numpy())
        
        sig1 = d1[(p1[:, 0], p1[:, 1])]
        sig2 = d2[(p2[:, 0], p2[:, 1])]
        
        if self.match_edges == "symmetric":
            sig1_2 = d2[(p1[:, 0], p1[:, 1])]
            sig2_1 = d1[(p2[:, 0], p2[:, 1])]
            return ((sig1 - sig1_2)**2).sum() + ((sig2 - sig2_1)**2).sum()
        return ((sig1 - sig2)**2).sum()


@register_model("topoae")
class TopoAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.topo_sig = TopoSignature(config.get("match_edges", "symmetric"))
        self.lam = config.get("topo_lambda", 1.0)
        self.recon_loss = nn.MSELoss()
        self.latent_norm = nn.Parameter(torch.ones(1), requires_grad=True)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, *args):
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        # Flatten for distance computation
        x_flat = x.view(x.size(0), -1)
        x_dist = torch.cdist(x_flat, x_flat)
        x_dist = x_dist / x_dist.max()
        z_dist = torch.cdist(z, z) / self.latent_norm
        
        topo_loss = self.topo_sig(x_dist, z_dist) / x.size(0)
        
        loss = rec_loss + self.lam * topo_loss
        return loss, {"recon_loss": rec_loss.item(), "topo_loss": topo_loss.item()}