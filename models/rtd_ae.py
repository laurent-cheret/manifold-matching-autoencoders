"""RTD Autoencoder (Trofimov et al., ICLR 2023)."""

import torch
import torch.nn as nn
import numpy as np
from .base import register_model, get_encoder, get_decoder

try:
    import ripserplusplus as rpp
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def get_rtd_indices(DX, rc, dim, card):
    dgm = rc['dgms'][dim]
    pairs = rc['pairs'][dim]
    indices, pers = [], []
    
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2][:, l2]), [len(s2), len(s2)])]
            indices += i1 + i2
            pers.append(dgm[i][1] - dgm[i][0])
    
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1].flatten())
    return indices[:4*card] + [0] * max(0, 4*card - len(indices))


def compute_rips(DX, dim, card):
    DX_ = DX.numpy()
    DX_ = (DX_ + DX_.T) / 2.0
    np.fill_diagonal(DX_, 0)
    
    if not HAS_RIPSER:
        return [[0] * (4 * card) for _ in range(dim)]
    
    rc = rpp.run(f"--format distance --dim {dim}", DX_)
    return [get_rtd_indices(DX_, rc, d, card) for d in range(1, dim + 1)]


class RTDModule(nn.Module):
    def __init__(self, dim=1, card=50, mode='minimum'):
        super().__init__()
        self.dim = max(1, dim)
        self.card = card
        self.mode = mode
    
    def forward(self, Dr1, Dr2, immovable=None):
        device = Dr1.device
        n = len(Dr1)
        Dzz = torch.zeros((n, n), device=device)
        
        if self.mode == 'minimum':
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat([torch.cat([Dzz, Dr1.T], 1), torch.cat([Dr1, Dr12], 1)], 0)
            DX_2 = DX if immovable is None else (
                torch.cat([torch.cat([Dzz, Dr1.T], 1), torch.cat([Dr1, Dr1], 1)], 0) if immovable == 2 else
                torch.cat([torch.cat([Dzz, Dr1.T], 1), torch.cat([Dr1, Dr2], 1)], 0)
            )
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat([torch.cat([Dzz, Dr12.T], 1), torch.cat([Dr12, Dr2], 1)], 0)
            DX_2 = DX
        
        all_ids = compute_rips(DX.detach().cpu(), self.dim, self.card)
        dgms = []
        for ids in all_ids:
            idx = np.reshape(ids, [2 * self.card, 2])
            if self.mode == 'minimum':
                dgm = torch.hstack([
                    DX[idx[::2, 0], idx[::2, 1]].reshape(-1, 1),
                    DX_2[idx[1::2, 0], idx[1::2, 1]].reshape(-1, 1)
                ])
            else:
                dgm = torch.hstack([
                    DX_2[idx[::2, 0], idx[::2, 1]].reshape(-1, 1),
                    DX[idx[1::2, 0], idx[1::2, 1]].reshape(-1, 1)
                ])
            dgms.append(dgm)
        return dgms


class RTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, is_sym=True, lp=1.0):
        super().__init__()
        self.is_sym = is_sym
        self.p = lp
        self.rtd = RTDModule(dim, card)
    
    def forward(self, x_dist, z_dist):
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        loss_xz = sum(torch.sum(torch.abs(d[:, 1] - d[:, 0]) ** self.p) for d in rtd_xz)
        
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist, immovable=2)
            loss_zx = sum(torch.sum(torch.abs(d[:, 1] - d[:, 0]) ** self.p) for d in rtd_zx)
        else:
            loss_zx = torch.tensor(0.0)
        
        return loss_xz, loss_zx, (loss_xz + loss_zx) / 2.0


@register_model("rtdae")
class RTDAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.rtd_loss = RTDLoss(
            dim=config.get("rtd_dim", 1),
            card=config.get("rtd_card", 50),
            is_sym=config.get("rtd_is_sym", True),
            lp=config.get("rtd_lp", 1.0)
        )
        self.lam = config.get("rtd_lambda", 1.0)
        self.recon_loss = nn.MSELoss()
        self.norm = nn.Parameter(torch.ones(1), requires_grad=True)
    
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
        x_dist = torch.cdist(x_flat, x_flat) / np.sqrt(x_flat.shape[1])
        z_dist = self.norm * torch.cdist(z, z) / np.sqrt(z.shape[1])
        
        loss_xz, loss_zx, rtd_loss = self.rtd_loss(x_dist, z_dist)
        
        loss = rec_loss + self.lam * rtd_loss
        return loss, {
            "recon_loss": rec_loss.item(),
            "rtd_loss": rtd_loss.item()
        }