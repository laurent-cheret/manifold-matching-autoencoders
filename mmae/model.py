"""Autoencoder with optional manifold-matching regularizer."""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """MLP encoder for flat-vector inputs.

    Each hidden block is `Linear -> BatchNorm1d -> ReLU` when `batchnorm=True`.
    The final bottleneck Linear has no BN/activation: its output IS the latent
    code, fed to both the decoder and the manifold-matching loss; normalizing
    it would distort the latent geometry.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=(512, 256, 128),
                 batchnorm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))  # no BN/activation here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)


class MLPDecoder(nn.Module):
    """Symmetric MLP decoder.

    BN sits between hidden layers only. The final Linear (reconstruction
    output) has no BN/activation -- the reconstruction loss compares it
    directly to the standardized input.
    """

    def __init__(self, latent_dim, output_dim, hidden_dims=(512, 256, 128),
                 batchnorm: bool = True):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))  # no BN/activation here
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def manifold_matching_loss(z, ref, normalize: bool = True):
    """Distance-preservation loss between latent codes z and reference embedding ref.

    Both z and ref are (B, d) tensors. We compute pairwise Euclidean distances
    of each batch and match them with MSE.

    Args:
        z, ref: (B, d) tensors of latent and reference coordinates.
        normalize: if True (default), z-score each distance matrix before
            matching. The loss becomes invariant to the scale and shift of the
            reference embedding -- only the *shape* of the distance
            distribution is matched. Set to False to match raw Euclidean
            distances directly: this anchors the latent to the same metric
            scale as the reference.
    """
    z_dist = torch.cdist(z, z, p=2)
    r_dist = torch.cdist(ref, ref, p=2)
    if normalize:
        z_dist = (z_dist - z_dist.mean()) / (z_dist.std() + 1e-8)
        r_dist = (r_dist - r_dist.mean()) / (r_dist.std() + 1e-8)
    return ((z_dist - r_dist) ** 2).mean()


class Autoencoder(nn.Module):
    """Autoencoder with optional manifold-matching regularizer.

    forward(x, ref=None) returns (total_loss, components_dict). When `ref` is
    provided and regularizer == 'mmae', the loss is
    `recon_mse + lam * manifold_matching_loss(z, ref, normalize=mm_normalize)`.

    BatchNorm is on by default (between hidden layers); pass `batchnorm=False`
    for the un-normalized network.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=(512, 256, 128),
                 regularizer="mmae", lam=1.0, mm_normalize: bool = True,
                 batchnorm: bool = True):
        super().__init__()
        assert regularizer in ("none", "mmae"), f"unknown regularizer: {regularizer}"
        self.encoder = MLPEncoder(input_dim, latent_dim, hidden_dims, batchnorm=batchnorm)
        self.decoder = MLPDecoder(latent_dim, input_dim, hidden_dims, batchnorm=batchnorm)
        self.regularizer = regularizer
        self.lam = lam
        self.mm_normalize = mm_normalize
        self.batchnorm = batchnorm
        self.recon_loss = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, ref=None):
        if x.dim() > 2:
            x = x.flatten(1)
        z = self.encode(x)
        x_hat = self.decode(z)
        recon = self.recon_loss(x_hat, x)

        if self.regularizer == "mmae" and ref is not None:
            mm = manifold_matching_loss(z, ref, normalize=self.mm_normalize)
            total = recon + self.lam * mm
            return total, {"recon": recon.item(), "mm": mm.item(), "total": total.item()}

        return recon, {"recon": recon.item(), "total": recon.item()}
