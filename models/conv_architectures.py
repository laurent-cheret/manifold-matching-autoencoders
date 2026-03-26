"""Convolutional Autoencoder Architectures.

Architectures adapted from TopoAE repository for fair comparison.
Uses BatchNorm for stable training.
"""

import torch
import torch.nn as nn
import numpy as np


class View(nn.Module):
    """Reshape module for Sequential."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvEncoder_MNIST(nn.Module):
    """Conv encoder for 28x28 grayscale images (MNIST/FashionMNIST)."""
    
    def __init__(self, latent_dim=2):
        super().__init__()
        # Input: [B, 1, 28, 28]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # [B, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B, 128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ConvDecoder_MNIST(nn.Module):
    """Conv decoder for 28x28 grayscale images (MNIST/FashionMNIST)."""
    
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128 * 4 * 4),
            nn.BatchNorm1d(128 * 4 * 4),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            View((-1, 128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # [B, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # [B, 1, 28, 28]
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z))


class ConvEncoder_CIFAR(nn.Module):
    """Conv encoder for 32x32 RGB images (CIFAR-10).
    
    Architecture from TopoAE paper with added BatchNorm.
    """
    
    def __init__(self, latent_dim=2):
        super().__init__()
        # Input: [B, 3, 32, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),   # [B, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [B, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [B, 48, 4, 4] = 768
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(True),
            nn.Linear(250, latent_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ConvDecoder_CIFAR(nn.Module):
    """Conv decoder for 32x32 RGB images (CIFAR-10).
    
    Architecture from TopoAE paper with added BatchNorm.
    """
    
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(True),
            nn.Linear(250, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            View((-1, 48, 4, 4)),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [B, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [B, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [B, 3, 32, 32]
            # nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z))


class ConvEncoder_COIL20(nn.Module):
    """Conv encoder for 128x128 grayscale images (COIL-20)."""
    
    def __init__(self, latent_dim=2):
        super().__init__()
        # Input: [B, 1, 128, 128]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),    # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), # [B, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ConvDecoder_COIL20(nn.Module):
    """Conv decoder for 128x128 grayscale images (COIL-20)."""
    
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            View((-1, 256, 4, 4)),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # [B, 1, 128, 128]
            # No Sigmoid - data is z-score normalized, can be negative
        )

    def forward(self, z):
        return self.deconv(self.fc(z))




def get_conv_encoder(dataset_name, latent_dim):
    """Factory for conv encoders."""
    if dataset_name in ['mnist', 'fmnist']:
        return ConvEncoder_MNIST(latent_dim)
    elif dataset_name == 'cifar10':
        return ConvEncoder_CIFAR(latent_dim)
    elif dataset_name == 'coil20':
        return ConvEncoder_COIL20(latent_dim)
    else:
        raise ValueError(f"No conv encoder for {dataset_name}")


def get_conv_decoder(dataset_name, latent_dim):
    """Factory for conv decoders."""
    if dataset_name in ['mnist', 'fmnist']:
        return ConvDecoder_MNIST(latent_dim)
    elif dataset_name == 'cifar10':
        return ConvDecoder_CIFAR(latent_dim)
    elif dataset_name == 'coil20':
        return ConvDecoder_COIL20(latent_dim)
    else:
        raise ValueError(f"No conv decoder for {dataset_name}")