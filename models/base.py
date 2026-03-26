"""Model registry and base architectures."""

import torch
import torch.nn as nn

# Registry
MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model class."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name):
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_models():
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())


# ============== MLP Architectures ==============

class MLPEncoder(nn.Module):
    """MLP Encoder for flat vector data.
    
    Matches DeepAE architecture from TopoAE paper when hidden_dims=[1000, 500, 250].
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 64], use_batchnorm=False):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class MLPDecoder(nn.Module):
    """MLP Decoder for flat vector data.
    
    Matches DeepAE architecture from TopoAE paper when hidden_dims=[1000, 500, 250].
    """
    
    def __init__(self, latent_dim, output_dim, hidden_dims=[64, 64], output_shape=None, use_batchnorm=False):
        super().__init__()
        self.output_shape = output_shape
        layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        # layers.append(nn.Sigmoid())  # Match DeepAE output activation
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        out = self.net(z)
        if self.output_shape is not None:
            out = out.view(-1, *self.output_shape)
        return out


# ============== Encoder/Decoder Factory ==============

def get_encoder(config):
    """Create encoder based on config."""
    arch_type = config.get('arch_type', 'mlp')
    latent_dim = config['latent_dim']
    
    if arch_type == 'conv':
        try:
            from .conv_architectures import get_conv_encoder
        except ImportError:
            from conv_architectures import get_conv_encoder
        return get_conv_encoder(config['dataset_name'], latent_dim)
    else:
        return MLPEncoder(
            input_dim=config['input_dim'],
            latent_dim=latent_dim,
            hidden_dims=config.get('hidden_dims', [64, 64]),
            use_batchnorm=config.get('use_batchnorm', False)
        )


def get_decoder(config):
    """Create decoder based on config."""
    arch_type = config.get('arch_type', 'mlp')
    latent_dim = config['latent_dim']
    
    if arch_type == 'conv':
        try:
            from .conv_architectures import get_conv_decoder
        except ImportError:
            from conv_architectures import get_conv_decoder
        return get_conv_decoder(config['dataset_name'], latent_dim)
    else:
        # For MLP, output is flat - don't reshape
        return MLPDecoder(
            latent_dim=latent_dim,
            output_dim=config['input_dim'],
            hidden_dims=config.get('hidden_dims', [64, 64]),
            output_shape=None,  # MLP outputs flat vectors
            use_batchnorm=config.get('use_batchnorm', False)
        )