"""Models module."""

from .base import get_model, list_models

# Import to register
from . import vanilla_ae
from . import topo_ae
from . import rtd_ae
from . import geom_ae
from . import ggae
from . import mmae
from . import spae
from . import mmae_variants  
from . import mmae_knn
def build_model(model_name, config):
    """Build model by name."""
    model_cls = get_model(model_name)
    return model_cls(config)