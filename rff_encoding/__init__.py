"""Transferable coordinate encoding with fixed Random Fourier Features."""

from .data import make_network, spatial_field
from .encoders import LearnedEmbeddingEncoder, RawEncoder, RFFEncoder
from .model import Model, mse, train

__version__ = "0.2.0"

__all__ = [
    "LearnedEmbeddingEncoder",
    "Model",
    "RFFEncoder",
    "RawEncoder",
    "make_network",
    "mse",
    "spatial_field",
    "train",
]
