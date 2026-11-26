"""The three ways of encoding a station that the demo compares.

1. RFFEncoder: fixed Random Fourier Features of the coordinates (transfers).
2. RawEncoder: the raw coordinates themselves (transfers, but spectral bias).
3. LearnedEmbeddingEncoder: one learned vector per station id (cannot transfer).
"""

import math

import torch
from torch import nn


class RFFEncoder(nn.Module):
    """Fixed Random Fourier Feature map of station coordinates.

        phi(x) = cos(x @ W + b),   W ~ N(0, sigma^2),   b ~ U[0, 2*pi)

    W and b are drawn once at construction and then FROZEN: they are stored
    with ``register_buffer``, so they are saved in the state dict but are not
    trainable parameters and the optimiser never sees them. Because phi is a
    pure function of the coordinates, identical coordinates give identical
    features in any network, which is what makes the encoding transferable.

    Note on scale: there is no explicit 2*pi factor inside the cosine. Writing
    cos(2*pi * x @ W' + b) with W' ~ N(0, s^2) is the same map with
    sigma = 2*pi*s, so sigma = 8 here corresponds to s = 8 / (2*pi), about 1.27.

    Parameters
    ----------
    out_dim : int
        Number of Fourier features D.
    sigma : float
        Standard deviation of the Gaussian frequencies. Larger sigma lets the
        downstream MLP represent higher spatial frequencies.
    in_dim : int
        Dimension of the coordinates (2 for (x, y)).
    """

    def __init__(self, out_dim=256, sigma=8.0, in_dim=2):
        super().__init__()
        # Gaussian frequencies: by Bochner's theorem this corresponds to a
        # Gaussian (RBF) kernel, see ``kernel_features`` below.
        W = torch.randn(in_dim, out_dim) * sigma
        b = torch.rand(out_dim) * 2.0 * math.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma

    def forward(self, coords):
        """coords: (N, in_dim) -> features: (N, out_dim), values in [-1, 1]."""
        return torch.cos(coords @ self.W + self.b)

    def kernel_features(self, coords):
        """Features scaled by sqrt(2 / D) so that their dot product
        approximates the Gaussian kernel exp(-sigma^2 * ||x - y||^2 / 2)
        (Rahimi and Recht, 2007). The model itself uses ``forward``; this
        method exists to make the theory checkable in the tests.
        """
        return math.sqrt(2.0 / self.out_dim) * self.forward(coords)


class RawEncoder(nn.Module):
    """Baseline A: pass the raw (x, y) coordinates straight through.

    Transferable (a function of coordinates), but a small MLP fed raw
    low-dimensional coordinates struggles to represent high-frequency
    structure. This is the spectral-bias problem the RFF lifting addresses.
    """

    out_dim = 2

    def forward(self, coords):
        return coords


class LearnedEmbeddingEncoder(nn.Module):
    """Baseline B: one learned embedding vector per training station.

    The embedding is indexed by station identity (a row number), so it has
    no meaning for a station outside the training network: looking up an
    unseen id raises an IndexError. This is the design the RFF map replaces.
    """

    def __init__(self, n_stations, out_dim=128):
        super().__init__()
        self.table = nn.Embedding(n_stations, out_dim)
        self.n_stations = n_stations
        self.out_dim = out_dim

    def forward(self, station_ids):
        return self.table(station_ids)
