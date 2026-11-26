"""Synthetic "world" used by the demo: a spatial field observed by stations.

The ground truth is a smooth but reasonably high-frequency field on the unit
square. A "network" is a set of random station positions, each observing the
field value at its location with additive Gaussian noise. The learning task
is to recover the clean value at each station from the noisy one.
"""

import numpy as np
import torch


def spatial_field(coords):
    """Target field f(x, y) on the unit square.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Station coordinates in [0, 1] x [0, 1].

    Returns
    -------
    np.ndarray, shape (N,)
    """
    x, y = coords[:, 0], coords[:, 1]
    return (
        np.sin(8.0 * np.pi * x) * np.cos(7.0 * np.pi * y)
        + 0.5 * np.sin(10.0 * np.pi * (x + y))
    )


def make_network(n_stations, rng, noise_std=0.05):
    """Create one network: random station positions plus their observations.

    The random generator is passed in explicitly (instead of being a hidden
    global) so that results are reproducible and tests can control it.

    Parameters
    ----------
    n_stations : int
        Number of stations in the network.
    rng : np.random.Generator
        Source of randomness for positions and noise.
    noise_std : float
        Standard deviation of the observation noise.

    Returns
    -------
    coords, clean, noisy : torch.Tensor
        Shapes (n_stations, 2), (n_stations,), (n_stations,), float32.
    """
    coords = rng.uniform(0.0, 1.0, size=(n_stations, 2))
    clean = spatial_field(coords)
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape)
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(clean, dtype=torch.float32),
        torch.tensor(noisy, dtype=torch.float32),
    )
