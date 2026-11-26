"""Tests for the synthetic data, the training loop and the demo entry point."""

import numpy as np
import pytest
import torch

from rff_encoding import Model, RFFEncoder, make_network, mse, spatial_field, train


def test_spatial_field_known_values():
    pts = np.array([[0.0, 0.0], [1.0 / 16.0, 0.0]])
    vals = spatial_field(pts)
    # f(0, 0) = 0. f(1/16, 0) = sin(pi/2) * cos(0) + 0.5 * sin(5*pi/8).
    expected = [0.0, 1.0 + 0.5 * np.sin(5.0 * np.pi / 8.0)]
    np.testing.assert_allclose(vals, expected, atol=1e-12)


def test_make_network_shapes_and_bounds():
    coords, clean, noisy = make_network(100, np.random.default_rng(1))
    assert coords.shape == (100, 2)
    assert clean.shape == noisy.shape == (100,)
    assert coords.dtype == torch.float32
    assert coords.min() >= 0.0 and coords.max() <= 1.0


def test_make_network_is_reproducible_with_same_seed():
    a = make_network(20, np.random.default_rng(42))
    b = make_network(20, np.random.default_rng(42))
    for ta, tb in zip(a, b, strict=True):
        assert torch.equal(ta, tb)


def test_noise_level_matches_noise_std():
    _, clean, noisy = make_network(20000, np.random.default_rng(0), noise_std=0.05)
    assert float((noisy - clean).std()) == pytest.approx(0.05, rel=0.05)


def test_mse_basic():
    assert mse(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0])) == pytest.approx(2.0)


def test_training_reduces_loss():
    torch.manual_seed(0)
    coords, _, noisy = make_network(200, np.random.default_rng(0))
    model = Model(RFFEncoder(out_dim=64), in_dim=64, hidden=64)
    _, losses = train(model, coords, noisy, epochs=200)
    assert len(losses) == 200
    assert losses[-1] < 0.5 * losses[0]


def test_demo_smoke_run():
    """The demo runs end to end on a short schedule and reports all encoders."""
    import rff_encoding_demo

    results = rff_encoding_demo.main(["--epochs", "20", "--no-figure"])
    assert set(results) == {"Learned embedding", "Raw coords", "RFF coords"}
    assert results["Learned embedding"]["test"] is None
    assert results["RFF coords"]["test"] is not None
