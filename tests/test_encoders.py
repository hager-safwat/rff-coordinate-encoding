"""Tests for the three encoders, focused on the properties the README claims."""

import math

import pytest
import torch

from rff_encoding import LearnedEmbeddingEncoder, Model, RawEncoder, RFFEncoder, train


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_rff_output_shape_and_range():
    enc = RFFEncoder(out_dim=64)
    feats = enc(torch.rand(10, 2))
    assert feats.shape == (10, 64)
    # cos(.) is bounded, so every feature lies in [-1, 1].
    assert feats.abs().max() <= 1.0


def test_rff_has_no_trainable_parameters_but_is_saved():
    """W and b are frozen buffers: not optimised, but stored in checkpoints."""
    enc = RFFEncoder(out_dim=32)
    assert list(enc.parameters()) == []
    state = enc.state_dict()
    assert set(state) == {"W", "b"}
    assert state["W"].shape == (2, 32)


def test_rff_buffers_unchanged_by_training():
    """Training the full model updates the MLP head but never W or b."""
    enc = RFFEncoder(out_dim=32)
    W0, b0 = enc.W.clone(), enc.b.clone()
    model = Model(enc, in_dim=32, hidden=16)
    head0 = model.head[0].weight.detach().clone()

    coords = torch.rand(50, 2)
    target = torch.sin(6 * coords[:, 0])
    train(model, coords, target, epochs=20)

    assert torch.equal(enc.W, W0)
    assert torch.equal(enc.b, b0)
    assert not torch.equal(model.head[0].weight, head0)


def test_rff_encoding_depends_only_on_coordinates():
    """The key transfer property: a station's features do not depend on which
    network it belongs to, or on the other stations in that network."""
    enc = RFFEncoder(out_dim=64)
    station = torch.tensor([[0.3, 0.7]])
    network_a = torch.cat([station, torch.rand(5, 2)])
    network_b = torch.cat([torch.rand(40, 2), station])

    feats_a = enc(network_a)[0]
    feats_b = enc(network_b)[-1]
    assert torch.allclose(feats_a, feats_b)


def test_rff_approximates_gaussian_kernel():
    """Bochner / Rahimi and Recht: with W ~ N(0, sigma^2) and b ~ U[0, 2pi),
    sqrt(2/D) cos(xW + b) has dot products close to exp(-sigma^2 |x-y|^2 / 2).

    Monte Carlo error scales like 1/sqrt(D); with D = 20000 a tolerance of
    0.03 is several standard deviations wide, and the seed is fixed.
    """
    sigma = 2.0
    enc = RFFEncoder(out_dim=20000, sigma=sigma)
    x = torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.5, -0.3], [1.0, 1.0]])
    z = enc.kernel_features(x)
    approx = z @ z.T

    sq_dist = torch.cdist(x, x) ** 2
    exact = torch.exp(-(sigma ** 2) * sq_dist / 2.0)
    assert torch.allclose(approx, exact, atol=0.03)


def test_raw_encoder_is_identity():
    coords = torch.rand(7, 2)
    assert torch.equal(RawEncoder()(coords), coords)


def test_learned_embedding_cannot_encode_unseen_station():
    """A station id outside the training table has no row to look up."""
    enc = LearnedEmbeddingEncoder(n_stations=10, out_dim=8)
    assert enc(torch.arange(10)).shape == (10, 8)
    with pytest.raises(IndexError):
        enc(torch.tensor([10]))


def test_rff_model_runs_unchanged_on_networks_of_any_size():
    """Same weights, different network sizes: no retraining, no resizing."""
    model = Model(RFFEncoder(out_dim=32), in_dim=32, hidden=16)
    for n in (1, 60, 500):
        assert model(torch.rand(n, 2)).shape == (n,)


def test_default_sigma_matches_documented_value():
    enc = RFFEncoder()
    assert enc.sigma == 8.0
    assert enc.out_dim == 256
    # Phases are drawn in [0, 2*pi).
    assert enc.b.min() >= 0.0 and enc.b.max() < 2 * math.pi
