"""
Transferable coordinate encoding with Random Fourier Features (RFF)
==================================================================

A minimal, self-contained demonstration of one idea: using a *fixed*,
parameter-free Random Fourier Feature map of spatial coordinates as a
positional encoding, instead of a *learned* per-node embedding table.

Why this matters
----------------
Many spatial models give each sensor / node / station its own learned
embedding vector. That embedding is indexed by the node's *identity*, so
the trained model is welded to the exact network it was trained on: a node
that did not exist at training time has no embedding, and the model cannot
say anything about it.

An RFF coordinate map removes that limitation. It encodes a node by *where
it is*, not *which node it is*, through a frozen random projection

        phi(x) = cos(2*pi * (W @ x) + b),      W, b fixed (never trained)

So identical coordinates always map to identical features, in any network.
The same trained weights therefore run unchanged on a network of different
size and geometry. (Rahimi & Recht 2007 justify the Gaussian frequency
draw; Tancik et al. 2020 show the lifting also cures the "spectral bias"
that makes a plain MLP struggle to fit high-frequency functions of raw
low-dimensional coordinates.)

This script compares three encoders on a synthetic spatial-regression task
and shows, on a *held-out network of new station positions*:

  1. learned per-node embedding  -> cannot transfer at all (no rows for
                                     unseen stations),
  2. raw coordinates -> MLP      -> transfers, but fits the high-frequency
                                     field poorly (spectral bias),
  3. RFF coordinates -> MLP      -> transfers AND fits the field well.

All data is synthetic. This is a mechanism demo, not a scientific result.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Reproducibility. A single seed controls the synthetic networks, the frozen
# RFF projection, and the weight initialisation, so the figure is identical
# on every run.
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Synthetic world
# ---------------------------------------------------------------------------
# The "ground truth" is a smooth but reasonably high-frequency spatial field
# defined on the unit square. Each station sits somewhere on that square and
# observes the field value at its location, corrupted by noise. The learning
# task is to recover the clean field value at each station from the noisy one.
def spatial_field(coords):
    """Target field f(x, y) on the unit square. Shape (N, 2) -> (N,)."""
    x, y = coords[:, 0], coords[:, 1]
    return (
        np.sin(8.0 * np.pi * x) * np.cos(7.0 * np.pi * y)
        + 0.5 * np.sin(10.0 * np.pi * (x + y))
    )


def make_network(n_stations, noise_std=0.05):
    """A 'network' = random station positions + their noisy observations."""
    coords = rng.uniform(0.0, 1.0, size=(n_stations, 2))
    clean = spatial_field(coords)
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape)
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(clean, dtype=torch.float32),
        torch.tensor(noisy, dtype=torch.float32),
    )


# The model is trained ONLY on the training network. The test network is a
# completely different set of station positions -- different count, different
# geometry -- that the model never sees during training.
train_coords, train_clean, train_noisy = make_network(n_stations=500)
test_coords, test_clean, test_noisy = make_network(n_stations=60)


# ---------------------------------------------------------------------------
# 2. The three encoders
# ---------------------------------------------------------------------------
class RFFEncoder(nn.Module):
    """Fixed Random Fourier Feature map of 2-D coordinates.

    W and b are drawn once from fixed distributions and then FROZEN
    (requires_grad=False). Nothing here is learned. Because it is a pure
    function of the input coordinates, it produces identical features for
    identical coordinates in any network -- which is what makes it
    transferable.
    """

    def __init__(self, out_dim=256, sigma=8.0):
        super().__init__()
        # Gaussian frequency matrix (Bochner's theorem -> RBF-like kernel).
        W = torch.randn(2, out_dim) * sigma
        b = torch.rand(out_dim) * 2.0 * np.pi
        # register_buffer stores them in the model (and the checkpoint) but
        # keeps them OUT of the trainable parameters.
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.out_dim = out_dim

    def forward(self, coords):
        return torch.cos(coords @ self.W + self.b)


class RawEncoder(nn.Module):
    """Baseline A: pass the raw (x, y) coordinates straight through.

    Also transferable (a function of coordinates), but a small MLP fed raw
    low-dimensional coordinates struggles to represent high-frequency
    structure -- the spectral-bias problem the RFF lifting is meant to fix.
    """

    out_dim = 2

    def forward(self, coords):
        return coords


class LearnedEmbeddingEncoder(nn.Module):
    """Baseline B: one learned embedding vector per training station.

    This is the design the RFF map replaces. The embedding is indexed by
    station *identity* (a row number), so it has no meaning for a station
    that was not in the training network. It cannot transfer -- there is no
    row to look up for an unseen station.
    """

    def __init__(self, n_stations, out_dim=128):
        super().__init__()
        self.table = nn.Embedding(n_stations, out_dim)
        self.out_dim = out_dim

    def forward(self, station_ids):
        return self.table(station_ids)


# ---------------------------------------------------------------------------
# 3. A shared prediction head
# ---------------------------------------------------------------------------
# Every model uses the SAME small MLP on top of its encoder, so the only
# thing that differs between models is how the station is encoded. That keeps
# the comparison fair: any difference in results comes from the encoding.
class Model(nn.Module):
    def __init__(self, encoder, in_dim, hidden=256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, encoder_input):
        z = self.encoder(encoder_input)
        return self.head(z).squeeze(-1)


def train(model, encoder_input, target, epochs=5000, lr=1e-3):
    """Full-batch training loop -- the dataset is tiny, so this is instant."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(encoder_input)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def mse(pred, target):
    return float(((pred - target) ** 2).mean())


# ---------------------------------------------------------------------------
# 4. Train each model on the training network
# ---------------------------------------------------------------------------
# Model B (learned embedding) is trained against station INDICES, not
# coordinates -- that is the whole point of why it cannot transfer.
train_ids = torch.arange(train_coords.shape[0])

rff_dim = 256
raw_model = train(Model(RawEncoder(), in_dim=RawEncoder.out_dim),
                  train_coords, train_noisy)
rff_model = train(Model(RFFEncoder(rff_dim), in_dim=rff_dim),
                  train_coords, train_noisy)
emb_model = train(Model(LearnedEmbeddingEncoder(train_coords.shape[0], rff_dim),
                        in_dim=rff_dim),
                  train_ids, train_noisy)


# ---------------------------------------------------------------------------
# 5. Evaluate -- the honest part
# ---------------------------------------------------------------------------
# We score every model against the CLEAN field (how well it recovered the
# true signal), first on its own training network, then on the unseen test
# network.
results = {}

# Learned embedding: fits its own network, but there is simply no way to run
# it on the test network -- there are no embedding rows for those stations.
with torch.no_grad():
    results["Learned embedding"] = {
        "train": mse(emb_model(train_ids), train_clean),
        "test": None,  # cannot transfer
    }

# Raw coordinates and RFF: both are functions of coordinates, so we can feed
# them the test-network coordinates directly -- no retraining, no changes.
for name, model in [("Raw coords", raw_model), ("RFF coords", rff_model)]:
    with torch.no_grad():
        results[name] = {
            "train": mse(model(train_coords), train_clean),
            "test": mse(model(test_coords), test_clean),
        }

print("\nRecovery error vs the clean field (lower is better)")
print("-" * 58)
print(f"{'encoder':<20}{'train network':>16}{'new network':>18}")
print("-" * 58)
for name, r in results.items():
    test_str = "cannot transfer" if r["test"] is None else f"{r['test']:.4f}"
    print(f"{name:<20}{r['train']:>16.4f}{test_str:>18}")
print("-" * 58)


# ---------------------------------------------------------------------------
# 6. Figure
# ---------------------------------------------------------------------------
# A dense grid so we can draw each model's learned field as a smooth image.
g = 120
gx, gy = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g))
grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)

with torch.no_grad():
    true_img = spatial_field(grid.numpy()).reshape(g, g)
    raw_img = raw_model(grid).numpy().reshape(g, g)
    rff_img = rff_model(grid).numpy().reshape(g, g)

vmin, vmax = true_img.min(), true_img.max()
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))

ax[0].imshow(true_img, extent=[0, 1, 0, 1], origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
ax[0].scatter(*train_coords.numpy().T, c="k", s=5, alpha=0.35, label="train stations")
ax[0].scatter(*test_coords.numpy().T, marker="^", c="lime", edgecolors="k",
              s=42, label="new (unseen) stations")
ax[0].set_title("True spatial field")
ax[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

ax[1].imshow(raw_img, extent=[0, 1, 0, 1], origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
ax[1].set_title(f"Raw coords -> MLP\nnew-network error {results['Raw coords']['test']:.3f}")

ax[2].imshow(rff_img, extent=[0, 1, 0, 1], origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
ax[2].set_title(f"RFF coords -> MLP\nnew-network error {results['RFF coords']['test']:.3f}")

for a in ax:
    a.set_xticks([]); a.set_yticks([])

fig.suptitle(
    "Fixed RFF coordinate encoding transfers to an unseen network and recovers the "
    "high-frequency field;\nthe learned per-station embedding cannot be evaluated on new "
    "stations at all.",
    fontsize=10.5,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figure.png", dpi=130)
print("\nSaved figure.png")
