"""
Transferable coordinate encoding with Random Fourier Features (RFF)
==================================================================

A minimal demonstration of one idea: using a *fixed*, parameter-free Random
Fourier Feature map of spatial coordinates as a positional encoding, instead
of a *learned* per-node embedding table.

        phi(x) = cos(x @ W + b),      W ~ N(0, sigma^2), b ~ U[0, 2*pi), both frozen

A learned per-node embedding is indexed by the node's identity, so a node that
did not exist at training time has no embedding. The RFF map encodes a node by
*where it is*, not *which node it is*, so the same trained weights run
unchanged on a network of different size and geometry.

This script compares three encoders on a synthetic spatial-regression task and
evaluates them on a held-out network of new station positions:

  1. learned per-node embedding  -> cannot transfer (no rows for unseen stations)
  2. raw coordinates -> MLP      -> transfers, but underfits the field (spectral bias)
  3. RFF coordinates -> MLP      -> transfers and fits the field well

All data is synthetic. This is a mechanism demo, not a scientific result.

Usage
-----
    python rff_encoding_demo.py                      # full run, writes figure.png
    python rff_encoding_demo.py --epochs 50 --no-figure   # quick smoke run
"""

import argparse

import numpy as np
import torch

from rff_encoding import (
    LearnedEmbeddingEncoder,
    Model,
    RawEncoder,
    RFFEncoder,
    make_network,
    mse,
    spatial_field,
    train,
)

SEED = 0
RFF_DIM = 256


def run_experiment(epochs=5000, seed=SEED):
    """Train the three models and return (results, trained models, data).

    The call order below matters for exact reproducibility: one numpy
    generator creates both networks, and one torch seed controls the frozen
    RFF draw and every weight initialisation, in this order.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # The model is trained ONLY on the training network. The test network is
    # a different set of station positions (different count and geometry).
    train_coords, train_clean, train_noisy = make_network(500, rng)
    test_coords, test_clean, _ = make_network(60, rng)

    # The learned-embedding model is trained on station INDICES, not
    # coordinates. That is exactly why it cannot transfer.
    train_ids = torch.arange(train_coords.shape[0])

    raw_model, _ = train(Model(RawEncoder(), in_dim=RawEncoder.out_dim),
                         train_coords, train_noisy, epochs=epochs)
    rff_model, _ = train(Model(RFFEncoder(RFF_DIM), in_dim=RFF_DIM),
                         train_coords, train_noisy, epochs=epochs)
    emb_model, _ = train(Model(LearnedEmbeddingEncoder(train_coords.shape[0], RFF_DIM),
                               in_dim=RFF_DIM),
                         train_ids, train_noisy, epochs=epochs)

    # Score every model against the CLEAN field, on its own training network
    # and then on the unseen test network, with no retraining.
    results = {}
    with torch.no_grad():
        results["Learned embedding"] = {
            "train": mse(emb_model(train_ids), train_clean),
            "test": None,  # no embedding rows exist for unseen stations
        }
        for name, model in [("Raw coords", raw_model), ("RFF coords", rff_model)]:
            results[name] = {
                "train": mse(model(train_coords), train_clean),
                "test": mse(model(test_coords), test_clean),
            }

    models = {"raw": raw_model, "rff": rff_model, "emb": emb_model}
    data = {"train_coords": train_coords, "test_coords": test_coords}
    return results, models, data


def print_results(results):
    print("\nRecovery error vs the clean field (lower is better)")
    print("-" * 58)
    print(f"{'encoder':<20}{'train network':>16}{'new network':>18}")
    print("-" * 58)
    for name, r in results.items():
        test_str = "cannot transfer" if r["test"] is None else f"{r['test']:.4f}"
        print(f"{name:<20}{r['train']:>16.4f}{test_str:>18}")
    print("-" * 58)


def save_figure(results, models, data, path="figure.png"):
    # Imported here so the package and tests do not depend on matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = 120
    gx, gy = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        true_img = spatial_field(grid.numpy()).reshape(g, g)
        raw_img = models["raw"](grid).numpy().reshape(g, g)
        rff_img = models["rff"](grid).numpy().reshape(g, g)

    vmin, vmax = true_img.min(), true_img.max()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    style = {"extent": [0, 1, 0, 1], "origin": "lower", "vmin": vmin, "vmax": vmax,
             "cmap": "RdBu_r"}

    ax[0].imshow(true_img, **style)
    ax[0].scatter(*data["train_coords"].numpy().T, c="k", s=5, alpha=0.35,
                  label="train stations")
    ax[0].scatter(*data["test_coords"].numpy().T, marker="^", c="lime",
                  edgecolors="k", s=42, label="new (unseen) stations")
    ax[0].set_title("True spatial field")
    ax[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax[1].imshow(raw_img, **style)
    ax[1].set_title(f"Raw coords -> MLP\nnew-network error {results['Raw coords']['test']:.3f}")

    ax[2].imshow(rff_img, **style)
    ax[2].set_title(f"RFF coords -> MLP\nnew-network error {results['RFF coords']['test']:.3f}")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(
        "Fixed RFF coordinate encoding transfers to an unseen network and recovers the "
        "high-frequency field;\nthe learned per-station embedding cannot be evaluated on new "
        "stations at all.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    print(f"\nSaved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=5000,
                        help="training epochs per model (default: 5000)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-figure", action="store_true",
                        help="skip writing figure.png")
    args = parser.parse_args(argv)

    results, models, data = run_experiment(epochs=args.epochs, seed=args.seed)
    print_results(results)
    if not args.no_figure:
        save_figure(results, models, data)
    return results


if __name__ == "__main__":
    main()
