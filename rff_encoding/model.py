"""Shared prediction head, training loop and metric.

Every model uses the SAME small MLP on top of its encoder, so the only thing
that differs between models is how a station is encoded. Any difference in
results therefore comes from the encoding, which keeps the comparison fair.
"""

import torch
from torch import nn


class Model(nn.Module):
    """Encoder followed by a 3-layer MLP that predicts one value per station."""

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
    """Full-batch Adam training with MSE loss.

    The dataset is tiny, so each epoch is one gradient step on all stations.
    Only ``model.parameters()`` are optimised; frozen buffers (the RFF W and
    b) are not parameters and are never updated.

    Returns the trained model and the list of training losses per epoch.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(encoder_input), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


@torch.no_grad()
def mse(pred, target):
    """Mean squared error as a Python float."""
    return float(((pred - target) ** 2).mean())
