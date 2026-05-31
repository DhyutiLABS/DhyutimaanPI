"""
model.py — MLP and Fourier-MLP architectures for 2D Heat PINN DoE.
"""

import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain tanh MLP: d_in → [hidden]×n_layers → 1."""

    def __init__(self, d_in: int = 2, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierMLP(nn.Module):
    """
    Random Fourier Feature MLP.
    Fixed projection B ~ N(0, σ²I), then concat [cos(2πBx), sin(2πBx)].
    The rest is a plain MLP on the 2*n_freq features.
    """

    def __init__(self, d_in: int = 2, n_freq: int = 32, sigma: float = 1.0,
                 hidden: int = 64, n_layers: int = 4):
        super().__init__()
        B = torch.randn(d_in, n_freq) * sigma
        self.register_buffer('B', B)
        d_ff = 2 * n_freq
        layers = [nn.Linear(d_ff, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B)          # (batch, n_freq)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.net(ff)


def build_model(fourier: bool, d_in: int = 2,
                hidden: int = 64, n_layers: int = 4,
                n_freq: int = 32, sigma: float = 1.0) -> nn.Module:
    """Factory: returns plain MLP or FourierMLP based on flag."""
    if fourier:
        return FourierMLP(d_in=d_in, n_freq=n_freq, sigma=sigma,
                          hidden=hidden, n_layers=n_layers)
    return MLP(d_in=d_in, hidden=hidden, n_layers=n_layers)
