"""
model.py — MLP and Fourier-MLP for 1D Burgers PINN.
Input: (x, t)  →  Output: scalar u
"""

import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain tanh MLP: 2 → [hidden]×n_layers → 1."""

    def __init__(self, d_in: int = 2, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)


class FourierMLP(nn.Module):
    """
    Random Fourier Feature MLP.
    Fixed B ~ N(0, σ²I); encoding: [cos(2πBx), sin(2πBx)].
    """

    def __init__(self, d_in: int = 2, n_freq: int = 32, sigma: float = 1.0,
                 hidden: int = 64, n_layers: int = 4):
        super().__init__()
        B = torch.randn(d_in, n_freq) * sigma
        self.register_buffer("B", B)
        d_ff = 2 * n_freq
        layers = [nn.Linear(d_ff, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (xt @ self.B)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.net(ff)


class HardModel(nn.Module):
    """Wraps a base model with the hard-constraint ansatz for Burgers."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        from problem import apply_hard
        return apply_hard(self.base(xt), xt)


def build_model(fourier: bool, hard: bool,
                d_in: int = 2, hidden: int = 64, n_layers: int = 4,
                n_freq: int = 32, sigma: float = 1.0) -> nn.Module:
    """Factory: build base MLP (or FourierMLP) optionally wrapped with hard constraint."""
    base = (FourierMLP(d_in=d_in, n_freq=n_freq, sigma=sigma,
                       hidden=hidden, n_layers=n_layers)
            if fourier
            else MLP(d_in=d_in, hidden=hidden, n_layers=n_layers))
    if hard:
        return HardModel(base)
    return base
