"""
model.py — MLP architecture for the 2D Heat PINN.

Default: 5 hidden layers × 64 neurons, tanh activation, Xavier uniform init.
No batch norm or dropout (incompatible with torch.func.vmap).
"""

import torch
import torch.nn as nn
from typing import Type


class MLP(nn.Module):
    """Fully-connected MLP: (batch, in_dim) → (batch, out_dim).

    Architecture from problem-spec §5:
        in_dim=2, out_dim=1, hidden=64, depth=5, activation=tanh
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 64,
        depth: int = 5,
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation()]
        layers.append(nn.Linear(hidden, out_dim))

        self.net = nn.Sequential(*layers)
        self._xavier_init()

    def _xavier_init(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(hidden: int = 64, depth: int = 5, device: torch.device | None = None) -> MLP:
    """Convenience factory used by train.py and verify.py."""
    model = MLP(in_dim=2, out_dim=1, hidden=hidden, depth=depth)
    if device is not None:
        model = model.to(device)
    return model
