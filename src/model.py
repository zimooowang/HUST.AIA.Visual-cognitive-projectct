from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Simple MLP for action regression."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *list(hidden_dims), output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

