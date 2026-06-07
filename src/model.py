from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Simple MLP for action regression.

    The model maps the concatenated visual-language feature vector to a
    7-dimensional robot action output. The hidden layer widths are provided
    through configuration so that the same code can be reused across baselines.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        # Build a feed-forward network of the form:
        # input -> Linear/ReLU/Dropout -> ... -> Linear -> output.
        dims = [input_dim, *list(hidden_dims), output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            # Dropout is used only on hidden layers to reduce overfitting.
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Forward propagation directly returns the predicted action vector.
        return self.network(inputs)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable and non-trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters())
