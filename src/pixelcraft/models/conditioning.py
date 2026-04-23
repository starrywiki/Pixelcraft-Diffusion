from __future__ import annotations

import torch
from torch import nn


class LabelConditioner(nn.Module):
    def __init__(self, num_classes: int, condition_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, condition_dim)

    def forward(self, condition_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(condition_ids)
