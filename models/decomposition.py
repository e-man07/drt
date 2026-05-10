"""DecompositionHead: project a 384-d embedding into k L2-normalized sub-vectors.

Two Linear → LayerNorm → GELU blocks (384 → 512 → 384), then reshape to
(..., k, sub_dim) and L2-normalize each sub-vector independently.

This is the only trainable component for Scale 1 (frozen encoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecompositionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        k: int = 6,
        sub_dim: int = 64,
    ):
        super().__init__()
        if k * sub_dim != embed_dim:
            raise ValueError(
                f"k * sub_dim ({k} * {sub_dim} = {k * sub_dim}) must equal "
                f"embed_dim ({embed_dim})"
            )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.sub_dim = sub_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.fc1(x)))
        h = F.gelu(self.norm2(self.fc2(h)))
        sub = h.reshape(*x.shape[:-1], self.k, self.sub_dim)
        return F.normalize(sub, p=2, dim=-1)
