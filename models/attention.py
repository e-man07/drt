"""QueryAttentionHead: produce per-slot importance weights from a query embedding.

Two-layer MLP (384 → 64 → k) with GELU and a final softmax. Operates on the
full pre-decomposition query embedding so the slot weighting is independent
of the decomposition pathway.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryAttentionHead(nn.Module):
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 64, k: int = 6):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, k)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(q))
        return F.softmax(self.fc2(h), dim=-1)
