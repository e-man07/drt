"""InfoNCE retrieval loss applied to DRT scores.

For a batch of B (query, positive) pairs, every query is contrasted against
its own positive plus every other query's positive (in-batch negatives) and
optionally a per-query set of explicit negatives.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    query_subs: torch.Tensor,
    query_alphas: torch.Tensor,
    pos_subs: torch.Tensor,
    neg_subs: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute InfoNCE over DRT scores.

    Args:
      query_subs:  (B, k, d) L2-normalized.
      query_alphas:(B, k) softmax weights.
      pos_subs:    (B, k, d) L2-normalized; row i is the positive for query i.
      neg_subs:    optional (B, N_neg, k, d) explicit negatives per row.
      temperature: scaling τ for the cross-entropy logits.

    Returns: scalar loss.
    """
    B = query_subs.shape[0]

    cos_pos = torch.einsum("ikd,jkd->ijk", query_subs, pos_subs)
    pos_scores = (query_alphas.unsqueeze(1) * cos_pos).sum(dim=-1)

    if neg_subs is None:
        logits = pos_scores
    else:
        cos_neg = (query_subs.unsqueeze(1) * neg_subs).sum(dim=-1)
        neg_scores = (query_alphas.unsqueeze(1) * cos_neg).sum(dim=-1)
        logits = torch.cat([pos_scores, neg_scores], dim=1)

    target = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits / temperature, target)
