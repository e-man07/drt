"""Barlow-Twins-inspired decorrelation loss for DRT sub-vectors.

For each pair of slots (i, j) with i < j, compute the cross-correlation matrix
C[i,j] = sub[:,i].T @ sub[:,j] / B  (shape (d, d)) and penalize ||C||_F^2.

Without this loss, all slots collapse to the same representation and DRT
degenerates into cosine similarity with extra steps.
"""
from __future__ import annotations

import torch


def decorrelation_loss(sub_vectors: torch.Tensor) -> torch.Tensor:
    """Off-diagonal cross-correlation penalty across slot pairs.

    sub_vectors: (B, k, d). Returns a scalar.
    """
    B, k, _ = sub_vectors.shape
    if k < 2 or B < 2:
        return sub_vectors.new_zeros(())

    cross = torch.einsum("bid,bje->ijde", sub_vectors, sub_vectors) / B
    sq_frob = (cross ** 2).sum(dim=(-1, -2))

    iu, ju = torch.triu_indices(k, k, offset=1, device=sub_vectors.device)
    n_pairs = iu.numel()
    return sq_frob[iu, ju].sum() / n_pairs
