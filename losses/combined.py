"""Combined DRT training loss: retrieval + λ·decorrelation, with slot dropout.

Decorrelation is computed on pre-dropout sub-vectors (we want the underlying
representation to specialize, not the dropout-corrupted version). Slot dropout
is then applied to both query and positive sub-vectors before the retrieval
loss, so the retrieval objective forces each surviving slot to carry useful
signal independently.
"""
from __future__ import annotations

import torch

from losses.contrastive import info_nce_loss
from losses.decorrelation import decorrelation_loss


def slot_dropout(
    sub_vectors: torch.Tensor,
    p: float = 0.15,
    training: bool = True,
) -> torch.Tensor:
    """Per-slot dropout: zero out entire sub-vectors with probability p.

    sub_vectors: (..., k, d). The mask is independent per leading-axis sample
    and is rescaled by 1 / mean(mask) so the expected value is preserved.
    No-op when not training or p <= 0.
    """
    if not training or p <= 0:
        return sub_vectors
    mask = torch.bernoulli(
        torch.full(sub_vectors.shape[:-1], 1 - p, device=sub_vectors.device)
    )
    scale = mask.mean(dim=-1, keepdim=True).clamp(min=1e-8)
    mask = mask / scale
    return sub_vectors * mask.unsqueeze(-1)


def combined_loss(
    query_subs: torch.Tensor,
    query_alphas: torch.Tensor,
    pos_subs: torch.Tensor,
    neg_subs: torch.Tensor | None = None,
    *,
    temperature: float = 0.05,
    lambda_decorr: float = 0.1,
    slot_dropout_p: float = 0.15,
    training: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L_total = L_retrieval + λ_decorr · L_decorr.

    Returns (total, {"retrieval": L_ret_detached, "decorrelation": L_dec_detached}).
    """
    L_dec = decorrelation_loss(torch.cat([query_subs, pos_subs], dim=0))

    q_drop = slot_dropout(query_subs, p=slot_dropout_p, training=training)
    p_drop = slot_dropout(pos_subs, p=slot_dropout_p, training=training)
    if neg_subs is not None and training and slot_dropout_p > 0:
        B, N, k, d = neg_subs.shape
        n_drop = slot_dropout(neg_subs.reshape(B * N, k, d), p=slot_dropout_p, training=training)
        neg_subs_eff = n_drop.reshape(B, N, k, d)
    else:
        neg_subs_eff = neg_subs

    L_ret = info_nce_loss(
        q_drop, query_alphas, p_drop, neg_subs=neg_subs_eff, temperature=temperature
    )

    L_total = L_ret + lambda_decorr * L_dec
    return L_total, {"retrieval": L_ret.detach(), "decorrelation": L_dec.detach()}
