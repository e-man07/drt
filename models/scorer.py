"""DRTScorer: end-to-end module wrapping decomposition + query attention.

score(q, d) = Σᵢ αᵢ(q) · cos(qᵢ, dᵢ)

Sub-vectors are L2-normalized inside DecompositionHead, so cos reduces to a
dot product. The same DecompositionHead is shared by query and document; only
the query goes through the QueryAttentionHead, giving the asymmetric scoring.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import QueryAttentionHead
from models.decomposition import DecompositionHead


class DRTScorer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        decomp_hidden: int = 512,
        attn_hidden: int = 64,
        k: int = 6,
        sub_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.sub_dim = sub_dim
        self.decomposition = DecompositionHead(embed_dim, decomp_hidden, k, sub_dim)
        self.attention = QueryAttentionHead(embed_dim, attn_hidden, k)

    def encode_doc(self, x: torch.Tensor) -> torch.Tensor:
        return self.decomposition(x)

    def encode_query(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decomposition(q), self.attention(q)

    @staticmethod
    def score(
        query_subs: torch.Tensor,
        query_alphas: torch.Tensor,
        doc_subs: torch.Tensor,
    ) -> torch.Tensor:
        """Σᵢ αᵢ · (qᵢ · dᵢ). Shapes broadcast on all-but-last-two dims.

        Paired:    (B,k,d) × (B,k,d) → (B,)
        Many-vs-1: (B,1,k,d) × (1,P,k,d) → (B,P)
        """
        cos_per_slot = (query_subs * doc_subs).sum(dim=-1)
        return (query_alphas * cos_per_slot).sum(dim=-1)

    def forward(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        q_subs, q_alphas = self.encode_query(query_emb)
        d_subs = self.encode_doc(doc_emb)
        return self.score(q_subs, q_alphas, d_subs)
