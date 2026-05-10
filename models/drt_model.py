"""DRTModel: end-to-end model = MiniLMEncoder + DecompositionHead + QueryAttentionHead.

Used for Scale 2 (encoder unfrozen). The encoder is shared between query and
document; the QueryAttentionHead consumes the same pre-decomposition embedding
as the DecompositionHead so the slot weighting can adapt to query type.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import QueryAttentionHead
from models.decomposition import DecompositionHead
from models.encoder import DEFAULT_MODEL, MiniLMEncoder


class DRTModel(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_seq_length: int = 256,
        decomp_hidden: int = 512,
        attn_hidden: int = 64,
        k: int = 6,
        sub_dim: int = 64,
    ):
        super().__init__()
        self.encoder = MiniLMEncoder(model_name=model_name, max_seq_length=max_seq_length)
        embed_dim = self.encoder.embed_dim
        self.decomposition = DecompositionHead(embed_dim, decomp_hidden, k, sub_dim)
        self.attention = QueryAttentionHead(embed_dim, attn_hidden, k)
        self.embed_dim = embed_dim
        self.k = k
        self.sub_dim = sub_dim

    @property
    def head_parameters(self):
        """Parameters of the decomposition + attention heads (separate from encoder)."""
        return list(self.decomposition.parameters()) + list(self.attention.parameters())

    def forward_text(self, texts: list[str], device: str | torch.device):
        """Tokenize on host, forward on device, return raw 384-d embeddings."""
        tok = self.encoder.tokenize(texts)
        tok = {k: v.to(device) for k, v in tok.items()}
        return self.encoder(tok["input_ids"], tok["attention_mask"])

    def encode_query(self, texts: list[str], device: str | torch.device):
        emb = self.forward_text(texts, device)
        return self.decomposition(emb), self.attention(emb)

    def encode_doc(self, texts: list[str], device: str | torch.device):
        emb = self.forward_text(texts, device)
        return self.decomposition(emb)

    @staticmethod
    def score(query_subs, query_alphas, doc_subs):
        """Σᵢ αᵢ · (qᵢ · dᵢ). Sub-vectors are L2-normalized so dot = cos."""
        cos_per_slot = (query_subs * doc_subs).sum(dim=-1)
        return (query_alphas * cos_per_slot).sum(dim=-1)
