"""MiniLMEncoder: HuggingFace BERT-base encoder with mean pooling.

Wraps `sentence-transformers/all-MiniLM-L6-v2` so we can train it end-to-end
in Scale 2. The output is the mean-pooled last hidden state, L2-normalized,
which exactly matches what `sentence_transformers.SentenceTransformer` returns
for this checkpoint with default settings — so a frozen pass should reproduce
Scale-1 embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, masking padding."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class MiniLMEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_seq_length: int = 256,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_dim = self.bert.config.hidden_size

    def freeze(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad = True

    def gradient_checkpointing_enable(self) -> None:
        self.bert.gradient_checkpointing_enable()

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pool(out.last_hidden_state, attention_mask)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def encode_text(self, texts: list[str], device: str | torch.device) -> torch.Tensor:
        """One-shot tokenize + forward for inference. Use sparingly — at training
        time the trainer calls tokenize + forward separately for batched control."""
        tok = self.tokenize(texts)
        tok = {k: v.to(device) for k, v in tok.items()}
        return self(tok["input_ids"], tok["attention_mask"])
