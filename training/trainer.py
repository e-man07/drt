"""Scale-1 DRT training loop on frozen embeddings.

Trains DecompositionHead + QueryAttentionHead on (query, positive) pairs
using in-batch negatives. No encoder forward pass — embeddings are
precomputed and loaded by the Dataset.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from losses.combined import combined_loss
from models.scorer import DRTScorer
from training.scheduler import cosine_warmup_schedule


@dataclass
class TrainConfig:
    embed_dim: int = 384
    decomp_hidden: int = 512
    attn_hidden: int = 64
    k: int = 6
    sub_dim: int = 64

    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    temperature: float = 0.05
    lambda_decorr: float = 0.1
    slot_dropout_p: float = 0.15

    num_epochs: int = 20
    warmup_ratio: float = 0.1

    device: str = "auto"
    seed: int = 42
    log_every: int = 10
    num_workers: int = 0


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_drt(train_dataset: Dataset, config: TrainConfig) -> DRTScorer:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = detect_device(config.device)
    print(f"Device: {device}")

    model = DRTScorer(
        embed_dim=config.embed_dim,
        decomp_hidden=config.decomp_hidden,
        attn_hidden=config.attn_hidden,
        k=config.k,
        sub_dim=config.sub_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )
    if len(loader) == 0:
        raise ValueError(
            f"Empty dataloader: dataset has {len(train_dataset)} samples but "
            f"batch_size={config.batch_size} with drop_last=True yields no batches."
        )

    total_steps = len(loader) * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    print(
        f"Steps/epoch: {len(loader)} | total steps: {total_steps} "
        f"| warmup: {warmup_steps}"
    )

    step = 0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_t0 = time.time()
        sums = {"loss": 0.0, "ret": 0.0, "dec": 0.0}

        nb = device == "cuda"
        for q, p in loader:
            q = q.to(device, non_blocking=nb)
            p = p.to(device, non_blocking=nb)

            q_subs, q_alphas = model.encode_query(q)
            p_subs = model.encode_doc(p)

            loss, comp = combined_loss(
                q_subs,
                q_alphas,
                p_subs,
                temperature=config.temperature,
                lambda_decorr=config.lambda_decorr,
                slot_dropout_p=config.slot_dropout_p,
                training=True,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            sums["loss"] += loss.item()
            sums["ret"] += comp["retrieval"].item()
            sums["dec"] += comp["decorrelation"].item()

            if step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  step {step:5d} | loss={loss.item():.4f} "
                    f"ret={comp['retrieval'].item():.4f} "
                    f"dec={comp['decorrelation'].item():.4f} | lr={lr:.2e}"
                )

        n = len(loader)
        elapsed = time.time() - epoch_t0
        print(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"loss={sums['loss'] / n:.4f} ret={sums['ret'] / n:.4f} "
            f"dec={sums['dec'] / n:.4f} | {elapsed:.1f}s"
        )

    return model
