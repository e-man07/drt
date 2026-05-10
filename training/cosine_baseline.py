"""Train a vanilla MiniLM bi-encoder as the fair-comparison baseline.

Same encoder weights, same training data (queries + BM25 hard negatives), same
epoch count and batch size as the DRT trainer. The only difference: scoring is
plain cosine (dot product over single 384-d vectors), and there is no
decomposition head, attention head, slot dropout, or decorrelation loss.

The point is to isolate the scoring-function change. If DRT beats this, the
gain is from the architecture, not from "we trained MiniLM on MS MARCO".
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.dataset_online import OnlineMSMarcoDataset, collate_text_triples
from models.encoder import DEFAULT_MODEL, MiniLMEncoder
from training.scheduler import cosine_warmup_schedule


@dataclass
class CosineBaselineConfig:
    model_name: str = DEFAULT_MODEL
    max_seq_length: int = 256

    encoder_lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    batch_size: int = 512
    grad_accum_steps: int = 1
    num_epochs: int = 5
    num_hard_negatives: int = 7

    temperature: float = 0.05
    grad_checkpoint: bool = True

    log_every: int = 50
    save_every_epoch: bool = True
    seed: int = 42
    num_workers: int = 4


def _baseline_forward_pass(
    encoder: MiniLMEncoder,
    queries: list[str],
    positives: list[str],
    flat_negatives: list[str],
    num_neg: int,
    device: str,
    temperature: float,
) -> torch.Tensor:
    B = len(queries)

    q_tok = encoder.tokenize(queries)
    p_tok = encoder.tokenize(positives)
    n_tok = encoder.tokenize(flat_negatives)
    q_tok = {k: v.to(device) for k, v in q_tok.items()}
    p_tok = {k: v.to(device) for k, v in p_tok.items()}
    n_tok = {k: v.to(device) for k, v in n_tok.items()}

    q = encoder(q_tok["input_ids"], q_tok["attention_mask"])  # (B, 384)
    p = encoder(p_tok["input_ids"], p_tok["attention_mask"])  # (B, 384)
    n = encoder(n_tok["input_ids"], n_tok["attention_mask"])  # (B*num_neg, 384)
    n = n.reshape(B, num_neg, -1)

    pos_scores = (q * p).sum(dim=-1, keepdim=True)              # (B, 1)
    inbatch_full = q @ p.t()                                    # (B, B)
    eye = torch.eye(B, dtype=torch.bool, device=q.device)
    in_batch_neg = inbatch_full.masked_select(~eye).reshape(B, B - 1)
    hard_neg = (q.unsqueeze(1) * n).sum(dim=-1)                 # (B, num_neg)

    logits = torch.cat([pos_scores, in_batch_neg, hard_neg], dim=1)
    target = torch.zeros(B, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits / temperature, target)


def train_cosine_baseline(
    train_dataset: OnlineMSMarcoDataset,
    config: CosineBaselineConfig,
    checkpoint_dir: Path,
    log_path: Path | None = None,
) -> MiniLMEncoder:
    torch.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder = MiniLMEncoder(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        normalize=True,
    ).to(device)

    if config.grad_checkpoint and device == "cuda":
        encoder.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.encoder_lr,
        weight_decay=config.weight_decay,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_text_triples,
        pin_memory=(device == "cuda"),
    )

    total_steps = (len(loader) // config.grad_accum_steps) * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(device == "cuda"))

    print(f"Steps/epoch: {len(loader)} | optim steps: {total_steps} | warmup: {warmup_steps}")

    log_f = open(log_path, "a", encoding="utf-8") if log_path else None
    optim_step = 0
    for epoch in range(config.num_epochs):
        encoder.train()
        epoch_t0 = time.time()
        sums = {"loss": 0.0, "n": 0}

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (queries, positives, flat_negatives, num_neg) in enumerate(loader):
            with autocast(enabled=(device == "cuda"), dtype=torch.float16):
                loss = _baseline_forward_pass(
                    encoder, queries, positives, flat_negatives, num_neg,
                    device, config.temperature,
                )
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optim_step += 1
                sums["loss"] += loss.item() * config.grad_accum_steps
                sums["n"] += 1

                if optim_step % config.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    line = (
                        f"  epoch {epoch + 1} | step {optim_step:6d} | "
                        f"loss={loss.item() * config.grad_accum_steps:.4f} | lr={lr:.2e}"
                    )
                    print(line, flush=True)
                    if log_f:
                        log_f.write(line + "\n"); log_f.flush()

        n = max(1, sums["n"])
        elapsed = time.time() - epoch_t0
        epoch_line = (
            f"Epoch {epoch + 1}/{config.num_epochs} | loss={sums['loss'] / n:.4f} | {elapsed:.0f}s"
        )
        print(epoch_line, flush=True)
        if log_f:
            log_f.write(epoch_line + "\n"); log_f.flush()

        if config.save_every_epoch:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"cosine_baseline_epoch{epoch + 1}.pt"
            torch.save(
                {"model": encoder.state_dict(), "config": config.__dict__, "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  saved {ckpt_path}", flush=True)

    if log_f:
        log_f.close()
    return encoder
