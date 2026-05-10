"""End-to-end DRT trainer with mixed-precision FP16 and differential LRs.

Used for Scale 2 — the encoder (MiniLM) is unfrozen and trained jointly with
the decomposition + attention heads. Encoder gets a low LR (5e-5), heads get
a high LR (2e-3). InfoNCE retrieval loss + decorrelation loss + slot dropout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.dataset_online import OnlineMSMarcoDataset, TokenizingCollator
from losses.combined import slot_dropout
from losses.decorrelation import decorrelation_loss
from models.drt_model import DRTModel
from training.scheduler import cosine_warmup_schedule


@dataclass
class E2EConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 256
    k: int = 6
    sub_dim: int = 64
    decomp_hidden: int = 512
    attn_hidden: int = 64

    encoder_lr: float = 5e-5
    head_lr: float = 2e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    batch_size: int = 512
    grad_accum_steps: int = 1
    num_epochs: int = 5
    num_hard_negatives: int = 7

    temperature: float = 0.05
    lambda_decorr: float = 0.1
    slot_dropout_p: float = 0.15
    grad_checkpoint: bool = True

    log_every: int = 50
    save_every_epoch: bool = True
    seed: int = 42
    num_workers: int = 4


def info_nce_with_explicit_negatives(
    pos_scores: torch.Tensor,    # (B,)
    in_batch_neg_scores: torch.Tensor,  # (B, B-1) -- other queries' positives
    hard_neg_scores: torch.Tensor,      # (B, num_neg)
    temperature: float,
) -> torch.Tensor:
    logits = torch.cat(
        [pos_scores.unsqueeze(1), in_batch_neg_scores, hard_neg_scores], dim=1
    )
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits / temperature, target)


def _drt_forward_pass(
    model: DRTModel,
    q_tok: dict,
    p_tok: dict,
    n_tok: dict,
    num_neg: int,
    device: str,
    config: E2EConfig,
):
    """Runs one forward pass and returns (loss, components_dict).

    Tokenization happens on DataLoader workers (TokenizingCollator). This
    function only does encoder forward + decomposition + attention + losses.
    """
    B = q_tok["input_ids"].shape[0]

    q_tok = {k: v.to(device, non_blocking=True) for k, v in q_tok.items()}
    p_tok = {k: v.to(device, non_blocking=True) for k, v in p_tok.items()}
    n_tok = {k: v.to(device, non_blocking=True) for k, v in n_tok.items()}

    q_emb = model.encoder(q_tok["input_ids"], q_tok["attention_mask"])  # (B, 384)
    p_emb = model.encoder(p_tok["input_ids"], p_tok["attention_mask"])  # (B, 384)
    n_emb = model.encoder(n_tok["input_ids"], n_tok["attention_mask"])  # (B*num_neg, 384)

    q_subs = model.decomposition(q_emb)
    p_subs = model.decomposition(p_emb)
    n_subs = model.decomposition(n_emb).reshape(B, num_neg, model.k, model.sub_dim)
    q_alphas = model.attention(q_emb)

    pre_drop_subs = torch.cat([q_subs, p_subs], dim=0)
    L_dec = decorrelation_loss(pre_drop_subs)

    q_d = slot_dropout(q_subs, p=config.slot_dropout_p, training=True)
    p_d = slot_dropout(p_subs, p=config.slot_dropout_p, training=True)
    n_d = slot_dropout(
        n_subs.reshape(B * num_neg, model.k, model.sub_dim),
        p=config.slot_dropout_p,
        training=True,
    ).reshape(B, num_neg, model.k, model.sub_dim)

    # Per-pair positive: (B,)
    pos_scores = (q_alphas * (q_d * p_d).sum(dim=-1)).sum(dim=-1)

    # In-batch negatives: each query against every other query's positive
    cos_inbatch = torch.einsum("ikd,jkd->ijk", q_d, p_d)
    inbatch_full = (q_alphas.unsqueeze(1) * cos_inbatch).sum(dim=-1)  # (B, B)
    eye = torch.eye(B, dtype=torch.bool, device=inbatch_full.device)
    in_batch_neg_scores = inbatch_full.masked_select(~eye).reshape(B, B - 1)

    # Hard negatives: (B, num_neg)
    cos_hard = (q_d.unsqueeze(1) * n_d).sum(dim=-1)
    hard_neg_scores = (q_alphas.unsqueeze(1) * cos_hard).sum(dim=-1)

    L_ret = info_nce_with_explicit_negatives(
        pos_scores, in_batch_neg_scores, hard_neg_scores, config.temperature
    )

    L_total = L_ret + config.lambda_decorr * L_dec
    return L_total, {"retrieval": L_ret.detach(), "decorrelation": L_dec.detach()}


def train_drt_e2e(
    train_dataset: OnlineMSMarcoDataset,
    config: E2EConfig,
    checkpoint_dir: Path,
    log_path: Path | None = None,
) -> DRTModel:
    torch.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | torch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = DRTModel(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        k=config.k,
        sub_dim=config.sub_dim,
        decomp_hidden=config.decomp_hidden,
        attn_hidden=config.attn_hidden,
    ).to(device)

    if config.grad_checkpoint and device == "cuda":
        model.encoder.gradient_checkpointing_enable()

    encoder_params = list(model.encoder.parameters())
    head_params = model.head_parameters
    n_enc = sum(p.numel() for p in encoder_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"Trainable params — encoder: {n_enc:,} | heads: {n_head:,} | total: {n_enc + n_head:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    collator = TokenizingCollator(model.encoder.tokenizer, max_seq_length=config.max_seq_length)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    total_steps = (len(loader) // config.grad_accum_steps) * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(device == "cuda"))

    print(f"Steps/epoch: {len(loader)} | optim steps: {total_steps} | warmup: {warmup_steps}")

    log_f = open(log_path, "a", encoding="utf-8") if log_path else None
    optim_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_t0 = time.time()
        sums = {"loss": 0.0, "ret": 0.0, "dec": 0.0, "n": 0}

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader):
            with autocast(enabled=(device == "cuda"), dtype=torch.float16):
                loss, comp = _drt_forward_pass(
                    model, batch["q_tok"], batch["p_tok"], batch["n_tok"], batch["num_neg"],
                    device, config,
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
                sums["ret"] += comp["retrieval"].item()
                sums["dec"] += comp["decorrelation"].item()
                sums["n"] += 1

                if optim_step % config.log_every == 0:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    line = (
                        f"  epoch {epoch + 1} | step {optim_step:6d} | "
                        f"loss={loss.item() * config.grad_accum_steps:.4f} "
                        f"ret={comp['retrieval'].item():.4f} "
                        f"dec={comp['decorrelation'].item():.4f} | "
                        f"enc_lr={lrs[0]:.2e} head_lr={lrs[1]:.2e}"
                    )
                    print(line, flush=True)
                    if log_f:
                        log_f.write(line + "\n"); log_f.flush()

        n = max(1, sums["n"])
        elapsed = time.time() - epoch_t0
        epoch_line = (
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"loss={sums['loss'] / n:.4f} ret={sums['ret'] / n:.4f} "
            f"dec={sums['dec'] / n:.4f} | {elapsed:.0f}s"
        )
        print(epoch_line, flush=True)
        if log_f:
            log_f.write(epoch_line + "\n"); log_f.flush()

        if config.save_every_epoch:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"drt_scale2_epoch{epoch + 1}.pt"
            torch.save(
                {"model": model.state_dict(), "config": config.__dict__, "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  saved {ckpt_path}", flush=True)

    if log_f:
        log_f.close()
    return model
