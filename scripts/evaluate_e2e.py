"""Run the Scale-2 comparison: cosine baseline vs DRT, both on dev queries.

Loads the latest epoch checkpoints unless explicit paths are given. Encodes
the full corpus through each model, computes top-100 per dev query, and prints
the side-by-side metrics table to stdout (also writes to --output if set).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from evaluation.evaluate_e2e import evaluate_baseline, evaluate_drt, print_comparison
from models.drt_model import DRTModel
from models.encoder import MiniLMEncoder


def _latest(dir_path: Path, prefix: str) -> Path | None:
    candidates = sorted(dir_path.glob(f"{prefix}*.pt"))
    return candidates[-1] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--checkpoint-drt", type=Path, default=None)
    ap.add_argument("--checkpoint-baseline", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw_full"))
    ap.add_argument("--output", type=Path, default=Path("logs/comparison.txt"))
    ap.add_argument("--score-batch", type=int, default=32)
    args = ap.parse_args()

    drt_ckpt = args.checkpoint_drt or _latest(args.checkpoint_dir, "drt_scale2_epoch")
    bl_ckpt = args.checkpoint_baseline or _latest(args.checkpoint_dir, "cosine_baseline_epoch")
    if drt_ckpt is None or bl_ckpt is None:
        print(f"ERROR: missing checkpoints in {args.checkpoint_dir}", file=sys.stderr)
        return 2
    print(f"DRT checkpoint:      {drt_ckpt}")
    print(f"Baseline checkpoint: {bl_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    corpus_p = args.data_dir / "corpus.jsonl"
    dev_q_p = args.data_dir / "dev_queries.jsonl"
    dev_qrels_p = args.data_dir / "qrels" / "dev.tsv"

    # Cosine baseline
    bl_state = torch.load(bl_ckpt, map_location=device, weights_only=False)
    bl_cfg = bl_state.get("config", {})
    encoder = MiniLMEncoder(
        model_name=bl_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        max_seq_length=bl_cfg.get("max_seq_length", 256),
        normalize=True,
    ).to(device)
    encoder.load_state_dict(bl_state["model"])
    print("\n[Cosine baseline]")
    bl_metrics = evaluate_baseline(
        encoder, corpus_p, dev_q_p, dev_qrels_p, device, score_batch=args.score_batch
    )
    print(f"  MRR@10={bl_metrics['MRR@10']:.4f} | nDCG@10={bl_metrics['nDCG@10']:.4f} | Recall@100={bl_metrics['Recall@100']:.4f}")
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    # DRT
    drt_state = torch.load(drt_ckpt, map_location=device, weights_only=False)
    drt_cfg = drt_state.get("config", {})
    model = DRTModel(
        model_name=drt_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        max_seq_length=drt_cfg.get("max_seq_length", 256),
        decomp_hidden=drt_cfg.get("decomp_hidden", 512),
        attn_hidden=drt_cfg.get("attn_hidden", 64),
        k=drt_cfg.get("k", 6),
        sub_dim=drt_cfg.get("sub_dim", 64),
    ).to(device)
    model.load_state_dict(drt_state["model"])
    print("\n[DRT]")
    drt_metrics = evaluate_drt(
        model, corpus_p, dev_q_p, dev_qrels_p, device, score_batch=args.score_batch
    )
    print(f"  MRR@10={drt_metrics['MRR@10']:.4f} | nDCG@10={drt_metrics['nDCG@10']:.4f} | Recall@100={drt_metrics['Recall@100']:.4f}")

    print()
    table = print_comparison(bl_metrics, drt_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"DRT checkpoint:      {drt_ckpt}\n")
        f.write(f"Baseline checkpoint: {bl_ckpt}\n\n")
        f.write(json.dumps({"baseline": bl_metrics, "drt": drt_metrics}, indent=2) + "\n\n")
        f.write(table + "\n")
    print(f"\nSaved → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
