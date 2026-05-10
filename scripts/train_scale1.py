"""Scale-1 training entry point: train DRT head on frozen MS MARCO embeddings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from data.dataset import FrozenEmbeddingDataset, load_query_ids, split_query_ids
from training.trainer import TrainConfig, train_drt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--qrels", type=Path, default=Path("data/raw/qrels/dev.tsv"))
    ap.add_argument(
        "--eval-qrels",
        type=Path,
        default=None,
        help="If set, --qrels is used in full for training and --eval-qrels for "
        "held-out evaluation (no 80/20 split). Use this for train+dev mode.",
    )
    ap.add_argument(
        "--query-prefix",
        default="",
        help="Prefix for query_*.npy files in --embeddings-dir. '' = dev queries, "
        "'train_' = train queries.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drt_scale1.pt"),
    )
    ap.add_argument("--train-frac", type=float, default=0.8)

    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sub-dim", type=int, default=64)
    ap.add_argument("--decomp-hidden", type=int, default=512)
    ap.add_argument("--attn-hidden", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-epochs", type=int, default=20)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lambda-decorr", type=float, default=0.1)
    ap.add_argument("--slot-dropout-p", type=float, default=0.15)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig(
        k=args.k,
        sub_dim=args.sub_dim,
        decomp_hidden=args.decomp_hidden,
        attn_hidden=args.attn_hidden,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        lambda_decorr=args.lambda_decorr,
        slot_dropout_p=args.slot_dropout_p,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        num_workers=args.num_workers,
    )

    if args.eval_qrels is None:
        train_qids, eval_qids = split_query_ids(args.qrels, args.train_frac, args.seed)
        print(
            f"Query split (seed={args.seed}): "
            f"{len(train_qids):,} train | {len(eval_qids):,} eval"
        )
    else:
        train_qids = load_query_ids(args.qrels)
        eval_qids = load_query_ids(args.eval_qrels)
        print(
            f"Train queries from {args.qrels.name}: {len(train_qids):,}  |  "
            f"Eval queries from {args.eval_qrels.name}: {len(eval_qids):,}"
        )

    train_dataset = FrozenEmbeddingDataset(
        args.embeddings_dir,
        args.qrels,
        query_ids=train_qids,
        query_prefix=args.query_prefix,
    )
    print(f"Training pairs: {len(train_dataset):,}")

    model = train_drt(train_dataset, cfg)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "train_qids": train_qids,
            "eval_qids": eval_qids,
        },
        args.checkpoint,
    )
    print(f"\nSaved checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
