"""Scale-2 entry point: end-to-end DRT training on full MS MARCO.

Reads hyperparams from configs/scale2.yaml. CLI flags override individual
fields. Saves checkpoints under --checkpoint-dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from data.dataset_online import OnlineMSMarcoDataset
from training.trainer_e2e import E2EConfig, train_drt_e2e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/scale2.yaml"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw_full"))
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--log-path", type=Path, default=Path("logs/drt_scale2.log"))

    # Common overrides
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-epochs", type=int, default=None)
    ap.add_argument("--encoder-lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--lambda-decorr", type=float, default=None)
    ap.add_argument("--slot-dropout-p", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--no-hard-negatives", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}

    # Apply CLI overrides
    overrides = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "lambda_decorr": args.lambda_decorr,
        "slot_dropout_p": args.slot_dropout_p,
        "k": args.k,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg_dict[k] = v
    if args.no_grad_checkpoint:
        cfg_dict["grad_checkpoint"] = False

    config = E2EConfig(**{k: v for k, v in cfg_dict.items() if k in E2EConfig.__dataclass_fields__})

    hn_path = None if args.no_hard_negatives else (args.data_dir / "hard_negatives.tsv")
    if hn_path is not None and not hn_path.exists():
        print(f"Hard negatives not found at {hn_path}; falling back to random negatives.")
        hn_path = None

    dataset = OnlineMSMarcoDataset(
        queries_path=args.data_dir / "train_queries.jsonl",
        corpus_path=args.data_dir / "corpus.jsonl",
        qrels_path=args.data_dir / "qrels" / "train.tsv",
        hard_negatives_path=hn_path,
        num_hard_negatives=config.num_hard_negatives,
        seed=config.seed,
    )
    print(f"Training pairs: {len(dataset):,}")

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    train_drt_e2e(dataset, config, args.checkpoint_dir, args.log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
