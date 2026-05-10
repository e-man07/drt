"""Download MS MARCO passage ranking data via HuggingFace `datasets`.

Outputs (default --out-dir data/raw):
  - corpus.jsonl         JSONL of {_id, title, text}, optionally subsampled
  - queries.jsonl        JSONL of {_id, text}, dev-small (~6,980 queries)
  - qrels/dev.tsv        TSV: query-id<TAB>corpus-id<TAB>score (header included)

With --include-train, additionally:
  - train_queries.jsonl  JSONL of {_id, text}, MS MARCO train queries (~500K)
  - qrels/train.tsv      TSV with train (query, positive-passage) pairs

When subsampling, every passage referenced in dev qrels is always kept so
evaluation remains correct; the remainder is filled by a seeded random sample.
Train qrels reference the full 8.84M corpus, so many train-positive passages
won't be in the subsampled corpus — the dataset class filters those out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset


def _pick_col(column_names, *candidates: str) -> str:
    for c in candidates:
        if c in column_names:
            return c
    raise ValueError(f"None of {candidates} found in columns {column_names}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument(
        "--max-passages",
        type=int,
        default=500_000,
        help="Cap corpus size after the must-keep set is taken; 0 = full corpus.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-train",
        action="store_true",
        help="Also fetch MS MARCO train queries + train qrels (for full-scale training).",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_corpus = out_dir / "corpus.jsonl"
    out_queries = out_dir / "queries.jsonl"
    out_qrels = out_dir / "qrels" / "dev.tsv"
    out_qrels.parent.mkdir(parents=True, exist_ok=True)

    dev_done = (
        not args.force
        and out_corpus.exists()
        and out_queries.exists()
        and out_qrels.exists()
    )

    out_train_queries = out_dir / "train_queries.jsonl"
    out_train_qrels = out_dir / "qrels" / "train.tsv"
    train_done = (
        not args.force
        and out_train_queries.exists()
        and out_train_qrels.exists()
    )

    if dev_done and (not args.include_train or train_done):
        print(f"All requested outputs already exist in {out_dir}/. Use --force to overwrite.")
        return 0

    if dev_done:
        print(f"Dev outputs already in {out_dir}/, skipping dev download.")
    else:
        _download_dev(out_corpus, out_queries, out_qrels, args.max_passages, args.seed)

    if args.include_train and not train_done:
        _download_train(out_train_queries, out_train_qrels)

    return 0


def _download_dev(
    out_corpus: Path,
    out_queries: Path,
    out_qrels: Path,
    max_passages: int,
    seed: int,
) -> None:
    print("Loading qrels (BeIR/msmarco-qrels, split=validation)...")
    qrels = load_dataset("BeIR/msmarco-qrels", split="validation")
    qcol = _pick_col(qrels.column_names, "query-id", "query_id")
    ccol = _pick_col(qrels.column_names, "corpus-id", "corpus_id")
    scol = _pick_col(qrels.column_names, "score", "relevance")

    qrels_query_ids = {str(x) for x in qrels[qcol]}
    must_keep_corpus_ids = {str(x) for x in qrels[ccol]}
    print(
        f"  qrels rows: {len(qrels):,} | unique queries: {len(qrels_query_ids):,} "
        f"| unique positive passages: {len(must_keep_corpus_ids):,}"
    )

    print("Loading queries (BeIR/msmarco, queries) and filtering to dev-small...")
    queries = load_dataset("BeIR/msmarco", "queries", split="queries")
    dev_queries: list[dict] = []
    for row in queries:
        qid = str(row["_id"])
        if qid in qrels_query_ids:
            dev_queries.append({"_id": qid, "text": row["text"]})
    print(f"  matched dev queries: {len(dev_queries):,}")

    print("Loading corpus (BeIR/msmarco, corpus) — this is the large one...")
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    n_total = len(corpus)
    print(f"  corpus rows: {n_total:,}")

    rng = np.random.default_rng(seed)
    if max_passages == 0 or max_passages >= n_total:
        keep_indices = np.arange(n_total, dtype=np.int64)
        n_must, n_sampled = n_total, 0
    else:
        all_ids = corpus["_id"]
        must_keep_idx: list[int] = []
        other_idx: list[int] = []
        for i, cid in enumerate(all_ids):
            (must_keep_idx if str(cid) in must_keep_corpus_ids else other_idx).append(i)
        must_keep_arr = np.asarray(must_keep_idx, dtype=np.int64)
        other_arr = np.asarray(other_idx, dtype=np.int64)
        n_to_sample = max(0, max_passages - must_keep_arr.size)
        if n_to_sample > 0 and other_arr.size > 0:
            n_to_sample = min(n_to_sample, other_arr.size)
            sampled = rng.choice(other_arr, size=n_to_sample, replace=False)
            keep_indices = np.concatenate([must_keep_arr, sampled])
        else:
            keep_indices = must_keep_arr
        keep_indices.sort()
        n_must, n_sampled = int(must_keep_arr.size), int(n_to_sample)

    print(
        f"  keeping {keep_indices.size:,} passages "
        f"(must-keep={n_must:,}, sampled={n_sampled:,})"
    )

    sub_corpus = corpus.select(keep_indices.tolist())

    print(f"Writing {out_corpus}...")
    with open(out_corpus, "w", encoding="utf-8") as f:
        for row in sub_corpus:
            f.write(
                json.dumps(
                    {
                        "_id": str(row["_id"]),
                        "title": row.get("title") or "",
                        "text": row["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Writing {out_queries}...")
    with open(out_queries, "w", encoding="utf-8") as f:
        for q in dev_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Writing {out_qrels}...")
    with open(out_qrels, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            f.write(f"{row[qcol]}\t{row[ccol]}\t{row[scol]}\n")

    print(
        f"\nDone (dev). Corpus: {keep_indices.size:,} | "
        f"Queries: {len(dev_queries):,} | Qrels: {len(qrels):,}"
    )


def _download_train(out_train_queries: Path, out_train_qrels: Path) -> None:
    print("Loading qrels (BeIR/msmarco-qrels, split=train)...")
    qrels = load_dataset("BeIR/msmarco-qrels", split="train")
    qcol = _pick_col(qrels.column_names, "query-id", "query_id")
    ccol = _pick_col(qrels.column_names, "corpus-id", "corpus_id")
    scol = _pick_col(qrels.column_names, "score", "relevance")

    train_query_ids = {str(x) for x in qrels[qcol]}
    print(
        f"  train qrels rows: {len(qrels):,} | unique queries: {len(train_query_ids):,}"
    )

    print("Loading queries (BeIR/msmarco, queries) and filtering to train set...")
    queries = load_dataset("BeIR/msmarco", "queries", split="queries")
    train_queries: list[dict] = []
    for row in queries:
        qid = str(row["_id"])
        if qid in train_query_ids:
            train_queries.append({"_id": qid, "text": row["text"]})
    print(f"  matched train queries: {len(train_queries):,}")

    print(f"Writing {out_train_queries}...")
    with open(out_train_queries, "w", encoding="utf-8") as f:
        for q in train_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Writing {out_train_qrels}...")
    with open(out_train_qrels, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            f.write(f"{row[qcol]}\t{row[ccol]}\t{row[scol]}\n")

    print(
        f"\nDone (train). Train queries: {len(train_queries):,} | "
        f"Train qrels: {len(qrels):,}"
    )


if __name__ == "__main__":
    sys.exit(main())
