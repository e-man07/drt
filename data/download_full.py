"""Download full MS MARCO passage ranking + BM25 hard negatives for Scale 2.

Outputs (under --out-dir, default data/raw_full):
  - corpus.jsonl                 8,841,823 passages, {_id, title, text}
  - train_queries.jsonl          ~502,939 train queries, {_id, text}
  - dev_queries.jsonl            6,980 dev-small queries, {_id, text}
  - qrels/train.tsv              ~533K rows, query-id<TAB>corpus-id<TAB>score
  - qrels/dev.tsv                ~7,437 rows
  - hard_negatives.tsv           qidpidtriples.train.small.tsv (~270M lines)

The first three datasets come from HuggingFace `BeIR/msmarco` and
`BeIR/msmarco-qrels`. The hard negatives come from the official MS MARCO
mirror — they aren't bundled in the BEIR HF dataset.

This is the Scale-2 download (full corpus, no subsampling). Total ~7-8 GB on
disk. Designed to run on the Akash A100 box where /workspace is the persistent
volume.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from datasets import load_dataset


# MS MARCO BM25 hard negatives. The "small" tarball is no longer hosted; the
# z22 mirror still serves the gzipped full set (~270M triples), which our
# loader caps at 7 negatives per (qid, pos) pair so most rows are discarded.
HARD_NEGATIVES_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz"
)


def _pick_col(column_names, *candidates: str) -> str:
    for c in candidates:
        if c in column_names:
            return c
    raise ValueError(f"None of {candidates} in {column_names}")


def write_qrels_tsv(qrels_dataset, out_path: Path) -> int:
    qcol = _pick_col(qrels_dataset.column_names, "query-id", "query_id")
    ccol = _pick_col(qrels_dataset.column_names, "corpus-id", "corpus_id")
    scol = _pick_col(qrels_dataset.column_names, "score", "relevance")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels_dataset:
            f.write(f"{row[qcol]}\t{row[ccol]}\t{row[scol]}\n")
    return len(qrels_dataset)


def write_queries_filtered(
    queries_dataset, allowed_qids: set[str], out_path: Path
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in queries_dataset:
            qid = str(row["_id"])
            if qid in allowed_qids:
                f.write(json.dumps({"_id": qid, "text": row["text"]}, ensure_ascii=False) + "\n")
                n += 1
    return n


def write_corpus(corpus_dataset, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in corpus_dataset:
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
            n += 1
    return n


def download_hard_negatives(out_path: Path) -> int:
    """Download qidpidtriples.train.full.2.tsv.gz and decompress to out_path.

    The MS MARCO mirror serves gzipped TSV (qid\\tpos_pid\\tneg_pid). We stream
    decompress to disk so we never hold the full ~8 GB uncompressed file in
    memory. Returns the number of rows written.
    """
    import gzip

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_gz = out_path.with_suffix(".tsv.gz")
    print(f"Downloading {HARD_NEGATIVES_URL}")
    print(f"  -> {tmp_gz}")
    with urllib.request.urlopen(HARD_NEGATIVES_URL) as resp, open(tmp_gz, "wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)

    print(f"Decompressing -> {out_path}")
    n = 0
    with gzip.open(tmp_gz, "rb") as src, open(out_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            n += chunk.count(b"\n")
    tmp_gz.unlink()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw_full"))
    ap.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip outputs that already exist (idempotent re-runs).",
    )
    ap.add_argument(
        "--no-hard-negatives",
        action="store_true",
        help="Skip downloading hard negatives (~3 GB). For smoke tests.",
    )
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    f_corpus = out / "corpus.jsonl"
    f_train_q = out / "train_queries.jsonl"
    f_dev_q = out / "dev_queries.jsonl"
    f_train_qrels = out / "qrels" / "train.tsv"
    f_dev_qrels = out / "qrels" / "dev.tsv"
    f_hn = out / "hard_negatives.tsv"

    def has(path: Path) -> bool:
        return args.skip_if_exists and path.exists() and path.stat().st_size > 0

    if not (has(f_train_qrels) and has(f_dev_qrels)):
        print("Loading qrels (train + validation)...")
        qrels_train = load_dataset("BeIR/msmarco-qrels", split="train")
        qrels_dev = load_dataset("BeIR/msmarco-qrels", split="validation")
        train_qids = {str(x) for x in qrels_train["query-id"]}
        dev_qids = {str(x) for x in qrels_dev["query-id"]}
        print(f"  train qrels: {len(qrels_train):,} ({len(train_qids):,} queries)")
        print(f"  dev   qrels: {len(qrels_dev):,} ({len(dev_qids):,} queries)")
        print(f"Writing {f_train_qrels}")
        write_qrels_tsv(qrels_train, f_train_qrels)
        print(f"Writing {f_dev_qrels}")
        write_qrels_tsv(qrels_dev, f_dev_qrels)
    else:
        print(f"qrels already present in {out / 'qrels'}, skipping.")
        train_qids = {row.split("\t")[0] for i, row in enumerate(open(f_train_qrels)) if i > 0}
        dev_qids = {row.split("\t")[0] for i, row in enumerate(open(f_dev_qrels)) if i > 0}

    if not (has(f_train_q) and has(f_dev_q)):
        print("Loading queries (filtering to train+dev)...")
        queries = load_dataset("BeIR/msmarco", "queries", split="queries")
        print(f"Writing {f_train_q}")
        n_train = write_queries_filtered(queries, train_qids, f_train_q)
        print(f"  matched train queries: {n_train:,}")
        print(f"Writing {f_dev_q}")
        n_dev = write_queries_filtered(queries, dev_qids, f_dev_q)
        print(f"  matched dev queries:   {n_dev:,}")
    else:
        print(f"queries already present, skipping.")

    if not has(f_corpus):
        print("Loading corpus (8.84M)...")
        corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
        print(f"  corpus rows: {len(corpus):,}")
        print(f"Writing {f_corpus}")
        n = write_corpus(corpus, f_corpus)
        print(f"  wrote {n:,} passages")
    else:
        print(f"corpus already present, skipping.")

    if not args.no_hard_negatives and not has(f_hn):
        print("Downloading hard negatives...")
        n = download_hard_negatives(f_hn)
        print(f"  wrote {n:,} hard-negative triples")
    elif args.no_hard_negatives:
        print("--no-hard-negatives set, skipping hard-negatives download.")
    else:
        print(f"hard negatives already present, skipping.")

    print(f"\nDone. Outputs in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
