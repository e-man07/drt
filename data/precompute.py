"""Precompute embeddings for MS MARCO corpus and queries with all-MiniLM-L6-v2.

Reads JSONL files from --in-dir (default data/raw) and writes parallel
.npy arrays to --out-dir (default data/embeddings):

  - passage_embeddings.npy   (N, 384) float32, unnormalized
  - passage_ids.npy          (N,) U16 string
  - query_embeddings.npy     (M, 384) float32, unnormalized
  - query_ids.npy            (M,) U16 string

Embeddings are saved unnormalized so per-sub-vector normalization can happen
inside the DecompositionHead at training time, and a cosine baseline can
normalize at use-time. Device defaults to MPS on Mac, CUDA otherwise, CPU last.

Encoding is chunked: the output array is allocated as a numpy memmap and
each chunk's results are written directly to disk and then released. This
keeps peak memory bounded — important on M4 where MPS shares system RAM.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


EMBED_DIM = 384


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_jsonl(path: Path, text_fn: Callable[[dict], str]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(str(row["_id"]))
            texts.append(text_fn(row))
    return ids, texts


def encode_to_memmap(
    model: SentenceTransformer,
    texts: list[str],
    out_path: Path,
    *,
    batch_size: int,
    chunk_size: int,
    device: str,
    label: str,
) -> None:
    """Encode `texts` in chunks; write directly to a numpy memmap on disk.

    Peak memory stays at one chunk's worth of embeddings (~chunk_size×384×4 bytes)
    plus the model. After each chunk we drop the returned tensor and (on MPS)
    flush the GPU cache.
    """
    n = len(texts)
    arr = np.lib.format.open_memmap(
        out_path, dtype=np.float32, shape=(n, EMBED_DIM), mode="w+"
    )
    n_chunks = (n + chunk_size - 1) // chunk_size
    t0 = time.time()
    for ci, start in enumerate(range(0, n, chunk_size), start=1):
        end = min(start + chunk_size, n)
        chunk_emb = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        arr[start:end] = chunk_emb.astype(np.float32, copy=False)
        del chunk_emb
        gc.collect()
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0.0
        eta = (n - end) / rate if rate > 0 else float("inf")
        print(
            f"  {label} chunk {ci}/{n_chunks} | rows {end:,}/{n:,} "
            f"| {elapsed:.0f}s elapsed, {rate:.0f} rows/s, ETA {eta:.0f}s",
            flush=True,
        )
    arr.flush()
    del arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Encode this many rows per memmap-write cycle.",
    )
    ap.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    p_emb = out / "passage_embeddings.npy"
    p_ids = out / "passage_ids.npy"
    q_emb = out / "query_embeddings.npy"
    q_ids = out / "query_ids.npy"

    train_jsonl = args.in_dir / "train_queries.jsonl"
    train_q_emb_pre = out / "train_query_embeddings.npy"
    train_q_ids_pre = out / "train_query_ids.npy"
    dev_done_pre = all(p.exists() for p in (p_emb, p_ids, q_emb, q_ids))
    train_pending = train_jsonl.exists() and not (train_q_emb_pre.exists() and train_q_ids_pre.exists())
    if not args.force and dev_done_pre and not train_pending:
        print(f"All embedding files already exist in {out}/. Use --force to overwrite.")
        return 0

    device = detect_device(args.device)
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    if args.force or not (p_emb.exists() and p_ids.exists()):
        passage_path = args.in_dir / "corpus.jsonl"
        print(f"Reading {passage_path}")
        passage_ids, passage_texts = read_jsonl(
            passage_path,
            lambda r: f"{r.get('title') or ''} {r['text']}".strip(),
        )
        print(f"  {len(passage_texts):,} passages")

        print("Encoding passages (memmap-streamed)...")
        encode_to_memmap(
            model, passage_texts, p_emb,
            batch_size=args.batch_size, chunk_size=args.chunk_size,
            device=device, label="passages",
        )
        np.save(p_ids, np.array(passage_ids, dtype="U16"))
        n_passages = len(passage_ids)
        del passage_texts, passage_ids
        gc.collect()
    else:
        n_passages = int(np.load(p_ids).shape[0])
        print(f"Passages: skipping (already exist; n={n_passages:,})")

    if args.force or not (q_emb.exists() and q_ids.exists()):
        query_path = args.in_dir / "queries.jsonl"
        print(f"Reading {query_path}")
        query_ids, query_texts = read_jsonl(query_path, lambda r: r["text"])
        print(f"  {len(query_texts):,} queries")

        print("Encoding queries (memmap-streamed)...")
        encode_to_memmap(
            model, query_texts, q_emb,
            batch_size=args.batch_size, chunk_size=args.chunk_size,
            device=device, label="queries",
        )
        np.save(q_ids, np.array(query_ids, dtype="U16"))
        del query_texts, query_ids
        gc.collect()
    else:
        print("Dev queries: skipping (already exist).")

    train_query_path = args.in_dir / "train_queries.jsonl"
    train_q_emb = out / "train_query_embeddings.npy"
    train_q_ids = out / "train_query_ids.npy"
    if train_query_path.exists() and (args.force or not (train_q_emb.exists() and train_q_ids.exists())):
        print(f"Reading {train_query_path}")
        train_query_ids, train_query_texts = read_jsonl(
            train_query_path, lambda r: r["text"]
        )
        print(f"  {len(train_query_texts):,} train queries")
        print("Encoding train queries (memmap-streamed)...")
        encode_to_memmap(
            model, train_query_texts, train_q_emb,
            batch_size=args.batch_size, chunk_size=args.chunk_size,
            device=device, label="train queries",
        )
        np.save(train_q_ids, np.array(train_query_ids, dtype="U16"))
        n_train_queries = len(train_query_ids)
        del train_query_texts, train_query_ids
        gc.collect()
    else:
        n_train_queries = None

    pe_check = np.load(p_emb, mmap_mode="r")
    qe_check = np.load(q_emb, mmap_mode="r")
    assert pe_check.shape == (n_passages, EMBED_DIM), pe_check.shape
    assert qe_check.shape[1] == EMBED_DIM, qe_check.shape

    print("\nDone.")
    print(f"  passage_embeddings.npy : {pe_check.shape} {pe_check.dtype}")
    print(f"  passage_ids.npy        : ({n_passages},) U16")
    print(f"  query_embeddings.npy   : {qe_check.shape} {qe_check.dtype}")
    if n_train_queries is not None:
        tqe_check = np.load(train_q_emb, mmap_mode="r")
        assert tqe_check.shape == (n_train_queries, EMBED_DIM), tqe_check.shape
        print(f"  train_query_embeddings.npy : {tqe_check.shape} {tqe_check.dtype}")
        print(f"  train_query_ids.npy        : ({n_train_queries},) U16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
