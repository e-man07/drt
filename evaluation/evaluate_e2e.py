"""End-to-end evaluation: encode the full corpus through the trained model,
then compute MRR@10 / nDCG@10 / Recall@100 on dev queries.

Designed to run on the same A100 the training used. Uses fp16 for the corpus
encoding (8.84M × 384 × 2 B ≈ 6.7 GB), brute-force matmul for scoring
(no FAISS — gives exact numbers and is plenty fast on A100).

Two callable surfaces:
  - `evaluate_drt(...)` — DRT model: encode through encoder + decomposition,
    compute weighted slot scores
  - `evaluate_baseline(...)` — vanilla bi-encoder: encode through encoder only,
    compute cosine
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from models.drt_model import DRTModel
from models.encoder import MiniLMEncoder


TOP_K = 100


class _TextDataset(Dataset):
    """Trivial Dataset wrapper around a list of strings."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class _TextTokenizerCollator:
    """Tokenizes a batch of strings on DataLoader worker processes."""

    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[str]):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )


def _load_jsonl(path: Path) -> list[tuple[str, str]]:
    """Return list of (id, text) preserving file order."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            title = row.get("title") or ""
            out.append((str(row["_id"]), f"{title} {text}".strip() if title else text))
    return out


def _load_qrels_grouped(path: Path) -> dict[str, dict[str, int]]:
    """Read qrels TSV → {qid: {pid: score}} (positive only)."""
    out = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            qid, pid, score = row[0], row[1], int(row[2])
            if score > 0:
                out[qid][pid] = score
    return out


def _topk_descending(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)[:, :k]
    idx_part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    sort_order = np.argsort(-scores[rows, idx_part], axis=1)
    return idx_part[rows, sort_order]


def _encode_corpus_baseline(
    encoder: MiniLMEncoder,
    corpus: list[tuple[str, str]],
    device: str,
    batch_size: int = 1024,
    num_workers: int = 4,
    fp16: bool = True,
    log_every_batches: int = 10,
) -> tuple[torch.Tensor, list[str]]:
    encoder.eval()
    pids = [pid for pid, _ in corpus]
    texts = [t for _, t in corpus]
    n = len(texts)
    all_emb = torch.empty(
        (n, encoder.embed_dim),
        dtype=torch.float16 if fp16 else torch.float32,
        device="cpu",
    )

    dataset = _TextDataset(texts)
    collator = _TextTokenizerCollator(encoder.tokenizer, encoder.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    t0 = time.time()
    cursor = 0
    with torch.no_grad():
        for bi, tok in enumerate(loader):
            bs = tok["input_ids"].shape[0]
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
            with torch.cuda.amp.autocast(enabled=fp16 and device == "cuda", dtype=torch.float16):
                emb = encoder(tok["input_ids"], tok["attention_mask"])
            all_emb[cursor : cursor + bs] = emb.detach().to(dtype=all_emb.dtype, device="cpu")
            cursor += bs
            if (bi + 1) % log_every_batches == 0 or cursor >= n:
                elapsed = time.time() - t0
                rate = cursor / elapsed if elapsed > 0 else 0.0
                eta = (n - cursor) / rate if rate > 0 else float("inf")
                print(
                    f"  baseline corpus {cursor:>9,}/{n:,} "
                    f"({100 * cursor / n:5.1f}%) | {elapsed:6.0f}s | "
                    f"{rate:>5.0f} rows/s | ETA {eta:5.0f}s",
                    flush=True,
                )
    return all_emb, pids


def _encode_corpus_drt(
    model: DRTModel,
    corpus: list[tuple[str, str]],
    device: str,
    batch_size: int = 1024,
    num_workers: int = 4,
    fp16: bool = True,
    log_every_batches: int = 10,
) -> tuple[torch.Tensor, list[str]]:
    model.eval()
    pids = [pid for pid, _ in corpus]
    texts = [t for _, t in corpus]
    n = len(texts)
    all_subs = torch.empty(
        (n, model.k, model.sub_dim),
        dtype=torch.float16 if fp16 else torch.float32,
        device="cpu",
    )

    dataset = _TextDataset(texts)
    collator = _TextTokenizerCollator(
        model.encoder.tokenizer, model.encoder.max_seq_length
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    t0 = time.time()
    cursor = 0
    with torch.no_grad():
        for bi, tok in enumerate(loader):
            bs = tok["input_ids"].shape[0]
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
            with torch.cuda.amp.autocast(enabled=fp16 and device == "cuda", dtype=torch.float16):
                emb = model.encoder(tok["input_ids"], tok["attention_mask"])
                subs = model.decomposition(emb)
            all_subs[cursor : cursor + bs] = subs.detach().to(dtype=all_subs.dtype, device="cpu")
            cursor += bs
            if (bi + 1) % log_every_batches == 0 or cursor >= n:
                elapsed = time.time() - t0
                rate = cursor / elapsed if elapsed > 0 else 0.0
                eta = (n - cursor) / rate if rate > 0 else float("inf")
                print(
                    f"  drt corpus      {cursor:>9,}/{n:,} "
                    f"({100 * cursor / n:5.1f}%) | {elapsed:6.0f}s | "
                    f"{rate:>5.0f} rows/s | ETA {eta:5.0f}s",
                    flush=True,
                )
    return all_subs, pids


def _evaluate_topk_metrics(
    topk_indices: np.ndarray,
    qrels_idx_dicts: list[dict[int, int]],
    qrels_idx_sets: list[set[int]],
) -> dict[str, float]:
    return {
        "MRR@10": mrr_at_k(topk_indices, qrels_idx_sets, k=10),
        "nDCG@10": ndcg_at_k(topk_indices, qrels_idx_dicts, k=10),
        "Recall@100": recall_at_k(topk_indices, qrels_idx_sets, k=100),
    }


def evaluate_baseline(
    encoder: MiniLMEncoder,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    device: str,
    score_batch: int = 64,
) -> dict[str, float]:
    corpus = _load_jsonl(corpus_path)
    queries = _load_jsonl(queries_path)
    qrels = _load_qrels_grouped(qrels_path)

    print(f"Encoding {len(corpus):,} passages (baseline)...")
    p_emb, pids = _encode_corpus_baseline(encoder, corpus, device)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    print(f"Encoding {len(queries):,} queries (baseline)...")
    q_emb, qids = _encode_corpus_baseline(encoder, queries, device)
    qrels_idx_dicts: list[dict[int, int]] = []
    for qid in qids:
        rel = qrels.get(qid, {})
        rel_idx = {pid_to_idx[p]: s for p, s in rel.items() if p in pid_to_idx}
        qrels_idx_dicts.append(rel_idx)
    qrels_idx_sets = [set(d.keys()) for d in qrels_idx_dicts]

    Q = q_emb.shape[0]
    topk = np.zeros((Q, TOP_K), dtype=np.int64)
    p_emb_t = p_emb.to(device)
    for s in range(0, Q, score_batch):
        e = min(s + score_batch, Q)
        qb = q_emb[s:e].to(device)
        with torch.no_grad():
            scores = (qb @ p_emb_t.t()).float().cpu().numpy()
        topk[s:e] = _topk_descending(scores, TOP_K)
    return _evaluate_topk_metrics(topk, qrels_idx_dicts, qrels_idx_sets)


def evaluate_drt(
    model: DRTModel,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    device: str,
    score_batch: int = 32,
) -> dict[str, float]:
    corpus = _load_jsonl(corpus_path)
    queries = _load_jsonl(queries_path)
    qrels = _load_qrels_grouped(qrels_path)

    print(f"Encoding {len(corpus):,} passages (DRT)...")
    doc_subs, pids = _encode_corpus_drt(model, corpus, device)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    print(f"Encoding {len(queries):,} queries (DRT)...")
    model.eval()
    q_subs_list = []
    q_alphas_list = []
    qids = [qid for qid, _ in queries]
    qtexts = [t for _, t in queries]
    chunk_size = 256
    with torch.no_grad():
        for s in range(0, len(qtexts), chunk_size):
            e = min(s + chunk_size, len(qtexts))
            with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float16):
                qs, qa = model.encode_query(qtexts[s:e], device)
            q_subs_list.append(qs.half().cpu())
            q_alphas_list.append(qa.half().cpu())
    q_subs = torch.cat(q_subs_list, dim=0)
    q_alphas = torch.cat(q_alphas_list, dim=0)

    qrels_idx_dicts: list[dict[int, int]] = []
    for qid in qids:
        rel = qrels.get(qid, {})
        rel_idx = {pid_to_idx[p]: s for p, s in rel.items() if p in pid_to_idx}
        qrels_idx_dicts.append(rel_idx)
    qrels_idx_sets = [set(d.keys()) for d in qrels_idx_dicts]

    Q = q_subs.shape[0]
    topk = np.zeros((Q, TOP_K), dtype=np.int64)
    doc_subs_t = doc_subs.to(device)
    for s in range(0, Q, score_batch):
        e = min(s + score_batch, Q)
        qs = q_subs[s:e].to(device)
        qa = q_alphas[s:e].to(device)
        with torch.no_grad():
            cos_per_slot = torch.einsum("bkd,pkd->bpk", qs, doc_subs_t)
            scores = (qa.unsqueeze(1) * cos_per_slot).sum(dim=-1).float().cpu().numpy()
        topk[s:e] = _topk_descending(scores, TOP_K)
    return _evaluate_topk_metrics(topk, qrels_idx_dicts, qrels_idx_sets)


def print_comparison(baseline: dict[str, float], drt: dict[str, float]) -> str:
    lines = []
    lines.append("=" * 58)
    lines.append(f"{'Metric':<14} {'Cosine BL':>12} {'DRT':>12} {'Δ':>12}")
    lines.append("-" * 58)
    for name in ["MRR@10", "nDCG@10", "Recall@100"]:
        c = baseline[name]
        d = drt[name]
        delta = d - c
        sign = "+" if delta >= 0 else ""
        lines.append(f"{name:<14} {c:>12.4f} {d:>12.4f} {sign}{delta:>11.4f}")
    text = "\n".join(lines)
    print(text)
    return text
