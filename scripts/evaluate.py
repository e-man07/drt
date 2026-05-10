"""Evaluate DRT vs cosine baseline on the held-out MS MARCO dev split.

Loads the checkpoint produced by scripts/train_scale1.py (which records the
seeded eval_qids), computes top-k indices for both scoring methods, and prints
a side-by-side metrics table.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from models.scorer import DRTScorer


TOP_K = 100


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            qid, cid, score = row[0], row[1], int(row[2])
            if score > 0:
                out[qid][cid] = score
    return out


def topk_descending(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k entries per row, sorted descending by score."""
    if k >= scores.shape[1]:
        idx_part = np.argsort(-scores, axis=1)
        return idx_part[:, :k]
    idx_part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    sort_order = np.argsort(-scores[rows, idx_part], axis=1)
    return idx_part[rows, sort_order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drt_scale1.pt"))
    ap.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--qrels", type=Path, default=Path("data/raw/qrels/dev.tsv"))
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--score-batch", type=int, default=64)
    args = ap.parse_args()

    device = detect_device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    eval_qids: list[str] = list(ckpt["eval_qids"])
    print(f"Eval queries from checkpoint: {len(eval_qids):,}")

    model = DRTScorer(
        embed_dim=cfg["embed_dim"],
        decomp_hidden=cfg["decomp_hidden"],
        attn_hidden=cfg["attn_hidden"],
        k=cfg["k"],
        sub_dim=cfg["sub_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pe = np.load(args.embeddings_dir / "passage_embeddings.npy")
    pi = np.load(args.embeddings_dir / "passage_ids.npy")
    qe = np.load(args.embeddings_dir / "query_embeddings.npy")
    qi = np.load(args.embeddings_dir / "query_ids.npy")

    pid_to_idx = {str(p): i for i, p in enumerate(pi)}
    qid_to_idx = {str(q): i for i, q in enumerate(qi)}

    eval_qid_list = [qid for qid in eval_qids if qid in qid_to_idx]
    if len(eval_qid_list) < len(eval_qids):
        print(
            f"Warning: {len(eval_qids) - len(eval_qid_list)} eval queries not in "
            f"precomputed embeddings; dropping them."
        )
    eval_q_idx = np.array([qid_to_idx[qid] for qid in eval_qid_list])

    qrels_all = load_qrels(args.qrels)
    qrels_idx_dicts: list[dict[int, int]] = []
    for qid in eval_qid_list:
        rel_str = qrels_all.get(qid, {})
        rel_idx = {pid_to_idx[c]: s for c, s in rel_str.items() if c in pid_to_idx}
        qrels_idx_dicts.append(rel_idx)
    qrels_idx_sets: list[set[int]] = [set(d.keys()) for d in qrels_idx_dicts]
    n_with_rel = sum(1 for s in qrels_idx_sets if s)
    print(f"Eval queries with at least one positive in corpus: {n_with_rel:,}")

    # ───────── Cosine baseline ─────────
    print("\n[Cosine baseline]")
    t0 = time.time()
    qe_eval = qe[eval_q_idx].astype(np.float32, copy=False)
    qe_norm = qe_eval / (np.linalg.norm(qe_eval, axis=1, keepdims=True) + 1e-8)
    pe_norm = pe.astype(np.float32, copy=False)
    pe_norm = pe_norm / (np.linalg.norm(pe_norm, axis=1, keepdims=True) + 1e-8)
    Q = qe_norm.shape[0]
    cos_topk = np.zeros((Q, TOP_K), dtype=np.int64)
    for s in range(0, Q, args.score_batch):
        e = min(s + args.score_batch, Q)
        chunk_scores = qe_norm[s:e] @ pe_norm.T
        cos_topk[s:e] = topk_descending(chunk_scores, TOP_K)
    print(f"  scored + topk in {time.time() - t0:.1f}s")
    cos_mrr = mrr_at_k(cos_topk, qrels_idx_sets, k=10)
    cos_ndcg = ndcg_at_k(cos_topk, qrels_idx_dicts, k=10)
    cos_recall = recall_at_k(cos_topk, qrels_idx_sets, k=100)
    print(f"  MRR@10={cos_mrr:.4f} | nDCG@10={cos_ndcg:.4f} | Recall@100={cos_recall:.4f}")

    # ───────── DRT scoring ─────────
    print("\n[DRT]")
    t0 = time.time()
    P = pe.shape[0]
    with torch.no_grad():
        pe_t = torch.from_numpy(pe).float()
        chunk = 8192
        doc_subs_parts: list[torch.Tensor] = []
        for s in range(0, P, chunk):
            ds = model.encode_doc(pe_t[s : s + chunk].to(device)).cpu()
            doc_subs_parts.append(ds)
        doc_subs = torch.cat(doc_subs_parts, dim=0)

        qe_t = torch.from_numpy(qe_eval).float().to(device)
        q_subs, q_alphas = model.encode_query(qe_t)
        q_subs = q_subs.cpu()
        q_alphas = q_alphas.cpu()

    Q = q_subs.shape[0]
    drt_topk = np.zeros((Q, TOP_K), dtype=np.int64)
    for s in range(0, Q, args.score_batch):
        e = min(s + args.score_batch, Q)
        qs = q_subs[s:e]
        qa = q_alphas[s:e]
        cos_per_slot = torch.einsum("bkd,pkd->bpk", qs, doc_subs)
        scores = (qa.unsqueeze(1) * cos_per_slot).sum(dim=-1).numpy()
        drt_topk[s:e] = topk_descending(scores, TOP_K)
    print(f"  encoded + scored in {time.time() - t0:.1f}s")
    drt_mrr = mrr_at_k(drt_topk, qrels_idx_sets, k=10)
    drt_ndcg = ndcg_at_k(drt_topk, qrels_idx_dicts, k=10)
    drt_recall = recall_at_k(drt_topk, qrels_idx_sets, k=100)
    print(f"  MRR@10={drt_mrr:.4f} | nDCG@10={drt_ndcg:.4f} | Recall@100={drt_recall:.4f}")

    # ───────── Summary ─────────
    print("\n" + "=" * 56)
    print(f"{'Metric':<14} {'Cosine':>12} {'DRT':>12} {'Δ':>12}")
    print("-" * 56)
    for name, c, d in [
        ("MRR@10", cos_mrr, drt_mrr),
        ("nDCG@10", cos_ndcg, drt_ndcg),
        ("Recall@100", cos_recall, drt_recall),
    ]:
        delta = d - c
        sign = "+" if delta >= 0 else ""
        print(f"{name:<14} {c:>12.4f} {d:>12.4f} {sign}{delta:>11.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
