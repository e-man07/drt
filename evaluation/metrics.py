"""Retrieval metrics computed from top-k ranked indices.

All functions accept `topk_indices` already sorted descending by score so the
full Q×P score matrix never has to be materialized at metric time.

  - mrr_at_k(topk, qrels_sets, k)              — first-hit reciprocal rank.
  - ndcg_at_k(topk, qrels_score_dicts, k)      — graded relevance, ideal-normalized.
  - recall_at_k(topk, qrels_sets, k)           — fraction of relevants retrieved.
"""
from __future__ import annotations

import math

import numpy as np


def mrr_at_k(
    topk_indices: np.ndarray,
    qrels: list[set[int]],
    k: int = 10,
) -> float:
    rr: list[float] = []
    for i, rel in enumerate(qrels):
        if not rel:
            continue
        score = 0.0
        for rank, idx in enumerate(topk_indices[i, :k], start=1):
            if int(idx) in rel:
                score = 1.0 / rank
                break
        rr.append(score)
    return float(np.mean(rr)) if rr else 0.0


def ndcg_at_k(
    topk_indices: np.ndarray,
    qrels: list[dict[int, int]],
    k: int = 10,
) -> float:
    ndcgs: list[float] = []
    for i, rel in enumerate(qrels):
        if not rel:
            continue
        dcg = sum(
            (2 ** rel.get(int(idx), 0) - 1) / math.log2(rank + 1)
            for rank, idx in enumerate(topk_indices[i, :k], start=1)
        )
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** r - 1) / math.log2(rank + 1)
            for rank, r in enumerate(ideal, start=1)
        )
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def recall_at_k(
    topk_indices: np.ndarray,
    qrels: list[set[int]],
    k: int = 100,
) -> float:
    recalls: list[float] = []
    for i, rel in enumerate(qrels):
        if not rel:
            continue
        retrieved = {int(idx) for idx in topk_indices[i, :k]}
        recalls.append(len(retrieved & rel) / len(rel))
    return float(np.mean(recalls)) if recalls else 0.0
