"""PyTorch Dataset over precomputed embeddings + qrels.

Yields (query_emb, positive_passage_emb) tensor pairs. The training loop adds
in-batch negatives implicitly, so this dataset does not produce explicit
triplets.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_query_ids(qrels_path: Path) -> list[str]:
    """Read a qrels TSV and return the unique query IDs in deterministic order."""
    qids: list[str] = []
    seen: set[str] = set()
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            qid = row[0]
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    qids.sort()
    return qids


def split_query_ids(
    qrels_path: Path,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Read qrels, return (train_qids, eval_qids) — disjoint, seeded shuffle."""
    qids = load_query_ids(qrels_path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qids))
    n_train = int(round(len(qids) * train_frac))
    train_qids = [qids[i] for i in perm[:n_train]]
    eval_qids = [qids[i] for i in perm[n_train:]]
    return train_qids, eval_qids


class FrozenEmbeddingDataset(Dataset):
    """Iterates positive (query, passage) qrels rows as embedding tensor pairs.

    Loads passage_embeddings.npy + passage_ids.npy + query_embeddings.npy +
    query_ids.npy from `embeddings_dir`, and qrels TSV from `qrels_path`.
    Optionally restrict to a subset of query IDs (e.g. the train split).
    """

    def __init__(
        self,
        embeddings_dir: Path,
        qrels_path: Path,
        query_ids: list[str] | None = None,
        query_prefix: str = "",
    ):
        """`query_prefix=""` loads query_embeddings.npy (dev). `"train_"` loads
        train_query_embeddings.npy. The same passage embeddings are used either way."""
        embeddings_dir = Path(embeddings_dir)
        self.passage_emb = np.load(embeddings_dir / "passage_embeddings.npy", mmap_mode="r")
        passage_ids = np.load(embeddings_dir / "passage_ids.npy")
        self.passage_id_to_idx: dict[str, int] = {
            str(pid): i for i, pid in enumerate(passage_ids)
        }

        self.query_emb = np.load(embeddings_dir / f"{query_prefix}query_embeddings.npy", mmap_mode="r")
        all_query_ids = np.load(embeddings_dir / f"{query_prefix}query_ids.npy")
        self.query_id_to_idx: dict[str, int] = {
            str(qid): i for i, qid in enumerate(all_query_ids)
        }

        allowed = set(query_ids) if query_ids is not None else None

        self.pairs: list[tuple[int, int]] = []
        skipped = 0
        with open(qrels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                qid, cid, score = row[0], row[1], int(row[2])
                if score <= 0:
                    continue
                if allowed is not None and qid not in allowed:
                    continue
                if qid not in self.query_id_to_idx or cid not in self.passage_id_to_idx:
                    skipped += 1
                    continue
                self.pairs.append(
                    (self.query_id_to_idx[qid], self.passage_id_to_idx[cid])
                )
        if skipped:
            print(
                f"FrozenEmbeddingDataset: skipped {skipped} qrels rows "
                f"(query or passage not in precomputed embeddings)"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        qi, pi = self.pairs[idx]
        q = torch.from_numpy(np.array(self.query_emb[qi], copy=True)).float()
        p = torch.from_numpy(np.array(self.passage_emb[pi], copy=True)).float()
        return q, p
