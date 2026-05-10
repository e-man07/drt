"""OnlineMSMarcoDataset: yields raw text triples for end-to-end training.

Unlike Scale 1's `FrozenEmbeddingDataset` which returned precomputed embeddings,
this dataset returns the raw query / passage texts. The trainer tokenizes and
encodes each batch through the (unfrozen) MiniLM encoder.

Loading the full corpus + queries into memory uses ~3-4 GB. Fits comfortably
on the A100 box (64 Gi host RAM).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from torch.utils.data import Dataset

from training.hard_negatives import load_hard_negative_triples


def _load_jsonl_id_to_text(path: Path, id_field: str = "_id") -> dict[str, str]:
    """Read a JSONL file of {_id, text} (passages also have title)."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            title = row.get("title") or ""
            out[str(row[id_field])] = (
                f"{title} {text}".strip() if title else text
            )
    return out


def _load_qrels_pairs(path: Path) -> list[tuple[str, str]]:
    """Read qrels TSV (header + qid<TAB>cid<TAB>score). Returns positive pairs only."""
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            qid, cid, score = row[0], row[1], int(row[2])
            if score > 0:
                pairs.append((qid, cid))
    return pairs


class OnlineMSMarcoDataset(Dataset):
    """Yields `(query_text, pos_text, [neg_text * num_hard_negatives])`.

    Each `__getitem__` returns three lists of strings. The trainer's collate
    function concatenates batches and tokenizes once per batch.

    Behavior:
      - If `hard_negatives_path` is given, training pairs come from the
        hard-negatives file (each pair gets its own list of BM25 negatives).
      - Otherwise, training pairs come from positive qrels and negatives are
        sampled at random from the corpus (with replacement).
    """

    def __init__(
        self,
        queries_path: Path,
        corpus_path: Path,
        qrels_path: Path,
        hard_negatives_path: Path | None = None,
        num_hard_negatives: int = 7,
        seed: int = 42,
    ):
        self.qid_to_text = _load_jsonl_id_to_text(queries_path)
        self.pid_to_text = _load_jsonl_id_to_text(corpus_path)
        self.num_hard_negatives = num_hard_negatives
        self.rng = random.Random(seed)

        if hard_negatives_path is not None and Path(hard_negatives_path).exists():
            triples = load_hard_negative_triples(
                hard_negatives_path,
                valid_pids=set(self.pid_to_text.keys()),
                valid_qids=set(self.qid_to_text.keys()),
                num_hard_negatives=num_hard_negatives,
            )
            self.triples: list[tuple[str, str, list[str] | None]] = [
                (q, p, n) for (q, p, n) in triples
            ]
            self.use_random_negs = False
        else:
            pairs = _load_qrels_pairs(qrels_path)
            pairs = [
                (qid, pid)
                for qid, pid in pairs
                if qid in self.qid_to_text and pid in self.pid_to_text
            ]
            self.triples = [(qid, pid, None) for qid, pid in pairs]
            self.use_random_negs = True

        self._all_pids = list(self.pid_to_text.keys())

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple[str, str, list[str]]:
        qid, pos_pid, negs = self.triples[idx]
        q_text = self.qid_to_text[qid]
        p_text = self.pid_to_text[pos_pid]
        if negs is not None:
            neg_pids = list(negs)
            if len(neg_pids) < self.num_hard_negatives:
                # Pad with random samples to keep batch shape uniform
                while len(neg_pids) < self.num_hard_negatives:
                    cand = self.rng.choice(self._all_pids)
                    if cand != pos_pid and cand not in neg_pids:
                        neg_pids.append(cand)
        else:
            neg_pids = []
            while len(neg_pids) < self.num_hard_negatives:
                cand = self.rng.choice(self._all_pids)
                if cand != pos_pid and cand not in neg_pids:
                    neg_pids.append(cand)
        neg_texts = [self.pid_to_text[pid] for pid in neg_pids]
        return q_text, p_text, neg_texts


def collate_text_triples(batch: list[tuple[str, str, list[str]]]):
    """Plain text collate: zip into parallel lists. Tokenization is left to caller."""
    queries = [b[0] for b in batch]
    positives = [b[1] for b in batch]
    negatives_per_row = [b[2] for b in batch]
    flat_negatives = [n for row in negatives_per_row for n in row]
    num_neg = len(negatives_per_row[0]) if negatives_per_row else 0
    return queries, positives, flat_negatives, num_neg


class TokenizingCollator:
    """Collate-fn that tokenizes on DataLoader worker processes.

    Without this, tokenization happens on the main training thread between
    GPU steps and the GPU sits idle ~50% of wall clock. With num_workers > 0,
    each worker holds a copy of the tokenizer (pickled across the fork) and
    runs tokenization in parallel, keeping the GPU fed.
    """

    def __init__(self, tokenizer, max_seq_length: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _tok(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

    def __call__(self, batch: list[tuple[str, str, list[str]]]):
        queries = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        negatives_per_row = [b[2] for b in batch]
        flat_negatives = [n for row in negatives_per_row for n in row]
        num_neg = len(negatives_per_row[0]) if negatives_per_row else 0
        return {
            "q_tok": self._tok(queries),
            "p_tok": self._tok(positives),
            "n_tok": self._tok(flat_negatives),
            "num_neg": num_neg,
        }
