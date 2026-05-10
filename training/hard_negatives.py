"""Loader for MS MARCO BM25 hard-negative triples.

Format of `qidpidtriples.train.small.tsv` (MS MARCO official):

    qid \\t pos_pid \\t neg_pid

Each row is one hard negative example for one (query, positive) pair. We group
by (qid, pos_pid) and assemble lists of `num_hard_negatives` negatives per pair.
Triples whose pids are absent from the corpus subset are dropped.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_hard_negative_triples(
    triples_path: Path,
    valid_pids: set[str] | None = None,
    valid_qids: set[str] | None = None,
    num_hard_negatives: int = 7,
    max_triples: int | None = None,
) -> list[tuple[str, str, list[str]]]:
    """Return list of `(qid, pos_pid, [neg_pid, …])`.

    `valid_pids` / `valid_qids`: if given, drop triples whose ids aren't present.
    `num_hard_negatives`: target number of negatives per (qid, pos) pair. Pairs
        with fewer surviving negatives after filtering are kept (oversample at
        train time) — but pairs with zero negatives are dropped.
    `max_triples`: stop reading after this many input rows (useful for smoke
        tests; default reads the full file).
    """
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows_seen = 0

    with open(triples_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            qid, pos_pid, neg_pid = row[0], row[1], row[2]
            rows_seen += 1
            if valid_qids is not None and qid not in valid_qids:
                continue
            if valid_pids is not None:
                if pos_pid not in valid_pids or neg_pid not in valid_pids:
                    continue
            key = (qid, pos_pid)
            if len(by_pair[key]) < num_hard_negatives:
                by_pair[key].append(neg_pid)
            if max_triples is not None and rows_seen >= max_triples:
                break

    out: list[tuple[str, str, list[str]]] = []
    for (qid, pos_pid), negs in by_pair.items():
        if not negs:
            continue
        out.append((qid, pos_pid, negs))
    return out
