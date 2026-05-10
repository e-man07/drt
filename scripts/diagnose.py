"""DRT diagnostics — analyze why DRT lost to the cosine baseline.

Four steps from the post-mortem:
  1. Modified scoring inference (uniform / per-slot / concat-cosine / top-2)
  2. Per-slot probes (query type, entity presence, length bucket)
  3. Per-query failure analysis (top wins vs top losses)
  4. Linear CKA between encoders' representations

Designed to run on the Akash box (where the corpus + checkpoints already
live). The first subcommand `encode` does the heavy GPU work (~75 min on
A100) and saves all encoded representations to /workspace/diagnostics/.
The remaining subcommands run from those saved arrays — fast enough to
iterate locally.

Usage:
    python -m scripts.diagnose encode      # encode corpus + queries through both
    python -m scripts.diagnose scoring     # 4 modified scoring variants
    python -m scripts.diagnose probes      # per-slot logistic regression probes
    python -m scripts.diagnose failures    # which queries DRT wins/loses on
    python -m scripts.diagnose cka         # representation similarity
    python -m scripts.diagnose all         # all of the above in order
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from models.drt_model import DRTModel
from models.encoder import MiniLMEncoder

DIAG_DIR = Path("/workspace/diagnostics")
DATA_DIR = Path("/workspace/data")
CKPT_DIR = Path("/workspace/checkpoints")
TOP_K = 100


# ───────────────────────────── shared helpers ─────────────────────────────


class _TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class _TextCollator:
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
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            title = row.get("title") or ""
            out.append((str(row["_id"]), f"{title} {text}".strip() if title else text))
    return out


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
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


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ───────────────────────────── 0. ENCODE ─────────────────────────────


def cmd_encode(args: argparse.Namespace) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"Device: {device}")

    corpus_path = args.data_dir / "corpus.jsonl"
    dev_q_path = args.data_dir / "dev_queries.jsonl"

    print(f"Loading corpus + dev queries from {args.data_dir}...")
    corpus = _load_jsonl(corpus_path)
    queries = _load_jsonl(dev_q_path)
    print(f"  corpus: {len(corpus):,}  queries: {len(queries):,}")

    pids = [pid for pid, _ in corpus]
    ptexts = [t for _, t in corpus]
    qids = [qid for qid, _ in queries]
    qtexts = [t for _, t in queries]

    np.save(DIAG_DIR / "corpus_pids.npy", np.array(pids, dtype="U16"))
    np.save(DIAG_DIR / "query_qids.npy", np.array(qids, dtype="U16"))
    print(f"  saved id arrays")

    # ── Baseline encoder ──
    bl_state = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    bl_cfg = bl_state.get("config", {})
    bl_enc = MiniLMEncoder(
        model_name=bl_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        max_seq_length=bl_cfg.get("max_seq_length", 256),
        normalize=True,
    ).to(device)
    bl_enc.load_state_dict(bl_state["model"])
    bl_enc.eval()
    print("Loaded baseline encoder")

    _encode_to_file(
        bl_enc, qtexts, DIAG_DIR / "baseline_query_emb.npy",
        shape=(len(qtexts), bl_enc.embed_dim), kind="baseline_query",
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    _encode_to_file(
        bl_enc, ptexts, DIAG_DIR / "baseline_corpus_emb.npy",
        shape=(len(ptexts), bl_enc.embed_dim), kind="baseline_corpus",
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    del bl_enc
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── DRT model ──
    drt_state = torch.load(args.drt_ckpt, map_location=device, weights_only=False)
    drt_cfg = drt_state.get("config", {})
    drt = DRTModel(
        model_name=drt_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        max_seq_length=drt_cfg.get("max_seq_length", 256),
        decomp_hidden=drt_cfg.get("decomp_hidden", 512),
        attn_hidden=drt_cfg.get("attn_hidden", 64),
        k=drt_cfg.get("k", 6),
        sub_dim=drt_cfg.get("sub_dim", 64),
    ).to(device)
    drt.load_state_dict(drt_state["model"])
    drt.eval()
    print("Loaded DRT model")

    # Save DRT meta (k, sub_dim) for later
    json.dump(
        {"k": drt.k, "sub_dim": drt.sub_dim, "embed_dim": drt.embed_dim},
        open(DIAG_DIR / "drt_meta.json", "w"),
    )

    # Queries through DRT (need raw 384 emb, subs, alphas)
    _encode_drt_queries(
        drt, qtexts, DIAG_DIR, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Corpus through DRT (subs only; concat-cosine is reshape, no extra)
    _encode_drt_corpus(
        drt, ptexts, DIAG_DIR / "drt_corpus_subs.npy",
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print("\nencode: done")


def _encode_to_file(
    encoder: MiniLMEncoder,
    texts: list[str],
    out_path: Path,
    shape: tuple[int, int],
    kind: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Generic baseline-encoder pipeline that writes a memmapped fp16 array."""
    arr = np.lib.format.open_memmap(out_path, dtype=np.float16, shape=shape, mode="w+")
    dataset = _TextDataset(texts)
    collator = _TextCollator(encoder.tokenizer, encoder.max_seq_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    n = len(texts)
    cursor = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, tok in enumerate(loader):
            bs = tok["input_ids"].shape[0]
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
            with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float16):
                emb = encoder(tok["input_ids"], tok["attention_mask"])
            arr[cursor : cursor + bs] = emb.detach().to(dtype=torch.float16, device="cpu").numpy()
            cursor += bs
            if (bi + 1) % 10 == 0 or cursor >= n:
                elapsed = time.time() - t0
                rate = cursor / elapsed if elapsed else 0.0
                eta = (n - cursor) / rate if rate else float("inf")
                print(
                    f"  [{kind}] {cursor:>9,}/{n:,} ({100*cursor/n:5.1f}%) "
                    f"| {elapsed:6.0f}s | {rate:>5.0f} rows/s | ETA {eta:5.0f}s",
                    flush=True,
                )
    arr.flush()
    del arr


def _encode_drt_queries(
    drt: DRTModel, texts: list[str], out_dir: Path,
    device: str, batch_size: int, num_workers: int,
) -> None:
    """Encode queries through DRT — save raw encoder output, sub_vectors, alphas."""
    n = len(texts)
    raw = np.lib.format.open_memmap(
        out_dir / "drt_query_raw.npy", dtype=np.float16, shape=(n, drt.embed_dim), mode="w+"
    )
    subs = np.lib.format.open_memmap(
        out_dir / "drt_query_subs.npy", dtype=np.float16, shape=(n, drt.k, drt.sub_dim), mode="w+"
    )
    alphas = np.lib.format.open_memmap(
        out_dir / "drt_query_alphas.npy", dtype=np.float16, shape=(n, drt.k), mode="w+"
    )

    dataset = _TextDataset(texts)
    collator = _TextCollator(drt.encoder.tokenizer, drt.encoder.max_seq_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    cursor = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, tok in enumerate(loader):
            bs = tok["input_ids"].shape[0]
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
            with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float16):
                emb = drt.encoder(tok["input_ids"], tok["attention_mask"])
                qs = drt.decomposition(emb)
                qa = drt.attention(emb)
            raw[cursor : cursor + bs] = emb.detach().to(dtype=torch.float16, device="cpu").numpy()
            subs[cursor : cursor + bs] = qs.detach().to(dtype=torch.float16, device="cpu").numpy()
            alphas[cursor : cursor + bs] = qa.detach().to(dtype=torch.float16, device="cpu").numpy()
            cursor += bs
            if (bi + 1) % 5 == 0 or cursor >= n:
                elapsed = time.time() - t0
                print(f"  [drt_query] {cursor:>5,}/{n:,} | {elapsed:6.0f}s", flush=True)
    raw.flush(); subs.flush(); alphas.flush()
    del raw, subs, alphas


def _encode_drt_corpus(
    drt: DRTModel, texts: list[str], out_path: Path,
    device: str, batch_size: int, num_workers: int,
) -> None:
    n = len(texts)
    arr = np.lib.format.open_memmap(
        out_path, dtype=np.float16, shape=(n, drt.k, drt.sub_dim), mode="w+"
    )
    dataset = _TextDataset(texts)
    collator = _TextCollator(drt.encoder.tokenizer, drt.encoder.max_seq_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    cursor = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, tok in enumerate(loader):
            bs = tok["input_ids"].shape[0]
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
            with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float16):
                emb = drt.encoder(tok["input_ids"], tok["attention_mask"])
                subs = drt.decomposition(emb)
            arr[cursor : cursor + bs] = subs.detach().to(dtype=torch.float16, device="cpu").numpy()
            cursor += bs
            if (bi + 1) % 10 == 0 or cursor >= n:
                elapsed = time.time() - t0
                rate = cursor / elapsed if elapsed else 0.0
                eta = (n - cursor) / rate if rate else float("inf")
                print(
                    f"  [drt_corpus] {cursor:>9,}/{n:,} ({100*cursor/n:5.1f}%) "
                    f"| {elapsed:6.0f}s | {rate:>5.0f} rows/s | ETA {eta:5.0f}s",
                    flush=True,
                )
    arr.flush()
    del arr


# ───────────────────────────── 1. SCORING VARIANTS ─────────────────────────────


def _build_qrels_idx(qids: list[str], pids: list[str], qrels_path: Path):
    qrels = _load_qrels(qrels_path)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    qrels_idx_dicts = []
    qrels_idx_sets = []
    for qid in qids:
        rel = qrels.get(qid, {})
        rel_idx = {pid_to_idx[p]: s for p, s in rel.items() if p in pid_to_idx}
        qrels_idx_dicts.append(rel_idx)
        qrels_idx_sets.append(set(rel_idx.keys()))
    return qrels_idx_dicts, qrels_idx_sets, pid_to_idx


def _metrics_from_topk(
    topk: np.ndarray,
    qrels_idx_dicts: list[dict[int, int]],
    qrels_idx_sets: list[set[int]],
) -> dict:
    return {
        "MRR@10": mrr_at_k(topk, qrels_idx_sets, k=10),
        "nDCG@10": ndcg_at_k(topk, qrels_idx_dicts, k=10),
        "Recall@100": recall_at_k(topk, qrels_idx_sets, k=100),
    }


def _score_baseline(
    q_emb: np.ndarray, p_emb: np.ndarray, device: str, score_batch: int = 64
) -> np.ndarray:
    """Cosine baseline: q @ p.T, return top-K indices."""
    q_t = torch.from_numpy(q_emb).to(device).float()
    p_t = torch.from_numpy(p_emb).to(device).float()
    Q = q_t.shape[0]
    topk = np.zeros((Q, TOP_K), dtype=np.int64)
    for s in range(0, Q, score_batch):
        e = min(s + score_batch, Q)
        with torch.no_grad():
            scores = (q_t[s:e] @ p_t.t()).float().cpu().numpy()
        topk[s:e] = _topk_descending(scores, TOP_K)
    return topk


def _score_drt_with_alphas(
    q_subs: np.ndarray,        # (Q, k, d) float16
    q_alphas: np.ndarray,      # (Q, k) float16  — caller supplies (variants override this)
    p_subs: np.ndarray,        # (P, k, d) float16
    device: str,
    score_batch: int = 32,
) -> np.ndarray:
    qs_t = torch.from_numpy(q_subs).to(device).float()
    qa_t = torch.from_numpy(q_alphas).to(device).float()
    ps_t = torch.from_numpy(p_subs).to(device).float()
    Q = qs_t.shape[0]
    topk = np.zeros((Q, TOP_K), dtype=np.int64)
    for s in range(0, Q, score_batch):
        e = min(s + score_batch, Q)
        with torch.no_grad():
            cos_per_slot = torch.einsum("bkd,pkd->bpk", qs_t[s:e], ps_t)  # (b, P, k)
            scores = (qa_t[s:e].unsqueeze(1) * cos_per_slot).sum(dim=-1).float().cpu().numpy()
        topk[s:e] = _topk_descending(scores, TOP_K)
    return topk


def cmd_scoring(args: argparse.Namespace) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"Device: {device}")

    qids = list(map(str, np.load(DIAG_DIR / "query_qids.npy")))
    pids = list(map(str, np.load(DIAG_DIR / "corpus_pids.npy")))
    qrels_idx_dicts, qrels_idx_sets, _ = _build_qrels_idx(
        qids, pids, args.data_dir / "qrels" / "dev.tsv"
    )

    # Mmap loads
    bl_q = np.load(DIAG_DIR / "baseline_query_emb.npy", mmap_mode="r")
    bl_p = np.load(DIAG_DIR / "baseline_corpus_emb.npy", mmap_mode="r")
    drt_qs = np.load(DIAG_DIR / "drt_query_subs.npy", mmap_mode="r")
    drt_qa = np.load(DIAG_DIR / "drt_query_alphas.npy", mmap_mode="r")
    drt_ps = np.load(DIAG_DIR / "drt_corpus_subs.npy", mmap_mode="r")
    print(f"  bl_q: {bl_q.shape} | bl_p: {bl_p.shape}")
    print(f"  drt_qs: {drt_qs.shape} | drt_qa: {drt_qa.shape} | drt_ps: {drt_ps.shape}")

    K = drt_qs.shape[1]
    D_sub = drt_qs.shape[2]
    Q = drt_qs.shape[0]
    results: dict = {}
    topk_store: dict[str, np.ndarray] = {}

    def _run(name: str, topk: np.ndarray):
        m = _metrics_from_topk(topk, qrels_idx_dicts, qrels_idx_sets)
        results[name] = m
        topk_store[name] = topk
        print(f"  {name:30s} → MRR@10={m['MRR@10']:.4f} nDCG@10={m['nDCG@10']:.4f} R@100={m['Recall@100']:.4f}")

    # A: cosine baseline (sanity)
    print("\n[A] cosine baseline")
    bl_q_arr = np.array(bl_q, dtype=np.float32)
    bl_p_arr = np.array(bl_p, dtype=np.float32)  # ~13 GB but fits 64 GB host RAM
    _run("cosine_baseline", _score_baseline(bl_q_arr, bl_p_arr, device, args.score_batch))
    del bl_p_arr

    # B: DRT proper (sanity)
    print("\n[B] DRT (learned alphas)")
    drt_qs_arr = np.array(drt_qs, dtype=np.float32)
    drt_qa_arr = np.array(drt_qa, dtype=np.float32)
    drt_ps_arr = np.array(drt_ps, dtype=np.float32)
    _run("drt_learned_alphas", _score_drt_with_alphas(drt_qs_arr, drt_qa_arr, drt_ps_arr, device, args.score_batch))

    # C: uniform alphas
    print("\n[C] DRT with uniform alphas (1/k)")
    uniform_alphas = np.full_like(drt_qa_arr, 1.0 / K)
    _run("drt_uniform_alphas", _score_drt_with_alphas(drt_qs_arr, uniform_alphas, drt_ps_arr, device, args.score_batch))

    # D: per-slot
    print("\n[D] DRT single slot (i in 0..k-1)")
    for i in range(K):
        a = np.zeros_like(drt_qa_arr)
        a[:, i] = 1.0
        _run(f"drt_slot_{i}", _score_drt_with_alphas(drt_qs_arr, a, drt_ps_arr, device, args.score_batch))

    # E: concatenated cosine — flatten (k, d) → (k*d,)
    print("\n[E] concatenated cosine (reshape DRT subs to 384-d)")
    drt_qs_flat = drt_qs_arr.reshape(Q, K * D_sub)
    drt_ps_flat = drt_ps_arr.reshape(drt_ps_arr.shape[0], K * D_sub)
    # Note: subs are L2-normalized per-slot, so the flattened vector has norm sqrt(k).
    # Cosine still works but normalize for cleanliness.
    drt_qs_flat = drt_qs_flat / (np.linalg.norm(drt_qs_flat, axis=1, keepdims=True) + 1e-8)
    drt_ps_flat = drt_ps_flat / (np.linalg.norm(drt_ps_flat, axis=1, keepdims=True) + 1e-8)
    _run("drt_concat_cosine", _score_baseline(drt_qs_flat, drt_ps_flat, device, args.score_batch))
    del drt_qs_flat, drt_ps_flat

    # F: top-2 slots per query
    print("\n[F] DRT top-2 slots per query")
    top2_alphas = np.zeros_like(drt_qa_arr)
    top2_idx = np.argsort(-drt_qa_arr, axis=1)[:, :2]  # top 2 slots per query
    for i in range(Q):
        for j in top2_idx[i]:
            top2_alphas[i, j] = drt_qa_arr[i, j]
        # renormalize so weights sum to 1
        s = top2_alphas[i].sum()
        if s > 0:
            top2_alphas[i] /= s
    _run("drt_top2_alphas", _score_drt_with_alphas(drt_qs_arr, top2_alphas, drt_ps_arr, device, args.score_batch))

    # Save results + topk for failure analysis
    json.dump(results, open(DIAG_DIR / "scoring_metrics.json", "w"), indent=2)
    np.savez_compressed(
        DIAG_DIR / "scoring_topk.npz",
        **{name: arr.astype(np.int32) for name, arr in topk_store.items()},
    )
    print(f"\nSaved metrics → {DIAG_DIR / 'scoring_metrics.json'}")
    print(f"Saved top-k arrays → {DIAG_DIR / 'scoring_topk.npz'}")


# ───────────────────────────── 2. PROBES ─────────────────────────────

# Heuristics for query labels (no LLM needed). These are noisy but
# good enough to compare per-slot information content.
_FACTOID_RE = re.compile(
    r"^\s*(what|who|whom|when|where|why|how|which|did|do|does|is|are|was|were|can|could|should)\b",
    flags=re.IGNORECASE,
)
# MS MARCO queries are lowercased so capitalized-word entity detection
# doesn't work. Use a proxy: queries with named-entity-like tokens —
# 4-digit years, dollar amounts, or specific entity-evoking patterns.
_ENTITY_PROXY_RE = re.compile(
    r"\b(\d{4}|\d+(\.\d+)?\s*(percent|million|billion)|"
    r"president|company|university|state|country|city|"
    r"american|english|chinese|french|german|"
    r"[a-z]+ ?[a-z]* (corporation|inc|llc|university|college|hospital))\b",
    flags=re.IGNORECASE,
)


def _label_query(text: str) -> dict:
    words = text.split()
    n_words = len(words)
    return {
        "factoid": int(bool(_FACTOID_RE.search(text))),
        "has_entity": int(bool(_ENTITY_PROXY_RE.search(text))),
        "length_bucket": 0 if n_words <= 5 else (1 if n_words <= 10 else 2),
    }


def cmd_probes(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    qids = list(map(str, np.load(DIAG_DIR / "query_qids.npy")))
    drt_qs = np.load(DIAG_DIR / "drt_query_subs.npy")  # (Q, k, d)
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    # Reload texts to label
    queries = _load_jsonl(args.data_dir / "dev_queries.jsonl")
    labels = {"factoid": [], "has_entity": [], "length_bucket": []}
    for qid, txt in queries:
        if qid not in qid_to_idx:
            continue
        for k, v in _label_query(txt).items():
            labels[k].append(v)
    for k in labels:
        labels[k] = np.array(labels[k])
    print(f"Label counts:")
    print(f"  factoid:        {labels['factoid'].sum():,} / {len(labels['factoid']):,}")
    print(f"  has_entity:     {labels['has_entity'].sum():,} / {len(labels['has_entity']):,}")
    bucket_counts = np.bincount(labels["length_bucket"], minlength=3)
    print(f"  length_bucket:  short={bucket_counts[0]} med={bucket_counts[1]} long={bucket_counts[2]}")

    K = drt_qs.shape[1]
    results: dict = {"per_slot": {}, "concat_baseline": {}}

    rng = np.random.default_rng(0)

    for slot in range(K):
        results["per_slot"][f"slot_{slot}"] = {}
        X = drt_qs[:, slot, :].astype(np.float32)
        for task, y in labels.items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
            acc = accuracy_score(y_te, clf.predict(X_te))
            # Majority-class baseline
            maj = max(np.bincount(y_te) / len(y_te))
            results["per_slot"][f"slot_{slot}"][task] = {"acc": float(acc), "majority": float(maj)}
            print(f"  slot {slot} | {task:14s}  acc={acc:.4f}  (majority={maj:.4f})")

    # As a sanity comparison, also run probes on the concatenated 384-d query vector
    X_concat = drt_qs.reshape(drt_qs.shape[0], -1).astype(np.float32)
    for task, y in labels.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X_concat, y, test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=2000, multi_class="auto").fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        maj = max(np.bincount(y_te) / len(y_te))
        results["concat_baseline"][task] = {"acc": float(acc), "majority": float(maj)}
        print(f"  concat 6×64 | {task:14s}  acc={acc:.4f}  (majority={maj:.4f})")

    json.dump(results, open(DIAG_DIR / "probes.json", "w"), indent=2)
    print(f"\nSaved → {DIAG_DIR / 'probes.json'}")


# ───────────────────────────── 3. FAILURE ANALYSIS ─────────────────────────────


def _per_query_mrr_at_10(topk: np.ndarray, rel_sets: list[set[int]]) -> np.ndarray:
    out = np.zeros(topk.shape[0], dtype=np.float64)
    for i, rel in enumerate(rel_sets):
        if not rel:
            continue
        for rank, idx in enumerate(topk[i, :10], start=1):
            if int(idx) in rel:
                out[i] = 1.0 / rank
                break
    return out


def cmd_failures(args: argparse.Namespace) -> None:
    qids = list(map(str, np.load(DIAG_DIR / "query_qids.npy")))
    pids = list(map(str, np.load(DIAG_DIR / "corpus_pids.npy")))
    queries = _load_jsonl(args.data_dir / "dev_queries.jsonl")
    qid_to_text = {q: t for q, t in queries}
    qrels_idx_dicts, qrels_idx_sets, _ = _build_qrels_idx(
        qids, pids, args.data_dir / "qrels" / "dev.tsv"
    )

    npz = np.load(DIAG_DIR / "scoring_topk.npz")
    bl_topk = npz["cosine_baseline"]
    drt_topk = npz["drt_learned_alphas"]

    bl_mrr = _per_query_mrr_at_10(bl_topk, qrels_idx_sets)
    drt_mrr = _per_query_mrr_at_10(drt_topk, qrels_idx_sets)
    delta = drt_mrr - bl_mrr

    # filter: only queries with rel docs in corpus
    has_rel = np.array([bool(r) for r in qrels_idx_sets])
    eligible = np.where(has_rel)[0]
    delta_e = delta[eligible]

    order = np.argsort(delta_e)
    losses = eligible[order[:100]]   # smallest delta = DRT lost most
    wins = eligible[order[-100:][::-1]]  # largest delta = DRT won most

    def _summarize(idx_list):
        out = {"avg_query_len": 0.0, "labels": defaultdict(int), "examples": []}
        lengths = []
        for i in idx_list:
            qid = qids[i]
            text = qid_to_text.get(qid, "")
            lengths.append(len(text.split()))
            lab = _label_query(text)
            for k, v in lab.items():
                out["labels"][f"{k}={v}"] += 1
            if len(out["examples"]) < 10:
                out["examples"].append({
                    "qid": qid,
                    "query": text,
                    "n_relevant": len(qrels_idx_sets[i]),
                    "bl_mrr": float(bl_mrr[i]),
                    "drt_mrr": float(drt_mrr[i]),
                    "delta": float(delta[i]),
                })
        out["avg_query_len"] = float(np.mean(lengths))
        out["labels"] = dict(out["labels"])
        return out

    summary = {
        "n_eligible": int(len(eligible)),
        "wins": _summarize(wins),
        "losses": _summarize(losses),
        "global_metrics": {
            "bl_mrr@10_mean": float(bl_mrr[eligible].mean()),
            "drt_mrr@10_mean": float(drt_mrr[eligible].mean()),
            "delta_mean": float(delta_e.mean()),
            "delta_median": float(np.median(delta_e)),
            "frac_drt_better": float((delta_e > 0).mean()),
            "frac_drt_worse":  float((delta_e < 0).mean()),
            "frac_tie":        float((delta_e == 0).mean()),
        },
    }
    json.dump(summary, open(DIAG_DIR / "failures.json", "w"), indent=2)

    print("\n=== Per-query DRT vs cosine baseline ===")
    for k, v in summary["global_metrics"].items():
        print(f"  {k:24s} {v:.4f}")
    print(f"\nWINS group avg query length: {summary['wins']['avg_query_len']:.1f} words")
    print(f"LOSSES group avg query length: {summary['losses']['avg_query_len']:.1f} words")
    print(f"\nWIN labels: {summary['wins']['labels']}")
    print(f"LOSS labels: {summary['losses']['labels']}")
    print(f"\n=== 10 example WINS (DRT > BL) ===")
    for e in summary["wins"]["examples"]:
        print(f"  Δ={e['delta']:+.3f}  bl={e['bl_mrr']:.2f}  drt={e['drt_mrr']:.2f}  | {e['query']}")
    print(f"\n=== 10 example LOSSES (DRT < BL) ===")
    for e in summary["losses"]["examples"]:
        print(f"  Δ={e['delta']:+.3f}  bl={e['bl_mrr']:.2f}  drt={e['drt_mrr']:.2f}  | {e['query']}")
    print(f"\nSaved → {DIAG_DIR / 'failures.json'}")


# ───────────────────────────── 4. CKA ─────────────────────────────


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019). X, Y are (n, d_x), (n, d_y).
    Returns scalar in [0, 1]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # |Y^T X|_F^2 / (|X^T X|_F * |Y^T Y|_F)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / (den + 1e-12))


def cmd_cka(args: argparse.Namespace) -> None:
    bl_q = np.load(DIAG_DIR / "baseline_query_emb.npy").astype(np.float32)
    drt_q_raw = np.load(DIAG_DIR / "drt_query_raw.npy").astype(np.float32)
    drt_q_subs = np.load(DIAG_DIR / "drt_query_subs.npy").astype(np.float32)
    drt_q_concat = drt_q_subs.reshape(drt_q_subs.shape[0], -1)

    results = {
        "baseline_vs_drt_raw_encoder": _linear_cka(bl_q, drt_q_raw),
        "baseline_vs_drt_concat_subs": _linear_cka(bl_q, drt_q_concat),
        "drt_raw_vs_drt_concat":        _linear_cka(drt_q_raw, drt_q_concat),
        "per_slot_vs_baseline": {},
        "per_slot_vs_drt_raw":  {},
    }
    K = drt_q_subs.shape[1]
    for i in range(K):
        results["per_slot_vs_baseline"][f"slot_{i}"] = _linear_cka(bl_q, drt_q_subs[:, i, :])
        results["per_slot_vs_drt_raw"][f"slot_{i}"] = _linear_cka(drt_q_raw, drt_q_subs[:, i, :])

    json.dump(results, open(DIAG_DIR / "cka.json", "w"), indent=2)

    print("\n=== Linear CKA on dev queries (n=6,980) ===")
    print(f"  baseline encoder  vs  DRT encoder (raw)        : {results['baseline_vs_drt_raw_encoder']:.4f}")
    print(f"  baseline encoder  vs  DRT concat sub-vectors   : {results['baseline_vs_drt_concat_subs']:.4f}")
    print(f"  DRT raw encoder   vs  DRT concat sub-vectors   : {results['drt_raw_vs_drt_concat']:.4f}")
    print(f"\n  per-slot vs baseline encoder:")
    for k, v in results["per_slot_vs_baseline"].items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  per-slot vs DRT raw encoder:")
    for k, v in results["per_slot_vs_drt_raw"].items():
        print(f"    {k}: {v:.4f}")
    print(f"\nSaved → {DIAG_DIR / 'cka.json'}")


# ───────────────────────────── ALL ─────────────────────────────


def cmd_all(args: argparse.Namespace) -> None:
    cmd_encode(args)
    cmd_scoring(args)
    cmd_probes(args)
    cmd_failures(args)
    cmd_cka(args)


# ───────────────────────────── CLI ─────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=DATA_DIR)
    common.add_argument("--baseline-ckpt", type=Path, default=CKPT_DIR / "cosine_baseline_epoch5.pt")
    common.add_argument("--drt-ckpt", type=Path, default=CKPT_DIR / "drt_scale2_epoch5.pt")
    common.add_argument("--batch-size", type=int, default=1024)
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--score-batch", type=int, default=64)

    for name, fn in [
        ("encode", cmd_encode),
        ("scoring", cmd_scoring),
        ("probes", cmd_probes),
        ("failures", cmd_failures),
        ("cka", cmd_cka),
        ("all", cmd_all),
    ]:
        p = sub.add_parser(name, parents=[common])
        p.set_defaults(func=fn)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
