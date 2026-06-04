<h1 align="center">DRT — Decomposed Relevance Tensors</h1>

<p align="center">
  <em>A retrieval-scoring architecture, a clean negative result, and a four-step diagnostic methodology for understanding why dense-retrieval architectures fail.</em>
</p>

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#method">Method</a> ·
  <a href="#diagnostic-methodology">Diagnostics</a> ·
  <a href="#reproducing-the-experiments">Reproduce</a> ·
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/result-negative-red">
  <img alt="Reproducible" src="https://img.shields.io/badge/artifacts-checkpointed-success">
</p>

---

## TL;DR

We trained **Decomposed Relevance Tensors (DRT)** — an architecture that replaces flat cosine similarity with a learned decomposition into `k=6` L2-normalized sub-vectors plus a query-adaptive softmax over slot-level similarities. The hypothesis: decomposition + query-adaptive weighting extracts more retrieval signal than cosine on the same encoder.

On the full MS MARCO passage dev set (8.84 M corpus, 6,980 queries), a vanilla cosine bi-encoder trained on identical data, with identical epochs and hyperparameters from the same MiniLM checkpoint, **beats DRT by −2% MRR@10** (0.3278 vs 0.3074).

A four-step post-mortem reveals *exactly* why:

1. **The decomposition machinery is a no-op** — concat-cosine on the DRT sub-vectors, DRT with learned attention weights, and DRT with uniform `1/k` weights all score **identically** (0.3076 / 0.3075 / 0.3076 MRR@10).
2. **The attention head learned nothing** — learned α and uniform α produce identical scores.
3. **Decorrelation made slots statistically uncorrelated but semantically interchangeable** — every slot scores within 1% on per-task probes.
4. **The encoder drifted, didn't break** — Linear CKA 0.945 between baseline and DRT encoders. The 2% deficit is exactly the cost of that drift.

The **diagnostic procedure itself** is reusable for any dense-retrieval architecture and may be the more durable contribution of this work.

---

## Results

### Headline — full MS MARCO dev (6,980 queries × 8.84 M corpus, brute-force scoring)

| Metric         | Cosine bi-encoder | DRT (end-to-end) | Δ          |
| -------------- | ----------------: | ---------------: | ---------: |
| **MRR@10**     |        **0.3278** |           0.3074 | **−0.0204** |
| nDCG@10        |            0.3884 |           0.3648 |     −0.0235 |
| Recall@100     |            0.8600 |           0.8315 |     −0.0285 |

### Diagnostic scoring variants — all on the same trained DRT checkpoint

| Variant                       | MRR@10 | Interpretation |
| ----------------------------- | -----: | -------------- |
| `cosine_baseline` (reference) | 0.3278 | what we are trying to beat |
| `drt_learned_alphas` (full DRT) | 0.3075 | with the trained attention head |
| `drt_uniform_alphas` (α = 1/k) | 0.3076 | **identical** — attention head adds nothing |
| `drt_concat_cosine` | 0.3076 | **identical** — decomposition is a no-op |
| `drt_top2_alphas` | 0.2869 | sparsifying hurts |
| `drt_slot_{0..5}` (single slot) | 0.256-0.261 | all slots within 1%; not specialized |

The three variants at **0.3075-0.3076** are within `10⁻⁴` of each other. That equivalence is what closes the diagnostic loop — neither pillar of the DRT design (sub-vector decomposition, query-adaptive weighting) contributes detectable signal.

> Full numbers in [`PAPER_NOTES.md`](./PAPER_NOTES.md) and `results/diagnostics/scoring_metrics.json`.

### Architecture diagram

A single-page Excalidraw covering architecture, training, results, and the four diagnostic findings:

```
DRT_architecture.excalidraw  →  open at https://excalidraw.com (drag & drop)
```

---

## What's in this repository

```
drt/
├── README.md                   ← you are here
├── PAPER_NOTES.md              ← paper-ready writeup (≈600 lines, all numbers cited)
├── DRT_architecture.excalidraw ← single-page diagram of the full arc
├── DRT_Research_Blueprint.html ← the original research design (motivating doc)
├── DRT_Scale2_Prompt.md        ← the Scale-2 spec
│
├── models/        ← MiniLM encoder, DecompositionHead, QueryAttentionHead, DRTModel
├── losses/        ← InfoNCE, Barlow-Twins-style decorrelation, slot dropout
├── data/          ← MS MARCO download + (precompute|online) dataset loaders
├── training/      ← Scale-1 frozen trainer, Scale-2 e2e trainer, cosine baseline trainer
├── evaluation/    ← MRR / nDCG / Recall metrics + the end-to-end evaluator
├── scripts/       ← train_scale1.py, train_baseline.py, train_scale2.py, evaluate_e2e.py,
│                    diagnose.py, run_pipeline.sh, make_diagram.py
├── configs/       ← Scale-1 and Scale-2 YAML configs
├── deploy/        ← Akash SDL + entrypoint for A100 training
│
├── results/
│   ├── diagnostics/    ← JSON outputs of all 4 diagnostic steps + scoring_topk.npz
│   └── logs/           ← every training, eval, and diagnostic run log
│
└── requirements.txt, LICENSE, .gitignore
```

---

## Quickstart

### Just look at the numbers

All raw artifacts are checked in. No code execution required.

```bash
git clone https://github.com/e-man07/drt.git && cd drt
cat PAPER_NOTES.md                                  # full writeup
cat results/logs/comparison.txt                     # headline result
python3 -c "import json; print(json.dumps(json.load(open('results/diagnostics/scoring_metrics.json')), indent=2))"
```

### Reproduce inference from the trained checkpoint

Checkpoints aren't committed (88 MB each, gitignored). Download them from the latest release, or re-train via the pipeline below. Inference once you have them:

```bash
python3 -m scripts.evaluate_e2e \
    --checkpoint-baseline checkpoints/cosine_baseline_epoch5.pt \
    --checkpoint-drt      checkpoints/drt_scale2_epoch5.pt \
    --data-dir            data/raw_full \
    --output              results/logs/comparison.txt
```

### Run the four-step diagnostic

```bash
python3 -m scripts.diagnose encode    # ~75 min on A100, ~CPU not feasible
python3 -m scripts.diagnose scoring   # all 11 scoring variants
python3 -m scripts.diagnose probes    # per-slot probes
python3 -m scripts.diagnose failures  # top wins / top losses
python3 -m scripts.diagnose cka       # representation similarity
# or
python3 -m scripts.diagnose all
```

---

## Method

### Architecture

```
input text → MiniLM-L6-v2 encoder (22.7M params) → 384-d embedding
                              ↓                              ↓
                  Decomposition Head (~394K)       Query Attention Head (~25K)
                Linear(384,512)→LN→GELU            Linear(384, 64)→GELU
                Linear(512,384)→LN→GELU            Linear(64, k=6)
                reshape (k=6, d=64)                softmax over slots
                L2-normalize each slot                  ↓
                              ↓                       α ∈ ℝ⁶
                       sub-vectors (k, d)
                              ↓
                                 score(q, d) = Σᵢ αᵢ(q) · (qᵢ · dᵢ)
```

Source: `models/{encoder,decomposition,attention,drt_model,scorer}.py`.

Total trainable parameters:

| Scale | Encoder | Heads | Total |
| ----- | ------: | ----: | ----: |
| 1 (frozen) | 0 | 419,398 | **419,398** |
| 2 (e2e)    | 22,713,216 | 419,398 | **23,132,614** |

### Losses

Three terms combined per training step.

**Retrieval — InfoNCE over DRT scores** (`losses/contrastive.py`).
For each query in batch `B`, the positive document scores against `B−1` in-batch negatives plus `N` BM25 hard negatives. Cross-entropy with target = the diagonal of the score matrix. Temperature τ = 0.05.

**Decorrelation — Barlow-Twins-inspired** (`losses/decorrelation.py`).
For every slot pair `(i, j)` with `i < j`, compute the cross-correlation matrix `C[i,j] = sub[:,i].T @ sub[:,j] / B` (shape `d × d`) and penalize the squared Frobenius norm `‖C[i,j]‖_F²`. λ_dec = 0.1.

**Slot dropout** (`losses/combined.py`).
At training time, each sub-vector is masked-to-zero with probability p = 0.15. Surviving slots are rescaled by `1 / E[mask]` to preserve expected magnitude. Applied **after** the decorrelation loss is computed on the pre-dropout slots, so structural regularization doesn't corrupt the decorrelation signal.

$$
\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\text{InfoNCE}} \;+\; 0.1 \cdot \mathcal{L}_{\text{Barlow}}
$$

### Training stages

| Stage | Where | Encoder | Corpus | Train queries | Epochs | Wall-clock |
| ----- | ----- | ------- | ------ | ------------- | ------ | ---------- |
| **Scale 1** | Mac M4 MPS | frozen | 500K subsample | 5,584 (dev 80/20) | 20 | ~30 sec |
| **Scale 2** | A100 80 GB | **unfrozen** | full 8.84 M | 502,939 → 418,010 (BM25-filtered) | 5 | ~2 hr each (baseline + DRT) |

Differential learning rates at Scale 2: encoder 5e-5, heads 2e-3. Mixed-precision fp16 + gradient checkpointing on the encoder. Hyperparameters in [`configs/scale2.yaml`](./configs/scale2.yaml).

### Data

- **MS MARCO Passage Ranking v1** via HuggingFace (`BeIR/msmarco`, `BeIR/msmarco-qrels`).
- **BM25 hard negatives** from the official MS MARCO mirror (`qidpidtriples.train.full.2.tsv.gz` — the `.small` tarball was deprecated; we use the full superset and the loader caps at 7 negatives per pair).
- Pipeline: `data/download_full.py` (full) and `data/download.py` (Scale-1 subsample).

---

## Diagnostic methodology

After the headline result, we ran four diagnostics on the **same** trained DRT checkpoint — no retraining. The procedure is implemented in [`scripts/diagnose.py`](./scripts/diagnose.py).

### 1. Modified scoring inference (`scripts/diagnose.py scoring`)

Six variants applied to the same checkpoint:

| Variant | What it tests |
| ------- | ------------- |
| `drt_learned_alphas` | full DRT — `Σ αᵢ(q) · (qᵢ · dᵢ)` |
| `drt_uniform_alphas` | does the attention head do anything? (replace α with 1/k) |
| `drt_concat_cosine` | does the decomposition do anything? (reshape subs → 384-d, plain cosine) |
| `drt_slot_i` (i ∈ 0..5) | is any single slot specialized? |
| `drt_top2_alphas` | does sparsifying help? |

→ If `drt_concat_cosine` recovers baseline → the *scoring function* is broken.
→ If `drt_concat_cosine` matches full DRT → the *encoder* is the bottleneck.

We saw the second.

### 2. Per-slot probes (`scripts/diagnose.py probes`)

For each slot `i ∈ 0..5`, train a logistic regression on its 64-d encoding (70/30 split, stratified, seed 42) to predict:
- query type (factoid yes/no, regex)
- entity-bearing query (heuristic keyword regex)
- length bucket (≤5 / 6-10 / 11+ words)

→ If slots specialize, accuracy spikes on one slot per task and is at majority baseline elsewhere.
→ If slots are interchangeable, every slot scores about the same.

We saw the second.

### 3. Failure analysis (`scripts/diagnose.py failures`)

Compute per-query MRR@10 for baseline and DRT. Sort by `Δ = MRR_drt − MRR_baseline`. Profile the top-100 wins and top-100 losses by query length, query-type distribution, and example queries.

→ Tells you which query types DRT helps vs hurts.

DRT loses harder on factoid queries (79% of top-100 losses are factoid, vs 59% of wins).

### 4. Linear CKA (`scripts/diagnose.py cka`)

Linear CKA (Kornblith et al., 2019) between several representations of the same 6,980 dev queries:

- baseline encoder vs DRT encoder (raw 384-d)
- baseline encoder vs DRT concat-subs (384-d)
- baseline encoder vs each DRT slot (64-d)

→ Distinguishes "encoder fine, scoring broken" (high CKA) from "encoder damaged" (low CKA).

We saw 0.945 (small drift, not catastrophic). The 2% MRR deficit corresponds to that drift.

---

## Reproducing the experiments

### Environment

- Python 3.10+
- PyTorch 2.5+ (with CUDA for Scale 2; MPS works for Scale 1)
- See [`requirements.txt`](./requirements.txt)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Scale 1 (laptop)

```bash
# 1. Download + subsample + encode (frozen MiniLM)
python3 -m data.download                # 500K corpus subsample
python3 -m data.precompute              # encode through frozen encoder

# 2. Train DecompositionHead + QueryAttentionHead
python3 -m scripts.train_scale1

# 3. Evaluate vs cosine baseline
python3 -m scripts.evaluate
```

Total wall-clock: ~35 min (mostly encoding).

### Scale 2 (single A100 80 GB)

The full pipeline is automated via [`scripts/run_pipeline.sh`](./scripts/run_pipeline.sh). On Akash, [`deploy/scale2.sdl.yml`](./deploy/scale2.sdl.yml) provisions an A100 box with a persistent volume and runs the pipeline on container start. Manually:

```bash
python -m data.download_full            # full MS MARCO + BM25 hard negatives
python -m scripts.train_baseline        # cosine bi-encoder, 5 epochs
python -m scripts.train_scale2          # DRT end-to-end, 5 epochs
python -m scripts.evaluate_e2e          # comparison on full dev
```

Total wall-clock: ~6 hr (mostly training).

### Diagnostics

```bash
python -m scripts.diagnose all          # ~80 min on A100 (75 min encoding + 5 min analyses)
```

Outputs land in `results/diagnostics/`: five JSON files plus `scoring_topk.npz` (top-100 indices per query per variant, ~24 MB).

---

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `models/encoder.py` | `MiniLMEncoder` — wraps `all-MiniLM-L6-v2`, mean-pool + L2-norm |
| `models/decomposition.py` | `DecompositionHead` — two-layer MLP → reshape (k,d) → L2-norm |
| `models/attention.py` | `QueryAttentionHead` — tiny MLP → softmax over k slots |
| `models/scorer.py` | `DRTScorer` — Scale-1 wrapper combining the heads + static `score()` |
| `models/drt_model.py` | `DRTModel` — Scale-2 e2e model with the encoder inside |
| `losses/contrastive.py` | `info_nce_loss` |
| `losses/decorrelation.py` | `decorrelation_loss` (Barlow-style) |
| `losses/combined.py` | `slot_dropout` + `combined_loss` |
| `data/download.py`, `data/download_full.py` | MS MARCO download (subsample / full) |
| `data/precompute.py` | Scale-1 frozen-encoder embedding precomputation |
| `data/dataset.py`, `data/dataset_online.py` | Scale-1 precomputed / Scale-2 online text dataset |
| `training/scheduler.py` | Cosine LR + warmup |
| `training/trainer.py`, `training/trainer_e2e.py` | Scale-1 / Scale-2 trainers |
| `training/cosine_baseline.py` | Vanilla bi-encoder baseline trainer |
| `training/hard_negatives.py` | BM25 hard-negatives loader |
| `evaluation/metrics.py` | MRR@k, nDCG@k, Recall@k |
| `evaluation/evaluate_e2e.py` | Full-corpus DRT vs cosine evaluator |
| `scripts/train_scale1.py` | Scale-1 entry point |
| `scripts/train_baseline.py` | Scale-2 cosine baseline entry point |
| `scripts/train_scale2.py` | Scale-2 DRT entry point |
| `scripts/evaluate.py`, `scripts/evaluate_e2e.py` | Eval entry points |
| `scripts/diagnose.py` | Four-step diagnostic procedure |
| `scripts/run_pipeline.sh` | One-shot: download → baseline → DRT → eval, with resume markers |
| `scripts/make_diagram.py` | Generates `DRT_architecture.excalidraw` |
| `configs/scale1.yaml`, `configs/scale2.yaml` | Hyperparameters |
| `deploy/scale2.sdl.yml` | Akash SDL for A100 80 GB training |
| `deploy/entrypoint.sh` | Container bootstrap (curl-fetched at startup) |
| `results/diagnostics/*.json` | All diagnostic outputs |
| `results/diagnostics/scoring_topk.npz` | Top-100 indices for each scoring variant |
| `results/logs/*` | Every run log, including `comparison.txt` |

---

## Saved artifacts

Everything required to inspect the headline result and the diagnostic findings is checked in. The full corpus encodings (≈14 GB) are *not* committed but are regeneratable in ~75 min on A100 by running `python -m scripts.diagnose encode`.

- [`PAPER_NOTES.md`](./PAPER_NOTES.md) — paper-ready writeup (~600 lines), every number cited from the saved JSONs.
- [`DRT_architecture.excalidraw`](./DRT_architecture.excalidraw) — single-page diagram of the full research arc.
- [`results/diagnostics/`](./results/diagnostics) — five JSON files + 24 MB `scoring_topk.npz`.
- [`results/logs/`](./results/logs) — training + eval + diagnostic logs.
- Final checkpoints (`cosine_baseline_epoch5.pt`, `drt_scale2_epoch5.pt`) — **not in git**; available on request or via the GitHub release.

---

## Limitations

- Single hyperparameter configuration (`k=6`, `λ_dec=0.1`, `p=0.15`). A `λ=0` ablation would close whether decorrelation specifically is the source of encoder drift. This was deferred when the main DRT result missed the success criterion.
- Single backbone (`all-MiniLM-L6-v2`). DPR-base, Contriever, or larger encoders may behave differently.
- Single domain (MS MARCO Passage). BEIR, NQ, or domain-specific corpora untested.
- One random seed per training run. No bootstrap confidence intervals on the metric deltas.
- Probe labels are regex heuristics. A better entity tagger would sharpen step 2.

See [`PAPER_NOTES.md` §7.3](./PAPER_NOTES.md) for the full limitations discussion.

---

## Future work

1. **λ_dec = 0 ablation** to confirm decorrelation is the source of encoder drift. ~2 hr on A100.
2. **Soft attention slotting** — replace the fixed `reshape(B, k, d)` with `k` learned attention heads over the encoder's token sequence. Gives the model a mechanism to actually differentiate slots.
3. **Reframe as a methodology contribution** — *"Statistical decorrelation does not produce semantic specialization in dense retrieval embeddings."* The four-step diagnostic generalizes to other dense-retrieval architectures.

See [`PAPER_NOTES.md` §8](./PAPER_NOTES.md) for details.

---

## Citation

If this work informs your research, please cite the accompanying article and this repository.

```bibtex
@misc{drt2026,
  author       = {e-man07},
  title        = {DRT: Decomposed Relevance Tensors --- A negative result and four-step diagnostic methodology for dense retrieval architectures},
  year         = {2026},
  howpublished = {\url{https://github.com/e-man07/drt}},
  note         = {Companion code and saved artifacts}
}
```

---

## References (direct precursors)

- Khattab & Zaharia, 2020 — *ColBERT.*
- Humeau et al., 2020 — *Poly-encoders.*
- Zbontar et al., 2021 — *Barlow Twins.*
- Bardes et al., 2022 — *VICReg.*
- Kusupati et al., 2022 — *Matryoshka Representation Learning.*
- Karpukhin et al., 2020 — *Dense Passage Retrieval (DPR).*
- Xiong et al., 2021 — *ANCE.*
- Izacard et al., 2022 — *Contriever.*
- Thakur et al., 2021 — *BEIR.*
- Kornblith et al., 2019 — *Similarity of Neural Network Representations Revisited (CKA).*
- Locatello et al., 2019 — *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.*

Full bibliographic entries are in [`DRT_Research_Blueprint.html`](./DRT_Research_Blueprint.html).

---

## Acknowledgments

- **Akash Network** for the A100 80 GB compute that made Scale 2 feasible at low cost.
- **HuggingFace** for the BeIR/MS MARCO datasets and the `sentence-transformers` ecosystem.
- The authors of the precursor works listed above.

---

## License

[MIT](./LICENSE) © 2026 e-man07
