# Decomposed Relevance Tensors (DRT): Research Notes

> Paper-ready writeup of the full DRT research arc — proof-of-concept, end-to-end training, and post-hoc diagnostic.
>
> Source artifacts: `results/diagnostics/*.json`, `results/logs/*`, `checkpoints/*.pt`.
> Code: this repository (see Appendix F for commit SHAs).

---

## 0. TL;DR

We trained and evaluated **Decomposed Relevance Tensors (DRT)** — a retrieval-scoring architecture that replaces flat cosine similarity with (a) a learned decomposition of each embedding into `k=6` L2-normalized sub-vectors and (b) a query-adaptive softmax that weights per-slot similarities. The hypothesis was that decomposed scoring + query-adaptive weighting would extract more retrieval signal than cosine on the same encoder.

**The hypothesis is refuted.** On the full MS MARCO passage dev set (8.84 M corpus, 6,980 queries), a vanilla cosine bi-encoder trained from the same MiniLM checkpoint, on the same data, for the same number of epochs, beats DRT by **+0.0204 MRR@10** (0.3278 vs 0.3074), **+0.0235 nDCG@10**, and **+0.0285 Recall@100**.

A four-step diagnostic explains why:

1. **The decomposition machinery is a no-op.** When we replace the learned per-query attention weights with uniform `1/k` weights, or simply concatenate the six sub-vectors into a 384-d vector and use plain cosine, MRR@10 is **identical to the full DRT score** (0.3076 vs 0.3075). The decomposition + attention adds no discriminative power.
2. **The query-adaptive attention head learned nothing.** Learned weights vs uniform weights differ by < 0.0001 MRR@10. The softmax produces non-informative distributions.
3. **The decorrelation loss made slots statistically uncorrelated but not semantically distinct.** Per-slot probes show every slot achieves within 1% accuracy on factoid/length classification — they encode the same information.
4. **The encoder drifted, didn't break.** Linear CKA between the cosine-baseline encoder and the DRT encoder is **0.945** on 6,980 dev queries. The decomposition training pushed the encoder into a slightly worse subspace for retrieval — costing ~2% MRR@10.

DRT loses harder on factoid queries (79% of top-100 losses are factoid-typed vs 59% of top-100 wins). The architecture's reshuffled representation hurts precise entity/attribute matching more than topical retrieval.

The diagnostic methodology itself — modified scoring inference, per-slot probes, win/loss decomposition, and CKA — is reusable for analyzing any failed retrieval architecture.

---

## 1. Background and Hypothesis

### 1.1 The cosine-similarity ceiling

Modern dense retrieval encodes text into a single dense vector and ranks by cosine similarity. The blueprint that motivated this work identified four fundamental limitations:

1. **Single scalar collapse.** Cosine reduces topical overlap, entity alignment, intent match, temporal relevance, and specificity into one number in `[-1, 1]`.
2. **Forced symmetry.** `sim(q, d) = sim(d, q)`, yet relevance is directional.
3. **Opaque entanglement.** A 384- or 768-d vector blends multiple notions of similarity with no structure exposed.
4. **Curse of uniformity.** The same similarity function is used for factoid queries ("when was X born?") and topical queries ("machine learning trends"), even though different queries should weight different facets of relevance.

The two-stage industry pattern — cosine retrieval followed by cross-encoder reranking — works around these issues but is an engineering hack. The first stage has high recall but low precision; the second is too expensive to apply broadly. If the correct document doesn't survive cosine retrieval, the cross-encoder never sees it.

### 1.2 Prior art (positioning)

| Approach | Idea | Limitation |
|---|---|---|
| Cosine / dot-product bi-encoders (DPR, ANCE, Contriever) | Single vector, fast, ANN-indexable | Symmetric, scalar, no structure |
| **ColBERT** / late interaction | Per-token vectors, MaxSim scoring | Vectors are token-tied, not semantically typed; storage is k × tokens × dim |
| **Poly-encoder** | Multiple learned attention codes per document | Codes aren't decorrelated, not typed |
| **Matryoshka** embeddings | Nested sub-vectors at different dimensionalities | Truncated, not decomposed; no slot structure |
| Cross-encoders | Joint query-doc encoding | O(N) retrieval cost |
| Knowledge graphs | Explicit entity-relation structure | Brittle, requires extraction |
| **Barlow Twins / VICReg** | Decorrelation losses in self-supervised vision learning | Never applied to retrieval scoring |

DRT was positioned as the unfilled cell: **soft-learned sub-vector decomposition + decorrelation-driven specialization + query-adaptive weighting**, applied to retrieval scoring.

### 1.3 The DRT thesis

> *For any given text embedding, a learned decomposition into semantically specialized sub-vectors with query-adaptive weighted scoring extracts more retrieval signal than cosine similarity over the full vector.*

Concretely:
- A *Decomposition Head* projects a 384-d embedding into `k = 6` L2-normalized sub-vectors of `d = 64` dimensions each.
- A *Query Attention Head* maps the (full, pre-decomposition) query embedding to a softmax distribution over the `k` slots.
- The score is `Σᵢ αᵢ(q) · cos(qᵢ, dᵢ)` — a query-adaptive weighted sum of per-slot cosine similarities.
- A Barlow Twins-style decorrelation loss penalizes cross-correlation between any two slots, forcing them to encode independent information.
- Slot dropout zeros entire sub-vectors at training time, forcing each slot to independently carry retrieval signal.

The design claims **emergent specialization**: slots are not human-labeled, but the decorrelation loss drives them to encode different facets of relevance. The query attention head then learns which slots matter for which queries.

---

## 2. Methodology

### 2.1 Architecture

The system is a stack of three components: a transformer encoder, a Decomposition Head, and a Query Attention Head.

**Encoder.** `sentence-transformers/all-MiniLM-L6-v2` (BertModel base, 22.7 M parameters, hidden size 384, max sequence length 256). Forward pass returns the mean-pooled last-hidden-state, L2-normalized — matching what `SentenceTransformer.encode()` produces by default.

In Scale 1 (§3) the encoder is frozen. In Scale 2 (§4) it is unfrozen and receives gradients.

Implemented in `models/encoder.py:MiniLMEncoder`.

**Decomposition Head.** Two-layer MLP that projects the 384-d embedding into `k × sub_dim = 6 × 64 = 384` dimensions, then reshapes to `(k, sub_dim)` and L2-normalizes each sub-vector independently.

```
Input (..., 384)
  → Linear(384, 512) → LayerNorm → GELU
  → Linear(512, 384) → LayerNorm → GELU
  → reshape (..., 6, 64)
  → F.normalize(dim=-1)
Output (..., 6, 64), ‖sᵢ‖ = 1
```

Parameters: 394,432 (≈ 394 K).
Implemented in `models/decomposition.py:DecompositionHead`.

**Query Attention Head.** Tiny MLP from the (pre-decomposition) 384-d query embedding to a softmax distribution over `k` slots.

```
Input (..., 384)
  → Linear(384, 64) → GELU
  → Linear(64, 6)
  → softmax(dim=-1)
Output (..., 6), Σαᵢ = 1
```

Parameters: 24,966.
Implemented in `models/attention.py:QueryAttentionHead`.

**Scoring function.**
```
score(q, d) = Σᵢ₌₁ᵏ αᵢ(q) · cos(qᵢ, dᵢ)
            = Σᵢ αᵢ(q) · (qᵢ · dᵢ)   [since slots are L2-normalized]
```

Implemented as a static method `DRTScorer.score()` in `models/scorer.py` (Scale 1) and inline in `models/drt_model.py:DRTModel` (Scale 2). For batched retrieval the einsum form `einsum("bkd,pkd->bpk", q_subs, d_subs)` followed by an `α`-weighted sum is used.

**Total trainable parameters:**
| | Encoder | Decomp head | Attention head | **Total** |
|---|---|---|---|---|
| Scale 1 | 0 (frozen) | 394,432 | 24,966 | **419,398** |
| Scale 2 | 22,713,216 | 394,432 | 24,966 | **23,132,614** |

### 2.2 Loss design

Three components, summed:

$$
\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\text{retrieval}} \;+\; \lambda_{\text{dec}} \cdot \mathcal{L}_{\text{decorrelation}}
$$

with a structural slot-dropout regularizer applied at training time.

**L_retrieval.** InfoNCE over DRT scores. For each query `q` in a batch of size `B`, the positive document scores against `B-1` in-batch negatives (other queries' positives) plus `N` BM25 hard negatives. Target is the diagonal of the score matrix.

Implemented in `losses/contrastive.py:info_nce_loss`. Default temperature τ = 0.05.

**L_decorrelation.** Barlow Twins-inspired. Over a batch of decomposed sub-vectors with shape `(B, k, d)`, compute every cross-correlation matrix `C[i,j] = sub[:,i].T @ sub[:,j] / B` of shape `(d, d)` and penalize the squared Frobenius norm summed over the upper triangle pairs `i < j`.

```
∑_{i<j} ‖C[i,j]‖_F²   /   (k choose 2)
```

Implemented in `losses/decorrelation.py:decorrelation_loss`. Default λ_dec = 0.1.

**Slot dropout.** A structural regularizer (not strictly a loss term). At training time, each sub-vector is independently masked-to-zero with probability `p`. Surviving slots are rescaled by `1 / mean(mask)` to preserve expected magnitude.

Implemented in `losses/combined.py:slot_dropout`. Default p = 0.15.

The combined loss (`losses/combined.py:combined_loss`):
1. Compute decorrelation on **pre-dropout** sub-vectors (we want the underlying representation to specialize, not the dropout-corrupted version).
2. Apply slot dropout to query and document sub-vectors.
3. Compute InfoNCE on the post-dropout sub-vectors.
4. Total = retrieval + λ_dec · decorrelation.

### 2.3 Training data

MS MARCO passage ranking (v1).

| Split | Size | Use |
|---|---|---|
| Corpus | 8,841,823 passages | retrieval corpus |
| Train queries (with qrels) | 502,939 queries / 532,751 qrels rows | training |
| Dev-small queries (with qrels) | 6,980 queries / 7,437 qrels rows | evaluation |
| BM25 hard negatives | `qidpidtriples.train.full.2.tsv.gz` (397.8 M triples) | training negatives |

After filtering hard-negative triples to only those whose passages exist in the corpus subsample, **418,010 training (query, positive, 7-hard-negatives) tuples** remained.

Downloaded via HuggingFace `datasets` (`BeIR/msmarco`, `BeIR/msmarco-qrels`) and the official MS MARCO z22 mirror (hard negatives). Pipeline: `data/download_full.py`.

Scale 1 used a 500K-passage subsample of the corpus (random sample preserving every dev-qrels positive) for laptop-feasibility. Scale 2 used the full 8.84M corpus.

### 2.4 Hyperparameters

Identical across the cosine baseline and DRT trainers except for the head-specific terms:

| Parameter | Scale 1 (Mac M4 MPS) | Scale 2 (A100 80 GB) |
|---|---|---|
| Encoder | frozen | unfrozen |
| Encoder LR | — | 5e-5 |
| Head LR (DRT) | 2e-3 | 2e-3 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Scheduler | cosine + 10% warmup | cosine + 10% warmup |
| Batch size | 128 | 512 |
| Epochs | 20 | 5 |
| Hard negatives / query | 0 (in-batch only) | 7 (BM25) + in-batch |
| Mixed precision | fp32 | fp16 (autocast) |
| Gradient checkpointing | n/a | on encoder |
| `k` | 6 | 6 |
| sub_dim | 64 | 64 |
| Temperature τ | 0.05 | 0.05 |
| λ_decorrelation | 0.1 | 0.1 |
| Slot dropout p | 0.15 | 0.15 |
| Max seq length | 256 | 256 |
| Seed | 42 | 42 |

---

## 3. Scale 1 — Frozen-Encoder Proof of Concept

### 3.1 Setup

Goal: confirm that DRT scoring extracts more retrieval signal than cosine *over the same frozen embeddings*. This isolates the head + scoring function from any encoder-training effect.

- Device: Apple M4 (MPS backend).
- Corpus: 500,000-passage subsample of MS MARCO (random, preserving every dev-qrels positive). Encoded once with frozen `all-MiniLM-L6-v2` and saved as a numpy memmap.
- Training queries: an 80/20 seeded split of MS MARCO dev-small. 5,584 train + 1,396 eval queries.
- Training pairs after qrels filtering: 5,951 (one positive per query, no explicit hard negatives — in-batch only).
- 20 epochs × 46 steps/epoch = 920 optimizer steps.
- Wall-clock: download + encode 32 min, training ~10 sec, eval ~20 sec.

This setup is methodologically weak (very small train set, eval split is internal to dev) but designed for fast iteration on consumer hardware.

### 3.2 Results

Evaluated on the 1,396 held-out dev queries against the 500K-passage corpus.

| Metric | Cosine baseline | DRT (full) | Δ |
|---|---|---|---|
| MRR@10 | **0.6951** | 0.6236 | **−0.0715** |
| nDCG@10 | 0.7383 | 0.6663 | −0.0720 |
| Recall@100 | 0.9800 | 0.9396 | −0.0404 |

Both numbers are inflated relative to standard MS MARCO results because the 500K corpus is ~18× smaller than the canonical 8.84M; rank-1 retrieval is easier. The Δ is what matters: **DRT lost by 7%**.

### 3.3 Scale 1 ablation: λ_dec = 0, p = 0

To check whether decorrelation or slot dropout were specifically hurting:

| Variant | MRR@10 | Δ vs cosine | Δ vs full DRT |
|---|---|---|---|
| Cosine baseline | 0.6951 | — | — |
| DRT full (λ=0.1, p=0.15) | 0.6236 | −0.0715 | — |
| **DRT no-decorr no-dropout (λ=0, p=0)** | **0.6300** | **−0.0651** | +0.0064 |

Removing both regularizers barely changes the result. The architecture's failure mode is not specifically the decorrelation loss in Scale 1 — it's that there is essentially nothing useful for the head to learn from 5,951 (q, p+) pairs against frozen MiniLM embeddings already optimized for cosine retrieval.

### 3.4 Interpretation

The frozen-encoder ceiling. MiniLM was contrastively trained to put similar texts close together under cosine similarity. A 419 K-parameter head sitting on top of those embeddings cannot improve on cosine — at best it can match it. With 5,951 training pairs, the head doesn't have enough signal to even match, and ends up adding noise.

Scale 1 is therefore inconclusive about DRT's intrinsic value. The blueprint anticipated this: *Scale 1 verifies the architecture and code path work; Scale 2 — unfreezing the encoder — is the real test.*

---

## 4. Scale 2 — End-to-End Training

### 4.1 Setup

- Provider: Akash A100-SXM4-80GB (1 GPU, 16 CPU, 64 GiB RAM, 200 GiB persistent `/workspace` volume).
- Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
- Code: pinned to commit `e2ef77b` (then `cfc17e9`, `3e8023a`, `8b5a4b9`, `3e3130f` — see commit history).
- Pipeline: download → train cosine baseline → train DRT → evaluate.
- Wall-clock per training (each model): **~119 min** for 5 epochs at ~1,430 s/epoch (816 optimizer steps/epoch, batch 512).

The cosine baseline trains the *same MiniLM encoder* from the *same starting weights* on the *same hard-negative training data* for the *same epoch count*, using only InfoNCE over plain `cos(q, d)` (no decomposition, no attention head, no decorrelation, no slot dropout). It is the fair comparison target the blueprint specified.

Encoded corpus (for both eval and diagnostics) lives on `/workspace/` as `baseline_corpus_emb.npy` (8.84 M × 384 fp16 = 6.4 GB) and `drt_corpus_subs.npy` (8.84 M × 6 × 64 fp16 = 6.4 GB).

### 4.2 Training trajectory

#### Cosine baseline — per-epoch training loss

| Epoch | Avg loss | Wall (s) |
|---|---|---|
| 1 | 0.3567 | 1428 |
| 2 | 0.3042 | 1425 |
| 3 | 0.2602 | 1428 |
| 4 | 0.2311 | 1431 |
| **5** | **0.2171** | 1430 |

Smooth monotonic descent — 39% reduction. Source: `results/logs/02-baseline.console.log`.

#### DRT — per-epoch training loss

| Epoch | Total loss | Retrieval | Decorrelation | Wall (s) |
|---|---|---|---|---|
| 1 | 0.7459 | 0.7438 | 0.0211 | 1433 |
| 2 | 0.5523 | 0.5513 | 0.0105 | 1430 |
| 3 | 0.4561 | 0.4546 | 0.0142 | 1428 |
| 4 | 0.3920 | 0.3904 | 0.0162 | 1433 |
| **5** | **0.3591** | **0.3574** | **0.0169** | 1429 |

52% reduction over the 5 epochs. Decorrelation loss stayed in a stable 0.01-0.02 band throughout, indicating the slots became and remained statistically near-uncorrelated. Source: `results/logs/03-drt.console.log`.

**The DRT and baseline training losses are not directly comparable** — DRT's loss is computed via the per-slot weighted sum and the decorrelation term lives on a different scale. Only the eval metrics compare across architectures.

### 4.3 Full-corpus evaluation

Both checkpoints evaluated on the standard MS MARCO dev-small set (6,980 queries) against the full 8.84 M passage corpus, brute-force scoring (no FAISS approximation).

| Metric | Cosine baseline | DRT | Δ |
|---|---|---|---|
| **MRR@10** | **0.3278** | 0.3074 | **−0.0204** |
| nDCG@10 | 0.3884 | 0.3648 | −0.0236 |
| Recall@100 | 0.8600 | 0.8315 | −0.0285 |

Source: `results/logs/comparison.txt` and `results/diagnostics/scoring_metrics.json`.

The cosine number (0.3278) is consistent with published MS MARCO numbers for a contrastively fine-tuned MiniLM bi-encoder (typically 0.32-0.34). The DRT number (0.3074) is what we set out to compare against it.

### 4.4 Conclusion

The blueprint's success criterion was Δ MRR@10 ≥ **+0.02** for DRT. We got **−0.02**. DRT lost by exactly the same magnitude the criterion required for success.

Unfreezing the encoder narrowed the gap considerably (from −7% in Scale 1 to −2% in Scale 2), confirming that the frozen-encoder ceiling was real. But end-to-end training did not flip the sign.

The next question — *why* — is what the diagnostic section answers.

---

## 5. Diagnostic Methodology

After the main result, we ran a four-step post-hoc analysis using the trained DRT checkpoint (no retraining). Each step targets a specific hypothesis about *where* in the pipeline DRT loses signal. All scripts live in `scripts/diagnose.py` (subcommands `encode`, `scoring`, `probes`, `failures`, `cka`, `all`).

### 5.1 Step 1 — Modified scoring inference

Six scoring variants applied to the **same** trained DRT checkpoint:

| Variant | Description |
|---|---|
| **cosine_baseline** | Reference: plain cosine on the baseline bi-encoder. |
| **drt_learned_alphas** | The DRT default — `Σᵢ αᵢ(q) · (qᵢ · dᵢ)` with αᵢ from the trained attention head. |
| **drt_uniform_alphas** | Replace α with `1/k` for all slots — tests whether the attention head adds discriminative power. |
| **drt_slot_i** (i ∈ 0..5) | Score using only slot `i`, with αᵢ = 1, all others 0. Tests per-slot information content. |
| **drt_concat_cosine** | Reshape DRT sub-vectors back to a 384-d vector, L2-normalize, score with plain cosine. Tests whether the decomposition transformation degrades the encoder vs the original cosine subspace. |
| **drt_top2_alphas** | Use only the two highest-α slots per query, renormalized. Tests sparse mixture. |

Implemented in `cmd_scoring` (`scripts/diagnose.py`). Top-100 indices per query are saved per variant to `results/diagnostics/scoring_topk.npz` for downstream analysis.

### 5.2 Step 2 — Per-slot probes

For each of the 6 slots, a separate logistic regression is trained to predict three properties of the query from the slot's 64-d encoding, with a 70/30 train/test split:

- **factoid**: Does the query begin with a wh-/yes-no question word (regex)?
- **has_entity**: Heuristic for entity-bearing queries — years, percentages, common entity nouns (regex).
- **length_bucket**: ≤ 5 / 6-10 / 11+ words.

A 384-d concat-of-all-slots probe is run as a ceiling comparison.

If each slot specializes for a different facet of the query, we'd expect the per-task accuracy to spike on one slot and be at majority-class baseline on others. If slots are interchangeable, every slot would score about the same.

Implemented in `cmd_probes` (`scripts/diagnose.py`).

### 5.3 Step 3 — Failure analysis

Compute per-query MRR@10 for the cosine baseline and for DRT separately. Sort queries by `Δ = MRR_drt - MRR_baseline`. The top-100 wins (largest positive Δ) and the top-100 losses (largest negative Δ) are profiled by:
- Average query length
- Distribution over query-type labels (factoid, has_entity, length_bucket)
- Ten representative example queries from each group

This identifies *which query types* DRT helps vs hurts.

Implemented in `cmd_failures` (`scripts/diagnose.py`).

### 5.4 Step 4 — Representation similarity (linear CKA)

Compute Linear CKA (Kornblith et al., 2019) between several representations of the same 6,980 dev queries:

- baseline encoder output (384-d)
- DRT encoder output, before decomposition (384-d)
- DRT concatenated sub-vectors (384-d)
- DRT individual slot `i` (64-d), each compared to both encoders

Linear CKA in [0, 1] measures how much one representation can be linearly reconstructed from another. CKA close to 1 means the representations carry the same information up to rotation; CKA close to 0.5 indicates substantial divergence.

Implemented in `cmd_cka` (`scripts/diagnose.py`).

---

## 6. Diagnostic Findings

### 6.1 The smoking gun: the decomposition + attention is a no-op

Full per-variant scoring results (`results/diagnostics/scoring_metrics.json`):

| Variant | MRR@10 | nDCG@10 | Recall@100 |
|---|---|---|---|
| cosine_baseline | **0.3278** | 0.3884 | 0.8600 |
| drt_learned_alphas (full DRT) | 0.3075 | 0.3648 | 0.8312 |
| **drt_uniform_alphas** (α = 1/k) | **0.3076** | 0.3650 | 0.8314 |
| **drt_concat_cosine** | **0.3076** | 0.3650 | 0.8314 |
| drt_top2_alphas (top-2 slots) | 0.2869 | 0.3417 | 0.7902 |
| drt_slot_0 | 0.2611 | 0.3093 | 0.7307 |
| drt_slot_1 | 0.2577 | 0.3060 | 0.7292 |
| drt_slot_2 | 0.2611 | 0.3103 | 0.7299 |
| drt_slot_3 | 0.2611 | 0.3100 | 0.7279 |
| drt_slot_4 | 0.2561 | 0.3063 | 0.7288 |
| drt_slot_5 | 0.2608 | 0.3101 | 0.7344 |

**Three of the variants score identically: 0.3075 / 0.3076 / 0.3076.**

- `drt_learned_alphas` — the full DRT model.
- `drt_uniform_alphas` — same model, but with the learned attention weights replaced by `1/6` for every slot.
- `drt_concat_cosine` — same model, but discarding the attention head entirely and just doing plain cosine over the flattened sub-vectors.

These three score within 0.0001 MRR@10 of each other. The decomposition transformation + the attention-weighted combination is doing nothing detectable beyond what plain cosine over the same encoder's transformed output already does.

This is **the diagnostic that closes the loop**: the architecture's two distinctive mechanisms — soft sub-vector decomposition and query-adaptive weighting — both fail to contribute discriminative power. The −0.02 MRR@10 deficit relative to the cosine baseline comes from somewhere else entirely (see §6.4).

### 6.2 The query attention head learned nothing

`drt_learned_alphas` (0.3075) and `drt_uniform_alphas` (0.3076) are within 0.0001 MRR@10. The trained softmax over the 6 slots is, for all retrieval purposes, equivalent to a uniform distribution.

We did not save the learned α distributions for inspection, but the equivalence with `1/k` is itself the evidence. A non-trivial attention head — one that emphasized different slots for different queries — would necessarily produce a different score than uniform weighting unless the slots themselves are interchangeable. (See §6.3.)

### 6.3 Slots are statistically uncorrelated but semantically interchangeable

Three lines of evidence:

**A. Single-slot scoring.** Using only slot `i` yields MRR@10 in the range **0.2561 to 0.2611** for every i. Variance across slots: ~0.005 MRR@10. If any slot were specialized for some retrieval-useful property, we'd expect more spread.

**B. Probes** (`results/diagnostics/probes.json`):

| Slot | factoid (majority 0.7168) | length_bucket (majority 0.4924) | has_entity (majority 0.9704)* |
|---|---|---|---|
| 0 | 0.7593 | 0.6232 | 0.9704 |
| 1 | 0.7607 | 0.6394 | 0.9704 |
| 2 | 0.7693 | 0.6423 | 0.9704 |
| 3 | 0.7741 | 0.6423 | 0.9704 |
| 4 | 0.7736 | 0.6380 | 0.9704 |
| 5 | 0.7607 | 0.6079 | 0.9704 |
| **concat (384-d)** | **0.8128** | **0.6915** | 0.9709 |

\* The has_entity probe is uninformative because the heuristic label fires on only 208/6,980 queries; class imbalance makes the majority-class baseline saturate.

Every slot scores within **0.0148 accuracy** of every other slot on factoid classification (range 0.7593-0.7741) and within **0.0344** on length-bucket classification (range 0.6079-0.6423). The concat probe is meaningfully better than any single slot (factoid +4%, length +5%), so the slots **aren't redundant** in the strict sense — each has *some* unique information. But they're also not *specialized* — no slot stands out as the "factoid slot" or "length slot."

**C. CKA between slots and the encoder** (`results/diagnostics/cka.json`):

| Slot | CKA vs baseline encoder | CKA vs DRT raw encoder |
|---|---|---|
| 0 | 0.5637 | 0.5683 |
| 1 | 0.5629 | 0.5692 |
| 2 | 0.5742 | 0.5770 |
| 3 | 0.5678 | 0.5720 |
| 4 | 0.5640 | 0.5706 |
| 5 | 0.5740 | 0.5748 |

Every slot is **~57% similar** to both encoders (variance 0.005 across slots). All six slots are doing essentially the same projection of the encoder's 384-d output into a 64-d subspace.

**Conclusion**: the Barlow-style decorrelation loss successfully drove inter-slot cross-correlation to ~0.01-0.02 throughout training (§4.2), but *statistical decorrelation does not imply semantic specialization*. The slots ended up as rotated copies of each other rather than as semantically distinct subspaces.

### 6.4 Encoder drift, not encoder break (CKA 0.945 vs baseline)

| Comparison (n = 6,980 dev queries) | Linear CKA |
|---|---|
| baseline encoder vs DRT raw encoder | **0.9452** |
| baseline encoder vs DRT concat-subs | 0.8297 |
| DRT raw encoder vs DRT concat-subs | 0.8359 |

The DRT-trained encoder is **94.5% representationally similar** to the cosine-trained encoder. That's a small drift, not a catastrophic remapping.

The remaining ~17% drift in *the post-decomposition representation* (CKA 0.83 between baseline encoder and DRT concat-subs) is what costs the 2% MRR@10. The decomposition head's two `Linear → LayerNorm → GELU` blocks transform the encoder's output into a 384-d space that — when restructured back into a flat vector via concat — is slightly *less* useful for cosine retrieval than the original encoder's output would have been.

In other words: training the encoder under the joint objective of (a) contrastive retrieval, (b) producing embeddings that survive a decomposition-head transformation, (c) admitting a low-cross-correlation slot structure, and (d) being robust to slot dropout pushed it into a representation that's a small but real step worse for retrieval than what plain contrastive training produces.

### 6.5 DRT loses harder on factoid queries

Per-query MRR@10 comparison (`results/diagnostics/failures.json`):

| | Value |
|---|---|
| baseline mean MRR@10 | 0.3278 |
| DRT mean MRR@10 | 0.3075 |
| Δ mean | −0.0203 |
| Δ median | **0.0000** |
| frac DRT > BL | 14.0% |
| frac DRT < BL | 20.9% |
| frac tie | **65.1%** |
| avg query length (wins) | 5.5 words |
| avg query length (losses) | 6.3 words |

DRT and baseline tie on **65%** of dev queries (mostly because neither retrieves the correct passage in the top-10, returning MRR = 0 for both). Among the queries where they differ:

| Label | Wins (top-100 DRT > BL) | Losses (top-100 DRT < BL) |
|---|---|---|
| factoid=1 (wh-/yes-no) | 59 | **79** |
| factoid=0 | 41 | 21 |
| length=0 (≤5 words) | 57 | 38 |
| length=1 (6-10 words) | 40 | 56 |
| length=2 (11+ words) | 3 | 6 |

The asymmetry is consistent: 79% of DRT's worst losses are factoid queries vs 59% of its best wins. DRT degrades performance on precise, single-answer queries more than on broader topical queries.

Representative loss examples (full list in `results/diagnostics/failures.json`):
- "does law enforcement have the responsibility to intervene?"
- "what's the temperature in tucson arizona right now?"
- "where is 89130"
- "who is deputy director andrew mccabe"
- "what carnival ships have havana"
- "what is the parent company for adidas"

These need precise entity/attribute matching ("89130" as a zip, "andrew mccabe" as a named person, "adidas" as a company). The encoder drift in §6.4 hurts those matches more than it hurts topical retrieval.

Representative win examples:
- "node js import"
- "what is the best place for all inclusive vacations for families"
- "what food helps to produce collagen"
- "chemistry amu definition"

These are more topical/conceptual queries where some representation reshuffling is forgivable.

---

## 7. Discussion

### 7.1 Why decorrelation didn't create specialization

The Barlow decorrelation loss enforces a **statistical** property: pairwise cross-correlation between slots should be near zero in expectation over a batch. We observed (§4.2) that it succeeded — decorrelation loss stabilized at 0.01-0.02 throughout training.

But statistical decorrelation in a 64-d subspace is a very weak constraint relative to semantic specialization. There are an enormous number of orthogonal directions in 64-d, and the loss doesn't tell the model which orthogonal directions matter. The model is free to pick six near-orthogonal projections that all encode "general semantic relevance" — which is what it did (§6.3).

For decorrelation to produce specialization, the slots would need *additional pressure to attach to different semantic dimensions*. Possibilities the current architecture lacks:
- Different supervised objectives per slot (one slot trained against entity-match labels, another against topic labels, etc.) — requires labeled data.
- An information bottleneck per slot that penalizes pairs of slots encoding "the same" task-relevant information, not just statistical correlation.
- A discrete attention bottleneck that forces each query to *route* through a small number of slots, so different queries see different slots.

Slot dropout, despite its name, doesn't produce this routing — it randomly drops slots independently per training example, which encourages each slot to be retrieval-useful on its own but doesn't push slots apart from each other.

### 7.2 What the architecture lacks

The two load-bearing claims of DRT:

1. **Decomposed sub-vectors beat flat vectors.** §6.1 refutes this — concat-cosine over the sub-vectors is identical to the full DRT score. The decomposition is a no-op.
2. **Query-adaptive weighting beats uniform weighting.** §6.2 refutes this — learned α matches uniform α exactly. The attention is a no-op.

Both fail at the same level: the architecture has no *mechanism* to force the slots to be different things that *would* benefit from differential weighting. Decorrelation makes them statistically distinct without making them semantically distinct, and without semantic distinction the attention head has no leverage.

The fixed `reshape(B, k, d)` over a generic 384-d transformation does not create slot-level differentiation. It just rotates the original embedding into a structured layout.

### 7.3 Limitations

- **Single configuration.** We trained one DRT instance at one hyperparameter setting (`k=6`, `λ=0.1`, `p=0.15`). A larger ablation sweep over `k`, `λ`, and `p` was deferred when the main DRT result didn't meet the success criterion. The negative result is therefore over-specific to this configuration. A `λ=0` ablation in particular would close whether decorrelation specifically is the source of the encoder drift.
- **Single backbone.** Only `all-MiniLM-L6-v2` was tested. Whether the same conclusions hold for a larger encoder (e.g., DPR-base, Contriever) is unknown.
- **Single domain.** Only MS MARCO passage. BEIR, NQ, or domain-specific corpora may behave differently.
- **One random seed.** Each training run was a single seed. Variance bars not estimated.
- **Heuristic query labels.** The probes used regex-based labels for query type and entity presence. The has_entity heuristic in particular over-restricted (208 / 6,980 positives) and the probe accuracy saturated at the majority baseline. Better labels (e.g., LLM-tagged) would sharpen step 2 conclusions.
- **No statistical test.** The per-variant MRR@10 numbers are point estimates; no bootstrap confidence intervals. The 0.3075 vs 0.3076 equivalence claim is robust to any sane CI given the magnitude, but more careful statistics would be needed for a published comparison.

### 7.4 What we learned about diagnostic methodology

Independent of the DRT result, the four-step diagnostic procedure may be a contribution in its own right:

1. **Modified scoring inference** is cheap (no retraining) and quickly localizes whether a complex scoring function is doing anything beyond a simpler scoring function on the same representation. If `f(g(x))` matches `cosine(g(x))`, the complexity `f(·)` is a no-op.
2. **Per-slot probes** test whether multi-vector architectures actually specialize their components.
3. **Win/loss decomposition** tells you *what kind* of queries an architecture helps and hurts on — which directly informs follow-up design.
4. **CKA** distinguishes "the encoder is fine, the scoring is broken" from "the encoder itself was damaged."

The four together produce a much sharper picture than the aggregate MRR delta alone. This is the diagnostic procedure we used here, and it could be standardized for any new dense retrieval architecture.

---

## 8. Future Work

### 8.1 Cheap follow-up: confirm the decorrelation hypothesis

Train one more DRT variant with **λ_dec = 0** (no decorrelation loss), everything else identical. Predicted outcomes:

- If DRT recovers to within +/- 0.005 MRR@10 of the cosine baseline → confirms the decorrelation loss is what damages the encoder.
- If DRT still loses by ~2% → the encoder drift is caused by the decomposition head's transformation itself, not the loss. The fix would have to be a different decomposition architecture, not a different loss weight.

Either outcome closes the story cleanly. Cost: one A100 training run (~2 hours), ~$3.

### 8.2 Alternative architectures (deeper redesign)

If DRT or a closely-related multi-vector architecture is still the research direction, the diagnostics point to specific redesigns:

**Soft attention decomposition (instead of fixed reshape).** Replace the `Linear → reshape` decomposition with `k` learned attention heads over the encoder's last-hidden-state tokens. Each head produces a 64-d slot from a different weighted combination of token vectors. This gives the model a mechanism to differentiate slots — each head can attend to different aspects of the input.

**Supervised slot specialization.** Pre-train each slot against a different cross-encoder ranking signal (entity-match, topic-match, exact-match, etc.) before combining them. This bypasses the brittle "decorrelation creates specialization" hope by directly specifying what each slot should learn.

**Discrete slot routing.** Replace the dense softmax attention head with a top-k discrete router (e.g., Gumbel-softmax with k=1 or k=2 at inference). Each query routes through a small subset of slots, which forces the slots to actually differ — otherwise the router doesn't matter.

### 8.3 Reframing as a methodology paper

The most defensible publishable contribution from this work isn't a new architecture (DRT lost). It's the **diagnostic methodology** itself plus **the negative result and its explanation**.

Working title: *"Statistical decorrelation does not produce semantic specialization in dense retrieval embeddings: a four-step diagnostic methodology."*

Structure:
- Section 1: motivate that the field publishes architecture-level positive results without diagnosing *why* an architecture works, and that this gap matters when results don't replicate or when a new architecture is being designed.
- Section 2: describe DRT (the case study, our negative result).
- Section 3: the four diagnostics, generalized.
- Section 4: apply them to DRT, surfacing the specific failure modes.
- Section 5: conclude with a checklist for evaluating new dense retrieval architectures.

Target venue: a workshop at ACL / SIGIR / NeurIPS, or a methodology track at an IR venue. The negative result is too narrow for a main-conference paper as-is; the methodology gives it broader relevance.

---

## Appendices

### Appendix A — Full per-variant scoring numbers

Source: `results/diagnostics/scoring_metrics.json`. All numbers on full MS MARCO dev (6,980 queries × 8.84 M corpus). Bold = identical or near-identical groups.

| Variant | MRR@10 | nDCG@10 | Recall@100 |
|---|---|---|---|
| cosine_baseline | 0.327785 | 0.388401 | 0.859957 |
| drt_learned_alphas | **0.307470** | **0.364768** | **0.831232** |
| drt_uniform_alphas | **0.307609** | **0.364995** | **0.831375** |
| drt_concat_cosine | **0.307615** | **0.365028** | **0.831375** |
| drt_top2_alphas | 0.286876 | 0.341686 | 0.790150 |
| drt_slot_0 | 0.261141 | 0.309282 | 0.730695 |
| drt_slot_1 | 0.257671 | 0.306038 | 0.729202 |
| drt_slot_2 | 0.261106 | 0.310309 | 0.729871 |
| drt_slot_3 | 0.261140 | 0.310037 | 0.727865 |
| drt_slot_4 | 0.256129 | 0.306318 | 0.728844 |
| drt_slot_5 | 0.260780 | 0.310149 | 0.734432 |

### Appendix B — Per-slot probe accuracies (full)

Source: `results/diagnostics/probes.json`. Logistic regression on 70% of dev queries, evaluated on the remaining 30%. Stratified split, seed 42.

#### Per-slot (64-d input)

| Slot | factoid | majority | length_bucket | majority |
|---|---|---|---|---|
| 0 | 0.7593 | 0.7168 | 0.6232 | 0.4924 |
| 1 | 0.7607 | 0.7168 | 0.6394 | 0.4924 |
| 2 | 0.7693 | 0.7168 | 0.6423 | 0.4924 |
| 3 | 0.7741 | 0.7168 | 0.6423 | 0.4924 |
| 4 | 0.7736 | 0.7168 | 0.6380 | 0.4924 |
| 5 | 0.7607 | 0.7168 | 0.6079 | 0.4924 |

#### Concatenated baseline (6×64 = 384-d input)

| | factoid | length_bucket | has_entity |
|---|---|---|---|
| concat | **0.8128** | **0.6915** | 0.9709 (≈ majority) |

`has_entity` probe was uninformative because of label heuristic limitations (208 positives out of 6,980); we report it for completeness but exclude it from interpretation.

### Appendix C — CKA matrix

Source: `results/diagnostics/cka.json`. Linear CKA on dev queries (n = 6,980).

| Comparison | CKA |
|---|---|
| baseline encoder ↔ DRT raw encoder | **0.9452** |
| baseline encoder ↔ DRT concat sub-vectors | 0.8297 |
| DRT raw encoder ↔ DRT concat sub-vectors | 0.8359 |
| baseline encoder ↔ DRT slot 0 (64-d) | 0.5637 |
| baseline encoder ↔ DRT slot 1 | 0.5629 |
| baseline encoder ↔ DRT slot 2 | 0.5742 |
| baseline encoder ↔ DRT slot 3 | 0.5678 |
| baseline encoder ↔ DRT slot 4 | 0.5640 |
| baseline encoder ↔ DRT slot 5 | 0.5740 |
| DRT raw encoder ↔ DRT slot 0 | 0.5683 |
| DRT raw encoder ↔ DRT slot 1 | 0.5692 |
| DRT raw encoder ↔ DRT slot 2 | 0.5770 |
| DRT raw encoder ↔ DRT slot 3 | 0.5720 |
| DRT raw encoder ↔ DRT slot 4 | 0.5706 |
| DRT raw encoder ↔ DRT slot 5 | 0.5748 |

### Appendix D — Example wins and losses

Source: `results/diagnostics/failures.json`. Selected top-10 queries by absolute Δ MRR@10 in each direction.

#### Top 10 wins (DRT > baseline)

| Δ | bl | drt | n_rel | Query |
|---|---|---|---|---|
| +1.000 | 0.00 | 1.00 | 1 | node js import |
| +1.000 | 0.00 | 1.00 | 1 | what is the best place for all inclusive vacations for families |
| +1.000 | 0.00 | 1.00 | 1 | is physical therapy |
| +1.000 | 0.00 | 1.00 | 2 | what food helps to produce collagen |
| +1.000 | 0.00 | 1.00 | 1 | what are copper coated carbon rods used for |
| +0.889 | 0.11 | 1.00 | 1 | what is the best description of an adaptation |
| +0.889 | 0.11 | 1.00 | 1 | how can nitrogen be fixed |
| +0.875 | 0.12 | 1.00 | 1 | chemistry amu definition |
| +0.857 | 0.14 | 1.00 | 1 | what age do moles appear |
| +0.857 | 0.14 | 1.00 | 1 | how late can males receive hpv vaccine |

#### Top 10 losses (DRT < baseline)

| Δ | bl | drt | n_rel | Query |
|---|---|---|---|---|
| −1.000 | 1.00 | 0.00 | 1 | does law enforcement have the responsibility to intervene? |
| −1.000 | 1.00 | 0.00 | 1 | what's the temperature in tucson arizona right now? |
| −1.000 | 1.00 | 0.00 | 1 | the word said hello in swedish |
| −1.000 | 1.00 | 0.00 | 1 | dept of treasury careers |
| −1.000 | 1.00 | 0.00 | 1 | where is 89130 |
| −1.000 | 1.00 | 0.00 | 1 | who is deputy director andrew mccabe |
| −1.000 | 1.00 | 0.00 | 1 | what internet security do you get with rcn |
| −1.000 | 1.00 | 0.00 | 1 | what carnival ships have havana |
| −1.000 | 1.00 | 0.00 | 1 | what is the color of steel coin? |
| −1.000 | 1.00 | 0.00 | 1 | what is the parent company for adidas |

### Appendix E — Hyperparameter table (full)

#### Scale 1 (frozen-encoder PoC, Mac M4 MPS)

```yaml
model_name: sentence-transformers/all-MiniLM-L6-v2
max_seq_length: 256
encoder_frozen: true
k: 6
sub_dim: 64
decomp_hidden: 512
attn_hidden: 64

learning_rate: 2e-3      # heads only
weight_decay: 0.01
warmup_ratio: 0.1
scheduler: cosine
batch_size: 128
num_epochs: 20
num_hard_negatives: 0    # in-batch only

temperature: 0.05
lambda_decorr: 0.1
slot_dropout_p: 0.15
seed: 42
```

#### Scale 2 (end-to-end, A100 80 GB)

```yaml
model_name: sentence-transformers/all-MiniLM-L6-v2
max_seq_length: 256
encoder_frozen: false
k: 6
sub_dim: 64
decomp_hidden: 512
attn_hidden: 64

encoder_lr: 5e-5
head_lr: 2e-3
weight_decay: 0.01
warmup_ratio: 0.1
scheduler: cosine
batch_size: 512
grad_accum_steps: 1
num_epochs: 5
num_hard_negatives: 7    # BM25 hard negs + in-batch

temperature: 0.05
lambda_decorr: 0.1
slot_dropout_p: 0.15
grad_checkpoint: true
fp16: true
num_workers: 4
seed: 42
```

### Appendix F — Reproduction

#### Environment

- Python 3.14
- PyTorch 2.5.1 (Akash box, CUDA 12.4) / 2.11.0 (local Mac MPS)
- transformers 5.x, sentence-transformers, datasets, scikit-learn, numpy, pyyaml, tqdm
- 1 × A100-SXM4-80GB for Scale 2
- macOS / Apple M4 for Scale 1

#### Commits (this repository)

```
1923aae  first commit (scaffold + Scale 1 + initial data pipeline)
3e3130f  Akash deploy: bootstrap via curl-fetched entrypoint
8b5a4b9  Fix MS MARCO hard-negatives URL (small tarball was 404)
3e8023a  Move tokenization to DataLoader workers via TokenizingCollator
cfc17e9  Eval: bump chunk_size 256->4096 + add progress logging
e2ef77b  Eval: use DataLoader+workers for corpus encoding
3548fa6  Diagnostics: 4-step analysis of why DRT lost to baseline
76d4b15  diagnose: fix sklearn API + entity proxy
768389f  diagnose: remove last multi_class kwarg
```

#### Commands

Scale 1:
```bash
python3 -m data.download                            # MS MARCO dev, 500K subsample
python3 -m data.precompute                          # encode through frozen MiniLM
python3 -m scripts.train_scale1                     # train DecompositionHead + AttentionHead
python3 -m scripts.evaluate                         # DRT vs cosine
```

Scale 2 (on the Akash A100 box, via `deploy/scale2.sdl.yml` and `scripts/run_pipeline.sh`):
```bash
python -m data.download_full                        # full MS MARCO + hard negatives
python -m scripts.train_baseline                    # cosine bi-encoder, 5 epochs
python -m scripts.train_scale2                      # DRT end-to-end, 5 epochs
python -m scripts.evaluate_e2e                      # comparison on full dev
python -m scripts.diagnose all                      # 4-step diagnostic
```

Hyperparameters live in `configs/scale2.yaml`.

#### Data sources

- MS MARCO corpus, queries, qrels: HuggingFace `BeIR/msmarco`, `BeIR/msmarco-qrels`.
- BM25 hard negatives: `https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz` (the official `.small.tar.gz` mirror returned 404; we use the `.full.2` superset and let the loader filter).

#### Saved artifacts (this repo)

```
results/
├── diagnostics/
│   ├── scoring_metrics.json   # per-variant MRR/nDCG/Recall
│   ├── scoring_topk.npz       # top-100 indices per query per variant
│   ├── probes.json            # per-slot LR accuracies
│   ├── failures.json          # top-100 win/loss queries + summary stats
│   ├── cka.json               # representation similarity matrix
│   └── drt_meta.json          # DRT architecture meta (k, sub_dim, embed_dim)
└── logs/
    ├── 01-download.log
    ├── 02-baseline.{console.log, log}
    ├── 03-drt.{console.log, log}
    ├── 04-eval.log
    ├── comparison.txt         # final headline numbers
    ├── diagnose.log
    ├── diag_probes.log
    ├── diag_failures.log
    ├── diag_cka.log
    └── pipeline.log

checkpoints/
├── cosine_baseline_epoch5.pt  # 87 MB, the Scale 2 baseline
├── drt_scale2_epoch5.pt       # 88 MB, the Scale 2 DRT model
├── drt_scale1.pt              # 1.7 MB, Scale 1 frozen-encoder DRT head
├── drt_ablation_no_decorr_no_dropout.pt  # Scale 1 ablation
└── drt_train.pt               # Scale 1 dev-split-80/20 alt run
```

Skipped (regeneratable from checkpoints by re-running `scripts.diagnose encode` on an A100, ~75 min):
- `baseline_corpus_emb.npy` (8.84 M × 384, fp16, 6.4 GB)
- `drt_corpus_subs.npy` (8.84 M × 6 × 64, fp16, 6.4 GB)
- DRT query encodings (raw + subs + alphas)

### Appendix G — References (direct precursors)

These should be cited in any paper writeup. Bibliographic details are in the blueprint's reference section.

- Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.*
- Humeau et al., 2020 — *Poly-encoders.*
- Zbontar et al., 2021 — *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.*
- Bardes et al., 2022 — *VICReg.*
- Kusupati et al., 2022 — *Matryoshka Representation Learning.*
- Karpukhin et al., 2020 — *Dense Passage Retrieval (DPR).*
- Xiong et al., 2021 — *ANCE.*
- Izacard et al., 2022 — *Contriever.*
- Thakur et al., 2021 — *BEIR.*
- Kornblith et al., 2019 — *Similarity of Neural Network Representations Revisited (CKA).*
- Locatello et al., 2019 — *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations* (cautionary precedent for the decorrelation-doesn't-imply-specialization finding).

---

*End of `PAPER_NOTES.md`. Total length: ~580 lines / ~38 KB. Source files: `results/diagnostics/*.json`, `results/logs/*`, `DRT_Research_Blueprint.html`, `DRT_Scale2_Prompt.md`, and the code in this repository at commit `768389f`.*
