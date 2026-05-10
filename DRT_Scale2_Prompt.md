# DRT Scale 2 — End-to-End Training on A100

## Context

Read `DRT_Research_Blueprint.html` for full project context. We completed Scale 1 (frozen encoder + decomposition head on Mac M4). Results: DRT lost to cosine by ~7-8% MRR@10 across all runs. Diagnosis: the frozen MiniLM encoder produces embeddings optimized for cosine — a small head can't restructure them. The architecture, losses, and training loop all work correctly. Decorrelation loss converges, slots decorrelate, attention head produces meaningful weights. The problem is the encoder never learned to produce decomposable embeddings.

Scale 2 fixes this by unfreezing the encoder and training end-to-end.

## What to Build

Restructure the training pipeline for end-to-end training. The key change: the encoder is no longer frozen. Passages and queries are encoded on the fly during training, not loaded from precomputed numpy arrays.

### Architecture (same as Scale 1, but encoder is trainable)

```
Input text → MiniLM-L6-v2 (UNFROZEN, 22.7M params) → 384-dim → DecompositionHead → 6 × 64 sub-vectors
                                                      384-dim → QueryAttentionHead → 6 attention weights
Score = Σ αᵢ · cos(qᵢ, dᵢ)
```

### Training Configuration

```yaml
# Model
encoder: sentence-transformers/all-MiniLM-L6-v2
encoder_frozen: false
num_slots: 6
slot_dim: 64

# Optimization — differential learning rates
encoder_lr: 5e-5
head_lr: 2e-3
optimizer: AdamW
weight_decay: 0.01
scheduler: cosine with warmup
warmup_ratio: 0.1
fp16: true  # mixed precision

# Data
dataset: MS MARCO passage ranking (full train set)
train_queries: ~502K (queries.train.tsv)
passages: ~8.8M (collection.tsv)
negatives: BM25 hard negatives (from MS MARCO official) + in-batch negatives
eval: MS MARCO dev (6,980 queries)

# Training
batch_size: 512  # per GPU, adjust if OOM — try 256 with gradient_accumulation_steps: 2
epochs: 5
max_seq_length: 256
num_hard_negatives: 7  # per query, from BM25 negatives file

# Losses (same as Scale 1)
lambda_decorrelation: 0.1
slot_dropout_p: 0.15
temperature: 0.05  # learnable, initialize at 0.05

# Hardware
device: cuda
gpu: A100 80GB
```

### Key Implementation Changes from Scale 1

1. **Online encoding**: No precomputed embeddings. Each training step tokenizes and encodes a batch of queries + passages through MiniLM. The encoder parameters receive gradients.

2. **Differential learning rates**: Use two param groups in the optimizer:
   - `{"params": encoder.parameters(), "lr": 5e-5}`
   - `{"params": [decomposition_head.parameters(), attention_head.parameters()], "lr": 2e-3}`

3. **Hard negatives loading**: Download the MS MARCO BM25 hard negatives file (`qidpidtriples.train.small.tsv` or equivalent). Each training example becomes (query, positive_passage, negative_passage_1, ..., negative_passage_7) plus in-batch negatives.

4. **Mixed precision**: Wrap the training loop with `torch.cuda.amp.GradScaler` and `autocast` for fp16 training. Critical for fitting batch_size 512 on A100.

5. **Gradient checkpointing**: Enable on the encoder if memory is tight: `encoder.gradient_checkpointing_enable()`.

6. **Passage encoding for eval**: At evaluation time, encode all ~500K passages in the sub-corpus through the trained encoder + decomposition head. Build a FAISS index on the concatenated sub-vectors. Retrieve top-100 per query, then rerank with full DRT scoring (weighted slots).

### Training Loop Pseudocode

```python
for epoch in range(5):
    for batch in train_dataloader:
        # batch contains: query_texts, pos_passage_texts, neg_passage_texts

        with autocast():
            # Encode everything through the UNFROZEN encoder
            q_emb = encoder(query_texts)          # (B, 384)
            p_emb = encoder(pos_passage_texts)     # (B, 384)
            n_emb = encoder(neg_passage_texts)     # (B*num_neg, 384)

            # Decompose
            q_subs = decomposition_head(q_emb)     # (B, 6, 64)
            p_subs = decomposition_head(p_emb)     # (B, 6, 64)
            n_subs = decomposition_head(n_emb)     # (B*num_neg, 6, 64)

            # Query-adaptive weights
            q_alphas = attention_head(q_emb)       # (B, 6)

            # Slot dropout (training only)
            q_subs = slot_dropout(q_subs)
            p_subs = slot_dropout(p_subs)
            n_subs = slot_dropout(n_subs)

            # Compute DRT scores
            pos_scores = drt_score(q_subs, q_alphas, p_subs)
            neg_scores = drt_score(q_subs, q_alphas, n_subs)

            # Losses
            L_retrieval = infonce_loss(pos_scores, neg_scores, temperature)
            L_decorr = decorrelation_loss(torch.cat([q_subs, p_subs]))
            L_total = L_retrieval + 0.1 * L_decorr

        scaler.scale(L_total).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # Evaluate on dev after each epoch
    evaluate(model, dev_queries, dev_qrels, passage_corpus)
```

### Evaluation

After training, run full evaluation:

1. **MS MARCO dev**: MRR@10, nDCG@10, Recall@100
2. **Cosine baseline**: Train a standard bi-encoder with the same MiniLM encoder, same data, same epochs, cosine + InfoNCE loss. This is the fair comparison — same encoder, same data, different scoring.
3. **Report both**: DRT vs cosine-trained bi-encoder on the same eval set.

### Ablations to Run After Main Training

After the main model is trained, run these ablations (each is a separate training run):

1. **k = {2, 4, 6, 8, 12}** — vary number of sub-vectors
2. **No decorrelation** — λ₁ = 0, keep everything else
3. **No attention** — uniform weights (1/k) instead of learned attention
4. **No slot dropout** — p = 0
5. **Decompose only** — no decorrelation, no attention, no dropout

### Success Criteria

- DRT beats the cosine-trained bi-encoder by ≥ 2% MRR@10 on MS MARCO dev
- If it beats by ≥ 3%, proceed to full BEIR benchmark
- If it loses, analyze which ablation variant performs best and report as findings

### File Organization

```
drt/
├── configs/
│   └── scale2.yaml
├── data/
│   ├── download_full.py        # Downloads full MS MARCO train + passages + hard negatives
│   └── dataset_online.py       # PyTorch Dataset that returns raw text (not precomputed embeddings)
├── models/
│   ├── encoder.py              # Encoder wrapper with freeze/unfreeze support
│   ├── decomposition.py        # Same DecompositionHead as Scale 1
│   ├── attention.py            # Same QueryAttentionHead as Scale 1
│   └── drt_model.py            # Full end-to-end model combining all components
├── losses/                     # Same as Scale 1
├── training/
│   ├── trainer_e2e.py          # End-to-end trainer with mixed precision
│   └── cosine_baseline.py      # Train standard bi-encoder for fair comparison
├── evaluation/
│   └── evaluate_e2e.py         # Encode corpus + retrieve + score
└── scripts/
    ├── train_scale2.py         # Entry point
    ├── train_baseline.py       # Entry point for cosine baseline
    └── run_ablations.py        # Run all ablation variants
```

## Execution Order

1. Download full MS MARCO data (train queries, collection, hard negatives, dev queries, dev qrels)
2. Train the cosine baseline bi-encoder first (this is your comparison target)
3. Train the full DRT model
4. Evaluate both on MS MARCO dev
5. Print comparison table
6. If DRT wins by ≥ 2%, run ablations
