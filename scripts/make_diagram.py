"""Generate DRT_architecture.excalidraw — paper-ready diagram of the full
research arc: architecture, training pipeline, results, and diagnostic
findings.

Run: `python3 scripts/make_diagram.py` → writes DRT_architecture.excalidraw
in the repo root. Import into excalidraw.com or the VS Code extension.
"""
from __future__ import annotations

import json
import random
import string
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "DRT_architecture.excalidraw"

random.seed(42)
NOW_MS = int(time.time() * 1000)

# ─────────────── helpers ───────────────


def _id(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _seed() -> int:
    return random.randint(1, 2**31 - 1)


def _base(elem_type: str, x: float, y: float, w: float, h: float, **overrides):
    out = {
        "id": _id(),
        "type": elem_type,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": _seed(),
        "version": 1,
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": NOW_MS,
        "link": None,
        "locked": False,
    }
    out.update(overrides)
    return out


def rect(x, y, w, h, bg="#a5d8ff", stroke="#1971c2", **kw):
    return _base("rectangle", x, y, w, h, backgroundColor=bg, strokeColor=stroke, **kw)


def diamond(x, y, w, h, bg="#fff3bf", stroke="#f08c00", **kw):
    return _base("diamond", x, y, w, h, backgroundColor=bg, strokeColor=stroke, **kw)


def ellipse(x, y, w, h, bg="#d3f9d8", stroke="#2f9e44", **kw):
    return _base("ellipse", x, y, w, h, backgroundColor=bg, strokeColor=stroke, **kw)


def text(
    x: float,
    y: float,
    content: str,
    size: int = 16,
    color: str = "#1e1e1e",
    align: str = "center",
    family: int = 1,
    container_id: str | None = None,
    width: float | None = None,
    height: float | None = None,
):
    """Standalone text element (not bound to a container)."""
    lines = content.split("\n")
    # Approximate text dimensions
    line_h = int(size * 1.25)
    w = width if width is not None else max(len(line) for line in lines) * (size * 0.6) + 20
    h = height if height is not None else line_h * len(lines)
    return _base(
        "text",
        x,
        y,
        w,
        h,
        strokeColor=color,
        fillStyle="solid",
        text=content,
        fontSize=size,
        fontFamily=family,
        textAlign=align,
        verticalAlign="middle",
        baseline=int(size * 0.85),
        containerId=container_id,
        originalText=content,
        lineHeight=1.25,
        roundness=None,
    )


def labeled_box(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    bg: str = "#a5d8ff",
    stroke: str = "#1971c2",
    font_size: int = 16,
    text_color: str = "#1e1e1e",
):
    """A rectangle with centered text bound to it."""
    box = rect(x, y, w, h, bg=bg, stroke=stroke)
    txt = text(x, y, label, size=font_size, color=text_color, container_id=box["id"], width=w, height=h)
    box["boundElements"] = [{"id": txt["id"], "type": "text"}]
    return [box, txt]


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    start_id: str | None = None,
    end_id: str | None = None,
    stroke: str = "#1e1e1e",
    width: int = 2,
    label: str | None = None,
    label_size: int = 14,
):
    """Straight arrow from (x1,y1) to (x2,y2). Optionally bound to elements."""
    a = _base(
        "arrow",
        min(x1, x2),
        min(y1, y2),
        abs(x2 - x1),
        abs(y2 - y1),
        strokeColor=stroke,
        strokeWidth=width,
        fillStyle="solid",
        backgroundColor="transparent",
        points=[[0, 0], [x2 - x1, y2 - y1]],
        lastCommittedPoint=[x2 - x1, y2 - y1],
        startBinding=({"elementId": start_id, "focus": 0, "gap": 4} if start_id else None),
        endBinding=({"elementId": end_id, "focus": 0, "gap": 4} if end_id else None),
        startArrowhead=None,
        endArrowhead="arrow",
        roundness={"type": 2},
        elbowed=False,
    )
    # arrow's actual stored x/y is its origin; points are relative
    a["x"] = x1
    a["y"] = y1
    a["points"] = [[0, 0], [x2 - x1, y2 - y1]]
    elements = [a]
    if label:
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        t = text(midx, midy - label_size, label, size=label_size, color=stroke)
        elements.append(t)
    return elements


# ─────────────── layout helpers ───────────────

PAGE_W = 2400
COL_PAD = 40

# Color palette
C_INPUT = "#ffec99"
C_INPUT_BORDER = "#f08c00"
C_ENCODER = "#a5d8ff"
C_ENCODER_BORDER = "#1971c2"
C_HEAD = "#d0bfff"
C_HEAD_BORDER = "#7048e8"
C_OUTPUT = "#d3f9d8"
C_OUTPUT_BORDER = "#2f9e44"
C_LOSS = "#ffc9c9"
C_LOSS_BORDER = "#e03131"
C_RESULT_GOOD = "#d3f9d8"
C_RESULT_GOOD_BORDER = "#2f9e44"
C_RESULT_BAD = "#ffc9c9"
C_RESULT_BAD_BORDER = "#e03131"
C_DIAG = "#fff3bf"
C_DIAG_BORDER = "#f08c00"
C_NEUTRAL = "#f1f3f5"
C_NEUTRAL_BORDER = "#868e96"


elements: list[dict] = []


def add(*items):
    for it in items:
        if isinstance(it, list):
            elements.extend(it)
        else:
            elements.append(it)


# ═════════════════════════════════════════════════════════════
# SECTION 1 — TITLE
# ═════════════════════════════════════════════════════════════
y = 40
add(text(PAGE_W // 2 - 600, y, "DRT — Decomposed Relevance Tensors", size=36, color="#1971c2", width=1200))
y += 60
add(
    text(
        PAGE_W // 2 - 700,
        y,
        "Architecture, training, results, and diagnostic findings",
        size=20,
        color="#495057",
        width=1400,
    )
)
y += 50


# ═════════════════════════════════════════════════════════════
# SECTION 2 — ARCHITECTURE
# ═════════════════════════════════════════════════════════════
y += 30
add(text(80, y, "▍ Architecture", size=26, color="#1971c2", align="left", width=400))
y += 50

# Input
input_box = labeled_box(160, y, 200, 80, "Input text\n(query or passage)", bg=C_INPUT, stroke=C_INPUT_BORDER)
add(input_box)
input_id = input_box[0]["id"]

# Encoder
enc_box = labeled_box(
    480, y, 320, 80, "MiniLM Encoder\nall-MiniLM-L6-v2 (22.7M params)\n→ 384-d embedding",
    bg=C_ENCODER, stroke=C_ENCODER_BORDER, font_size=15,
)
add(enc_box)
enc_id = enc_box[0]["id"]
add(arrow(360, y + 40, 480, y + 40, start_id=input_id, end_id=enc_id))

# Decomposition head
decomp_box = labeled_box(
    920, y - 80, 360, 100,
    "Decomposition Head (~394K)\nLinear(384,512)→LN→GELU\nLinear(512,384)→LN→GELU\nReshape → (k=6, d=64) → L2-norm",
    bg=C_HEAD, stroke=C_HEAD_BORDER, font_size=13,
)
add(decomp_box)
decomp_id = decomp_box[0]["id"]
add(arrow(800, y + 30, 920, y - 30, start_id=enc_id, end_id=decomp_id))

# Attention head
attn_box = labeled_box(
    920, y + 80, 360, 100,
    "Query Attention Head (~25K)\nLinear(384,64)→GELU\nLinear(64,6)\n→ softmax → α∈ℝ⁶, Σα=1",
    bg=C_HEAD, stroke=C_HEAD_BORDER, font_size=13,
)
add(attn_box)
attn_id = attn_box[0]["id"]
add(arrow(800, y + 50, 920, y + 130, start_id=enc_id, end_id=attn_id))

# Outputs
subs_box = labeled_box(
    1400, y - 80, 240, 100, "Sub-vectors\n(B, k=6, d=64)\n‖sᵢ‖ = 1",
    bg=C_OUTPUT, stroke=C_OUTPUT_BORDER, font_size=14,
)
add(subs_box)
subs_id = subs_box[0]["id"]
add(arrow(1280, y - 30, 1400, y - 30, start_id=decomp_id, end_id=subs_id))

alpha_box = labeled_box(
    1400, y + 80, 240, 100, "Slot weights\nα ∈ ℝ⁶\nquery-adaptive",
    bg=C_OUTPUT, stroke=C_OUTPUT_BORDER, font_size=14,
)
add(alpha_box)
alpha_id = alpha_box[0]["id"]
add(arrow(1280, y + 130, 1400, y + 130, start_id=attn_id, end_id=alpha_id))

# Scoring
score_box = labeled_box(
    1760, y, 480, 100,
    "DRT score(q,d)  =  Σᵢ αᵢ(q) · (qᵢ · dᵢ)\nweighted sum of slot-level cosines",
    bg=C_NEUTRAL, stroke=C_NEUTRAL_BORDER, font_size=15,
)
add(score_box)
score_id = score_box[0]["id"]
add(arrow(1640, y - 30, 1760, y + 30, start_id=subs_id, end_id=score_id))
add(arrow(1640, y + 130, 1760, y + 70, start_id=alpha_id, end_id=score_id))

y += 220


# ═════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING LOSSES
# ═════════════════════════════════════════════════════════════
y += 40
add(text(80, y, "▍ Training losses", size=26, color="#1971c2", align="left", width=400))
y += 50

# Three loss boxes
loss_w = 480
loss_h = 130
loss_gap = 40
loss_x_start = 160

infonce_box = labeled_box(
    loss_x_start, y, loss_w, loss_h,
    "L_retrieval (InfoNCE)\n\nfor each query q in batch B:\n   pos = DRT score with positive doc\n   negs = scores with in-batch + BM25 hard\n   cross-entropy(logits/τ, target=0)\n\nτ = 0.05",
    bg=C_LOSS, stroke=C_LOSS_BORDER, font_size=13,
)
add(infonce_box)

decorr_box = labeled_box(
    loss_x_start + (loss_w + loss_gap), y, loss_w, loss_h,
    "L_decorrelation (Barlow Twins)\n\nfor all pairs (i,j), i<j:\n   C[i,j] = sub[:,i]ᵀ sub[:,j] / B   (d×d)\n   penalty += ‖C[i,j]‖²_F\nloss = mean over pairs\n\nλ_dec = 0.1",
    bg=C_LOSS, stroke=C_LOSS_BORDER, font_size=13,
)
add(decorr_box)

dropout_box = labeled_box(
    loss_x_start + 2 * (loss_w + loss_gap), y, loss_w, loss_h,
    "Slot dropout (regularizer)\n\nper sample, mask each slot ~ Bernoulli(1-p)\nrescale by 1/E[mask]\napplied BEFORE retrieval loss\n(decorrelation sees raw subs)\n\np = 0.15",
    bg=C_LOSS, stroke=C_LOSS_BORDER, font_size=13,
)
add(dropout_box)

y += loss_h + 30

# Total loss equation
add(
    text(
        loss_x_start,
        y,
        "L_total  =  L_retrieval  +  0.1 · L_decorrelation       (slot dropout applied as structural regularizer)",
        size=18,
        color="#1e1e1e",
        align="left",
        width=loss_x_start + 3 * (loss_w + loss_gap) - 200,
    )
)

y += 60


# ═════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING TWO STAGES
# ═════════════════════════════════════════════════════════════
y += 30
add(text(80, y, "▍ Training stages", size=26, color="#1971c2", align="left", width=500))
y += 50

stage_w = 780
stage_h = 200

scale1_box = labeled_box(
    160, y, stage_w, stage_h,
    "SCALE 1 — Proof of Concept (Mac M4 MPS)\n\n• Encoder FROZEN\n• 500K corpus subsample (random, dev-positives preserved)\n• Dev queries split 80/20 (5,584 train / 1,396 eval)\n• 5,951 training pairs (in-batch negs only)\n• 20 epochs × 46 steps  =  920 optim steps\n• Wall-clock: ~10 sec training, ~20 sec eval\n• 419K trainable params (heads only)",
    bg=C_NEUTRAL, stroke=C_NEUTRAL_BORDER, font_size=14,
)
add(scale1_box)

scale2_box = labeled_box(
    160 + stage_w + 80, y, stage_w, stage_h,
    "SCALE 2 — End-to-End on A100 80GB (Akash)\n\n• Encoder UNFROZEN (differential LR: enc=5e-5, head=2e-3)\n• Full 8.84M-passage corpus + 502K train queries\n• 397.8M BM25 hard-neg triples → 418,010 train tuples\n• 5 epochs × 816 steps  =  4,080 optim steps\n• Mixed-precision fp16 + gradient checkpointing\n• Wall-clock: ~2 hr per training, ~50 min eval\n• 23.1M trainable params (full encoder + heads)",
    bg=C_NEUTRAL, stroke=C_NEUTRAL_BORDER, font_size=14,
)
add(scale2_box)

y += stage_h + 50


# ═════════════════════════════════════════════════════════════
# SECTION 5 — HEADLINE RESULTS
# ═════════════════════════════════════════════════════════════
y += 30
add(text(80, y, "▍ Headline results — DRT lost to baseline at both scales", size=26, color="#e03131", align="left", width=1400))
y += 60

# Results table
tbl_x = 160
tbl_y = y
col_w = [280, 280, 280, 280, 280]
row_h = 60
n_cols = len(col_w)
x_positions = [tbl_x]
for w in col_w[:-1]:
    x_positions.append(x_positions[-1] + w)

# Header row
header_bg = "#dee2e6"
for i, (hdr, w) in enumerate(zip(["", "MRR@10", "nDCG@10", "Recall@100", "Δ MRR@10"], col_w)):
    add(labeled_box(x_positions[i], tbl_y, w, row_h, hdr, bg=header_bg, stroke="#495057", font_size=18))

# Scale 1 baseline row
y_r = tbl_y + row_h
add(labeled_box(x_positions[0], y_r, col_w[0], row_h, "Scale 1 — Cosine BL", bg=C_RESULT_GOOD, stroke=C_RESULT_GOOD_BORDER, font_size=16))
for i, val in enumerate(["0.6951", "0.7383", "0.9800", "—"]):
    add(labeled_box(x_positions[i + 1], y_r, col_w[i + 1], row_h, val, bg="#ffffff", stroke="#adb5bd", font_size=16))

# Scale 1 DRT row
y_r += row_h
add(labeled_box(x_positions[0], y_r, col_w[0], row_h, "Scale 1 — DRT", bg=C_RESULT_BAD, stroke=C_RESULT_BAD_BORDER, font_size=16))
for i, val in enumerate(["0.6236", "0.6663", "0.9396", "−0.0715"]):
    bg = "#ffe3e3" if i == 3 else "#ffffff"
    add(labeled_box(x_positions[i + 1], y_r, col_w[i + 1], row_h, val, bg=bg, stroke="#adb5bd", font_size=16))

# Scale 2 baseline
y_r += row_h
add(labeled_box(x_positions[0], y_r, col_w[0], row_h, "Scale 2 — Cosine BL", bg=C_RESULT_GOOD, stroke=C_RESULT_GOOD_BORDER, font_size=16))
for i, val in enumerate(["0.3278", "0.3884", "0.8600", "—"]):
    add(labeled_box(x_positions[i + 1], y_r, col_w[i + 1], row_h, val, bg="#ffffff", stroke="#adb5bd", font_size=16))

# Scale 2 DRT
y_r += row_h
add(labeled_box(x_positions[0], y_r, col_w[0], row_h, "Scale 2 — DRT", bg=C_RESULT_BAD, stroke=C_RESULT_BAD_BORDER, font_size=16))
for i, val in enumerate(["0.3074", "0.3648", "0.8315", "−0.0204"]):
    bg = "#ffe3e3" if i == 3 else "#ffffff"
    add(labeled_box(x_positions[i + 1], y_r, col_w[i + 1], row_h, val, bg=bg, stroke="#adb5bd", font_size=16))

y_r += row_h + 20
add(
    text(
        tbl_x,
        y_r,
        "Success criterion was Δ MRR@10  ≥  +0.02 (blueprint). Got −0.02.  Result is robust across scales.",
        size=16,
        color="#e03131",
        align="left",
        width=1500,
    )
)

y = y_r + 60


# ═════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSTIC FINDINGS (4 PANELS)
# ═════════════════════════════════════════════════════════════
y += 50
add(text(80, y, "▍ 4-step diagnostic — why DRT lost", size=26, color="#1971c2", align="left", width=900))
y += 50

# 2×2 grid of finding panels
panel_w = 1080
panel_h = 240
panel_gap_x = 60
panel_gap_y = 50
panel_x_start = 160

def find_panel(col, row, title, body, accent="#e03131"):
    x = panel_x_start + col * (panel_w + panel_gap_x)
    yy = y + row * (panel_h + panel_gap_y)
    border = labeled_box(x, yy, panel_w, 50, title, bg=C_DIAG, stroke=accent, font_size=18, text_color=accent)
    add(border)
    add(
        text(
            x + 20,
            yy + 60,
            body,
            size=14,
            color="#1e1e1e",
            align="left",
            width=panel_w - 40,
            height=panel_h - 70,
        )
    )

find_panel(
    0,
    0,
    "1. The decomposition machinery is a NO-OP",
    """\
Three independent scoring variants on the same trained DRT checkpoint:

  drt_learned_alphas    →  MRR@10 = 0.3075
  drt_uniform_alphas    →  MRR@10 = 0.3076
  drt_concat_cosine     →  MRR@10 = 0.3076

Within 0.0001 of each other.

The decomposition + attention head adds zero discriminative power
beyond what plain cosine over the same encoder's output produces.""",
    accent="#e03131",
)

find_panel(
    1,
    0,
    "2. The attention head learned NOTHING",
    """\
Learned α  vs  uniform α (1/k):
   MRR@10 0.3075   vs   0.3076   (Δ < 0.0001)

The trained softmax over 6 slots is, for retrieval purposes,
equivalent to a uniform distribution. Query-adaptive weighting
is the second pillar of DRT — and it's a no-op.""",
    accent="#e03131",
)

find_panel(
    0,
    1,
    "3. Slots are statistically uncorrelated but semantically interchangeable",
    """\
Single-slot MRR@10 (using only slot i, αi=1, others=0):
   slot 0..5  →  0.2611 / 0.2577 / 0.2611 / 0.2611 / 0.2561 / 0.2608
   range:  0.0050

Per-slot probe (logistic reg, factoid classification):
   slots 0..5  →  0.759 / 0.761 / 0.769 / 0.774 / 0.774 / 0.761
   range:  0.015  (concat 6×64 reaches 0.813)

Decorrelation made them statistically independent, NOT semantically distinct.""",
    accent="#e03131",
)

find_panel(
    1,
    1,
    "4. The encoder drifted (didn't break)",
    """\
Linear CKA on 6,980 dev queries:
   baseline encoder  vs  DRT raw encoder      :  0.9452
   baseline encoder  vs  DRT concat sub-vecs  :  0.8297

The encoder is 94.5% similar to the cosine-trained one — a small
shift, but enough to cost 2% MRR@10. DRT loses HARDER on factoid
queries (79 of top-100 losses are factoid vs 59 of top-100 wins).""",
    accent="#e03131",
)

y += 2 * panel_h + panel_gap_y + 80


# ═════════════════════════════════════════════════════════════
# SECTION 7 — TAKEAWAYS
# ═════════════════════════════════════════════════════════════
y += 30
add(text(80, y, "▍ Takeaways", size=26, color="#1971c2", align="left", width=400))
y += 50

takeaway_box = labeled_box(
    160,
    y,
    2240,
    230,
    """Both load-bearing claims of DRT are refuted by the data:
  • "Decomposed sub-vectors beat flat vectors"     →  concat cosine == DRT (Finding 1)
  • "Query-adaptive weighting beats uniform"       →  learned α == uniform α (Finding 2)

The decorrelation loss made slots statistically uncorrelated without making them semantically specialized.
The encoder drifted ~5% under the decomposition + decorrelation pressure, losing ~2% MRR@10.

Paths forward:
  • Cheap closure: λ_dec = 0 ablation to confirm decorrelation is the source of encoder drift.
  • Pivot:         Reframe as methodology paper: "Statistical decorrelation does not produce semantic specialization in retrieval embeddings."
  • Redesign:      Soft attention slotting (k learned attention heads over encoder tokens) instead of fixed reshape — gives the model a mechanism to differentiate slots.""",
    bg="#f8f9fa",
    stroke="#1971c2",
    font_size=16,
)
add(takeaway_box)


# ─────────────── final file ───────────────


doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://github.com/e-man07/drt — scripts/make_diagram.py",
    "elements": elements,
    "appState": {
        "gridSize": 20,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}


OUT.write_text(json.dumps(doc, indent=2))
print(f"Wrote {OUT}  ({len(elements)} elements, {OUT.stat().st_size:,} bytes)")
