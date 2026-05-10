#!/bin/bash
# DRT Scale 2 — full pipeline runner.
#
# Steps (each leaves a marker under /workspace/.done so reruns skip work):
#   1. Download MS MARCO (corpus, train queries, dev queries, qrels, hard negatives)
#   2. Train cosine baseline bi-encoder
#   3. Train DRT end-to-end
#   4. Evaluate both on dev, print comparison table
#
# Designed to run on the Akash A100 box, with /workspace as the persistent volume.
# Reruns after a redeploy resume from the last completed step.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DRT_DIR="${DRT_DIR:-${WORKSPACE}/drt}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/data}"
CKPT_DIR="${CKPT_DIR:-${WORKSPACE}/checkpoints}"
LOG_DIR="${LOG_DIR:-${WORKSPACE}/logs}"
DONE_DIR="${DONE_DIR:-${WORKSPACE}/.done}"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOG_DIR" "$DONE_DIR"

if [ -d "${WORKSPACE}/venv" ]; then
    # shellcheck disable=SC1091
    source "${WORKSPACE}/venv/bin/activate"
fi

cd "$DRT_DIR"
export PYTHONPATH="${DRT_DIR}:${PYTHONPATH:-}"

mark_done() { touch "${DONE_DIR}/$1"; }
is_done()   { [ -f "${DONE_DIR}/$1" ]; }

# ─────────── 1. DOWNLOAD ───────────
if is_done 01-download; then
    echo "[skip] download (marker present)"
else
    echo "[run] download MS MARCO (corpus + queries + qrels + hard negatives)"
    python -m data.download_full --out-dir "$DATA_DIR" --skip-if-exists 2>&1 \
        | tee "${LOG_DIR}/01-download.log"
    mark_done 01-download
fi

# ─────────── 2. COSINE BASELINE ───────────
if is_done 02-baseline; then
    echo "[skip] cosine baseline (marker present)"
else
    echo "[run] train cosine baseline"
    python -m scripts.train_baseline \
        --config configs/scale2.yaml \
        --data-dir "$DATA_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --log-path "${LOG_DIR}/02-baseline.log" 2>&1 \
        | tee -a "${LOG_DIR}/02-baseline.console.log"
    mark_done 02-baseline
fi

# ─────────── 3. DRT TRAINING ───────────
if is_done 03-drt; then
    echo "[skip] DRT training (marker present)"
else
    echo "[run] train DRT end-to-end"
    python -m scripts.train_scale2 \
        --config configs/scale2.yaml \
        --data-dir "$DATA_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --log-path "${LOG_DIR}/03-drt.log" 2>&1 \
        | tee -a "${LOG_DIR}/03-drt.console.log"
    mark_done 03-drt
fi

# ─────────── 4. EVALUATE ───────────
echo "[run] evaluate DRT vs cosine baseline on dev"
python -m scripts.evaluate_e2e \
    --checkpoint-dir "$CKPT_DIR" \
    --data-dir "$DATA_DIR" \
    --output "${LOG_DIR}/comparison.txt" 2>&1 \
    | tee "${LOG_DIR}/04-eval.log"
mark_done 04-eval

echo
echo "============================================================"
echo "Pipeline complete. Comparison table:"
echo "============================================================"
cat "${LOG_DIR}/comparison.txt"
