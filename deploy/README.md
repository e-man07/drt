# DRT Scale 2 — Akash deploy

End-to-end DRT training on Akash A100 80GB. The SDL boots a container that
auto-runs the full pipeline (download → cosine baseline → DRT → eval) and
also exposes SSH + JupyterLab for live access. Everything lands on a 200Gi
persistent volume, so a redeploy resumes from the last completed step.

## What gets deployed

`scale2.sdl.yml` provisions one container with:

- **GPU**: 1× NVIDIA A100 (80GB profile)
- **CPU/RAM**: 16 cores, 64Gi
- **Storage**:
  - 100Gi root (ephemeral)
  - 200Gi `/workspace` — **persistent** (`class: beta3`). Holds MS MARCO data,
    venv, HF cache, pip cache, checkpoints, logs.
  - 1Gi `/dev/shm` (ephemeral)
- **Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **Ports**: 22 (SSH), 8888 (JupyterLab) — both globally exposed.

Persistence is what matters: Scale 2 is multi-day training. If the provider
restarts the container, the venv survives, the corpus survives, the DRT
checkpoint from epoch 3 survives. The pipeline picks up where it left off
because each completed step writes a marker under `/workspace/.done/`.

## Deploy procedure

### 1. Push the drt repo somewhere the Akash container can `git clone`

The SDL needs `DRT_REPO_URL` to be reachable from inside the container. Either
make the repo public, or use a deploy-key approach (set up via SSH and a
volume-mounted private key).

### 2. Generate secrets

```bash
openssl rand -base64 24       # → SSH_PASSWORD
# HUGGING_FACE_HUB_TOKEN — create at https://huggingface.co/settings/tokens
```

### 3. Paste SDL into Akash Console

1. Console: <https://console.akash.network/> → "Deploy" → "Build your template"
2. Paste `deploy/scale2.sdl.yml`
3. Fill the env vars in the Console UI (do NOT commit them to git):
   - `SSH_PASSWORD`
   - `HUGGING_FACE_HUB_TOKEN`
   - `DRT_REPO_URL`
   - `DRT_REPO_REF` (commit SHA / tag / `main`)
   - optional: `WANDB_API_KEY`, `WANDB_PROJECT`
4. Click "Create Deployment", review bids, accept one with an A100.

### 4. Watch boot

The container takes ~5 min on first boot to apt-install, build the venv,
pull pip wheels, clone the repo, and start the pipeline. After that:

- `https://<provider>:8888` — JupyterLab (no auth — keep this short-lived,
  destroy the deployment when finished)
- `ssh -p <port> root@<provider>` — shell in
- Inside the container:
  - `tail -f /workspace/logs/pipeline.log` — running output
  - `ls /workspace/.done/` — completed pipeline steps
  - `nvidia-smi` — GPU activity

### 5. Retrieve results

After eval finishes (`/workspace/.done/04-eval` exists):

```bash
# From your laptop
scp -P <port> root@<provider>:/workspace/logs/comparison.txt ./
scp -P <port> root@<provider>:/workspace/checkpoints/drt_scale2_epoch5.pt ./
scp -P <port> root@<provider>:/workspace/checkpoints/cosine_baseline_epoch5.pt ./
```

Then close the deployment in the Console.

## What persists vs. what doesn't

| Path | Persists across restart? | Notes |
|---|---|---|
| `/workspace/data/` | ✅ | Raw MS MARCO; ~10 GB after download |
| `/workspace/cache/huggingface/` | ✅ | Model weights cache; ~1 GB |
| `/workspace/cache/pip/` | ✅ | Wheel cache; cuts re-deploy time to ~30 sec |
| `/workspace/venv/` | ✅ | Python venv with all training deps |
| `/workspace/checkpoints/` | ✅ | All epoch checkpoints, ~500 MB total |
| `/workspace/logs/` | ✅ | Training logs + comparison table |
| `/workspace/.done/` | ✅ | Step markers (skip already-completed steps) |
| `/workspace/drt/` | ✅ | The cloned repo |
| `/dev/shm` | ❌ | Ephemeral RAM disk for DataLoader workers |
| Container root FS (`/usr`, `/etc`, `/root`) | ❌ | Lost on restart |

If you want to **force a fresh run**: SSH in and `rm -rf /workspace/.done/*`.
That re-runs every step on the next pipeline launch.

## Pipeline flow (auto-run)

`scripts/run_pipeline.sh` runs in the background after boot:

1. **download** — `data/download_full.py`. ~30 min first time, instant after.
2. **cosine baseline** — `scripts/train_baseline.py`. ~24-30 hr A100.
3. **DRT** — `scripts/train_scale2.py`. ~24-30 hr A100.
4. **eval** — `scripts/evaluate_e2e.py`. Writes `logs/comparison.txt`.

Total wall-clock first run: ~50-60 hours. The persistent volume means a
redeploy doesn't lose this — restarted containers pick up at the next
unfinished step.

## Troubleshooting

- **"Bid not found / no providers offering A100"** — A100 80GB on Akash is
  scarce and pricier than H100. Adjust the `pricing.training.amount` in the
  SDL; current value (`100000` uakt) may be too low. Or switch to H100 by
  changing `model: a100` → `model: h100`.
- **"OOM at batch 512"** — drop `batch_size` in `configs/scale2.yaml` to 256
  and set `grad_accum_steps: 2`. Restart with `rm /workspace/.done/02-baseline /workspace/.done/03-drt`
  to redo training (downloads stay).
- **Pipeline stuck mid-train** — SSH in, `tail -f /workspace/logs/03-drt.console.log`.
  If it's silent, kill the python process; the next pipeline launch will
  retry from the last completed epoch checkpoint (the trainer reads epoch
  numbers off the saved files).
- **Need to redo everything** — `rm -rf /workspace/.done /workspace/checkpoints`
  then trigger pipeline again (`bash /workspace/drt/scripts/run_pipeline.sh`).

## Cost estimate

A100 80GB on Akash typically bids around 100-200 µAKT/block. At ~6-block-per-
minute and ~$3 per AKT (varies), one 24-hour training is roughly **$20-50**.
Two trainings + ~6 hr of misc → **~$50-150 per full Scale-2 run**. Cheap vs.
the comparable AWS p4d.24xlarge ($30+/hr).

Verify pricing in the Console "Bids" panel before accepting.
