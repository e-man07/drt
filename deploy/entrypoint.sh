#!/bin/bash
# DRT Scale 2 entrypoint, fetched by scale2.sdl.yml at container start.
#
# What it does, in order:
#   1. Install system deps (ssh, git)
#   2. Start sshd with the SSH_PASSWORD from env
#   3. Set up /workspace dirs + a persistent Python venv
#   4. Install Python deps into the venv (cached)
#   5. Clone / pull the DRT repo on the persistent volume
#   6. Start JupyterLab in the background
#   7. Kick off scripts/run_pipeline.sh in the background
#   8. Sleep forever so the container stays up for SSH inspection
#
# Persistent: /workspace/{data,checkpoints,logs,cache,venv,drt} survive
# restarts. /workspace/.done/ markers let the pipeline resume work.

set -e

WORKSPACE=/workspace

# 1. System deps (idempotent)
apt-get update -qq
apt-get install -y -qq openssh-server git wget curl > /dev/null 2>&1
rm -rf /var/lib/apt/lists/*

# 2. SSH (fail-closed)
mkdir -p /run/sshd
if [ -z "${SSH_PASSWORD}" ] || [ "${SSH_PASSWORD}" = "<set-in-console>" ]; then
    echo "ERROR: SSH_PASSWORD must be set in Akash Console" >&2
    exit 1
fi
echo "root:${SSH_PASSWORD}" | chpasswd
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
/usr/sbin/sshd

# 3. Persistent dirs
mkdir -p "${WORKSPACE}/data" \
         "${WORKSPACE}/checkpoints" \
         "${WORKSPACE}/logs" \
         "${WORKSPACE}/cache/huggingface" \
         "${WORKSPACE}/cache/pip" \
         "${WORKSPACE}/venv"

# 4. venv on persistent volume (survives container restart)
if [ ! -f "${WORKSPACE}/venv/pyvenv.cfg" ]; then
    python -m venv "${WORKSPACE}/venv" --system-site-packages
    "${WORKSPACE}/venv/bin/pip" install --upgrade pip
fi
# shellcheck disable=SC1091
source "${WORKSPACE}/venv/bin/activate"

pip install -q --cache-dir="${WORKSPACE}/cache/pip" \
    sentence-transformers \
    datasets \
    transformers \
    tokenizers \
    huggingface-hub \
    wandb \
    pyyaml \
    tqdm \
    jupyterlab \
    numpy

# 5. Clone / refresh DRT repo on persistent volume
if [ ! -d "${WORKSPACE}/drt/.git" ]; then
    git clone "${DRT_REPO_URL}" "${WORKSPACE}/drt"
fi
cd "${WORKSPACE}/drt"
git fetch --all
git checkout "${DRT_REPO_REF:-main}"
git pull --ff-only || true

# 6. JupyterLab in background
nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir="${WORKSPACE}" \
    > "${WORKSPACE}/logs/jupyter.log" 2>&1 &

# 7. Pipeline (auto-run, idempotent on redeploy)
chmod +x scripts/run_pipeline.sh
nohup bash scripts/run_pipeline.sh > "${WORKSPACE}/logs/pipeline.log" 2>&1 &

echo "=== DRT Scale 2 container ready ==="
echo "  SSH:     port 22  (root / ${SSH_PASSWORD:0:3}***)"
echo "  Jupyter: port 8888 (no token)"
echo "  Pipeline log:    ${WORKSPACE}/logs/pipeline.log"
echo "  Comparison out:  ${WORKSPACE}/logs/comparison.txt"

# 8. Stay alive for inspection
while true; do sleep 86400; done
