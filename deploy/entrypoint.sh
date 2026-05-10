#!/bin/bash
# DRT Scale 2 — Akash entrypoint.
#
# This is the canonical version of the bash that runs in scale2.sdl.yml's
# `args:` field. The SDL inlines this same logic so Akash containers can
# bootstrap without a custom image, but if you want to tweak the boot
# sequence locally (Docker), source this script.

set -e

WORKSPACE="${WORKSPACE:-/workspace}"

mkdir -p "${WORKSPACE}"/{data,checkpoints,logs,cache/huggingface,cache/pip,venv}
mkdir -p /run/sshd

# System deps (idempotent)
apt-get update -qq
apt-get install -y -qq openssh-server git wget curl > /dev/null 2>&1
rm -rf /var/lib/apt/lists/*

# SSH — fail-closed
if [ -z "${SSH_PASSWORD:-}" ] || [ "${SSH_PASSWORD}" = "<set-in-console>" ]; then
    echo "ERROR: SSH_PASSWORD must be set" >&2
    exit 1
fi
echo "root:${SSH_PASSWORD}" | chpasswd
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
/usr/sbin/sshd

# venv on persistent volume
if [ ! -f "${WORKSPACE}/venv/pyvenv.cfg" ]; then
    python -m venv "${WORKSPACE}/venv" --system-site-packages
    "${WORKSPACE}/venv/bin/pip" install --upgrade pip
fi
# shellcheck disable=SC1091
source "${WORKSPACE}/venv/bin/activate"

pip install -q --cache-dir="${WORKSPACE}/cache/pip" \
    sentence-transformers datasets transformers tokenizers \
    huggingface-hub wandb pyyaml tqdm jupyterlab numpy

# Repo
if [ ! -d "${WORKSPACE}/drt/.git" ]; then
    git clone "${DRT_REPO_URL}" "${WORKSPACE}/drt"
fi
cd "${WORKSPACE}/drt"
git fetch --all
git checkout "${DRT_REPO_REF:-main}"
git pull --ff-only || true

# JupyterLab
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' \
    --notebook-dir="${WORKSPACE}" > "${WORKSPACE}/logs/jupyter.log" 2>&1 &

# Pipeline
chmod +x scripts/run_pipeline.sh
nohup bash scripts/run_pipeline.sh > "${WORKSPACE}/logs/pipeline.log" 2>&1 &

echo "=== DRT Scale 2 ready (SSH 22, Jupyter 8888) ==="
while true; do sleep 86400; done
