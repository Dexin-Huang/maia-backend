#!/usr/bin/env bash
set -euo pipefail

echo "==================================="
echo "SAM 3D Pipeline - Starting up"
echo "==================================="

# Show NVIDIA GPU info
echo "Checking for NVIDIA GPU..."
nvidia-smi || echo "Warning: nvidia-smi failed. GPU may not be available."

# Authenticate with Hugging Face if token is provided
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "Hugging Face token found, authenticating..."
    python3 - <<'PY'
import os
from huggingface_hub import login, snapshot_download

# Login to Hugging Face
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    try:
        login(token=token)
        print("✓ Authenticated with Hugging Face")
    except Exception as e:
        print(f"✗ HF authentication failed: {e}")

# Download gated models
cache_dir = os.environ.get("HF_HOME", "/models")

for repo in ["facebook/sam-3d-body-dinov3"]:
    try:
        print(f"Downloading {repo}...")
        snapshot_download(repo, cache_dir=cache_dir)
        print(f"✓ Downloaded {repo}")
    except Exception as e:
        print(f"✗ Failed to download {repo}: {e}")
PY
else
    echo "Warning: HUGGINGFACE_HUB_TOKEN not set. Skipping model download."
fi

echo "==================================="
echo "Starting FastAPI server..."
echo "==================================="

# Export CONDA_PREFIX for SAM 3D Objects compatibility
export CONDA_PREFIX=/usr/local

# Start uvicorn server
exec uvicorn app.sam3d_api:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --workers 1 \
    --log-level info
