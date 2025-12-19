#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Nvwa Backend - RunPod Setup"
echo "=========================================="

# Check for required environment variables
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: HUGGINGFACE_HUB_TOKEN not set"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    echo "Set it with: export HUGGINGFACE_HUB_TOKEN=hf_your_token_here"
    exit 1
fi

# Create workspace directories
echo "Creating workspace directories..."
mkdir -p /workspace/{checkpoints,data,logs,repos}

# ============================================
# Clone SAM 3D Body (Meta's model)
# ============================================
if [[ ! -d "/workspace/repos/sam-3d-body" ]]; then
    echo "Cloning SAM 3D Body..."
    git clone https://github.com/facebookresearch/sam-3d-body.git /workspace/repos/sam-3d-body
    pip install -e /workspace/repos/sam-3d-body
else
    echo "SAM 3D Body already cloned"
fi

# ============================================
# Clone SAM-Body4D (Full pipeline)
# ============================================
if [[ ! -d "/workspace/repos/sam-body4d" ]]; then
    echo "Cloning SAM-Body4D..."
    git clone https://github.com/gaomingqi/sam-body4d.git /workspace/repos/sam-body4d

    # Install SAM-Body4D dependencies
    cd /workspace/repos/sam-body4d
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    cd -
else
    echo "SAM-Body4D already cloned"
fi

# ============================================
# Install Nvwa backend dependencies
# ============================================
echo "Installing Nvwa backend dependencies..."
pip install -r requirements.txt

# ============================================
# Download model checkpoints
# ============================================
echo "Downloading model checkpoints..."
python3 << 'EOF'
import os
from huggingface_hub import login, snapshot_download

token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if token:
    login(token=token)
    print("Logged in to HuggingFace")

# SAM 3D Body (required)
print("\n[1/4] Downloading SAM 3D Body...")
try:
    snapshot_download(
        "facebook/sam-3d-body-dinov3",
        cache_dir="/workspace/checkpoints",
        local_dir="/workspace/checkpoints/sam-3d-body-dinov3",
    )
    print("  ✓ SAM 3D Body downloaded")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  Make sure you have access to facebook/sam-3d-body-dinov3")

# SAM3 (for video segmentation / tracking)
print("\n[2/4] Downloading SAM3...")
try:
    # Note: SAM3 may not be publicly available yet
    # This will be updated when it's released
    # snapshot_download("facebook/sam3", ...)
    print("  ⚠ SAM3 not yet available - skipping")
except Exception as e:
    print(f"  ⚠ Skipped: {e}")

# Depth Anything V2 (optional - for depth estimation)
print("\n[3/4] Downloading Depth Anything V2...")
try:
    snapshot_download(
        "depth-anything/Depth-Anything-V2-Large",
        cache_dir="/workspace/checkpoints",
    )
    print("  ✓ Depth Anything V2 downloaded")
except Exception as e:
    print(f"  ⚠ Skipped (optional): {e}")

# MoGe (optional - for monocular geometry)
print("\n[4/4] Checking MoGe...")
try:
    # MoGe checkpoint location varies
    print("  ⚠ MoGe - manual download may be required")
except Exception as e:
    print(f"  ⚠ Skipped (optional): {e}")

print("\n" + "="*50)
print("Checkpoint download complete!")
print("="*50)
EOF

# ============================================
# Set environment variables
# ============================================
echo ""
echo "Setting environment variables..."

export PYTHONPATH="/workspace/repos/sam-3d-body:/workspace/repos/sam-body4d:$PYTHONPATH"
export CHECKPOINTS_DIR="/workspace/checkpoints"
export DATA_DIR="/workspace/data"
export SAM3D_BODY_PATH="/workspace/repos/sam-3d-body"
export SAM_BODY4D_PATH="/workspace/repos/sam-body4d"

# Add to bashrc for persistence
cat >> ~/.bashrc << 'ENVEOF'

# Nvwa environment
export PYTHONPATH="/workspace/repos/sam-3d-body:/workspace/repos/sam-body4d:$PYTHONPATH"
export CHECKPOINTS_DIR="/workspace/checkpoints"
export DATA_DIR="/workspace/data"
export SAM3D_BODY_PATH="/workspace/repos/sam-3d-body"
export SAM_BODY4D_PATH="/workspace/repos/sam-body4d"
ENVEOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  /workspace/repos/sam-3d-body    - SAM 3D Body model"
echo "  /workspace/repos/sam-body4d     - SAM-Body4D pipeline"
echo "  /workspace/checkpoints          - Model weights"
echo "  /workspace/data                 - Input/output data"
echo ""
echo "Start the server with:"
echo "  cd /workspace/nvwa/backend"
echo "  uvicorn app.sam3d_api:app --host 0.0.0.0 --port 8000"
echo ""
