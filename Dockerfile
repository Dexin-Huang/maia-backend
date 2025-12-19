# SAM 3D Body Pipeline Backend
# Base: NVIDIA CUDA 12.1 with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 with CUDA 12.1 support (matches SAM 3D Body requirements)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Set working directory for SAM 3D Body
WORKDIR /opt

# Clone SAM 3D Body repository
RUN git clone https://github.com/facebookresearch/sam-3d-body.git

# Install SAM 3D Body dependencies
WORKDIR /opt/sam-3d-body
RUN pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs \
    cython jsonlines pytest xtcocotools loguru optree fvcore black \
    pycocotools tensorboard huggingface_hub trimesh

# Install detectron2 (specific commit for SAM 3D Body compatibility)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# Install backend application dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy application code
COPY app /app/app

# Create entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment variables
ENV HF_HOME=/models
ENV SAM3D_BODY_PATH=/opt/sam-3d-body
ENV PYTHONPATH=/opt/sam-3d-body:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run entrypoint
ENTRYPOINT ["/entrypoint.sh"]
