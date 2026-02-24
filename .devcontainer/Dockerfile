# GPU-enabled image (CUDA 12.8 + cuDNN) suitable for PyTorch cu128 wheels
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- system packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python-is-python3 \
      git \
      curl \
      ca-certificates \
      build-essential \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---- python packages ----
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch matching CUDA 12.8 wheels (cu128)
RUN python -m pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128

# Runtime deps inferred from the codebase
RUN python -m pip install --no-cache-dir \
      numpy \
      pandas \
      tqdm \
      pyyaml \
      omegaconf \
      lightning \
      torchmetrics \
      flake8 \
      black \
      wandb

# ---- app ----
WORKDIR /app
COPY . /app

# Make `src.*` imports work when running scripts/
ENV PYTHONPATH=/app

# Default to a shell; run training like:
#   python scripts/main.py --config path/to/config.yaml
CMD ["bash"]
