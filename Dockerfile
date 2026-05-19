# syntax=docker/dockerfile:1.7
#
# GridLoraTester runtime image.
#
# Layout the container expects (bind-mount /workspace from the host):
#     /workspace/datasets   ← settings.dataset_root
#     /workspace/outputs    ← settings.lora_root      (folders of .safetensors to test)
#     /workspace/grids      ← settings.tests_root     (grid-test HTML + images output)
#     /workspace/data       ← glt.db (SQLite)
#     /workspace/cache      ← HF / torchinductor / int8_convrot caches
#
# Typical invocation:
#     docker run --gpus all -p 3000:3000 \
#         -v /host/glt-workspace:/workspace \
#         -e HF_TOKEN=hf_xxx \
#         ghcr.io/<you>/glt:latest

ARG CUDA_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=24
ARG TORCH_VERSION=2.11.0
# Nightly: only dev pre-releases exist for 1.27 on the Azure feed; the
# ".dev0" lower bound makes pip's PEP 440 ordering accept them.
ARG ORT_VERSION=1.27.0.dev0
# Ampere (3090/A100) + Ada (4090/L40) by default. Add 12.0 for Blackwell.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
# Set to 1 to compile SageAttention 2.x from source (slow, ~10-15 min).
ARG BUILD_SAGE=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --- system packages --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        build-essential \
        pkg-config \
        git \
        curl \
        ca-certificates \
        tini \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ffmpeg \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# --- Node.js (NodeSource LTS) ----------------------------------------------
# Needed at runtime: adapter-node serves the SvelteKit dashboard; better-sqlite3
# and sharp ship prebuilt binaries for Node 24 (CLAUDE.md → Stack section).
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
 && apt-get update && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && node --version && npm --version

# --- Python venv ------------------------------------------------------------
ENV VENV=/opt/glt-venv
RUN python${PYTHON_VERSION} -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"
RUN pip install --upgrade pip setuptools wheel

# --- PyTorch (CUDA 13 official wheels) --------------------------------------
# Done BEFORE requirements.txt so transitive `torch` deps resolve against the
# cu130 wheel we want, not a CPU build pulled from PyPI.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==${TORCH_VERSION} \
        torchvision

# --- onnxruntime-gpu cu130 (Microsoft Azure DevOps nightly feed) ------------
# CLAUDE.md pins this feed; stable PyPI ORT doesn't ship CUDA 13 binaries yet.
RUN pip install \
        --pre \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
        "onnxruntime-gpu>=${ORT_VERSION},<1.28"

# --- Project Python deps ----------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- Optional SageAttention build (off by default) --------------------------
# Builds native CUDA kernels for the archs listed in TORCH_CUDA_ARCH_LIST. Skip
# unless you actually want sage — the int8_convrot path doesn't need it and
# benches negative on Ampere (.claude/quant_int8.md → SageAttention bench).
RUN if [ "${BUILD_SAGE}" = "1" ]; then \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        pip install --no-build-isolation \
            "git+https://github.com/thu-ml/SageAttention.git@main" ; \
    fi

# --- Project source ---------------------------------------------------------
WORKDIR /app
COPY . /app

# --- UI build ---------------------------------------------------------------
# adapter-node emits ui/build/ — `node build` runs the prod server.
WORKDIR /app/ui
RUN npm ci && npm run build

# --- Runtime environment ----------------------------------------------------
ENV GLT_ROOT=/app \
    GLT_DB_PATH=/workspace/data/glt.db \
    GLT_TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor \
    GLT_INT8_CONVROT_CACHE_DIR=/workspace/cache/int8_convrot \
    HF_HOME=/workspace/cache/hf \
    HF_XET_HIGH_PERFORMANCE=1 \
    NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000

# Workspace tree exists at image-build time so a fresh `docker run` without a
# bind mount still has somewhere to write (transient). Bind a host directory
# onto /workspace to make state survive.
RUN mkdir -p \
        /workspace/datasets \
        /workspace/outputs \
        /workspace/grids \
        /workspace/data \
        /workspace/cache/hf \
        /workspace/cache/torchinductor \
        /workspace/cache/int8_convrot

# Settings preseed + server launcher
COPY docker/seed-settings.cjs /app/docker/seed-settings.cjs
COPY docker/entrypoint.sh /usr/local/bin/glt-entrypoint
RUN chmod +x /usr/local/bin/glt-entrypoint

WORKDIR /app/ui

EXPOSE 3000
VOLUME ["/workspace"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/glt-entrypoint"]
CMD []
