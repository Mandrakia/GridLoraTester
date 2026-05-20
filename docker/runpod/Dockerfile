# syntax=docker/dockerfile:1.7
# --------------------------------------------------------------------------
# GridLoraTester + ostris/ai-toolkit — RunPod multi-service container.
#
# Mirrors the working RunPod "template" (filebrowser / exposed SSH /
# ai-toolkit) and swaps the legacy Gradio grid UI for the GridLoraTester
# (GLT) SvelteKit dashboard. ai-toolkit trains; GLT tests the results.
#
# Both halves SHARE their dataset + output folders on the persistent volume,
# so a LoRA trained in ai-toolkit shows up in GLT for grid-testing with no
# copying:
#     /workspace/datasets   ← ai-toolkit DATASETS_FOLDER  == GLT dataset_root
#     /workspace/outputs     ← ai-toolkit TRAINING_FOLDER  == GLT lora_root
#     /workspace/grids       ← GLT tests_root (grid HTML + images)
#     /workspace/data        ← GLT glt.db (SQLite)
#     /workspace/aitk_db.db  ← ai-toolkit SQLite (settings + job history)
#
# Everything re-fetchable lives on the ephemeral container disk under
# /opt/scratch (HF / insightface / torch / inductor / int8 caches) — NEVER
# on /workspace, which is the PAID persistent network volume.
#
# Two Python venvs side-by-side (different torch versions, isolated):
#   /opt/venv               → torch 2.11+cu130 — GLT's `glt/` package + deps
#   /opt/ai-toolkit/venv    → torch 2.9.1+cu128 — ai-toolkit's pin
#
# Base: nvidia/cuda:13.0 cudnn-devel on Ubuntu 24.04. Drivers ≥ 580 on the
# host (RunPod L40S/H100/B200 are fine; old A100 driver builds may not load
# cu130 — pick a "newer drivers" node).
#
# Build (context = repo root):
#     docker build -f docker/runpod/Dockerfile -t <you>/glt-runpod:latest .
# --------------------------------------------------------------------------
ARG CUDA_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=24
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
# Nightly: only dev pre-releases exist for 1.27 on the Azure feed; the
# ".dev0" lower bound makes PEP 440 ordering accept them.
ARG ORT_VERSION=1.27.0.dev0
# SageAttention 2.x source build (off by default — int8_convrot doesn't need
# it and it benches negative on Ampere). The arch list below ONLY gates this
# build: torch + onnxruntime wheels already cover every arch, so inference runs
# on any RunPod GPU regardless of this value.
#   Ampere 8.0/8.6 · Ada 8.9 · Hopper 9.0 · Blackwell 10.0 (B200) / 12.0 (5090)
# When you enable Sage, narrow the list to your rented GPU for a far faster
# build, e.g. --build-arg BUILD_SAGE=1 --build-arg TORCH_CUDA_ARCH_LIST=8.9
# (and if the build chokes on 10.0, SageAttention may lack B200 kernels — drop it).
ARG BUILD_SAGE=0
ARG SAGE_GIT_REF=main
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

# System deps:
# - python 3.12 + build chain for git+ wheel builds
# - libgl/glib for opencv-headless + insightface, ffmpeg for ai-toolkit previews
# - tini for signal handling
# - openssh-server + openssl: enable `ssh root@pod` when $PUBLIC_KEY is set at
#   runtime (matches the runpod-workers/comfyui-base pattern)
# - rsync: companion for SSH file transfers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip \
        libgl1 libglib2.0-0 libsndfile1 ffmpeg \
        ca-certificates curl git build-essential pkg-config \
        tini \
        openssh-server openssh-client openssl rsync \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3

# FileBrowser — web file manager for /workspace at :8080. Single static
# binary, ~15 MB. Official one-line installer.
RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

# Node.js — adapter-node serves GLT's dashboard; better-sqlite3 + sharp ship
# prebuilt binaries for Node 24 (CLAUDE.md → Stack). ai-toolkit's UI (Next.js
# + a TypeScript worker) also runs here.
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x -o /tmp/nodesource_setup.sh && \
    bash /tmp/nodesource_setup.sh && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* /tmp/nodesource_setup.sh && \
    node --version && npm --version

# uv — ~10× faster than pip for these multi-GB install passes.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --------------------------------------------------------------------------
# Layer order = stable → volatile, so editing app code doesn't invalidate the
# giant torch / ai-toolkit / deps layers below.
# --------------------------------------------------------------------------

# GLT venv at /opt/venv — torch 2.11+cu130 (matches the user's local stack).
RUN uv venv /opt/venv --python python${PYTHON_VERSION} && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu130

# ostris/ai-toolkit in its own venv at /opt/ai-toolkit/venv (torch 2.9.1+cu128,
# different from GLT's — isolated). The worker auto-discovers
# `<TOOLKIT_ROOT>/venv/bin/python` at runtime, so the venv MUST be named `venv`.
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit && \
    uv venv /opt/ai-toolkit/venv --python python${PYTHON_VERSION} && \
    uv pip install --python /opt/ai-toolkit/venv/bin/python --no-cache \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --python /opt/ai-toolkit/venv/bin/python --no-cache \
        -r /opt/ai-toolkit/requirements.txt

# Pre-build ai-toolkit's Next.js UI + worker. `update_db` runs Prisma's
# `db push` against the SQLite at /opt/ai-toolkit/aitk_db.db (creating the
# schema). At runtime, start.sh moves/symlinks that file to /workspace so user
# settings survive restarts. AI_TOOLKIT_AUTH is read at runtime, not inlined.
WORKDIR /opt/ai-toolkit/ui
RUN npm install && npm run update_db && npm run build

# --------------------------------------------------------------------------
# GLT Python deps into /opt/venv. onnxruntime-gpu cu130 nightly comes from the
# Azure feed (stable PyPI ORT has no CUDA 13 binaries yet); CLAUDE.md pins it.
# --------------------------------------------------------------------------
RUN uv pip install --python /opt/venv/bin/python --no-cache \
        --prerelease=allow \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
        "onnxruntime-gpu>=${ORT_VERSION},<1.28"

COPY requirements.txt /tmp/glt-requirements.txt
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /tmp/glt-requirements.txt

# Optional SageAttention 2.x from source (off by default — see ARG above).
RUN if [ "${BUILD_SAGE}" = "1" ]; then \
        echo "[sage-build] arches=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} ref=${SAGE_GIT_REF}" && \
        uv pip install --python /opt/venv/bin/python --no-cache setuptools wheel packaging ninja && \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        uv pip install --python /opt/venv/bin/python --no-cache --no-build-isolation \
            "git+https://github.com/thu-ml/SageAttention.git@${SAGE_GIT_REF}"; \
    else \
        echo "[sage-build] skipped (BUILD_SAGE=0)."; \
    fi

# --------------------------------------------------------------------------
# Application code (changes most often — keep last). The whole GLT repo lands
# at /app; .dockerignore keeps node_modules / build / data / .git out.
# --------------------------------------------------------------------------
WORKDIR /app
COPY . /app

# Build GLT's SvelteKit dashboard. adapter-node emits ui/build/ — `node build`
# runs the prod server.
WORKDIR /app/ui
RUN npm ci && npm run build

# --------------------------------------------------------------------------
# Storage tiering — /workspace is the PAID persistent network volume; keep
# only the irreplaceable there. Everything re-fetchable goes to /opt/scratch
# (ephemeral container disk: FREE, faster local NVMe, lost on pod destroy).
#
# Pick Container Disk ≥ 80 GB on the RunPod template: the HF cache for FLUX-2
# Klein (inference) + Klein-base (training) + Qwen3 text encoder can reach
# 50 GB+, and ai-toolkit's latent cache adds a few more. Container Disk is
# included in the hourly price — bigger is free.
# --------------------------------------------------------------------------
ENV GLT_ROOT=/app \
    GLT_PYTHON_BIN=/opt/venv/bin/python \
    GLT_DB_PATH=/workspace/data/glt.db \
    GLT_DATASET_ROOT=/workspace/datasets \
    GLT_LORA_ROOT=/workspace/outputs \
    GLT_TESTS_ROOT=/workspace/grids \
    GLT_TORCHINDUCTOR_CACHE_DIR=/opt/scratch/torchinductor \
    GLT_INT8_CONVROT_CACHE_DIR=/opt/scratch/int8_convrot \
    HF_HOME=/opt/scratch/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    INSIGHTFACE_HOME=/opt/scratch/insightface \
    TORCH_HOME=/opt/scratch/torch \
    XDG_CACHE_HOME=/opt/scratch/cache \
    NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000

# Persistent (volume) tree + ephemeral (scratch) cache tree. Both exist at
# build time so a fresh `docker run` without a bind mount still has somewhere
# to write; bind /workspace to make state survive.
RUN mkdir -p \
        /workspace/datasets /workspace/outputs /workspace/grids /workspace/data \
        /opt/scratch/huggingface /opt/scratch/insightface /opt/scratch/torch \
        /opt/scratch/cache /opt/scratch/torchinductor /opt/scratch/int8_convrot \
        /opt/scratch/latents

RUN chmod +x /app/docker/runpod/start.sh

# Service ports:
#   22   → SSH (active only when $PUBLIC_KEY is set at runtime)
#   3000 → GLT SvelteKit dashboard
#   8080 → FileBrowser (web file manager for /workspace)
#   8675 → ai-toolkit Next.js UI
EXPOSE 22 3000 8080 8675

VOLUME ["/workspace"]

# tini reaps zombies + forwards signals so SIGTERM stops the services cleanly
# on `docker stop` / RunPod pod stop.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/runpod/start.sh"]
CMD []
