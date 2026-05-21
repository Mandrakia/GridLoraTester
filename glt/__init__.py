"""glt — Grid-based LoRA/LoKr tester for FLUX-family diffusion models."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


# Persistent torch.compile cache.
#
# PyTorch's `torch.compile` (FX graph cache + AOTAutograd cache) stores
# its compiled-kernel artifacts on disk and reuses them across process
# invocations as long as the cache directory persists and the cache key
# matches (same function source + input shapes/dtypes + torch/CUDA
# versions). On a hit, the per-shape "compile warmup" drops from
# ~45-70 s to a few seconds — the difference between "compile is only
# rentable on 20+ image grids" and "compile is rentable on everything".
#
# The default cache dir is `/tmp/torchinductor_<user>/`, which gets
# wiped on every system reboot. We redirect to `~/.cache/glt/torchinductor/`
# so the cache survives reboots and lives alongside glt's other caches
# (`~/.cache/glt/{fp8,int8_convrot,embeds.db}`).
#
# Override with `GLT_TORCHINDUCTOR_CACHE_DIR` for non-standard layouts.
# Disable entirely by exporting `TORCHINDUCTOR_CACHE_DIR=` (empty) before
# launching glt — the cache survives across runs but ~adds 200 MB - 2 GB
# of disk depending on how many (shape, batch_size, quant) combos a user
# exercises.
#
# This MUST happen before any `import torch` in subprocess code paths;
# setting it in glt/__init__.py guarantees that any glt entrypoint
# (`python -m glt …`, `from glt import …`) imports this module first.
_default_compile_cache = Path.home() / ".cache" / "glt" / "torchinductor"
_cache_dir = os.environ.get(
    "GLT_TORCHINDUCTOR_CACHE_DIR",
    str(_default_compile_cache),
)
if _cache_dir and "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    Path(_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir


# Quiet torchao's benign import noise.
#
# torchao 0.17 is the release built FOR torch 2.11+, but it ships optional CUDA
# kernels GLT never calls — `_C_mxfp8` (tagged cp310-only, so it can't load on a
# cp312 venv) and `_C_cutlass_90a` (Hopper sm_90a) — and its own quant_api.py
# trips a SyntaxWarning ('\.' in a non-raw string). GLT's fp8 path uses
# torch-native fp8 (Float8WeightOnly), NOT those kernels, so the "Failed to
# load …" / "Unable to import torchao Tensor objects" lines are pure noise with
# no functional impact — and no torchao version fixes the packaging (0.17 IS the
# torch-2.11 release). Silence it so it stops cluttering the per-job log. Like
# the cache dir above, this runs before any entrypoint imports torchao.
import logging as _logging
import warnings as _warnings

_warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"torchao.*")
_logging.getLogger("torchao").setLevel(_logging.ERROR)
