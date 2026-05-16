"""Preload CUDA 12 libs before onnxruntime is imported.

onnxruntime-gpu 1.x links against CUDA 12 SONAMEs (libcublasLt.so.12). If the
active venv has CUDA 13 (e.g. PyTorch's nvidia-*-cu13 wheels), the loader
can't find `.so.12` and silently falls back to CPU. We scan a few well-known
places for cu12 .so files and preload them with RTLD_GLOBAL so onnxruntime's
later dlopen finds them already resolved. cu12 and cu13 SONAMEs differ →
both coexist fine in the same process.
"""
from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path


def preload_cuda12_libs(extra_search_dirs: list[str | Path] | None = None) -> int:
    """Returns the number of `.so` files successfully preloaded (0 = no-op).

    Search order:
      1. `extra_search_dirs` (caller-provided)
      2. `$CUDA12_NVIDIA_DIR` env var
      3. `site-packages/nvidia/` of every site dir of the active interpreter
    """
    search_dirs: list[Path] = []
    if extra_search_dirs:
        search_dirs += [Path(p).expanduser() for p in extra_search_dirs]
    env_dir = os.environ.get("CUDA12_NVIDIA_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))

    site_dirs: list[str] = []
    try:
        site_dirs += list(site.getsitepackages())
    except Exception:
        pass
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    for sd in site_dirs:
        cand = Path(sd) / "nvidia"
        if cand.is_dir():
            search_dirs.append(cand)

    nvidia_dir = next((d for d in search_dirs if (d / "cublas" / "lib").is_dir()), None)
    if nvidia_dir is None:
        return 0

    # cublas/cublasLt must come before cudnn (cudnn calls into it). We match
    # any versioned SONAME (.so.11, .so.12, .so.10): onnxruntime-gpu can mix
    # CUDA major versions across sub-libs (cuBLAS 12 + cuFFT 11 in one release).
    subs = ("cublas", "cuda_runtime", "cufft", "curand", "cusolver",
            "cusparse", "nccl", "nvjitlink", "cuda_nvrtc", "cudnn")
    loaded = 0
    for sub in subs:
        lib_dir = nvidia_dir / sub / "lib"
        if not lib_dir.is_dir():
            continue
        for so in sorted(lib_dir.glob("lib*.so.*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                pass
    if loaded:
        print(f"[cuda] preloaded {loaded} lib(s) from {nvidia_dir}")
    return loaded
