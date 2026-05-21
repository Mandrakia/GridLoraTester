"""INT8 W8A8 + ConvRot quantization for Ampere-class GPUs (sm80, sm86).

Native FP8 tensor cores were introduced with Ada (sm89). On Ampere, FP8
weight-only quant via torchao saves VRAM but offers no compute speedup —
the dequant happens before a bf16 matmul. INT8, by contrast, has been a
native tensor-core dtype since sm80, and `torch._int_mm` dispatches to
cuBLASLt IMMA on every supported GPU.

This package provides a clean PyTorch implementation:

    - `quantize_transformer(t)` — walks `t`'s `nn.Linear` modules and
      replaces eligible ones in-place with `LinearInt8ConvRot`. The
      excluded-by-name list skips Flux2's small embedding / modulation
      projections that are sensitivity-critical (img_in, time_in, ...).

    - `LinearInt8ConvRot` — drop-in `nn.Linear` replacement. Storage =
      INT8 + per-row fp32 scale. Forward = online Hadamard rotation +
      per-row activation quant + `torch._int_mm` + dequant + bias.

    - `recommended_quant_for_gpu()` — auto-pick fp8/int8/none based on
      the current device's compute capability.

References
----------
QuaRot — Outlier-Free 4-Bit Inference in Rotated LLMs
    https://arxiv.org/abs/2404.00456

PyTorch native INT8 matmul docs
    https://pytorch.org/docs/stable/generated/torch._int_mm.html

torchao quantization recipes (BSD-licensed)
    https://github.com/pytorch/ao/tree/main/torchao/quantization

License attribution
-------------------
This package is an independent re-implementation, not a port. The
algorithmic ideas (group-wise Hadamard rotation per ConvRot/QuaRot, INT8
W8A8 per-row symmetric quantization) come from published papers and
PyTorch/Triton public documentation, both of which are not subject to
license restrictions. No code is copied from third-party projects.
"""
from __future__ import annotations

import torch

from .cache import (
    cache_path_for,
    load_into_transformer_from_cache,
    save_quantized_transformer,
)
from .hadamard import build_hadamard, rotate_activation, rotate_weight
from .linear import (
    CONVROT_GROUP_SIZE,
    LinearInt8ConvRot,
    dequantize_weight,
    quantize_int8_per_row,
    quantize_weight_per_row,
)
from .lora import (
    backup_quantized_baseline,
    bake_lora_delta,
    has_baseline,
    load_lora_into_quantized,
    resolve_targets_zimage,
    restore_quantized_baseline,
)
from .quantize import (
    FLUX2_EXCLUDE_SUBSTRINGS,
    ZIMAGE_EXCLUDE_SUBSTRINGS,
    quantize_transformer,
)


__all__ = [
    "CONVROT_GROUP_SIZE",
    "FLUX2_EXCLUDE_SUBSTRINGS",
    "ZIMAGE_EXCLUDE_SUBSTRINGS",
    "LinearInt8ConvRot",
    "backup_quantized_baseline",
    "bake_lora_delta",
    "build_hadamard",
    "cache_path_for",
    "dequantize_weight",
    "has_baseline",
    "load_into_transformer_from_cache",
    "load_lora_into_quantized",
    "quantize_int8_per_row",
    "quantize_transformer",
    "quantize_weight_per_row",
    "recommended_quant_for_gpu",
    "resolve_targets_zimage",
    "restore_quantized_baseline",
    "rotate_activation",
    "rotate_weight",
    "save_quantized_transformer",
]


def recommended_quant_for_gpu() -> str:
    """Return the recommended quant mode for the current CUDA device.

    Returns
    -------
    One of:
      - ``"int8_convrot"``  — Ampere (sm80, sm86): native INT8 IMMA path,
        FP8 would have no tensor-core acceleration.
      - ``"fp8_weight"``    — Ada (sm89), Hopper (sm90+), Blackwell
        (sm100+): native FP8 tensor cores, torchao Float8WeightOnly.
      - ``"none"``          — pre-Ampere or no CUDA: quantization either
        unsupported or not worth the work (these GPUs typically can't run
        Flux2 9B anyway).

    Notes
    -----
    The recommendation reflects the community consensus as of 2026
    (Q1-Q2) based on benchmarks across multiple Flux2 variants. NVFP4
    (sm100+) is not yet wired up; when added, it would be preferred over
    fp8_weight on Blackwell.
    """
    if not torch.cuda.is_available():
        return "none"
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:
        return "none"
    # Ampere — INT8 native, no FP8 tensor cores.
    if cap in ((8, 0), (8, 6)):
        return "int8_convrot"
    # Ada (sm89), Hopper (sm90+), Blackwell (sm100+) — FP8 native.
    if cap == (8, 9) or cap[0] >= 9:
        return "fp8_weight"
    # Pre-Ampere — INT8 IMMA is sm75+ technically, but Flux2 9B doesn't
    # fit in 11 GB anyway. No reasonable path.
    return "none"
