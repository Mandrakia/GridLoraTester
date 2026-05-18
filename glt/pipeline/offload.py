"""Module iteration helper.

The old `setup_text_encoder_offload` and `enable_model_cpu_offload`
wrappers are gone — the new two-phase build (Qwen → encode → del →
transformer) makes per-forward CPU↔GPU paging obsolete. `iter_quantizable_modules`
remains because both the fp8 and int8_convrot quantizers walk the
pipeline's nn.Modules to decide what to quantize.
"""
from __future__ import annotations


def iter_quantizable_modules(pipe):
    """Yield `(name, module)` for every weight-heavy nn.Module in the
    pipeline that should be quantized.

    Skipped:
      - **vae**: kept in bf16 for image quality
      - **schedulers**: no weights
      - **text_encoder**: lives in the Phase 1 pipeline only, gone by
        the time the quantizer runs in Phase 2
    """
    from torch import nn
    seen = set()
    for name, module in vars(pipe).items():
        if not isinstance(module, nn.Module) or name in seen:
            continue
        if name == "vae" or "scheduler" in name or name == "text_encoder":
            continue
        seen.add(name)
        yield name, module
