"""CPU offload + module iteration helpers."""
from __future__ import annotations


def iter_quantizable_modules(pipe):
    """Yield `(name, module)` for every weight-heavy nn.Module in the pipeline,
    skipping:

      - **vae**: kept in bf16 for image quality
      - **schedulers**: no weights to quantize
      - **text_encoder**: encoded once per prompt then offloaded, so the ~4 GB
        savings from FP8 don't matter for the steady-state VRAM picture.
        Quantizing it also produces a ~3% rel-error on the cond embeddings vs
        ComfyUI's pure-fp16 forward (verified via debug-dump 2026-05-13) which
        propagates noticeably through the 4-step distilled trajectory.

    For CPU offload (a different concern), use `iter_offloadable_encoders`
    instead — that one includes the text_encoder.
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


def iter_offloadable_encoders(pipe):
    """Yield `(name, module)` for every encoder-like submodule eligible for
    `accelerate.cpu_offload_with_hook`. Includes the text_encoder (which is
    NOT in `iter_quantizable_modules` but DOES need to be hooked so it
    auto-moves to CUDA at forward time). Excludes the transformer (kept on
    GPU) and the VAE (also on GPU, runs only at decode time)."""
    from torch import nn
    seen = set()
    for name, module in vars(pipe).items():
        if not isinstance(module, nn.Module) or name in seen:
            continue
        if name in ("transformer", "vae") or "scheduler" in name:
            continue
        seen.add(name)
        yield name, module


def setup_text_encoder_offload(pipe):
    """Keep transformer + VAE on GPU, page text encoder(s) to RAM. Each
    `forward()` auto-moves the encoder to GPU and the previous one back to
    CPU. Prints final VRAM allocation."""
    import torch
    from accelerate import cpu_offload_with_hook

    for name in ("transformer", "vae"):
        m = getattr(pipe, name, None)
        if m is not None:
            m.to("cuda")

    hook = None
    n = 0
    for name, module in iter_offloadable_encoders(pipe):
        module.to("cpu")
        module, hook = cpu_offload_with_hook(module, execution_device="cuda", prev_module_hook=hook)
        setattr(pipe, name, module)
        print(f"[offload] {name} -> cpu (auto-swap on .forward)")
        n += 1

    pipe._final_offload_hook = hook
    if n == 0:
        print("[warn] no text encoder found to offload — falling back to all-on-GPU")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"[vram] allocated after text-encoder offload: {mem:.2f} GB")
