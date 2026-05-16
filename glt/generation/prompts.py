"""Prompt-encoding helpers + on-disk embeds cache."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..pipeline.offload import iter_quantizable_modules


def prompts_embeds_cache_path(prompts_path: str | Path) -> Path:
    """Cache file for prompt embeds keyed by SHA256 of the prompts file
    contents. Lets us skip loading the text encoder entirely when the prompts
    file hasn't changed. Stored at `~/.cache/glt/embeds/` (override with
    `GLT_EMBEDS_CACHE_DIR`). If you change the text encoder source or dtype,
    `rm` the cache — KISS, no auto-invalidation on that."""
    cache_dir = Path(os.environ.get(
        "GLT_EMBEDS_CACHE_DIR",
        str(Path.home() / ".cache" / "glt" / "embeds"),
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = Path(prompts_path)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return cache_dir / f"{p.stem}_{h}.embeds.pt"


def encode_prompts(pipe, prompts: list[str], offload_mode: str = "full",
                   prompts_path: str | Path | None = None):
    """Pre-encode every prompt ONCE (text encoder runs N times instead of
    n_adapters × N times). Returns a list of `prompt_embeds` tensors aligned
    with `prompts`.

    If `prompts_path` is given, embeds are cached keyed by SHA256 of the file
    contents. Cache hit → `torch.load` + skip the text encoder entirely (the
    caller can also have skipped its load in `build_pipeline` via
    `skip_text_encoder`).

    Behaviour depends on `offload_mode`:

      - `"full"` / `"text-encoder"`: park the transformer to CPU during
        encoding (to free VRAM for the text encoder), then page the text
        encoder back to CPU at the end. Saves VRAM at the cost of one
        bidirectional swap.
      - `"none"`: large-VRAM setup — do NOT page the text encoder to CPU and
        do NOT park the transformer. Two reasons:

        1. `Pipeline._execution_device` reads from the first component's
           device; leaving the text encoder on CPU after encoding makes the
           pipeline think it's on CPU and try to allocate noise tensors on
           CPU — clashing with our CUDA generators (`Cannot generate a cpu
           tensor from a generator of type cuda`).
        2. With `torch.compile` enabled on the transformer, the CPU↔GPU
           round-trip invalidates the cached compile graph.
    """
    import torch

    def vram(label: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[vram] {label}: {mem:.2f} GB")

    embeds_cache = prompts_embeds_cache_path(prompts_path) if prompts_path else None
    if embeds_cache is not None and embeds_cache.exists():
        print(f"[embeds:cache] hit {embeds_cache}")
        cached = torch.load(embeds_cache, weights_only=False, map_location="cuda")
        try:
            target_dtype = next(pipe.transformer.parameters()).dtype
        except Exception:
            target_dtype = torch.bfloat16
        out = [e.to(device="cuda", dtype=target_dtype).contiguous() for e in cached]
        if len(out) != len(prompts):
            print(f"[embeds:cache] WARN cached={len(out)} != prompts={len(prompts)} — "
                  f"falling through to encode")
        else:
            print(f"[embeds:cache] loaded {len(out)} embeds, skipping text encoder entirely")
            return out

    print(f"[encode] one-shot encoding of {len(prompts)} prompt(s)  (offload_mode={offload_mode})")

    transformer = getattr(pipe, "transformer", None)
    # torch.compile wraps the module in OptimizedModule, which exposes
    # `_orig_mod`. Detect that and skip parking — moving a compiled module
    # to CPU and back invalidates its compile cache.
    is_compiled = transformer is not None and hasattr(transformer, "_orig_mod")

    # Park policy is offload-mode-driven, NOT free-memory-driven:
    #   - "none"          → user has plenty of VRAM, never park
    #   - "text-encoder"  → user explicitly wants TE swap; park the transformer
    #                       during encode so Qwen3 + lm_head dequant scales fit
    #   - "full"          → accelerate already manages it
    park_transformer = (
        offload_mode != "none"
        and not is_compiled
        and transformer is not None
    )

    free_mem = 0
    if torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info()
        except Exception:
            pass

    parked_transformer = False
    if park_transformer:
        try:
            on_gpu = any(p.device.type == "cuda" for p in transformer.parameters())
        except Exception:
            on_gpu = False
        if on_gpu:
            print("[encode] parking transformer on CPU to free VRAM for text encoder")
            transformer.to("cpu")
            parked_transformer = True
            vram("after parking transformer")
    elif is_compiled:
        print("[encode] transformer is torch.compile'd — skipping park/restore "
              "(would invalidate compile cache)")
    elif offload_mode == "none":
        print(f"[encode] --no-offload mode + {free_mem / 1024**3:.1f} GB free — "
              f"not parking transformer")

    # The transformer expects embeds in its own dtype. If the user loaded the
    # text encoder in a different dtype (e.g. fp16 to match ComfyUI while the
    # transformer is bf16), embeds come out in TE dtype → matmul fails. We
    # cast at encode time to keep the denoise loop dtype-pure.
    try:
        target_dtype = next(pipe.transformer.parameters()).dtype
    except Exception:
        target_dtype = torch.bfloat16
    embeds = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            prompt_embeds, _text_ids = pipe.encode_prompt(prompt=p)
            prompt_embeds = prompt_embeds.to(
                device="cuda", dtype=target_dtype, non_blocking=False,
            ).contiguous()
            embeds.append(prompt_embeds)
            print(f"  [{i + 1}/{len(prompts)}] shape={tuple(prompt_embeds.shape)} "
                  f"device={prompt_embeds.device} dtype={prompt_embeds.dtype} :: {p[:50]}")

    # Only page the text encoder to CPU when the caller's offload mode wants
    # it. With --no-offload (96 GB cards etc.) keeping it on GPU is faster
    # AND avoids the device-mismatch trap in pipe._execution_device.
    if offload_mode != "none":
        final_hook = getattr(pipe, "_final_offload_hook", None) or getattr(pipe, "final_offload_hook", None)
        if final_hook is not None:
            try:
                final_hook.offload()
            except Exception:
                pass
        else:
            for name, module in iter_quantizable_modules(pipe):
                if name == "transformer":
                    continue
                try:
                    module.to("cpu")
                except Exception:
                    pass
        vram("after text encoder paged out")

    if parked_transformer:
        print("[encode] restoring transformer to CUDA")
        transformer.to("cuda")
        vram("after restoring transformer")

    if embeds_cache is not None:
        try:
            torch.save([e.detach().cpu().contiguous() for e in embeds], embeds_cache)
            size_mb = embeds_cache.stat().st_size / 1e6
            print(f"[embeds:cache] wrote {size_mb:.1f} MB → {embeds_cache}")
        except Exception as e:
            print(f"[embeds:cache] save failed: {type(e).__name__}: {e}")

    return embeds
