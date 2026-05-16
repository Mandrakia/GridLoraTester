"""FP8 quantization (torchao / quanto) + on-disk cache + peft compatibility patch."""
from __future__ import annotations

import os
from pathlib import Path

from .offload import iter_quantizable_modules


def patch_peft_torchao_fp8(config_factory) -> None:
    """Workaround for a peft 0.19.x bug with the torchao bridge.

    `TorchaoLoraLinear.__init__` requires kwarg `get_apply_tensor_subclass`,
    but peft's dispatcher (`dispatch_torchao`) doesn't inject it, so every
    LoRA load fails. We default it to the same config we quantized with.
    """
    try:
        from peft.tuners.lora import torchao as _peft_torchao
    except ImportError:
        return
    cls = _peft_torchao.TorchaoLoraLinear
    if getattr(cls, "_glt_patched", False):
        return
    _orig_init = cls.__init__

    def _patched(self, *args, get_apply_tensor_subclass=None, **kwargs):
        if get_apply_tensor_subclass is None:
            get_apply_tensor_subclass = lambda: config_factory()
        return _orig_init(self, *args, get_apply_tensor_subclass=get_apply_tensor_subclass, **kwargs)

    cls.__init__ = _patched
    cls._glt_patched = True
    print(f"[patch] peft.TorchaoLoraLinear: default get_apply_tensor_subclass={config_factory.__name__}")


def has_fp8_tensor_cores() -> bool:
    """`True` when the current GPU has native FP8 tensor cores: Hopper (sm90),
    Ada Lovelace (sm89), Blackwell (sm100+). Ampere (sm80/sm86) does NOT."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(0)
        return cap[0] >= 9 or cap == (8, 9)
    except Exception:
        return False


def fp8_cache_path_for(source: str) -> Path:
    """Resolve the on-disk cache file for a torchao-quantized transformer.

    `source` is either an HF repo id (e.g. `"black-forest-labs/FLUX.2-klein-9B"`)
    or an absolute path to a local `.safetensors`. Either way it gets slugified
    into a filename under `~/.cache/glt/fp8/` (override with `GLT_FP8_CACHE_DIR`).
    No version / mtime / dtype in the key — KISS. To invalidate, `rm` the file.
    """
    cache_dir = Path(os.environ.get(
        "GLT_FP8_CACHE_DIR",
        str(Path.home() / ".cache" / "glt" / "fp8"),
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = source.replace("/", "_").replace(":", "_").lstrip("_")
    if slug.endswith(".safetensors"):
        slug = slug[: -len(".safetensors")]
    return cache_dir / f"{slug}.fp8cache.pt"


def load_fp8_cached_transformer(cache_path: Path, config_source: str):
    """Materialize a Float8-quantized `Flux2Transformer2DModel` directly from
    a cache file, skipping the bf16 load + torchao quantize compute entirely.

    Pattern: meta init → `quantize_` stubs (still on meta) → `torch.load` to
    cuda → `load_state_dict(assign=True)`. `assign=True` replaces meta params
    with the cached cuda Float8Tensors without going through a bf16
    intermediate, so VRAM peaks at the final fp8 size (~9 GB for Klein)
    instead of bf16 (~22 GB). Verified bit-identical to fresh quantize output.
    """
    import torch
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel
    from torchao.quantization import quantize_, Float8WeightOnlyConfig

    print(f"[fp8:cache] hit {cache_path}")
    config = Flux2Transformer2DModel.load_config(config_source, subfolder="transformer")
    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(config)
    quantize_(transformer, Float8WeightOnlyConfig())
    sd = torch.load(cache_path, weights_only=False, map_location="cuda:0")
    missing, unexpected = transformer.load_state_dict(sd, strict=False, assign=True)
    if missing:
        print(f"[fp8:cache]   {len(missing)} missing keys (first: {missing[:3]})")
    if unexpected:
        print(f"[fp8:cache]   {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    torch.cuda.synchronize()
    return transformer


def save_fp8_cached_transformer(transformer, cache_path: Path) -> None:
    """Serialize a torchao-quantized transformer's state_dict to disk so the
    next run can skip the ~25 s CPU quantize."""
    import torch
    print(f"[fp8:cache] writing {cache_path} ...")
    torch.save(transformer.state_dict(), cache_path)
    size_gb = cache_path.stat().st_size / 1e9
    print(f"[fp8:cache] wrote {size_gb:.2f} GB → {cache_path}")


def fp8_quantize_torchao(pipe, mode: str = "weight-only") -> None:
    """FP8 via torchao. Two modes:

      - `'weight-only'` (default): stores weights as FP8, dequant-to-bf16 on
        the fly inside each matmul. Saves ~50% of weight VRAM. NO compute
        speedup (the matmul is still bf16). Quality essentially identical.
      - `'dynamic'` (Hopper/Ada/Blackwell only): also quantizes activations
        on every forward and uses native FP8 tensor cores via
        `torch._scaled_mm`. ~1.5-2× faster on supported GPUs; on Ampere it
        falls back to a software path with no speedup — sometimes slower
        than weight-only.

    Both modes patch peft's `TorchaoLoraLinear` so LoRAs load cleanly on top.
    """
    from torchao.quantization import quantize_

    if mode == "dynamic":
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
            config_factory = Float8DynamicActivationFloat8WeightConfig
        except ImportError:
            from torchao.quantization import float8_dynamic_activation_float8_weight as config_factory
        if not has_fp8_tensor_cores():
            try:
                import torch
                cap = torch.cuda.get_device_capability(0)
            except Exception:
                cap = "unknown"
            print(f"[fp8:dynamic] WARNING: GPU compute cap {cap} has no FP8 tensor cores. "
                  f"Dynamic activation FP8 will fall back to a software path — expect NO "
                  f"speedup (sometimes slower than --fp8-mode weight-only).")
    else:
        try:
            from torchao.quantization import Float8WeightOnlyConfig
            config_factory = Float8WeightOnlyConfig
        except ImportError:
            from torchao.quantization import float8_weight_only as config_factory  # older API

    for name, module in iter_quantizable_modules(pipe):
        print(f"[quantize:torchao] {name} -> fp8 ({mode})")
        quantize_(module, config_factory())
    patch_peft_torchao_fp8(config_factory)


def fp8_quantize_quanto(pipe) -> None:
    """FP8 via optimum-quanto — JIT-compiles a CUDA kernel, requires CUDA toolkit."""
    from optimum.quanto import freeze, qfloat8, quantize
    for name, module in iter_quantizable_modules(pipe):
        print(f"[quantize:quanto] {name} -> fp8")
        quantize(module, weights=qfloat8)
        freeze(module)
