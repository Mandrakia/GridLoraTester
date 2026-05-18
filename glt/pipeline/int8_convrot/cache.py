"""On-disk cache for INT8 ConvRot-quantized transformer state.

Skips the ~25s CPU quantize step on cold start by storing the
(weight=int8, weight_scale=fp32, bias=bf16) tuple per module in a single
safetensors file. Keyed by the source transformer path (HF repo id or
absolute `.safetensors` file), in line with the existing FP8 cache
([[kiss-cache-keys]] in user memory: simple slug, manual `rm` to
invalidate; no dtype/version in the key).

Reload pattern mirrors `pipeline.fp8.load_fp8_cached_transformer`:

    bf16 init on meta -> swap in LinearInt8ConvRot stubs -> assign=True
    state_dict load -> no bf16 intermediate

This keeps peak VRAM at the final INT8 size (~9 GB for Klein 9B)
instead of bf16 (~18 GB) on the cold-start path.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor, nn
from safetensors.torch import load_file, save_file

from .linear import CONVROT_GROUP_SIZE, LinearInt8ConvRot
from .quantize import (
    FLUX2_EXCLUDE_SUBSTRINGS,
    _is_eligible,
    _is_excluded,
)


# Top-level slug that distinguishes our quant from the existing fp8
# cache (which uses `.fp8cache.pt`). Adding this prefix prevents
# accidental cross-load when both modes have been used against the
# same source path.
_CACHE_EXTENSION = ".int8convrot.safetensors"


def cache_path_for(source: str) -> Path:
    """Resolve the on-disk cache path for an INT8 ConvRot transformer.

    `source` is either an HF repo id (e.g. ``"black-forest-labs/FLUX.2-klein-9B"``)
    or an absolute path to a local `.safetensors`. Lives under
    ``~/.cache/glt/int8_convrot/`` by default; override with
    ``GLT_INT8_CONVROT_CACHE_DIR``. No dtype / version in the key — to
    invalidate, ``rm`` the file.
    """
    cache_dir = Path(os.environ.get(
        "GLT_INT8_CONVROT_CACHE_DIR",
        str(Path.home() / ".cache" / "glt" / "int8_convrot"),
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = source.replace("/", "_").replace(":", "_").lstrip("_")
    if slug.endswith(".safetensors"):
        slug = slug[: -len(".safetensors")]
    return cache_dir / f"{slug}{_CACHE_EXTENSION}"


def save_quantized_transformer(transformer: nn.Module, cache_path: Path) -> dict:
    """Serialize every `LinearInt8ConvRot` module's state to disk.

    Key convention in the safetensors file:
        <qual_name>.weight        int8 (out, in)
        <qual_name>.weight_scale  fp32 (out, 1)
        <qual_name>.bias          bf16 (out,)        — only if bias exists

    Returns a small report dict: {modules: N, mb_on_disk: F}.

    The cache is for the INT8 layers ONLY — non-quantized layers
    (img_in, time_in, etc.) keep their bf16 weights and reload via the
    standard from_pretrained path. This means the cache file is small
    relative to the full model (~9 GB vs ~18 GB for Klein 9B).
    """
    sd: dict[str, Tensor] = {}
    n_modules = 0
    for name, m in transformer.named_modules():
        if not isinstance(m, LinearInt8ConvRot):
            continue
        sd[f"{name}.weight"] = m.weight.detach().cpu().contiguous()
        sd[f"{name}.weight_scale"] = m.weight_scale.detach().cpu().contiguous()
        if m.bias is not None:
            sd[f"{name}.bias"] = m.bias.detach().cpu().contiguous()
        n_modules += 1
    save_file(sd, str(cache_path))
    mb = cache_path.stat().st_size / 1e6
    return {"modules": n_modules, "mb_on_disk": mb}


def load_into_transformer_from_cache(
    transformer: nn.Module,
    cache_path: Path,
    group_size: int = CONVROT_GROUP_SIZE,
    exclude_substrings: tuple[str, ...] = FLUX2_EXCLUDE_SUBSTRINGS,
) -> dict:
    """Replace eligible `nn.Linear` modules with `LinearInt8ConvRot`
    populated from the cached state_dict — no rotation or per-row
    quantization performed at runtime.

    The eligibility predicate must match what was used when the cache
    was written (default Flux2 list + group_size). Mismatched keys are
    reported in the return dict.
    """
    sd = load_file(str(cache_path), device="cpu")

    # Collect targets to swap, mirroring quantize_transformer's walk.
    pending: list[tuple[nn.Module, str, nn.Linear, str]] = []
    parents: dict[str, tuple[nn.Module, str]] = {}
    for qual_name, _module in transformer.named_modules():
        if not qual_name:
            continue
        parent_name, _, attr_name = qual_name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        parents[qual_name] = (parent, attr_name)

    for qual_name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_excluded(qual_name, exclude_substrings):
            continue
        ok, _ = _is_eligible(module, group_size)
        if not ok:
            continue
        parent, attr_name = parents[qual_name]
        pending.append((parent, attr_name, module, qual_name))

    loaded = 0
    missing: list[str] = []
    for parent, attr_name, src_linear, qual_name in pending:
        wk = f"{qual_name}.weight"
        sk = f"{qual_name}.weight_scale"
        if wk not in sd or sk not in sd:
            missing.append(qual_name)
            continue
        device = src_linear.weight.device
        layer = LinearInt8ConvRot(
            in_features=src_linear.in_features,
            out_features=src_linear.out_features,
            group_size=group_size,
            bias=src_linear.bias is not None,
            device=device,
        )
        layer.weight = sd[wk].to(device)
        layer.weight_scale = sd[sk].to(device)
        if src_linear.bias is not None:
            bk = f"{qual_name}.bias"
            if bk in sd:
                layer.bias = nn.Parameter(sd[bk].to(device), requires_grad=False)
        setattr(parent, attr_name, layer)
        loaded += 1

    return {
        "loaded": loaded,
        "expected": len(pending),
        "missing": missing[:10],
        "missing_count": len(missing),
    }
