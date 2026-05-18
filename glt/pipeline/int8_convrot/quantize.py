"""Walk a transformer module tree and replace eligible `nn.Linear` layers
with `LinearInt8ConvRot`.

A "Flux2 exclusion list" of layer-name substrings stays in bf16 — these
are small embedding / modulation projections (img_in, time_in, txt_in,
etc.) where:

  1. The compute saved by quantizing them is negligible (they run once
     per call, not 4 timesteps × N attention heads).
  2. Their numerical sensitivity is high: time-step modulation feeds
     into AdaLayerNorm, where a 1-2% quantization error compounds across
     every block. ConvRot's accuracy guarantees break down on these
     tiny, outlier-sensitive matrices.

The exclusion list is determined empirically by the diffusion community
(visible in any Flux2 quant config — e.g. NVIDIA ModelOpt, GGUF Q8 quant
recipes, BobJohnson24's INT8-Fast loader). Not a copyrighted artifact.
"""
from __future__ import annotations

from torch import nn

from .linear import CONVROT_GROUP_SIZE, LinearInt8ConvRot


# Layer-name substrings to skip when quantizing a Flux2 transformer.
# Match against the FULLY-QUALIFIED module name (e.g.
# "transformer_blocks.0.attn.to_q"), not just the leaf name. Substring
# match keeps the list short and resilient to diffusers naming churn.
FLUX2_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "img_in",
    "time_in",
    "guidance_in",
    "txt_in",
    "double_stream_modulation_img",
    "double_stream_modulation_txt",
    "single_stream_modulation",
)


def _is_excluded(qual_name: str, exclude_substrings: tuple[str, ...]) -> bool:
    """Check if `qual_name` contains any substring from the exclusion list."""
    return any(s in qual_name for s in exclude_substrings)


def _is_eligible(linear: nn.Linear, group_size: int) -> tuple[bool, str]:
    """Decide whether `linear` can be quantized as INT8 ConvRot.

    Returns ``(eligible, reason_if_not)``. Most rejections are shape
    constraints — Hadamard rotation needs `in_features % group_size == 0`.
    """
    if linear.in_features < group_size:
        return False, f"in_features={linear.in_features} < group_size={group_size}"
    if linear.in_features % group_size != 0:
        return False, (
            f"in_features={linear.in_features} not divisible by "
            f"group_size={group_size}"
        )
    # 1-D layers (in_features=1 or out_features=1) are usually scalars/
    # bias-like — quantizing them costs more than it saves.
    if linear.in_features == 1 or linear.out_features == 1:
        return False, "1-D linear (dim==1)"
    return True, ""


def quantize_transformer(
    transformer: nn.Module,
    group_size: int = CONVROT_GROUP_SIZE,
    exclude_substrings: tuple[str, ...] = FLUX2_EXCLUDE_SUBSTRINGS,
    verbose: bool = False,
) -> dict[str, int]:
    """In-place: replace eligible `nn.Linear` modules with `LinearInt8ConvRot`.

    Walks the module tree, builds the rotated INT8 weight on the same
    device as each source linear, and swaps the module via its parent's
    `__setattr__`. Returns a small report dict useful for logging:

        {
          "quantized": <int>,           # replaced
          "excluded_by_name": <int>,    # matched exclusion list
          "excluded_by_shape": <int>,   # failed _is_eligible
          "total_linears": <int>,       # all nn.Linear seen
        }

    The implementation iterates over `named_modules()` once, defers all
    replacements until after the walk (mutating mid-walk would shift
    indices), then applies them. Replacement is via the parent module's
    `setattr` so the new module slots into the original attribute name.
    """
    quantized = 0
    excluded_by_name = 0
    excluded_by_shape = 0
    total_linears = 0

    # Collect (parent_module, attr_name, source_linear, qual_name)
    pending: list[tuple[nn.Module, str, nn.Linear, str]] = []
    # Map qualified name -> (parent, attr_name) for setattr lookup.
    parents: dict[str, tuple[nn.Module, str]] = {}
    for qual_name, module in transformer.named_modules():
        if not qual_name:
            continue
        parent_name, _, attr_name = qual_name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        parents[qual_name] = (parent, attr_name)

    for qual_name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        total_linears += 1
        if _is_excluded(qual_name, exclude_substrings):
            excluded_by_name += 1
            if verbose:
                print(f"[int8_convrot] excluded by name: {qual_name}")
            continue
        ok, reason = _is_eligible(module, group_size)
        if not ok:
            excluded_by_shape += 1
            if verbose:
                print(f"[int8_convrot] excluded by shape: {qual_name} ({reason})")
            continue
        parent, attr_name = parents[qual_name]
        pending.append((parent, attr_name, module, qual_name))

    # Apply the replacements. The conversion runs on the source layer's
    # device — CPU-resident bf16 weights stay on CPU, GPU-resident stay
    # on GPU. The caller is responsible for the final `.to("cuda")` move.
    for parent, attr_name, src_linear, qual_name in pending:
        new_layer = LinearInt8ConvRot.from_bf16_linear(src_linear, group_size=group_size)
        setattr(parent, attr_name, new_layer)
        quantized += 1
        if verbose:
            print(
                f"[int8_convrot] quantized {qual_name} "
                f"({src_linear.in_features}->{src_linear.out_features})"
            )

    return {
        "quantized": quantized,
        "excluded_by_name": excluded_by_name,
        "excluded_by_shape": excluded_by_shape,
        "total_linears": total_linears,
    }
