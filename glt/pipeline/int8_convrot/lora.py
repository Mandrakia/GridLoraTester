"""LoRA bake-in for `LinearInt8ConvRot` modules.

`pip install peft` plus diffusers' `pipe.load_lora_weights()` does NOT
work here — peft only knows how to wrap `nn.Linear` (and a handful of
specific classes), and our `LinearInt8ConvRot` is rejected outright.
Even if we made peft accept it, the dynamic-LoRA pattern (base forward
+ B@A delta at runtime) would lose the INT8 IMMA speedup on the delta
path.

Instead, we BAKE the LoRA delta directly into the quantized weight:

    dequantize → add α * B @ rotate(A) → re-quantize

where the rotation is applied to A (the down-projection along the
in-feature dim) so the delta sits in the same Hadamard-rotated basis as
the existing weight. The math:

    W_rotated = W @ H^T                (offline, in quantize_transformer)
    new_W_rotated = W_rotated + α * B @ A_rot
                  = (W + α B A) @ H^T

So inference uses the same rotated forward path — no per-layer LoRA
branch to recompile, no extra GEMM. Compile survives swaps (the
underlying `weight` Parameter's identity is preserved by `.copy_()`).

Key-format handling
-------------------
BFL / ai-toolkit save LoRAs against the fused BFL transformer layout
(`diffusion_model.double_blocks.{i}.img_attn.qkv.lora_B.weight` with
out_dim=12288=3×4096). Diffusers' `Flux2Transformer2DModel` exposes the
same maths split into 8 separate linears for the double blocks
(`attn.to_q`, `.to_k`, `.to_v`, `.add_q_proj`, ...). Single blocks keep
the QKV+MLP fusion on both sides, so they map 1-to-1.

The converter slices the LoRA's B matrix along its output dim when a
single BFL key maps to multiple diffusers modules. `lora_A` is shared
across the slices (same compression).
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from .hadamard import build_hadamard, rotate_activation
from .linear import LinearInt8ConvRot, dequantize_weight, quantize_weight_per_row


# BFL double-block sub-path -> diffusers attribute path under
# `transformer_blocks.{idx}.`. A list value indicates "split the LoRA B
# along output dim across N targets" (fused QKV in BFL, split in diffusers).
_DOUBLE_BLOCK_MAP: dict[str, str | list[str]] = {
    "img_attn.qkv":  ["attn.to_q", "attn.to_k", "attn.to_v"],
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.qkv":  ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
    "txt_attn.proj": "attn.to_add_out",
    "img_mlp.0":     "ff.linear_in",
    "img_mlp.2":     "ff.linear_out",
    "txt_mlp.0":     "ff_context.linear_in",
    "txt_mlp.2":     "ff_context.linear_out",
}

# BFL single-block sub-path -> diffusers attribute path under
# `single_transformer_blocks.{idx}.`. Both formats keep QKV+MLP fused.
_SINGLE_BLOCK_MAP: dict[str, str] = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
}


def _resolve_targets(bfl_key: str) -> list[tuple[str, int | None]]:
    """Map a BFL LoRA key to ``[(diffusers_name, split_idx), ...]``.

    `split_idx` is None for 1-to-1 mappings and 0/1/2 for fused-QKV
    splits (Q=0, K=1, V=2 — the convention used by both BFL and
    diffusers when this fusion exists).

    Returns an empty list when the key doesn't match any known BFL
    layout — for example, modulation/embedder layers that BFL stores
    under `diffusion_model.img_in.*`, which our quantizer also skips
    via the Flux2 exclusion list. Unmapped keys are reported in the
    `load_lora_into_quantized()` summary so the caller can warn.
    """
    parts = bfl_key.split(".")
    if len(parts) < 4 or parts[0] != "diffusion_model":
        return []
    block_type = parts[1]
    block_idx = parts[2]
    # The middle parts identify the sub-layer; the final two are
    # `lora_{A|B}.weight`.
    sub_path = ".".join(parts[3:-2])

    if block_type == "double_blocks":
        target = _DOUBLE_BLOCK_MAP.get(sub_path)
        if target is None:
            return []
        if isinstance(target, list):
            return [
                (f"transformer_blocks.{block_idx}.{t}", i)
                for i, t in enumerate(target)
            ]
        return [(f"transformer_blocks.{block_idx}.{target}", None)]

    if block_type == "single_blocks":
        target = _SINGLE_BLOCK_MAP.get(sub_path)
        if target is None:
            return []
        return [(f"single_transformer_blocks.{block_idx}.{target}", None)]

    return []


def _pair_lora_keys(
    state_dict: dict[str, Tensor],
) -> dict[str, dict[str, Tensor]]:
    """Group state_dict entries into ``{base_key: {"A": ..., "B": ..., "alpha": ...}}``.

    `base_key` is the prefix before ``.lora_A/.lora_B/.alpha``. Entries
    that aren't part of a LoRA pair are silently dropped — the
    state_dict often contains metadata or unrelated keys.
    """
    pairs: dict[str, dict[str, Tensor]] = {}
    for k, v in state_dict.items():
        if k.endswith(".lora_A.weight"):
            base = k[: -len(".lora_A.weight")]
            pairs.setdefault(base, {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            base = k[: -len(".lora_B.weight")]
            pairs.setdefault(base, {})["B"] = v
        elif k.endswith(".alpha"):
            base = k[: -len(".alpha")]
            pairs.setdefault(base, {})["alpha"] = v
    return pairs


def bake_lora_delta(
    module: LinearInt8ConvRot,
    A: Tensor,
    B: Tensor,
    alpha: float = 1.0,
) -> None:
    """In-place: bake ``α · B @ A`` (LoRA delta) into ``module``'s INT8 weight.

    The delta is computed in float32, rotated to match the existing
    weight's Hadamard basis, added, and re-quantized per-row. The
    Parameter / Buffer identities are preserved via `.copy_()` so
    `torch.compile`'d graphs stay valid across LoRA swaps.

    Shapes:
      A: (rank, in_features) — projects activation to a low-rank space
      B: (out_features, rank) — projects back to the output space

    Both should be bf16/fp16; we cast through fp32 for the arithmetic.
    """
    device = module.weight.device
    a_f32 = A.to(device=device, dtype=torch.float32)
    b_f32 = B.to(device=device, dtype=torch.float32)

    # Dequant gives the bf16-precision rotated weight.
    w_rot_bf16 = dequantize_weight(module.weight, module.weight_scale)
    w_rot_f32 = w_rot_bf16.to(torch.float32)

    # Rotate A along its in-feature dim (last dim) so the delta lives in
    # the same basis: ΔW_rot = α · B @ (A · H).
    h = build_hadamard(module.group_size, device=device, dtype=torch.float32)
    a_rot = rotate_activation(a_f32, h, module.group_size)
    delta = (b_f32 @ a_rot) * float(alpha)

    w_new = w_rot_f32.add_(delta)
    new_int8, new_scale = quantize_weight_per_row(w_new)
    module.weight.copy_(new_int8)
    module.weight_scale.copy_(new_scale)


def load_lora_into_quantized(
    transformer: nn.Module,
    state_dict: dict[str, Tensor],
    alpha_scale: float = 1.0,
    verbose: bool = False,
) -> dict[str, int]:
    """Walk `state_dict` and bake every recognized LoRA pair into
    matching `LinearInt8ConvRot` modules.

    `alpha_scale` is the user-facing strength multiplier (1.0 = full).
    The per-layer `alpha` from the state_dict (when present) is divided
    by the LoRA rank, matching the standard ``α/r`` scaling convention.

    Returns
    -------
    Counts dict:
      - ``applied``           — number of (target_module, lora_pair) bakes done
      - ``unmatched_key``     — BFL keys with no known diffusers target
      - ``missing_module``    — known target name not present in transformer
      - ``incompatible_type`` — target was found but isn't INT8 ConvRot
      - ``total_pairs``       — total (A, B) pairs found in state_dict
    """
    pairs = _pair_lora_keys(state_dict)
    applied = 0
    unmatched_key = 0
    missing_module = 0
    incompatible_type = 0

    for base_key, parts in pairs.items():
        a = parts.get("A")
        b = parts.get("B")
        if a is None or b is None:
            continue
        # Re-synthesize a typical lora_A.weight string so `_resolve_targets`
        # has the canonical layout to parse.
        synthetic = base_key + ".lora_A.weight"
        targets = _resolve_targets(synthetic)
        if not targets:
            unmatched_key += 1
            if verbose:
                print(f"[int8_convrot:lora] unmatched key: {base_key}")
            continue

        # Per-layer alpha scaling: state_dict's `.alpha` (if present)
        # divided by rank gives the strength factor. The user-supplied
        # `alpha_scale` multiplies on top.
        rank = a.shape[0]
        layer_alpha_t = parts.get("alpha")
        if layer_alpha_t is not None:
            layer_alpha = float(layer_alpha_t.item())
            local_scale = (layer_alpha / max(rank, 1)) * alpha_scale
        else:
            local_scale = alpha_scale

        for module_name, split_idx in targets:
            try:
                module = transformer.get_submodule(module_name)
            except AttributeError:
                missing_module += 1
                if verbose:
                    print(f"[int8_convrot:lora] missing module: {module_name}")
                continue
            if not isinstance(module, LinearInt8ConvRot):
                incompatible_type += 1
                if verbose:
                    print(
                        f"[int8_convrot:lora] target not INT8 ConvRot: "
                        f"{module_name} ({type(module).__name__})"
                    )
                continue

            if split_idx is None:
                a_slice, b_slice = a, b
            else:
                # Fused-QKV split: B has shape (3 * out_per, rank); take
                # the chunk for q/k/v. A is shared (same projection).
                out_total = b.shape[0]
                if out_total % 3 != 0:
                    if verbose:
                        print(
                            f"[int8_convrot:lora] expected 3-way fused B "
                            f"for {base_key}, got {out_total} rows"
                        )
                    continue
                out_per = out_total // 3
                b_slice = b[split_idx * out_per : (split_idx + 1) * out_per]
                a_slice = a

            bake_lora_delta(module, a_slice, b_slice, alpha=local_scale)
            applied += 1

    return {
        "applied": applied,
        "unmatched_key": unmatched_key,
        "missing_module": missing_module,
        "incompatible_type": incompatible_type,
        "total_pairs": len(pairs),
    }


# ---- Baseline backup/restore for LoRA swap ------------------------------


def backup_quantized_baseline(transformer: nn.Module) -> int:
    """Snapshot the post-quantize INT8 state to CPU memory.

    Call once *after* `quantize_transformer()` and *before* any LoRA is
    baked, so subsequent swaps can restore the clean baseline. CPU
    storage avoids the 9 GB VRAM cost of keeping a second copy on GPU
    (the user's 24 GB / 32 GB cards are already tight with the live
    transformer + activations).

    Returns the number of modules snapshotted.
    """
    n = 0
    for m in transformer.modules():
        if isinstance(m, LinearInt8ConvRot):
            m._convrot_baseline_weight = m.weight.detach().cpu().clone()
            m._convrot_baseline_scale = m.weight_scale.detach().cpu().clone()
            n += 1
    return n


def restore_quantized_baseline(transformer: nn.Module) -> int:
    """Restore weights from the baseline backup. Call before applying
    a new LoRA when an old one has been baked in.

    No-op for modules without a recorded baseline (e.g. if called before
    `backup_quantized_baseline`). Returns the number of modules restored.
    """
    n = 0
    for m in transformer.modules():
        if not isinstance(m, LinearInt8ConvRot):
            continue
        bw = getattr(m, "_convrot_baseline_weight", None)
        bs = getattr(m, "_convrot_baseline_scale", None)
        if bw is None or bs is None:
            continue
        m.weight.copy_(bw.to(m.weight.device, non_blocking=True))
        m.weight_scale.copy_(bs.to(m.weight_scale.device, non_blocking=True))
        n += 1
    return n


def has_baseline(transformer: nn.Module) -> bool:
    """True iff at least one `LinearInt8ConvRot` module has a baseline
    backup. Used by the swap path to skip restore on first LoRA load."""
    for m in transformer.modules():
        if isinstance(m, LinearInt8ConvRot) and hasattr(m, "_convrot_baseline_weight"):
            return True
    return False
