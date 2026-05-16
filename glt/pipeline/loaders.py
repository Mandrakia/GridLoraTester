"""Single-file weight loaders for transformer / VAE / text encoder + dtype helper."""
from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


def torch_dtype(name: str):
    """String → `torch.dtype`. `'fp8'` is an alias for `'fp8_e4m3fn'` (default
    inference variant). `'fp8_mixed'` loads as bf16; tensors carrying fp8
    metadata stay fp8 inside the safetensors and get unpacked on demand."""
    import torch
    return {
        "bf16":       torch.bfloat16,
        "fp16":       torch.float16,
        "fp8":        torch.float8_e4m3fn,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2":   torch.float8_e5m2,
        "fp8_mixed":  torch.bfloat16,
    }[name]


def load_transformer_from_file(path: str | Path, target_dtype, target_dtype_name: str,
                               config_source: str = DEFAULT_MODEL_ID):
    """Load a FLUX-2 transformer from a single `.safetensors` file, handling
    both:

      - **Standard bf16 BFL format**: straight `from_single_file`.
      - **BFL official scaled-FP8 release** (fp8_e4m3fn weights + per-tensor
        `weight_scale` + `input_scale` scalars, same format ComfyUI loads
        directly): dequantize to bf16 in-memory, write a temp plain-bf16
        safetensors, then feed to `from_single_file`. This avoids a known
        diffusers bug — `convert_flux2_double_stream_blocks` blindly does
        `torch.chunk(t, 3, dim=0)` on the 0-D `weight_scale` tensors thinking
        they're fused Q/K/V weights, which crashes with
        `RuntimeError: chunk expects at least a 1-dimensional tensor`.

    The dequant result is mathematically equivalent to weight-only-FP8
    inference: `out = activation @ (weight_fp8.to(bf16) * weight_scale).T` —
    which is what ComfyUI does internally for FLUX FP8 inference.
    """
    import os
    import tempfile

    import torch
    from diffusers import Flux2Transformer2DModel
    from safetensors.torch import load_file, save_file

    path = Path(path)
    sd = load_file(str(path))
    has_scales = any(k.endswith(".weight_scale") for k in sd)

    if not has_scales:
        # `config=` pins the Klein architecture spec — otherwise diffusers
        # falls back to the default Flux2 Pro layer sizes and errors with a
        # shape mismatch (Klein layers are 1.5x smaller than the full FLUX-2).
        return Flux2Transformer2DModel.from_single_file(
            str(path), torch_dtype=target_dtype,
            config=config_source, subfolder="transformer",
        )

    print(f"[load]   BFL scaled-FP8 format detected in {path.name}  "
          f"(fp8_e4m3fn weights + weight_scale/input_scale scalars)")
    print(f"[load]   workaround for diffusers' Flux2 converter bug: dequant to bf16 "
          f"in memory then feed via from_single_file. "
          f"--transformer-dtype={target_dtype_name} is honoured as 'source format', "
          f"actual inference runs in bf16 host.")

    new_sd: dict = {}
    n_dequant = 0
    n_dropped = 0
    for k, v in sd.items():
        if k.endswith(".input_scale") or k.endswith(".weight_scale"):
            n_dropped += 1
            continue
        if v.dtype == torch.float8_e4m3fn and k.endswith(".weight"):
            scale_key = k[: -len(".weight")] + ".weight_scale"
            scale = sd.get(scale_key)
            if scale is not None and scale.numel() == 1:
                # bf16_weight = fp8_weight × weight_scale (equivalent to
                # ComfyUI's weight-only-FP8 dequant at forward time)
                new_sd[k] = v.to(torch.bfloat16) * scale.to(torch.bfloat16)
                n_dequant += 1
                continue
        new_sd[k] = v.to(torch.bfloat16) if v.dtype == torch.float8_e4m3fn else v

    print(f"[load]   dequant: {n_dequant} fp8 weights × weight_scale → bf16, "
          f"{n_dropped} scale scalars dropped")

    # Pick a temp location with plenty of space: /opt/scratch on RunPod-style
    # containers, /tmp on a host machine. System default otherwise.
    tmp_dirs = ["/opt/scratch", "/tmp"]
    tmp_root = next((d for d in tmp_dirs if os.path.isdir(d) and os.access(d, os.W_OK)), None)
    with tempfile.NamedTemporaryFile(
        suffix=".dequant.safetensors", delete=False, dir=tmp_root,
    ) as f:
        tmp_path = f.name
    try:
        save_file(new_sd, tmp_path)
        size_gb = os.path.getsize(tmp_path) / 1024**3
        print(f"[load]   wrote bf16 temp → {tmp_path} ({size_gb:.2f} GB)")
        transformer = Flux2Transformer2DModel.from_single_file(
            tmp_path, torch_dtype=torch.bfloat16,
            config=config_source, subfolder="transformer",
        )
        return transformer
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def load_vae_from_file(path: str | Path, config_source: str, target_dtype):
    """Build `AutoencoderKLFlux2` from `config_source`'s config and load
    weights from a single `.safetensors`. Diffusers' `from_single_file`
    doesn't register `AutoencoderKLFlux2` in `SINGLE_FILE_LOADABLE_CLASSES`
    (cf diffusers issues #9053 / #9141), so we do it manually."""
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file
    print(f"[load] vae ← {path}  (config from {config_source})")
    model = AutoencoderKLFlux2.from_pretrained(
        config_source, subfolder="vae", torch_dtype=target_dtype,
    )
    state_dict = load_file(str(path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load]   vae: {len(missing)} missing keys (truncated): {missing[:5]}")
    if unexpected:
        print(f"[load]   vae: {len(unexpected)} unexpected keys (truncated): {unexpected[:5]}")
    return model


def load_text_encoder_from_file(path: str | Path, config_source: str, target_dtype):
    """Build a Qwen3ForCausalLM from `config_source`'s config and load weights
    from a single `.safetensors`. `transformers` doesn't expose
    `from_single_file` on Qwen3 — we instantiate from config then
    `load_state_dict`."""
    from transformers import Qwen3ForCausalLM, AutoConfig
    from safetensors.torch import load_file
    config = AutoConfig.from_pretrained(config_source, subfolder="text_encoder")
    print(f"[load] text_encoder ← {path}  (config from {config_source})")
    model = Qwen3ForCausalLM(config).to(dtype=target_dtype)
    state_dict = load_file(str(path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load]   text_encoder: {len(missing)} missing keys (truncated): {missing[:5]}")
    if unexpected:
        print(f"[load]   text_encoder: {len(unexpected)} unexpected keys (truncated): {unexpected[:5]}")
    return model
