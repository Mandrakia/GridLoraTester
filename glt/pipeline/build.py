"""Top-level pipeline construction: load + quantize + offload + (optional) compile."""
from __future__ import annotations

from .attention import install_sage_attention
from .fp8 import (
    fp8_cache_path_for,
    fp8_quantize_quanto,
    fp8_quantize_torchao,
    load_fp8_cached_transformer,
    patch_peft_torchao_fp8,
    save_fp8_cached_transformer,
)
from .loaders import (
    DEFAULT_MODEL_ID,
    load_text_encoder_from_file,
    load_transformer_from_file,
    load_vae_from_file,
    torch_dtype as _dtype,
)
from .offload import setup_text_encoder_offload


def build_pipeline(
    use_fp8: bool,
    offload_mode: str,
    backend: str,
    fp8_mode: str = "weight-only",
    compile_transformer: bool = False,
    sage_attention: bool = False,
    model_path: str = DEFAULT_MODEL_ID,
    transformer_file: str | None = None,
    transformer_dtype: str = "bf16",
    text_encoder_file: str | None = None,
    text_encoder_dtype: str = "bf16",
    vae_file: str | None = None,
    vae_dtype: str = "bf16",
    skip_text_encoder: bool = False,
):
    """Construct a `Flux2KleinPipeline` with the requested combination of
    options. Handles BFL-format scaled-FP8 files, torchao/quanto FP8 quant,
    CPU offload modes, SageAttention, and optional `torch.compile`.

    See `README.md` and CLI `--help` for option semantics.
    """
    import torch
    from diffusers import Flux2KleinPipeline

    default_dtype = torch.bfloat16

    overrides: dict = {}
    fp8_source: list[str] = []  # components whose source is already fp8 →
                                # torchao quant must NOT re-quantize them.

    if transformer_file:
        td = _dtype(transformer_dtype)
        print(f"[load] transformer ← {transformer_file}  "
              f"(dtype={transformer_dtype}, config={model_path})")
        overrides["transformer"] = load_transformer_from_file(
            transformer_file, td, transformer_dtype, config_source=model_path,
        )
        if transformer_dtype.startswith("fp8"):
            fp8_source.append("transformer")
    if vae_file:
        vd = _dtype(vae_dtype)
        overrides["vae"] = load_vae_from_file(vae_file, model_path, vd)
        if vae_dtype.startswith("fp8"):
            fp8_source.append("vae")
    if text_encoder_file:
        ed = _dtype(text_encoder_dtype)
        overrides["text_encoder"] = load_text_encoder_from_file(
            text_encoder_file, config_source=model_path, target_dtype=ed,
        )
        if text_encoder_dtype.startswith("fp8"):
            fp8_source.append("text_encoder")

    # Skip Qwen3 load entirely when the caller already has cached embeds for
    # every prompt. Saves ~26s of "Loading pipeline components". The caller
    # MUST guarantee a full cache hit — pipe.encode_prompt will crash if
    # called with text_encoder=None.
    if skip_text_encoder and "text_encoder" not in overrides:
        overrides["text_encoder"] = None
        overrides["tokenizer"] = None
        print("[load] skipping text_encoder + tokenizer (embeds cache hit)")

    # When the source was fp8 (BFL release) we dequantized to bf16 in-memory
    # to get past diffusers' converter. Applying torchao on top would be a
    # SECOND fp8→bf16→fp8' round-trip that adds quantization noise vs the
    # original BFL weights — and would NOT match ComfyUI's behavior (Comfy
    # keeps fp8+scale and dequants at forward, no re-quant). So auto-skip
    # torchao to match ComfyUI exactly.
    if use_fp8 and fp8_source:
        print(f"[load] fp8 source detected ({', '.join(fp8_source)}) — skipping "
              f"torchao to match ComfyUI exactly (no second quant round-trip). "
              f"Pass --transformer-dtype bf16 to opt into torchao re-quant for "
              f"VRAM/perf at the cost of drifting from ComfyUI.")
        use_fp8 = False

    # FP8 weight-only cache: if a prior run already quantized this transformer,
    # materialize it directly from disk and inject as an override so
    # from_pretrained below doesn't even load the bf16 transformer.
    # Saves ~25 s (the CPU quantize) per startup.
    fp8_cache_path = None
    fp8_cache_hit = False
    if (
        use_fp8 and backend == "torchao" and fp8_mode == "weight-only"
        and "transformer" not in overrides
    ):
        cache_source = transformer_file if transformer_file else model_path
        fp8_cache_path = fp8_cache_path_for(cache_source)
        if fp8_cache_path.exists():
            overrides["transformer"] = load_fp8_cached_transformer(
                fp8_cache_path, config_source=model_path,
            )
            fp8_cache_hit = True

    print(f"[load] {model_path} (dtype={default_dtype}, fp8={use_fp8}, backend={backend}, "
          f"fp8_mode={fp8_mode}, offload={offload_mode}, compile={compile_transformer}, "
          f"sage={sage_attention}, "
          f"fp8_cache={'hit' if fp8_cache_hit else 'miss' if fp8_cache_path else 'n/a'})")
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=default_dtype, **overrides)

    if use_fp8 and fp8_cache_hit:
        # Cache hit skipped fp8_quantize_torchao entirely → peft's
        # TorchaoLoraLinear bridge patch wasn't installed. Apply it now so
        # subsequent load_lora_weights() doesn't fail on missing kwarg.
        try:
            from torchao.quantization import Float8WeightOnlyConfig
            patch_peft_torchao_fp8(Float8WeightOnlyConfig)
        except ImportError:
            from torchao.quantization import float8_weight_only as _f8wo
            patch_peft_torchao_fp8(_f8wo)

    if use_fp8 and not fp8_cache_hit:
        try:
            if backend == "torchao":
                fp8_quantize_torchao(pipe, mode=fp8_mode)
                if fp8_cache_path is not None:
                    save_fp8_cached_transformer(pipe.transformer, fp8_cache_path)
            elif backend == "quanto":
                fp8_quantize_quanto(pipe)
            else:
                raise ValueError(f"unknown fp8 backend: {backend}")
        except ImportError as e:
            hint = ("pip install torchao" if backend == "torchao"
                    else "pip install optimum-quanto  # also needs the CUDA toolkit + CUDA_HOME")
            print(f"[warn] FP8 backend '{backend}' unavailable ({e}). "
                  f"Continuing in bf16. Install: {hint}")

    if offload_mode == "full":
        pipe.enable_model_cpu_offload()
    elif offload_mode == "text-encoder":
        setup_text_encoder_offload(pipe)
    elif offload_mode == "none":
        pipe.to("cuda")
        try:
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[vram] allocated after load: {mem:.2f} GB")
        except Exception:
            pass
    else:
        raise ValueError(f"unknown offload mode: {offload_mode}")

    # Install sage BEFORE compile so the compiled graph captures the patched
    # SDPA. The patch is global (F.scaled_dot_product_attention).
    if sage_attention:
        install_sage_attention()

    if compile_transformer:
        if pipe.transformer is None:
            print("[compile] no transformer attribute on pipeline, skipping")
        elif offload_mode == "full":
            print("[compile] WARNING: torch.compile is poorly compatible with "
                  "--full-offload (transformer moves CPU↔GPU). You'll likely see "
                  "recompiles. Prefer --no-offload or --offload-text-encoder.")
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)
        else:
            print("[compile] torch.compile(transformer, mode='reduce-overhead', fullgraph=False)")
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)

    return pipe
