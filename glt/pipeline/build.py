"""Two-phase Klein pipeline construction.

The Qwen3 text encoder and the Flux2 transformer are NEVER co-resident
on the GPU. The lifecycle is:

  Phase 1 — `build_te_pipeline(...)`
    Load Qwen3 + tokenizer onto GPU. Caller encodes every missing prompt
    via `pipe.encode_prompt(...)`, persists the result to the embeds
    cache, then deletes the pipeline. Skipped entirely on a full cache
    hit (no Qwen3 load at all — saves ~28 s).

  Phase 2 — `build_transformer_pipeline(quant=...)`
    Load Flux2 transformer + VAE, apply the requested quant
    (fp8_weight / int8_convrot / ...), move to GPU. No text encoder.
    Lives for the entire grid run.

This eliminates every form of CPU↔GPU offload paging: Qwen is briefly
on GPU during encode, then gone forever, leaving the full 24 GB for
the transformer + activations. No `setup_text_encoder_offload`, no
`enable_model_cpu_offload`, no per-step parking. The 2× speedup from
int8_convrot lands without any VRAM math required.

Quant values
------------
  ``"auto"``         pick per GPU capability (Ampere → int8_convrot,
                      Ada/Hopper → fp8_weight)
  ``"none"``         no quant, bf16 throughout
  ``"fp8_weight"``   torchao Float8WeightOnly
  ``"fp8_dynamic"``  torchao Float8DynamicActivationFloat8Weight
  ``"fp8_quanto"``   optimum-quanto fp8
  ``"int8_convrot"`` glt's INT8 W8A8 + Hadamard rotation (Ampere best)
"""
from __future__ import annotations

import torch
from diffusers import Flux2KleinPipeline

from .attention import install_sage_attention
from .fp8 import (
    fp8_cache_path_for,
    fp8_quantize_quanto,
    fp8_quantize_torchao,
    load_fp8_cached_transformer,
    patch_peft_torchao_fp8,
    save_fp8_cached_transformer,
)
from .int8_convrot import (
    cache_path_for as int8_convrot_cache_path_for,
    load_into_transformer_from_cache as int8_convrot_load_cached,
    quantize_transformer as quantize_transformer_int8_convrot,
    recommended_quant_for_gpu,
    save_quantized_transformer as int8_convrot_save_cache,
)
from .loaders import (
    DEFAULT_MODEL_ID,
    load_text_encoder_from_file,
    load_transformer_from_file,
    load_vae_from_file,
    torch_dtype as _dtype,
)


QUANT_CHOICES = ("auto", "none", "fp8_weight", "fp8_dynamic", "fp8_quanto", "int8_convrot")


def resolve_quant(quant: str) -> str:
    """Map ``"auto"`` to the GPU's recommended mode; pass-through otherwise."""
    if quant not in QUANT_CHOICES:
        raise ValueError(f"invalid quant={quant!r}; choices are {QUANT_CHOICES}")
    if quant == "auto":
        chosen = recommended_quant_for_gpu()
        print(f"[quant] auto -> {chosen} (per GPU compute capability)")
        return chosen
    return quant


def _vram(label: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"[vram] {label}: {mem:.2f} GB")


# ---- Phase 1: text-encoder-only pipeline --------------------------------


def build_te_pipeline(
    model_path: str = DEFAULT_MODEL_ID,
    qwen_dtype: str = "bf16",
    qwen_file: str | None = None,
) -> "Flux2KleinPipeline":
    """Load Qwen3 + tokenizer ONLY, on GPU. Transformer + VAE = None.

    `qwen_dtype` accepts ``"bf16"`` (default, 16 GB on GPU, fits the
    24 GB 3090 with nothing else) or any of ``"fp16"`` / ``"fp8_e4m3fn"``
    / ``"fp8_e5m2"`` for smaller cards. fp8 cuts the load to ~8 GB at
    the cost of a tiny encode-time quality hit.

    Caller is expected to encode every missing prompt then `del` the
    pipeline. Nothing else is loaded so there's no offload to manage.
    """
    overrides: dict = {"transformer": None, "vae": None}
    if qwen_file:
        ed = _dtype(qwen_dtype)
        overrides["text_encoder"] = load_text_encoder_from_file(
            qwen_file, config_source=model_path, target_dtype=ed,
        )

    print(f"[load:te] {model_path} (qwen_dtype={qwen_dtype})")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, **overrides,
    )
    if pipe.text_encoder is not None:
        pipe.text_encoder.to("cuda")
    _vram("after Qwen load")
    return pipe


# ---- Phase 2: transformer + VAE pipeline --------------------------------


def build_transformer_pipeline(
    quant: str = "auto",
    model_path: str = DEFAULT_MODEL_ID,
    transformer_file: str | None = None,
    transformer_dtype: str = "bf16",
    vae_file: str | None = None,
    vae_dtype: str = "bf16",
    compile_transformer: bool = False,
    sage_attention: bool = False,
) -> "Flux2KleinPipeline":
    """Load Flux2 transformer + VAE, apply `quant`, move to GPU.

    No text encoder is loaded — the caller is expected to have already
    encoded prompts via `build_te_pipeline()` and persisted them to the
    embeds cache (or to have a full cache hit, in which case Phase 1
    was skipped entirely).
    """
    quant = resolve_quant(quant)

    overrides: dict = {"text_encoder": None, "tokenizer": None}
    fp8_source: list[str] = []  # components already in fp8 storage —
                                # torchao quant must NOT re-quantize.

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

    # BFL scaled-fp8 transformer was dequantized to bf16 in-memory to
    # get past diffusers' converter. Applying torchao on top is a
    # second fp8→bf16→fp8' round-trip that drifts from ComfyUI. Skip.
    if quant.startswith("fp8") and fp8_source:
        print(f"[load] fp8 source detected ({', '.join(fp8_source)}) — "
              f"skipping torchao re-quant to stay bit-identical to ComfyUI. "
              f"Pass --transformer-dtype bf16 to opt in.")
        quant = "none"

    # fp8 cache: on a hit, materialize the transformer directly from the
    # quantized state file, skipping the ~25 s CPU quantize.
    fp8_cache_path = None
    fp8_cache_hit = False
    if quant == "fp8_weight" and "transformer" not in overrides:
        cache_source = transformer_file if transformer_file else model_path
        fp8_cache_path = fp8_cache_path_for(cache_source)
        if fp8_cache_path.exists():
            overrides["transformer"] = load_fp8_cached_transformer(
                fp8_cache_path, config_source=model_path,
            )
            fp8_cache_hit = True

    print(f"[load] {model_path} (dtype=bf16, quant={quant}, "
          f"compile={compile_transformer}, sage={sage_attention}, "
          f"fp8_cache={'hit' if fp8_cache_hit else 'miss' if fp8_cache_path else 'n/a'})")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, **overrides,
    )
    pipe._glt_quant = quant  # downstream LoRA path selection

    if quant == "fp8_weight" and fp8_cache_hit:
        # peft's TorchaoLoraLinear bridge needs the patch even on cache
        # hit (no quantize call happened to apply it).
        try:
            from torchao.quantization import Float8WeightOnlyConfig
            patch_peft_torchao_fp8(Float8WeightOnlyConfig)
        except ImportError:
            from torchao.quantization import float8_weight_only as _f8wo
            patch_peft_torchao_fp8(_f8wo)

    if quant in ("fp8_weight", "fp8_dynamic") and not fp8_cache_hit:
        try:
            fp8_quantize_torchao(
                pipe, mode="dynamic" if quant == "fp8_dynamic" else "weight-only",
            )
            if fp8_cache_path is not None:
                save_fp8_cached_transformer(pipe.transformer, fp8_cache_path)
        except ImportError as e:
            print(f"[warn] torchao unavailable ({e}); continuing in bf16. "
                  f"pip install torchao to fix.")

    elif quant == "fp8_quanto":
        try:
            fp8_quantize_quanto(pipe)
        except ImportError as e:
            print(f"[warn] optimum-quanto unavailable ({e}); continuing in bf16.")

    elif quant == "int8_convrot":
        if pipe.transformer is None:
            print("[int8_convrot] no transformer on pipeline, skipping")
        else:
            cache_source = transformer_file if transformer_file else model_path
            cache_path = int8_convrot_cache_path_for(cache_source)
            if cache_path.exists():
                report = int8_convrot_load_cached(pipe.transformer, cache_path)
                tag = "hit" if not report.get("missing_count") else "partial"
                print(
                    f"[int8_convrot] cache {tag} ({cache_path.name}): "
                    f"{report['loaded']}/{report['expected']} module(s) loaded"
                    + (f", {report['missing_count']} missing" if report.get("missing_count") else "")
                )
                if report.get("missing_count"):
                    quantize_transformer_int8_convrot(pipe.transformer)
            else:
                report = quantize_transformer_int8_convrot(pipe.transformer)
                print(
                    f"[int8_convrot] quantized {report['quantized']}/"
                    f"{report['total_linears']} linears "
                    f"(excluded by name: {report['excluded_by_name']}, "
                    f"by shape: {report['excluded_by_shape']})"
                )
                try:
                    save_report = int8_convrot_save_cache(pipe.transformer, cache_path)
                    print(
                        f"[int8_convrot] wrote cache {cache_path.name} "
                        f"({save_report['modules']} modules, "
                        f"{save_report['mb_on_disk']:.0f} MB)"
                    )
                except Exception as e:
                    print(f"[int8_convrot] WARN cache write failed: {e}")

    pipe.to("cuda")
    _vram("after transformer load + quant")

    if sage_attention:
        install_sage_attention()

    if compile_transformer:
        if pipe.transformer is None:
            print("[compile] no transformer attribute on pipeline, skipping")
        else:
            # `mode="default"` rather than `reduce-overhead`: the latter
            # uses CUDA graphs (`cudagraph_trees`) which keep large
            # private memory pools (~5+ GB at batch=3 1024^2) that the
            # global allocator can't reuse for the VAE decode that runs
            # after each denoise. Net effect: OOM on a 24 GB card. Mode
            # `default` keeps inductor's kernel fusion + autotune
            # (the bulk of the speedup) without the CUDA-graph pool tax.
            # Empirical on Klein int8_convrot @ B=1 1024x1024:
            # reduce-overhead = 0.62 s/step, default ≈ 0.85 s/step,
            # both vs no-compile = 1.07 s/step.
            print("[compile] torch.compile(transformer, mode='default', fullgraph=False)")
            pipe.transformer = torch.compile(
                pipe.transformer, mode="default", fullgraph=False,
            )

    return pipe
