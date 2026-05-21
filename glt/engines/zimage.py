"""Z-Image-Turbo engine — Tongyi's 6B single-stream DiT (`ZImageTransformer2DModel`).

Diverges from FLUX-2 in four validated places (verified against the installed
diffusers `ZImagePipeline`, not assumed):

  1. **embeds are a list**: `__call__(prompt_embeds=...)` wants a *list* of
     per-prompt tensors (variable length, concatenated single-stream inside
     the transformer), so `denoise_batch` passes the list directly instead of
     `torch.cat`-ing into one padded `[B, seq, dim]` tensor.
  2. **encode_prompt returns a list**: `(prompt_embeds, neg)` where
     prompt_embeds is one element per prompt → take `[0]` for a single prompt.
  3. **raw latent output**: with `output_type="latent"` the pipeline returns
     UN-scaled latents — `(latents / scaling_factor) + shift_factor` is applied
     only on the pixel branch (FLUX-2 denormalizes *before* the latent
     short-circuit). So `decode()` must apply that scaling itself. No
     packing/unpatchify — single-stream latents stay `[B, C, H, W]`.
  4. **true CFG**: Turbo is guidance-distilled and runs CFG-free at
     `guidance_scale=1.0` → `do_classifier_free_guidance=False` at encode time,
     so no negative embeds and the cache stays one-tensor-per-prompt.

int8_convrot lands in a later phase (the single-stream layout — separate
to_q/to_k/to_v, `feed_forward.w{1,2,3}` — needs its own exclusion list +
1-to-1 LoRA resolver). Until then this engine offers bf16 + fp8_weight.
"""
from __future__ import annotations

from typing import Any

from .base import EngineSpec, ModelEngine


class ZImageEngine(ModelEngine):
    spec = EngineSpec(
        family="zimage",
        label="Z-Image Turbo (6B)",
        default_model_id="Tongyi-MAI/Z-Image-Turbo",
        quant_choices=("auto", "none", "fp8_weight", "int8_convrot"),
        default_steps=8,        # Turbo is distilled — 8 steps is the sweet spot
        # CFG-free: ZImagePipeline treats guidance_scale>0 as CFG-on (needs a
        # negative + a 2nd forward pass), which a distilled Turbo neither wants
        # nor we provide. guidance_scale=0.0 is the single-pass path (matches
        # the pipeline's own Turbo example).
        default_guidance=0.0,
        latent_multiple=16,     # same as FLUX-2 (vae_scale_factor * 2)
    )

    # ---- quant resolution ------------------------------------------------

    def _resolve_quant(self, quant: str) -> str:
        q = (quant or "auto").strip() or "auto"
        if q == "auto":
            # Same GPU-capability heuristic as FLUX-2: Ampere → int8_convrot
            # (native INT8 IMMA, ~2× denoise), Ada/Hopper → fp8_weight, else
            # none. All three are supported for Z-Image.
            from ..pipeline.int8_convrot import recommended_quant_for_gpu

            rec = recommended_quant_for_gpu()
            chosen = rec if rec in self.spec.quant_choices else "none"
            print(f"[zimage] quant auto -> {chosen} (per GPU compute capability)")
            return chosen
        if q not in self.spec.quant_choices:
            print(f"[zimage] quant={q!r} not supported; falling back to 'none'. "
                  f"Choices: {self.spec.quant_choices}")
            return "none"
        return q

    # ---- Phase 1: text encoder -------------------------------------------

    def te_model_id(self, model_path: str, te_file: str | None, te_dtype: str) -> str:
        from ..generation.prompt_cache import compute_te_model_id

        # Family-prefixed: belt-and-suspenders so a Z-Image embed never aliases
        # a FLUX-2 one even if a user points both at an identically-named dir.
        return f"zimage::{compute_te_model_id(model_path, te_file, te_dtype)}"

    def build_te_pipeline(self, *, model_path: str, te_dtype: str, te_file: str | None) -> Any:
        import torch
        from diffusers import ZImagePipeline

        from ..pipeline.loaders import load_text_encoder_from_file, torch_dtype as _dtype

        overrides: dict = {"transformer": None, "vae": None}
        if te_file:
            overrides["text_encoder"] = load_text_encoder_from_file(
                te_file, config_source=model_path, target_dtype=_dtype(te_dtype),
            )

        print(f"[load:te] {model_path} (ZImage Qwen3, te_dtype={te_dtype})")
        pipe = ZImagePipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **overrides,
        )
        if pipe.text_encoder is not None:
            pipe.text_encoder.to("cuda")
        return pipe

    def encode_single(self, pipe_te: Any, prompt: str) -> Any:
        # CFG-free → no negative embeds; the returned prompt_embeds is a list
        # with one element per prompt (single prompt here → element 0).
        prompt_embeds, _ = pipe_te.encode_prompt(
            prompt=prompt, do_classifier_free_guidance=False,
        )
        if isinstance(prompt_embeds, (list, tuple)):
            prompt_embeds = prompt_embeds[0]
        return prompt_embeds

    # ---- Phase 2: transformer + VAE --------------------------------------

    def build_transformer_pipeline(
        self,
        *,
        quant: str,
        model_path: str,
        transformer_file: str | None,
        transformer_dtype: str,
        vae_file: str | None,
        vae_dtype: str,
        compile_transformer: bool,
        sage_attention: bool,
    ) -> Any:
        import torch
        from diffusers import ZImagePipeline

        quant = self._resolve_quant(quant)
        if transformer_file or vae_file:
            print("[zimage] single-file transformer/vae override is not supported "
                  "yet (Z-Image ships in diffusers layout); using model_path components.")
        if sage_attention:
            print("[zimage] sage_attention is FLUX-2-specific; ignored for Z-Image.")

        print(f"[load] {model_path} (ZImage transformer + VAE, quant={quant}, "
              f"compile={compile_transformer})")
        pipe = ZImagePipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            text_encoder=None, tokenizer=None,
        )
        pipe._glt_quant = quant

        if quant == "fp8_weight":
            try:
                from ..pipeline.fp8 import fp8_quantize_torchao

                fp8_quantize_torchao(pipe, mode="weight-only")
            except ImportError as e:
                print(f"[warn] torchao unavailable ({e}); continuing in bf16.")

        elif quant == "int8_convrot":
            if pipe.transformer is None:
                print("[int8_convrot] no transformer on pipeline, skipping")
            else:
                self._apply_int8_convrot(pipe.transformer, model_path)

        pipe.to("cuda")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[vram] after Z-Image transformer load: "
                  f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        if compile_transformer:
            # SKIP — torch.compile is net-negative for Z-Image. Unlike FLUX-2
            # (which pads prompt embeds to a fixed sequence length → one stable
            # shape → compile once), Z-Image is single-stream and concatenates
            # variable-length text+image tokens. Each differently-sized prompt
            # batch is a NEW shape, so dynamo's guards fail and it recompiles
            # per batch — measured ~100-120 s recompiles vs ~8-12 s eager steady.
            # Running eager is strictly faster here. (Padding embeds to a fixed
            # length, or dynamic-shape compile, could reclaim it later.)
            print("[compile] SKIPPED for Z-Image: variable-length single-stream "
                  "tokens force a torch.compile recompile per prompt-batch shape "
                  "(net-negative). Running eager.")
        return pipe

    def denoise_batch(
        self,
        pipe: Any,
        embeds: list,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        comfyui_noise: bool,
        debug_dump: str | None,
    ) -> Any:
        import torch

        from ..generation.latents import generator_device

        B = len(embeds)
        # Per-item generators with the SAME seed → each prompt is rendered as if
        # alone at that seed (GLT's batch convention). CPU generators when
        # comfyui_noise so the RNG sequence is host-stable. The FLUX-2 fp32
        # prepare_latents detour for bit-exact ComfyUI parity is deferred:
        # Z-Image's prepare_latents passes dtype positionally, so the patch
        # needs validating against weights before enabling.
        gen_dev = "cpu" if comfyui_noise else generator_device(pipe)
        generators = [torch.Generator(device=gen_dev).manual_seed(seed) for _ in range(B)]

        if debug_dump:
            print("[zimage] debug_dump not wired for Z-Image yet; ignoring.")

        # Turbo is CFG-free: guidance_scale>0 would require negative embeds + a
        # 2nd forward pass we don't supply. Clamp to 0 (single pass).
        gs = float(guidance)
        if gs > 0 and not getattr(self, "_cfg_warned", False):
            print(f"[zimage] guidance_scale={gs} requested, but Turbo runs CFG-free; "
                  f"using guidance_scale=0 (single pass, no negatives).")
            self._cfg_warned = True
        gs = max(0.0, gs) if gs <= 0 else 0.0

        return pipe(
            prompt_embeds=embeds,           # LIST — not torch.cat'd
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=gs,
            generator=generators,
            output_type="latent",
        ).images

    def decode(self, pipe: Any, latents_list: list, chunk_size: int = 4) -> list:
        # output_type="latent" returns RAW latents; apply the denormalization
        # the pixel branch would have done before VAE decode.
        from ..generation.decode import batch_decode_latents

        sf = float(pipe.vae.config.scaling_factor)
        shift = float(getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0)
        vdt = pipe.vae.dtype
        scaled = [
            None if l is None else ((l.to(vdt) / sf) + shift)
            for l in latents_list
        ]
        return batch_decode_latents(pipe, scaled, chunk_size=chunk_size)

    # ---- int8_convrot (single-stream) ------------------------------------

    def _apply_int8_convrot(self, transformer, model_path: str) -> None:
        """Quantize the Z-Image transformer to INT8 ConvRot, using the disk
        cache when present. Mirrors the FLUX-2 build path but with the
        single-stream exclusion list. Cache slug is the model_path, so the
        Z-Image cache never collides with the FLUX-2 one."""
        from ..pipeline.int8_convrot import (
            ZIMAGE_EXCLUDE_SUBSTRINGS,
            cache_path_for,
            load_into_transformer_from_cache,
            quantize_transformer,
            save_quantized_transformer,
        )

        cache_path = cache_path_for(model_path)
        if cache_path.exists():
            report = load_into_transformer_from_cache(transformer, cache_path)
            tag = "hit" if not report.get("missing_count") else "partial"
            print(
                f"[int8_convrot] cache {tag} ({cache_path.name}): "
                f"{report['loaded']}/{report['expected']} module(s) loaded"
                + (f", {report['missing_count']} missing" if report.get("missing_count") else "")
            )
            if report.get("missing_count"):
                quantize_transformer(transformer, exclude_substrings=ZIMAGE_EXCLUDE_SUBSTRINGS)
            return

        report = quantize_transformer(transformer, exclude_substrings=ZIMAGE_EXCLUDE_SUBSTRINGS)
        print(
            f"[int8_convrot] quantized {report['quantized']}/{report['total_linears']} "
            f"linears (excluded by name: {report['excluded_by_name']}, "
            f"by shape: {report['excluded_by_shape']})"
        )
        try:
            save_report = save_quantized_transformer(transformer, cache_path)
            print(
                f"[int8_convrot] wrote cache {cache_path.name} "
                f"({save_report['modules']} modules, {save_report['mb_on_disk']:.0f} MB)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[int8_convrot] WARN cache write failed: {e}")

    def convrot_exclude_substrings(self) -> tuple[str, ...]:
        from ..pipeline.int8_convrot import ZIMAGE_EXCLUDE_SUBSTRINGS

        return tuple(ZIMAGE_EXCLUDE_SUBSTRINGS)

    def convrot_lora_resolver(self):
        from ..pipeline.int8_convrot import resolve_targets_zimage

        return resolve_targets_zimage
