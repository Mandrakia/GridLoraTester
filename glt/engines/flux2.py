"""FLUX.2-klein engine — thin wrapper over the original `pipeline.build` +
`generation` code paths. This is the reference behavior: routing grid.py
through it must be byte-identical to the pre-engine code."""
from __future__ import annotations

from typing import Any

from .base import EngineSpec, ModelEngine


class Flux2Engine(ModelEngine):
    spec = EngineSpec(
        family="flux2",
        label="FLUX.2 Klein 9B",
        default_model_id="black-forest-labs/FLUX.2-klein-9B",
        # full set incl. int8_convrot — the original FLUX-2-tuned modes.
        quant_choices=("auto", "none", "fp8_weight", "fp8_dynamic", "fp8_quanto", "int8_convrot"),
        default_steps=4,
        default_guidance=1.0,
        latent_multiple=16,
    )

    def te_model_id(self, model_path: str, te_file: str | None, te_dtype: str) -> str:
        # NOT family-prefixed: preserves the embed-cache key shipped before the
        # engine refactor, so existing installs keep their cached embeds. The
        # model_path (klein vs another repo) already disambiguates families.
        from ..generation.prompt_cache import compute_te_model_id

        return compute_te_model_id(model_path, te_file, te_dtype)

    def build_te_pipeline(self, *, model_path: str, te_dtype: str, te_file: str | None) -> Any:
        from ..pipeline.build import build_te_pipeline

        return build_te_pipeline(model_path=model_path, qwen_dtype=te_dtype, qwen_file=te_file)

    def encode_single(self, pipe_te: Any, prompt: str) -> Any:
        prompt_embeds, _ = pipe_te.encode_prompt(prompt=prompt)
        return prompt_embeds

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
        from ..pipeline.build import build_transformer_pipeline

        return build_transformer_pipeline(
            quant=quant,
            model_path=model_path,
            transformer_file=transformer_file,
            transformer_dtype=transformer_dtype,
            vae_file=vae_file,
            vae_dtype=vae_dtype,
            compile_transformer=compile_transformer,
            sage_attention=sage_attention,
        )

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
        from ..generation.latents import generate_latents_batched

        return generate_latents_batched(
            pipe, embeds,
            width=width, height=height,
            steps=steps, guidance=guidance, seed=seed,
            comfyui_noise=comfyui_noise, debug_dump=debug_dump,
        )

    def convrot_exclude_substrings(self) -> tuple[str, ...]:
        from ..pipeline.int8_convrot import FLUX2_EXCLUDE_SUBSTRINGS

        return tuple(FLUX2_EXCLUDE_SUBSTRINGS)

    def convrot_lora_resolver(self):
        # FLUX-2 / BFL fused-QKV de-fusion resolver (the original behavior).
        from ..pipeline.int8_convrot.lora import _resolve_targets

        return _resolve_targets
