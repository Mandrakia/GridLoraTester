"""Model-family engine abstraction.

GLT generates LoRA×prompt grids against more than one base diffusion model
(FLUX.2-klein and Z-Image-Turbo so far). The two share GLT's whole outer
loop — embed cache, per-shape batching, face scoring, run history — but
diverge in the diffusers plumbing:

  - **pipeline class + components**: `Flux2KleinPipeline` (transformer +
    `AutoencoderKLFlux2`) vs `ZImagePipeline` (transformer + plain
    `AutoencoderKL`); both encode with a Qwen3 text encoder.
  - **prompt embeds shape**: FLUX-2 pads to one `[B, seq, dim]` tensor and
    batches with `torch.cat`; Z-Image keeps a *list* of variable-length
    per-prompt tensors (single-stream concatenation happens inside the
    transformer).
  - **`prepare_latents` return**: FLUX-2 returns `(latents, latent_ids)`;
    Z-Image returns `latents` alone — so the `comfyui_noise` fp32 detour
    patch differs.
  - **int8_convrot layout**: FLUX-2 fuses QKV in double/single blocks
    (LoRA targets need de-fusion slicing); Z-Image is single-stream with
    separate `to_q`/`to_k`/`to_v` (1-to-1 mapping). Different exclusion
    lists too.

A `ModelEngine` owns exactly those divergent bits behind one interface so
`modes/grid.py` stays model-agnostic. The text-encoder phase, embed cache,
and decode are shared by default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EngineSpec:
    """Static metadata for a model family. Drives CLI defaults + UI hints."""

    family: str                       # "flux2" | "zimage"
    label: str                        # human label for the UI
    default_model_id: str             # HF repo id used when --model-path is unset
    quant_choices: tuple[str, ...]    # quant modes this family supports
    default_steps: int                # denoise steps when none requested
    default_guidance: float           # guidance_scale when none requested
    latent_multiple: int = 16         # width/height must be a multiple of this


class ModelEngine(ABC):
    """One model family's diffusers plumbing. Stateless — holds no pipeline.

    Methods split along GLT's two-phase lifecycle (see `pipeline.build`):
    Phase 1 loads the text encoder alone and encodes every prompt; Phase 2
    loads transformer + VAE and denoises. The two are never co-resident.
    """

    spec: EngineSpec

    # ---- Phase 1: text-encoder -------------------------------------------

    @abstractmethod
    def te_model_id(self, model_path: str, te_file: str | None, te_dtype: str) -> str:
        """Embed-cache key component. Distinct base models get distinct keys
        because `model_path` differs; FLUX-2 keeps its pre-engine key verbatim
        so existing caches keep hitting."""

    @abstractmethod
    def build_te_pipeline(self, *, model_path: str, te_dtype: str, te_file: str | None) -> Any:
        """Load the text encoder + tokenizer ONLY, on GPU (transformer/VAE
        left None). Caller encodes then `del`s it."""

    @abstractmethod
    def encode_single(self, pipe_te: Any, prompt: str) -> Any:
        """Encode ONE prompt → one embed tensor (raw device/dtype; the caller
        moves it to cuda/bf16 and caches). For families whose `encode_prompt`
        returns a list, return that prompt's single element."""

    # ---- Phase 2: transformer + VAE --------------------------------------

    @abstractmethod
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
        """Load transformer + VAE, apply `quant`, move to GPU. No text encoder.
        Tags the pipeline with `_glt_quant` for downstream LoRA-path selection."""

    @abstractmethod
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
        """Denoise a batch of prompts (one `pipe()` call) → `[B, C, H, W]`
        latents. Owns the embed-combination (cat vs list) and the
        `comfyui_noise` `prepare_latents` patch, both family-specific."""

    def decode(self, pipe: Any, latents_list: list, chunk_size: int = 4) -> list:
        """Latents → PIL images. Shape-aware batching is generic across
        families (VAE decode + image_processor.postprocess), so this is shared;
        override only if a family needs special handling."""
        from ..generation.decode import batch_decode_latents

        return batch_decode_latents(pipe, latents_list, chunk_size=chunk_size)

    # ---- int8_convrot config (Phase 4) -----------------------------------

    def convrot_exclude_substrings(self) -> tuple[str, ...]:
        """Module-name substrings to skip when quantizing (embeddings,
        modulation, final projection — sensitivity-critical small layers).
        Raises if the family has no int8_convrot support."""
        raise NotImplementedError(
            f"int8_convrot not supported for engine {self.spec.family!r}"
        )

    def convrot_lora_resolver(self):
        """Return the LoRA-key → ``[(module_name, split_idx), ...]`` resolver
        used to bake LoRA deltas into this family's int8_convrot weights
        (FLUX-2 de-fuses QKV; Z-Image maps 1-to-1). Raises if unsupported."""
        raise NotImplementedError(
            f"int8_convrot not supported for engine {self.spec.family!r}"
        )
