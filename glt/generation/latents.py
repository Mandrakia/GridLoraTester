"""Denoise loops: single-prompt + batched, plus ComfyUI-parity noise mode."""
from __future__ import annotations

from .debug_dump import setup_debug_dump


def generator_device(pipe):
    """Pick a `torch.Generator` device that matches where the pipeline will
    actually allocate the initial noise.

    Inspects `pipe._execution_device` (the canonical "where does the noise
    land" device in diffusers), with a fallback to scanning the transformer
    params and finally to CPU (CPU generators are always tolerated by
    `randn_tensor` — they just get copied to the target device)."""
    import torch
    try:
        dev = pipe._execution_device
        if dev is not None:
            return dev
    except Exception:
        pass
    try:
        return next(pipe.transformer.parameters()).device
    except Exception:
        return torch.device("cpu")


def generate_latent(pipe, prompt_embeds, width: int, height: int,
                    steps: int, guidance: float, seed: int):
    """Run denoising for ONE prompt — returns `[1, C, H, W]` latent.
    VAE decoding is deferred and batched at end-of-row."""
    import torch
    generator = torch.Generator(device=generator_device(pipe)).manual_seed(seed)
    return pipe(
        prompt_embeds=prompt_embeds,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="latent",
    ).images


def generate_latents_batched(pipe, prompt_embeds_list, width: int, height: int,
                             steps: int, guidance: float, seed: int,
                             comfyui_noise: bool = False,
                             debug_dump: str | None = None):
    """Run denoising for a BATCH of prompts in one `pipe()` call. Returns
    `[B, C, H, W]` latent.

    Noise device:

      - `comfyui_noise=False` (default): generators on `pipe`'s device (CUDA
        in the normal case). Fast noise generation but a different RNG
        sequence from ComfyUI for the same seed integer.
      - `comfyui_noise=True`: generators on CPU, same seed for each batch
        item. Matches what ComfyUI does (KSampler runs each prompt with a
        fresh `torch.manual_seed(seed)` on CPU). Each batch item gets the
        same CPU-generated noise, bit-identical to running that prompt
        alone in ComfyUI with the same seed.
    """
    import torch
    B = len(prompt_embeds_list)
    batch_embeds = torch.cat(prompt_embeds_list, dim=0)

    gen_dev = "cpu" if comfyui_noise else generator_device(pipe)
    generators = [torch.Generator(device=gen_dev).manual_seed(seed) for _ in range(B)]
    if not getattr(generate_latents_batched, "_noise_logged", False):
        sample = torch.randn((4,), generator=torch.Generator(device=str(gen_dev)).manual_seed(seed),
                             device="cpu" if str(gen_dev) == "cpu" else "cuda")
        print(f"[noise] comfyui_noise={comfyui_noise} gen_dev={gen_dev}  "
              f"first 4 samples for seed={seed}: {sample.cpu().tolist()}")
        generate_latents_batched._noise_logged = True

    # ComfyUI noise mode: also force the randn dtype to fp32 then cast.
    # `torch.randn(shape, dtype=bf16)` ≠ `torch.randn(shape, dtype=fp32).to(bf16)`
    # at identical seed/generator — different code paths advance the Philox
    # state differently. ComfyUI's prepare_noise does fp32→cast; diffusers'
    # `randn_tensor` passes the model dtype directly. Patching `prepare_latents`
    # to insert the fp32 detour gives bit-identical-modulo-bf16-truncation
    # noise to ComfyUI for the same seed.
    patched = None
    if comfyui_noise:
        _orig_prepare_latents = pipe.prepare_latents

        def patched_prepare_latents(*a, **kw):
            target_dtype = kw.get("dtype")
            if target_dtype is not None and target_dtype != torch.float32:
                kw = dict(kw)
                kw["dtype"] = torch.float32
                latents, latent_ids = _orig_prepare_latents(*a, **kw)
                latents = latents.to(target_dtype)
            else:
                latents, latent_ids = _orig_prepare_latents(*a, **kw)
            return latents, latent_ids

        pipe.prepare_latents = patched_prepare_latents
        patched = _orig_prepare_latents

    pipe_kwargs: dict = {}
    if debug_dump:
        cb = setup_debug_dump(pipe, debug_dump, batch_embeds)
        if cb is not None:
            pipe_kwargs["callback_on_step_end"] = cb

    try:
        return pipe(
            prompt_embeds=batch_embeds,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generators,
            output_type="latent",
            **pipe_kwargs,
        ).images
    finally:
        if patched is not None:
            pipe.prepare_latents = patched
