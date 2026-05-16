"""ComfyUI-parity tensor dumping for noise / latents / sigmas / embeds.

Pair with a Comfy-side dump at the same seed + resolution to bisect any
numerical divergence between the two pipelines.
"""
from __future__ import annotations

from pathlib import Path


def setup_debug_dump(pipe, debug_dir: str, batch_embeds):
    """Wire up the dump. Fires ONCE per process — re-run the script to dump
    again. Returns a `callback_on_step_end` to pass to `pipe(...)`, or `None`
    if a previous call already dumped this run.

    Files written to `debug_dir`:

      - `prompt_embeds.pt`        — `[B, seq, hidden]` from cached TE output
      - `noise_initial.pt`        — `[B, h*w, 128]` post-pack noise
      - `latent_ids.pt`           — `[B, h*w, 3]` positional ids
      - `sigmas.pt`               — `[steps+1]` final shifted schedule
      - `latents_after_step_N.pt` — one per denoising step
    """
    import torch
    if getattr(setup_debug_dump, "_done", False):
        return None
    out = Path(debug_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(batch_embeds.detach().cpu(), out / "prompt_embeds.pt")
    print(f"[debug-dump] prompt_embeds.pt shape={tuple(batch_embeds.shape)} "
          f"dtype={batch_embeds.dtype} -> {out}")

    _orig_pl = pipe.prepare_latents

    def patched_pl(*a, **kw):
        latents, latent_ids = _orig_pl(*a, **kw)
        torch.save(latents.detach().cpu(), out / "noise_initial.pt")
        torch.save(latent_ids.detach().cpu(), out / "latent_ids.pt")
        print(f"[debug-dump] noise_initial.pt shape={tuple(latents.shape)} "
              f"dtype={latents.dtype}")
        pipe.prepare_latents = _orig_pl  # restore — once is enough
        return latents, latent_ids
    pipe.prepare_latents = patched_pl

    sigmas_logged = [False]

    def callback_on_step_end(p, step, timestep, kwargs_dict):
        latents = kwargs_dict.get("latents")
        if latents is not None:
            torch.save(latents.detach().cpu(), out / f"latents_after_step_{step}.pt")
        if not sigmas_logged[0]:
            sigmas = getattr(p.scheduler, "sigmas", None)
            if sigmas is not None:
                torch.save(sigmas.detach().cpu(), out / "sigmas.pt")
                print(f"[debug-dump] sigmas.pt values={sigmas.cpu().tolist()}")
            sigmas_logged[0] = True
        return kwargs_dict

    setup_debug_dump._done = True
    return callback_on_step_end
