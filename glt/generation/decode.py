"""Batch VAE decoding (latents → PIL images)."""
from __future__ import annotations


def batch_decode_latents(pipe, latents_list, chunk_size: int = 4):
    """Decode a list of `[1, C, H, W]` latents through the VAE in batches and
    postprocess to PIL images. Entries that are None pass through."""
    import torch
    indices_with_latents = [(i, l) for i, l in enumerate(latents_list) if l is not None]
    if not indices_with_latents:
        return [None] * len(latents_list)

    pils_by_index: dict[int, "object"] = {}
    with torch.no_grad():
        for start in range(0, len(indices_with_latents), chunk_size):
            chunk = indices_with_latents[start:start + chunk_size]
            batch = torch.cat([l for _, l in chunk], dim=0)
            decoded = pipe.vae.decode(batch, return_dict=False)[0]
            pils = pipe.image_processor.postprocess(decoded, output_type="pil")
            for (orig_idx, _), pil in zip(chunk, pils):
                pils_by_index[orig_idx] = pil

    return [pils_by_index.get(i) for i in range(len(latents_list))]
