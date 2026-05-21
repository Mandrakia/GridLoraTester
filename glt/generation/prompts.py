"""Prompt encoding + SQLite embeds cache.

Cache is keyed by `(prompt_text, te_model_id)` so trigger-substituted
prompts get distinct entries and a Qwen swap (different file / dtype)
invalidates only the relevant rows. Lives at `~/.cache/glt/embeds.db`
(override with `GLT_EMBEDS_CACHE_DB`).

This module assumes the **two-phase pipeline lifecycle** (see
`pipeline.build`). The caller passes a TE-only pipeline (Qwen on GPU,
transformer/vae = None). We encode every missing prompt, persist to
the cache, return the aligned embeds list. The caller is expected to
`del pipe_te` afterwards to free Qwen entirely before loading the
transformer.

No offload juggling, no transformer parking — Qwen is the ONLY thing
on the GPU during this phase, so there's nothing to make room for.
"""
from __future__ import annotations

import sqlite3

import torch

from . import prompt_cache


def encode_prompts(
    pipe_te,
    prompts: list[str],
    cache_conn: sqlite3.Connection,
    te_model_id: str,
    engine=None,
) -> list[torch.Tensor]:
    """Pre-encode every prompt, hitting the embeds cache where possible.

    Returns a list of `prompt_embeds` tensors on CUDA in bf16, aligned
    one-to-one with `prompts` (duplicates produce shared tensors — fine,
    they're read-only inputs to the transformer).

    On a full cache hit `pipe_te` is not touched (it can be `None`
    even — the caller didn't need to build one). On a partial hit, only
    missing prompts go through Qwen.
    """
    distinct = list(dict.fromkeys(prompts))  # preserve order, dedupe
    cached_blobs = prompt_cache.get_many(cache_conn, distinct, te_model_id)
    n_hit = len(cached_blobs)
    print(
        f"[embeds:cache] {n_hit}/{len(distinct)} distinct prompt(s) cached "
        f"(te={te_model_id})"
    )

    target_dtype = torch.bfloat16
    cached_tensors: dict[str, torch.Tensor] = {
        text: prompt_cache.deserialize_embed(blob, device="cuda", dtype=target_dtype)
        for text, blob in cached_blobs.items()
    }

    missing = [t for t in distinct if t not in cached_tensors]
    if missing:
        if pipe_te is None or pipe_te.text_encoder is None:
            raise RuntimeError(
                f"{len(missing)} prompt(s) not in cache but no TE pipeline "
                f"was supplied — the caller should `build_te_pipeline(...)` "
                f"before calling encode_prompts() when missing > 0."
            )
        print(f"[encode] running text encoder on {len(missing)} missing prompt(s)")
        with torch.no_grad():
            for i, p in enumerate(missing):
                if engine is not None:
                    prompt_embeds = engine.encode_single(pipe_te, p)
                else:
                    prompt_embeds, _ = pipe_te.encode_prompt(prompt=p)
                prompt_embeds = prompt_embeds.to(
                    device="cuda", dtype=target_dtype, non_blocking=False,
                ).contiguous()
                cached_tensors[p] = prompt_embeds
                try:
                    blob = prompt_cache.serialize_embed(prompt_embeds)
                    prompt_cache.put(cache_conn, p, te_model_id, blob)
                except Exception as e:
                    print(f"[embeds:cache] WARN save failed: {type(e).__name__}: {e}")
                print(
                    f"  [{i + 1}/{len(missing)}] shape={tuple(prompt_embeds.shape)} "
                    f":: {p[:60]}"
                )

    return [cached_tensors[p] for p in prompts]


def all_prompts_cached(
    cache_conn: sqlite3.Connection,
    prompts: list[str],
    te_model_id: str,
) -> bool:
    """Pre-check: True iff `build_te_pipeline()` can be skipped entirely."""
    distinct = list(dict.fromkeys(prompts))
    hits = prompt_cache.get_many(cache_conn, distinct, te_model_id)
    return len(hits) == len(distinct)
