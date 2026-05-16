"""SageAttention SDPA monkey-patch + telemetry."""
from __future__ import annotations


def install_sage_attention() -> None:
    """Replace `torch.nn.functional.scaled_dot_product_attention` with a
    SageAttention-backed version when the call fits its supported shape
    (no attention mask, no causal, no dropout — exactly the FLUX joint
    attention case). Falls through to the original SDPA when the call has a
    mask / `is_causal` / `dropout > 0`, so the Qwen3 text encoder (causal)
    keeps working unmodified.

    Global monkey-patch. Apply ONCE before any forward pass.
    """
    import torch.nn.functional as F
    try:
        from sageattention import sageattn
    except ImportError as e:
        raise ImportError(
            "--sage-attention requires the `sageattention` pip package. "
            "Install with: pip install sageattention"
        ) from e

    if getattr(F, "_sage_patched", False):
        return  # already patched, e.g. by a previous benchmark iteration

    _orig_sdpa = F.scaled_dot_product_attention
    stats = {
        "calls": 0,
        "sage": 0,
        "fallback": 0,
        "errors": 0,
        "masked": 0,
        "causal": 0,
        "dropout": 0,
        "first_error": None,
    }

    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, **kwargs):
        stats["calls"] += 1
        # Sage handles only the no-mask, no-causal, no-dropout case (diffusion
        # cross-attn). Anything else falls back to the stock impl. Wrap in
        # try/except so a sage failure (unsupported head dim, etc.) doesn't
        # crash the run — we just degrade to original SDPA.
        if attn_mask is None and dropout_p == 0.0 and not is_causal:
            try:
                out = sageattn(query, key, value)
                stats["sage"] += 1
                return out
            except Exception as e:
                stats["errors"] += 1
                if stats["first_error"] is None:
                    stats["first_error"] = f"{type(e).__name__}: {e}"
        else:
            if attn_mask is not None:
                stats["masked"] += 1
            if is_causal:
                stats["causal"] += 1
            if dropout_p != 0.0:
                stats["dropout"] += 1
        stats["fallback"] += 1
        return _orig_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, **kwargs,
        )

    F.scaled_dot_product_attention = _sage_sdpa
    F._sage_patched = True
    F._sage_stats = stats
    print("[sage] SDPA patched (sage for diffusion attn, fallback for masked/causal)")


def sage_attention_stats_snapshot() -> dict | None:
    try:
        import torch.nn.functional as F
        stats = getattr(F, "_sage_stats", None)
    except Exception:
        return None
    return dict(stats) if stats else None


def print_sage_attention_stats(label: str = "stats", since: dict | None = None) -> None:
    stats = sage_attention_stats_snapshot()
    if not stats:
        return
    if since:
        stats = {
            k: (stats.get(k, 0) - since.get(k, 0))
            for k in ("calls", "sage", "fallback", "errors", "masked", "causal", "dropout")
        } | {"first_error": stats.get("first_error")}
    calls = stats["calls"]
    if calls == 0:
        print(f"[sage] {label}: no SDPA calls observed")
        return
    sage_pct = 100.0 * stats["sage"] / calls
    fallback_pct = 100.0 * stats["fallback"] / calls
    msg = (
        f"[sage] {label}: calls={calls} sage={stats['sage']} ({sage_pct:.1f}%) "
        f"fallback={stats['fallback']} ({fallback_pct:.1f}%) "
        f"errors={stats['errors']} masked={stats['masked']} "
        f"causal={stats['causal']} dropout={stats['dropout']}"
    )
    if stats["errors"] and stats["first_error"]:
        msg += f" first_error={stats['first_error']}"
    print(msg)
