"""SQLite-backed prompt-embedding cache.

Replaces the per-file `.pt` cache that used to live alongside the prompts
file. Key is the (final post-substitution) prompt text + text-encoder
identity — so changing the trigger word produces a different cache entry,
and swapping text encoders invalidates the relevant entries automatically.

The DB lives at `~/.cache/glt/embeds.db` by default (override with the
`GLT_EMBEDS_CACHE_DB` env var). It is **regenerable** — no user state in
it; `rm` it at any time, the next run repopulates. Kept entirely separate
from `ui/data/glt.db` (which has the curated dataset state).

Storage is `torch.save`'d bytes: dtype and shape are self-describing,
so changing torch's default dtype later doesn't require schema work.
"""
from __future__ import annotations

import io
import os
import sqlite3
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_embeds (
    prompt_text  TEXT NOT NULL,
    te_model_id  TEXT NOT NULL,
    embed_blob   BLOB NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (prompt_text, te_model_id)
);
"""


def default_cache_path() -> Path:
    p = os.environ.get("GLT_EMBEDS_CACHE_DB")
    if p:
        return Path(p)
    return Path.home() / ".cache" / "glt" / "embeds.db"


def open_cache(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else default_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    return conn


def get_many(
    conn: sqlite3.Connection,
    prompts: list[str],
    te_model_id: str,
) -> dict[str, bytes]:
    """Bulk lookup: returns {prompt_text: blob_bytes} for cache hits."""
    if not prompts:
        return {}
    # SQLite has a parameter cap (typically 999). Chunk if needed.
    out: dict[str, bytes] = {}
    CHUNK = 500
    for i in range(0, len(prompts), CHUNK):
        chunk = prompts[i:i + CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT prompt_text, embed_blob
              FROM prompt_embeds
             WHERE te_model_id = ?
               AND prompt_text IN ({placeholders})
            """,
            (te_model_id, *chunk),
        ).fetchall()
        for r in rows:
            out[r[0]] = r[1]
    return out


def put(
    conn: sqlite3.Connection,
    prompt_text: str,
    te_model_id: str,
    blob: bytes,
) -> None:
    """Insert or refresh the cached embedding bytes for one prompt."""
    conn.execute(
        """
        INSERT INTO prompt_embeds (prompt_text, te_model_id, embed_blob)
        VALUES (?, ?, ?)
        ON CONFLICT(prompt_text, te_model_id) DO UPDATE SET
            embed_blob = excluded.embed_blob,
            created_at = datetime('now')
        """,
        (prompt_text, te_model_id, blob),
    )


# ---- Tensor serialization helpers -------------------------------------

def serialize_embed(tensor) -> bytes:
    """torch.save the tensor to bytes. Self-describing: dtype + shape are
    embedded so a later load round-trips without schema knowledge."""
    import torch
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buf)
    return buf.getvalue()


def deserialize_embed(blob: bytes, device: str = "cuda", dtype=None):
    """Inverse of serialize_embed. Moves to `device` and optionally casts."""
    import torch
    buf = io.BytesIO(blob)
    t = torch.load(buf, weights_only=False, map_location="cpu")
    t = t.to(device=device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t.contiguous()


def compute_te_model_id(
    model_path: str,
    text_encoder_file: str | None,
    text_encoder_dtype: str,
) -> str:
    """Identity string for the text encoder used to encode prompts.
    Changes in any of these invalidate the relevant cache entries
    automatically (different te_model_id → different PK)."""
    src = text_encoder_file or model_path
    return f"{src}::{text_encoder_dtype}"
