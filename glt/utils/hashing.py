"""Cheap content fingerprints (size + head + tail) for fast collision checks."""
from __future__ import annotations

import hashlib
from pathlib import Path


def quick_file_hash(p: Path) -> str:
    """SHA-256 of (size || first 64 KB || last 64 KB). Good enough to tell two
    safetensors files apart in seconds — a full sha256 of a 500 MB file would
    take minutes. False positives (same fingerprint, different content) are
    vanishingly unlikely for genuine weight files."""
    h = hashlib.sha256()
    size = p.stat().st_size
    h.update(size.to_bytes(8, "little"))
    with p.open("rb") as f:
        h.update(f.read(64 * 1024))
        if size > 128 * 1024:
            f.seek(-64 * 1024, 2)
            h.update(f.read(64 * 1024))
    return h.hexdigest()
