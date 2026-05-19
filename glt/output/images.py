"""Atomic file writes + image saving helpers."""
from __future__ import annotations

from pathlib import Path


def save_image(img, path: Path, fmt: str, jpg_quality: int) -> None:
    """Persist a PIL image. PNG uses `compress_level=1` (much faster than the
    PIL default of 6, with negligible size impact on natural images)."""
    if fmt == "jpg":
        img.convert("RGB").save(path, format="JPEG", quality=jpg_quality, optimize=False)
    else:
        img.save(path, format="PNG", optimize=False, compress_level=1)


def atomic_write(path: Path, content: str) -> None:
    """Write to a sibling `.tmp` file and rename. A reader (e.g. a browser
    auto-refreshing `index.html`) never sees a truncated/half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
