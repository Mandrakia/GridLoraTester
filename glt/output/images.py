"""Atomic file writes + image saving helpers."""
from __future__ import annotations

from pathlib import Path


def save_image(img, path: Path, fmt: str, quality: int) -> None:
    """Persist a PIL image.

    - `jpg` / `webp`: lossy at `quality`. These outputs are regenerable
      (seed + LoRA + prompt are stored), so lossless is wasted bytes — and
      face scoring runs on the in-memory PIL, not the saved file, so a lossy
      save never changes a cell's score.
    - `png`: lossless fallback for the rare keeper. `compress_level=1` is much
      faster than PIL's default 6, with negligible size cost on natural images.
    """
    if fmt == "jpg":
        img.convert("RGB").save(path, format="JPEG", quality=quality, optimize=False)
    elif fmt == "webp":
        img.save(path, format="WEBP", quality=quality, method=4)
    else:
        img.save(path, format="PNG", optimize=False, compress_level=1)


def atomic_write(path: Path, content: str) -> None:
    """Write to a sibling `.tmp` file and rename. A reader (e.g. a browser
    auto-refreshing `index.html`) never sees a truncated/half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
