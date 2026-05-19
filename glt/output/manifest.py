"""Manifest serialization + per-row artifact orchestration."""
from __future__ import annotations

import json
from pathlib import Path

from .images import atomic_write
from .html import write_html


def write_grid_artifacts(output_dir: Path, manifest: dict, write_html_flag: bool = True) -> None:
    """Persist `manifest.json` and (optionally) `index.html`. Designed to be
    safe to call after every row update — uses atomic rename writes so a
    browser auto-refreshing the dashboard never sees a half-written file."""
    atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )
    if write_html_flag:
        write_html(output_dir, manifest)
