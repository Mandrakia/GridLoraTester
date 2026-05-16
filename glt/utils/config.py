"""Config and prompt loading helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_prompts(path: Path) -> list[str]:
    """One prompt per line; blank lines and `#`-prefixed lines are ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not prompts:
        sys.exit(f"[error] no prompts found in {path}")
    return prompts


def load_config(path: Path | None, default_search_parent: Path | None = None) -> dict:
    """Load a `config.json` (sticky base LoRAs, face-recognition options, ...).

    Falls back to `<default_search_parent>/config.json` when `path` is None.
    Returns `{}` if nothing is found / parseable."""
    if path is None and default_search_parent is not None:
        candidate = default_search_parent / "config.json"
        if candidate.exists():
            path = candidate
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] could not parse config {path}: {e}")
        return {}
