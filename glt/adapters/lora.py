"""Sticky base-LoRA loading via peft adapters."""
from __future__ import annotations

import re
from pathlib import Path


def load_base_loras(pipe, base_entries: list[dict]) -> tuple[list[str], list[float], list[dict]]:
    """Load 'sticky' LoRAs that stay active for the whole run on top of every
    test adapter. Each entry: `{"path": str, "weight": float, "name": str?}`.

    Returns `(adapter_names, weights, manifest_entries)` — manifest entries are
    safe to embed in run metadata.
    """
    if not base_entries:
        return [], [], []

    names: list[str] = []
    weights: list[float] = []
    manifest_entries: list[dict] = []
    for i, entry in enumerate(base_entries):
        path = entry.get("path")
        if not path:
            print(f"[base-lora] entry {i} has no 'path', skipping")
            continue
        p = Path(path).expanduser()
        if not p.exists():
            print(f"[base-lora] file not found: {p}, skipping")
            continue
        weight = float(entry.get("weight", 1.0))
        nick = entry.get("name") or p.stem
        adapter = f"base_{i}_{re.sub(r'[^A-Za-z0-9_]+', '_', nick)}"
        try:
            pipe.load_lora_weights(str(p), adapter_name=adapter)
            names.append(adapter)
            weights.append(weight)
            manifest_entries.append({"name": nick, "path": str(p), "weight": weight})
            print(f"[base-lora] loaded {p.name} as '{adapter}' (weight={weight})")
        except Exception as e:
            print(f"[base-lora] failed to load {p.name}: {e}")
    return names, weights, manifest_entries
