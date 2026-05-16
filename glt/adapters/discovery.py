"""LoRA / LoKr file discovery, naming, and step parsing."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from ..utils.hashing import quick_file_hash


_LORA_STEP_RE = re.compile(r"^(.*?)_(\d+)$")


def safe_stem(p: Path) -> str:
    """Sanitized output-file stem for an adapter. Same naming whether the input
    came from one dir or several — collisions are resolved up front in
    `discover_loras()` (identical content = dedup, different content = abort)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("_") or "lora"


def lora_display_name(p: Path) -> str:
    """How an adapter is identified in the manifest + HTML row label.
    Bare filename — clean and consistent across single/multi-dir runs."""
    return p.name


def parse_lora_step(p: Path) -> tuple[str, int | float]:
    """Extract `(basename, step)` from an adapter filename.

    `'my-lora_000001400.safetensors'` → `('my-lora', 1400)`
    `'my-lora.safetensors'`           → `('my-lora', math.inf)`  (final)
    """
    stem = p.stem
    m = _LORA_STEP_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, math.inf


def discover_loras(input_dirs: list[Path]) -> list[Path]:
    """Walk one or more directories for `.safetensors` files; deduplicate by
    filename across dirs:

      - same filename, identical content → keep one (deterministic pick)
      - same filename, different content → abort with a helpful message
        (the user must rename one or restructure)

    Returns a list sorted by `(basename, step)`.
    """
    by_name: dict[str, list[Path]] = {}
    for d in input_dirs:
        for p in d.glob("*.safetensors"):
            if p.is_file():
                by_name.setdefault(p.name, []).append(p)

    if not by_name:
        joined = " | ".join(str(d) for d in input_dirs)
        sys.exit(f"[error] no .safetensors files found in: {joined}")

    deduped: list[Path] = []
    for fname, paths in by_name.items():
        if len(paths) == 1:
            deduped.append(paths[0])
            continue
        # Collision across input dirs — hash to decide.
        hashes = {p: quick_file_hash(p) for p in paths}
        if len({*hashes.values()}) == 1:
            # All copies identical (e.g. user symlinked or duplicated the dir).
            chosen = paths[0]
            others = ", ".join(str(p) for p in paths[1:])
            print(f"[discover] '{fname}' present in multiple dirs with identical "
                  f"content; keeping {chosen}, ignoring duplicates ({others})")
            deduped.append(chosen)
        else:
            details = "\n  ".join(f"{p}  (hash={hashes[p][:12]}...)" for p in paths)
            sys.exit(
                f"[error] LoRA filename collision — '{fname}' exists in multiple "
                f"input dirs with DIFFERENT content:\n  {details}\n"
                f"Rename one of them (e.g. add a suffix) or restructure your dirs."
            )

    return sorted(deduped, key=lambda p: (parse_lora_step(p)[0], parse_lora_step(p)[1]))
