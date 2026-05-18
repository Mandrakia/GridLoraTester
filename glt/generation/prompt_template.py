"""Prompt template expansion: substitute [trigger] and parse leading [W:H]
aspect-ratio tags into concrete (width, height) values derived from a
megapixel budget.

Convention — prompts in a prompt_set are written in FLUX.2 Klein prose
style with two optional substitutions:

    [trigger]   → the test's trigger word (LoRA activation token).
                  Empty trigger keeps the placeholder verbatim.

    [W:H]       → aspect ratio tag at the very start of the prompt.
                  Removed from the text before encoding. Width and height
                  are computed so the image area approximates the test's
                  resolution (e.g. 1MP, 2MP) AND both dims are multiples
                  of 16 (Flux2 VAE 8× × patch 2 = `check_inputs` floor).

Anything inside the prompt that isn't one of these two tokens is
left untouched.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Flux2 Klein's `check_inputs` requires height and width to be divisible
# by `vae_scale_factor * 2` = 8 * 2 = 16. Anything else raises on the first
# pipeline call.
SHAPE_MULTIPLE = 16

# Leading "[W:H] ..." — matches "[3:4]", "[16:9]", " [ 4 : 3 ]". The tag is
# consumed (including its trailing whitespace); everything after is the
# rest of the prompt.
AR_TAG_RE = re.compile(r"^\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*")
# `[trigger]` placeholder. Case-sensitive on purpose — matches what the
# user types into prompts in the dashboard.
TRIGGER_TAG = "[trigger]"


@dataclass(frozen=True)
class ExpandedPrompt:
    """Output of expansion: ready-to-encode prompt + concrete dims."""
    text: str
    width: int
    height: int
    # Original tag we found, e.g. "3:4". Empty when no tag was present
    # (default 1:1 was applied). Kept so the run history can surface
    # "this cell was requested at AR X".
    ar_tag: str


def round_to_multiple(n: float, multiple: int) -> int:
    """Round n to the nearest positive multiple of `multiple`, clamped to
    at least one step (avoids 0 on tiny areas)."""
    return max(multiple, int(round(n / multiple) * multiple))


def dims_for_ratio(target_area: int, w_ratio: float, h_ratio: float) -> tuple[int, int]:
    """Compute (width, height) such that w*h ≈ target_area and
    w/h ≈ w_ratio/h_ratio, with both rounded to SHAPE_MULTIPLE.

    Works in both orientations — caller passes ratios as-is, we don't
    swap. So `dims_for_ratio(1MP, 3, 4)` is portrait (taller than wide),
    `dims_for_ratio(1MP, 16, 9)` is landscape.
    """
    if w_ratio <= 0 or h_ratio <= 0:
        raise ValueError(f"AR components must be positive, got {w_ratio}:{h_ratio}")
    if target_area <= 0:
        raise ValueError(f"target_area must be positive, got {target_area}")
    # h^2 * (w_r/h_r) = area → h = sqrt(area * h_r / w_r)
    h_exact = math.sqrt(target_area * h_ratio / w_ratio)
    w_exact = h_exact * (w_ratio / h_ratio)
    return round_to_multiple(w_exact, SHAPE_MULTIPLE), round_to_multiple(h_exact, SHAPE_MULTIPLE)


def parse_resolution_mp(resolution: str) -> float:
    """Accept '1MP', '2MP', '1.5MP', '1mp', or a bare number ('1', '2.5').
    Returns the megapixel value as a float. Raises on garbage."""
    s = resolution.strip().lower().rstrip("mp").rstrip()
    if not s:
        raise ValueError(f"empty resolution: {resolution!r}")
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"bad resolution {resolution!r}: {e}") from e


def expand_prompts(
    prompts: list[str],
    trigger: str,
    resolution: str,
) -> list[ExpandedPrompt]:
    """Expand a list of raw prompt strings into ready-to-encode prompts.

    Steps per prompt:
      1. Strip leading `[W:H]` tag (if any) → records the AR.
      2. Substitute every `[trigger]` occurrence with the trigger word.
         Empty trigger → placeholder kept (caller can re-run with a
         different trigger and the cache key will differ accordingly).
      3. Compute (w, h) = dims_for_ratio(MP_area, w_ratio, h_ratio),
         falling back to square (1:1) when no AR tag was present.
    """
    mp = parse_resolution_mp(resolution)
    target_area = int(round(mp * 1_000_000))

    out: list[ExpandedPrompt] = []
    for raw in prompts:
        m = AR_TAG_RE.match(raw)
        if m:
            w_ratio = float(m.group(1))
            h_ratio = float(m.group(2))
            ar_tag = f"{m.group(1)}:{m.group(2)}"
            body = raw[m.end():]
        else:
            w_ratio, h_ratio = 1.0, 1.0
            ar_tag = ""
            body = raw

        if trigger:
            body = body.replace(TRIGGER_TAG, trigger)
        # else: leave [trigger] verbatim. The user's choice.

        w, h = dims_for_ratio(target_area, w_ratio, h_ratio)
        out.append(ExpandedPrompt(text=body, width=w, height=h, ar_tag=ar_tag))
    return out
