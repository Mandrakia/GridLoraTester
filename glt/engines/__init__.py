"""Model-family engine registry.

`get_engine(family)` returns the singleton `ModelEngine` for a base model.
`'flux2'` is the default so every legacy invocation (no family specified)
behaves exactly as before. Z-Image registers itself here too.
"""
from __future__ import annotations

from .base import EngineSpec, ModelEngine
from .flux2 import Flux2Engine

DEFAULT_FAMILY = "flux2"

_ENGINES: dict[str, ModelEngine] = {}


def _register(engine: ModelEngine) -> None:
    _ENGINES[engine.spec.family] = engine


_register(Flux2Engine())

# Z-Image is import-guarded: a diffusers without ZImagePipeline (older
# install) simply won't offer the family rather than crashing the module.
try:
    from .zimage import ZImageEngine

    _register(ZImageEngine())
except Exception:  # noqa: BLE001 — missing diffusers symbol is non-fatal
    pass


def get_engine(family: str | None) -> ModelEngine:
    fam = (family or DEFAULT_FAMILY).strip().lower()
    try:
        return _ENGINES[fam]
    except KeyError:
        raise ValueError(
            f"unknown model family {family!r}; known: {sorted(_ENGINES)}"
        )


def available_families() -> list[str]:
    return list(_ENGINES)


def engine_specs() -> list[EngineSpec]:
    """All registered specs — for the UI / CLI to advertise choices."""
    return [e.spec for e in _ENGINES.values()]


__all__ = [
    "DEFAULT_FAMILY",
    "EngineSpec",
    "ModelEngine",
    "available_families",
    "engine_specs",
    "get_engine",
]
