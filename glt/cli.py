"""Top-level CLI: dispatch to grid / centroid / rescore mode.

Mode selection mirrors the original `flux_lora_grid.py` flags so existing
invocations keep working unchanged:

  - `--rescore`          → `modes.rescore`
  - `--compute-centroid` → `modes.centroid`
  - otherwise            → `modes.grid` (the default grid generation)
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    # The grid parser uses `-h` for image height, so it disables the default
    # `--help`. Provide a top-level catch for `--help` / `-?` here that prints
    # the grid help (the most common case) when no mode flag is present.
    if "--rescore" in argv:
        from .modes import rescore
        rescore.main(argv)
        return
    if "--compute-centroid" in argv:
        from .modes import centroid
        centroid.main(argv)
        return
    if "--serve" in argv:
        from .ipc import server
        server.main(argv)
        return
    from .modes import grid
    # `--grid` is the default mode and not an arg the grid parser recognizes
    # — consume it here so explicit `python -m glt --grid …` invocations
    # (and the dashboard's job handler) still work.
    if "--grid" in argv:
        argv = [a for a in argv if a != "--grid"]
    # `grid.run()` calls its own `parse_args()` which reads sys.argv directly.
    # We restore sys.argv so the parser sees exactly what the user typed,
    # regardless of how main() was invoked.
    if argv is not sys.argv[1:]:
        sys.argv = [sys.argv[0], *argv]
    grid.run()


if __name__ == "__main__":
    main()
