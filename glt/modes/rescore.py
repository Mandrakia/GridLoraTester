"""`glt --rescore` — re-score an existing grid without regenerating images."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..analysis.face import FaceScorer
from ..analysis.metrics import compute_row_metrics
from ..output.manifest import write_grid_artifacts
from ..utils.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="glt --rescore",
        description="Re-score an existing output directory's grid using the current "
                    "`face_recognition` centroid in `config.json`. Reads `manifest.json`, "
                    "rescores every image from disk, rewrites manifest + index.html. "
                    "No adapters loaded, no generation performed.",
    )
    parser.add_argument("--rescore", action="store_true", required=True,
                        help="Enable rescore-only mode")
    parser.add_argument("-o", dest="output_dir", required=True,
                        help="Existing output directory containing manifest.json")
    parser.add_argument("--config", required=True,
                        help="Path to config.json with a `face_recognition` section")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip rewriting index.html (only update manifest.json)")
    parser.add_argument("--force-rescore", "--force", action="store_true",
                        help="Recompute scores for ALL rows, including ones already "
                             "scored. Default: append-mode (skip rows whose `scores` "
                             "field is already populated).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        sys.exit(f"[error] output dir not found: {output_dir}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"[error] no manifest.json in {output_dir} — nothing to rescore")

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        sys.exit(f"[error] config not found: {config_path}")
    config = load_config(config_path)
    face_cfg = config.get("face_recognition") or {}
    if not face_cfg or not (face_cfg.get("centroid") or face_cfg.get("centroid_b64")):
        sys.exit("[error] config.face_recognition.centroid (or centroid_b64) is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if not rows:
        sys.exit(f"[error] manifest has no rows: {manifest_path}")

    scorer = FaceScorer(face_cfg)
    manifest.setdefault("meta", {})["face_recognition"] = {
        "model_name": scorer.model_name,
        "providers": scorer.providers,
        "centroid_dim": int(scorer.centroid.shape[0]),
        "thresholds": scorer.thresholds,
    }

    mode_label = "force-rescore (recompute all)" if args.force_rescore else "append (skip already-scored)"
    print(f"[rescore] {len(rows)} row(s) in {manifest_path}  mode={mode_label}")
    t_total = time.time()
    n_skipped = 0
    n_scored = 0
    for r_idx, row in enumerate(rows):
        # Append-mode: a row is considered "already scored" if it has a
        # `scores` field with at least one numeric value. A row with all-None
        # scores is treated as not-scored (e.g. no faces detected in any
        # image — but the centroid may have changed since, worth a recompute).
        existing_scores = row.get("scores")
        already_scored = (
            existing_scores is not None
            and any(s is not None for s in existing_scores)
        )
        if already_scored and not args.force_rescore:
            n_skipped += 1
            m = row.get("metrics") or {}
            print(f"  [{r_idx + 1}/{len(rows)}] {row.get('lora', '?')}: skip "
                  f"(already scored — median={m.get('median', float('nan')):.3f} "
                  f"faces={m.get('n_faces', 0)}/{m.get('n_total', 0)})")
            continue

        images = row.get("images") or []
        t0 = time.time()
        scores = []
        for name in images:
            if not name:
                scores.append(None)
                continue
            scores.append(scorer.score_path(output_dir / name))
        row["scores"] = scores
        row["metrics"] = compute_row_metrics(scores)
        m = row["metrics"]
        dt = time.time() - t0
        n_scored += 1
        if m["n_faces"]:
            print(f"  [{r_idx + 1}/{len(rows)}] {row.get('lora', '?')}: "
                  f"median={m['median']:.3f} p20={m['p20']:.3f} "
                  f"mean={m['mean']:.3f} std={m['std']:.3f} "
                  f"faces={m['n_faces']}/{m['n_total']} ({dt:.1f}s)")
        else:
            print(f"  [{r_idx + 1}/{len(rows)}] {row.get('lora', '?')}: "
                  f"no faces detected ({m['n_total']} img, {dt:.1f}s)")

    write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
    print(f"[manifest] {manifest_path}")
    print(f"\n[done] rescore complete in {time.time() - t_total:.1f}s "
          f"({n_scored} scored, {n_skipped} skipped)")
