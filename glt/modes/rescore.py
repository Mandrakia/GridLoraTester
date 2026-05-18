"""`glt --rescore` — re-score an existing grid without regenerating images.

Two modes:

  - **DB mode** (`--test-id N --db PATH`): the dashboard-launched path.
    Reads the latest `test_runs` row for the given test, walks every
    `test_run_cells` row that has an `image_filename`, scores it from
    disk via FaceScorer + the centroid stored in `centroids` for the
    test's dataset, writes `face_score` back via `set_cell_score`, then
    re-aggregates per-row metrics into `test_run_rows.metrics_json`.

  - **Legacy CLI mode** (`-o <output_dir> --config <path>`): kept for
    runs that pre-date the test_runs DB (read `manifest.json` from disk,
    score, rewrite manifest). Will be removed once the manifest-era
    artifacts are gone from the wild.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .. import db as glt_db
from ..analysis.face import FaceScorer
from ..analysis.metrics import compute_row_metrics
from ..output.manifest import write_grid_artifacts
from ..utils.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="glt --rescore",
        description="Re-score generated images against a face centroid. "
                    "DB mode: --test-id + --db. Legacy mode: -o + --config.",
    )
    parser.add_argument("--rescore", action="store_true", required=True,
                        help="Enable rescore-only mode")

    # DB mode (dashboard-driven)
    parser.add_argument("--test-id", type=int, default=None,
                        help="DB mode: id of the test row to rescore.")
    parser.add_argument("--db", default=None,
                        help="Path to ui/data/glt.db (required with --test-id).")
    parser.add_argument("--run-id", type=int, default=None,
                        help="DB mode: rescore a specific test_runs row "
                             "(default: latest run for this test).")
    parser.add_argument("--tests-root", default=None,
                        help="DB mode: tests root for image paths. Defaults to "
                             "the `tests_root` row in `settings`.")

    # Legacy CLI mode
    parser.add_argument("-o", dest="output_dir", default=None,
                        help="Legacy CLI mode: existing output dir containing manifest.json")
    parser.add_argument("--config", default=None,
                        help="Legacy CLI mode: path to config.json with face_recognition")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip rewriting index.html (legacy CLI mode only)")
    parser.add_argument("--force-rescore", "--force", action="store_true",
                        help="Recompute scores for ALL cells/rows (default: skip already-scored).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.test_id is not None:
        if not args.db:
            sys.exit("[error] --test-id requires --db pointing at ui/data/glt.db")
        _rescore_db(args)
        return
    if args.output_dir and args.config:
        _rescore_legacy(args)
        return
    sys.exit("[error] either --test-id + --db (DB mode) or -o + --config (legacy) required")


# ---- DB mode -------------------------------------------------------------


def _rescore_db(args) -> None:
    """Read the test_run_cells from the DB, score each image from disk,
    write `face_score` back per cell + recompute per-row metrics."""
    conn = glt_db.connect(args.db)
    try:
        test_def = glt_db.load_test_def(conn, args.test_id)
    except LookupError as e:
        sys.exit(f"[error] {e}")
    print(f"[db] test #{args.test_id}: {test_def['name']}")

    face_cfg = glt_db.load_face_cfg_for_test(conn, args.test_id)
    if face_cfg is None:
        sys.exit(
            f"[error] no centroid available for test #{args.test_id} — "
            f"analyze the dataset first (computes a centroid in the centroids table)."
        )

    # Resolve which run(s) to rescore. Default = every run for this test
    # (the dashboard's /tests/[id] page surfaces history across all runs,
    # so the Rescore button should cover the same set). `--run-id N`
    # narrows to a specific row.
    if args.run_id is not None:
        run_rows = conn.execute(
            "SELECT id, status FROM test_runs WHERE id = ? AND test_id = ?",
            (args.run_id, args.test_id),
        ).fetchall()
    else:
        run_rows = conn.execute(
            "SELECT id, status FROM test_runs WHERE test_id = ? ORDER BY id ASC",
            (args.test_id,),
        ).fetchall()
    if not run_rows:
        sys.exit(f"[error] no test_runs row for test #{args.test_id}")

    tests_root = args.tests_root or test_def["tests_root"]
    if not tests_root:
        sys.exit("[error] tests_root not configured (--tests-root or settings)")

    print(
        f"[db] rescoring {len(run_rows)} run(s) "
        f"({'force' if args.force_rescore else 'skip already-scored'} mode)"
    )

    # FaceScorer is heavy to instantiate (loads buffalo_l) — keep one
    # across runs. The centroid lives on the test's dataset and doesn't
    # change between runs.
    scorer = FaceScorer(face_cfg)
    t_total = time.time()
    grand_scored = 0
    grand_skipped = 0
    grand_missing_dirs = 0

    for run_row in run_rows:
        run_id = run_row["id"]
        output_dir = Path(tests_root).resolve() / test_def["name"] / f"run_{run_id}"
        if not output_dir.is_dir():
            print(f"  [run #{run_id}] skipped — dir missing on disk: {output_dir}")
            grand_missing_dirs += 1
            continue

        cells = conn.execute(
            "SELECT lora_idx, prompt_idx, image_filename, face_score "
            "  FROM test_run_cells "
            " WHERE run_id = ? AND image_filename IS NOT NULL "
            " ORDER BY lora_idx, prompt_idx",
            (run_id,),
        ).fetchall()
        if not cells:
            print(f"  [run #{run_id}] skipped — no cells with images")
            continue

        by_lora: dict[int, list[dict]] = {}
        for c in cells:
            by_lora.setdefault(c["lora_idx"], []).append({
                "prompt_idx": c["prompt_idx"],
                "image_filename": c["image_filename"],
                "face_score": c["face_score"],
            })

        glt_db.set_run_face_meta(conn, run_id, {
            "model_name": scorer.model_name,
            "providers": scorer.providers,
            "centroid_dim": int(scorer.centroid.shape[0]),
            "thresholds": scorer.thresholds,
            "rescored_at": _now_iso(),
        })

        print(
            f"  [run #{run_id}] status={run_row['status']}, "
            f"{sum(len(v) for v in by_lora.values())} cell(s) across "
            f"{len(by_lora)} LoRA(s)"
        )
        run_scored = 0
        run_skipped = 0
        t_run = time.time()
        for lora_idx in sorted(by_lora.keys()):
            items = by_lora[lora_idx]
            scores: list[float | None] = []
            t_row = time.time()
            for it in items:
                if it["face_score"] is not None and not args.force_rescore:
                    scores.append(it["face_score"])
                    run_skipped += 1
                    continue
                img_path = output_dir / it["image_filename"]
                s = scorer.score_path(img_path)
                glt_db.set_cell_score(conn, run_id, lora_idx, it["prompt_idx"], s)
                scores.append(s)
                run_scored += 1
            metrics = compute_row_metrics(scores)
            glt_db.set_row_metrics(conn, run_id, lora_idx, metrics)
            dt = time.time() - t_row
            if metrics["n_faces"]:
                print(
                    f"    lora_idx={lora_idx}: median={metrics['median']:.3f} "
                    f"p20={metrics['p20']:.3f} "
                    f"faces={metrics['n_faces']}/{metrics['n_total']} ({dt:.1f}s)"
                )
            else:
                print(
                    f"    lora_idx={lora_idx}: no faces detected "
                    f"({metrics['n_total']} img, {dt:.1f}s)"
                )
        grand_scored += run_scored
        grand_skipped += run_skipped
        print(
            f"    -> run #{run_id} done: {run_scored} scored, {run_skipped} skipped "
            f"({time.time() - t_run:.1f}s)"
        )

    print(
        f"\n[done] rescore complete in {time.time() - t_total:.1f}s "
        f"across {len(run_rows)} run(s) "
        f"({grand_scored} scored, {grand_skipped} skipped, "
        f"{grand_missing_dirs} run dir(s) missing — pass --force to recompute all)"
    )


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---- Legacy manifest mode ------------------------------------------------


def _rescore_legacy(args) -> None:
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
    print(f"[rescore:legacy] {len(rows)} row(s) in {manifest_path}  mode={mode_label}")
    t_total = time.time()
    n_skipped = 0
    n_scored = 0
    for r_idx, row in enumerate(rows):
        existing_scores = row.get("scores")
        already_scored = (
            existing_scores is not None
            and any(s is not None for s in existing_scores)
        )
        if already_scored and not args.force_rescore:
            n_skipped += 1
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
                  f"faces={m['n_faces']}/{m['n_total']} ({dt:.1f}s)")
        else:
            print(f"  [{r_idx + 1}/{len(rows)}] {row.get('lora', '?')}: "
                  f"no faces detected ({m['n_total']} img, {dt:.1f}s)")

    write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
    print(f"\n[done] {n_scored} scored, {n_skipped} skipped in {time.time() - t_total:.1f}s")
