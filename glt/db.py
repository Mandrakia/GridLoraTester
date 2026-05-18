"""SQLite access layer used by the grid script to persist run history.

The Python side talks to `ui/data/glt.db` (or whichever path the caller
passes) via sqlite3 stdlib — no extra dep. The dashboard owns schema
creation; this module assumes the tables already exist (it will raise on
SELECT against a missing table, with a clear hint).

Three tables are touched here:

    test_runs       -- one row per invocation of glt --grid for a test
    test_run_rows   -- one row per LoRA in the grid, with aggregate metrics
    test_run_cells  -- one row per (LoRA, prompt) cell, with filename + score

Writes are committed eagerly per call. Concurrent readers (the SvelteKit
dashboard with WAL enabled) see partial progress, which is by design —
the user watches a running grid live.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the GLT SQLite DB. Enables FK enforcement (matches dashboard).
    `isolation_level=None` puts us in autocommit so explicit BEGIN/COMMIT
    blocks elsewhere can shape transactions where atomicity matters."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(
            f"DB not found at {p}. The dashboard creates it on first boot — "
            f"point --db at ui/data/glt.db."
        )
    conn = sqlite3.connect(str(p), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ---- test_runs --------------------------------------------------------

def create_run(
    conn: sqlite3.Connection,
    test_id: int,
    config: dict[str, Any],
) -> int:
    """Insert a fresh run row in status='running'. Returns the new run_id.
    `config` is the snapshot of every effective knob — written verbatim
    so a later definition edit doesn't change historical run semantics."""
    cur = conn.execute(
        """
        INSERT INTO test_runs (test_id, status, config_json)
        VALUES (?, 'running', ?)
        """,
        (test_id, json.dumps(config, ensure_ascii=False)),
    )
    return int(cur.lastrowid)


def set_run_base_loras(
    conn: sqlite3.Connection,
    run_id: int,
    base_loras: list[dict[str, Any]],
) -> None:
    conn.execute(
        "UPDATE test_runs SET base_loras_json = ? WHERE id = ?",
        (json.dumps(base_loras, ensure_ascii=False), run_id),
    )


def set_run_face_meta(
    conn: sqlite3.Connection,
    run_id: int,
    meta: dict[str, Any] | None,
) -> None:
    conn.execute(
        "UPDATE test_runs SET face_meta_json = ? WHERE id = ?",
        (json.dumps(meta, ensure_ascii=False) if meta else None, run_id),
    )


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    error: str | None = None,
) -> None:
    """Transition a run out of 'running'. status in
    {'completed','failed','cancelled'}."""
    if status not in {"completed", "failed", "cancelled"}:
        raise ValueError(f"bad run status: {status}")
    conn.execute(
        """
        UPDATE test_runs
           SET status = ?,
               finished_at = datetime('now'),
               error = ?
         WHERE id = ?
        """,
        (status, error, run_id),
    )


# ---- test_run_rows ----------------------------------------------------

def upsert_row(
    conn: sqlite3.Connection,
    run_id: int,
    lora_idx: int,
    lora_display: str,
) -> None:
    """Insert (or replace label of) a LoRA row. Called once per LoRA at
    plan time so the cells query can FK back even before any cell lands."""
    conn.execute(
        """
        INSERT INTO test_run_rows (run_id, lora_idx, lora_display)
        VALUES (?, ?, ?)
        ON CONFLICT(run_id, lora_idx) DO UPDATE SET
            lora_display = excluded.lora_display
        """,
        (run_id, lora_idx, lora_display),
    )


def set_row_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    lora_idx: int,
    metrics: dict[str, Any] | None,
) -> None:
    conn.execute(
        "UPDATE test_run_rows SET metrics_json = ? WHERE run_id = ? AND lora_idx = ?",
        (
            json.dumps(metrics, ensure_ascii=False) if metrics else None,
            run_id,
            lora_idx,
        ),
    )


# ---- test_run_cells ---------------------------------------------------

def upsert_cell(
    conn: sqlite3.Connection,
    run_id: int,
    lora_idx: int,
    prompt_idx: int,
    prompt_text: str,
    prompt_width: int,
    prompt_height: int,
    image_filename: str | None = None,
    face_score: float | None = None,
) -> None:
    """Insert or update a cell row. Pre-populated with NULL filename when
    a cell is scheduled; updated with the filename + score when generated.

    Idempotent on re-run: a later call with the same key overwrites with
    the new image / score, matching the user-facing "regenerate this cell"
    semantics."""
    conn.execute(
        """
        INSERT INTO test_run_cells
            (run_id, lora_idx, prompt_idx, prompt_text,
             prompt_width, prompt_height, image_filename, face_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, lora_idx, prompt_idx) DO UPDATE SET
            prompt_text   = excluded.prompt_text,
            prompt_width  = excluded.prompt_width,
            prompt_height = excluded.prompt_height,
            image_filename = COALESCE(excluded.image_filename, test_run_cells.image_filename),
            face_score     = COALESCE(excluded.face_score, test_run_cells.face_score)
        """,
        (
            run_id,
            lora_idx,
            prompt_idx,
            prompt_text,
            prompt_width,
            prompt_height,
            image_filename,
            face_score,
        ),
    )


def set_cell_image(
    conn: sqlite3.Connection,
    run_id: int,
    lora_idx: int,
    prompt_idx: int,
    image_filename: str | None,
) -> None:
    conn.execute(
        """
        UPDATE test_run_cells
           SET image_filename = ?
         WHERE run_id = ? AND lora_idx = ? AND prompt_idx = ?
        """,
        (image_filename, run_id, lora_idx, prompt_idx),
    )


def set_cell_score(
    conn: sqlite3.Connection,
    run_id: int,
    lora_idx: int,
    prompt_idx: int,
    face_score: float | None,
) -> None:
    conn.execute(
        """
        UPDATE test_run_cells
           SET face_score = ?
         WHERE run_id = ? AND lora_idx = ? AND prompt_idx = ?
        """,
        (face_score, run_id, lora_idx, prompt_idx),
    )


# ---- Test definition reads (used at run setup) ------------------------

def load_test_def(conn: sqlite3.Connection, test_id: int) -> dict[str, Any]:
    """Pull a test's definition + its prompts (from prompt_sets when set,
    else the legacy free-text path) and the user's tests_root setting.
    Caller passes the merged dict into the run loop.

    Returns {name, lora_path, trigger, resolution, batch_size, quant,
    compile_mode, advanced, prompts, tests_root}. Raises LookupError if
    the test row is missing."""
    t = conn.execute(
        """
        SELECT id, name, lora_path, dataset_path, dataset_group_id,
               prompts_path, prompt_set_id,
               trigger, resolution, batch_size, quant, compile_mode, advanced_json
          FROM tests
         WHERE id = ?
        """,
        (test_id,),
    ).fetchone()
    if t is None:
        raise LookupError(f"test {test_id} not found")

    # Prompts come from a prompt_set when set, else the legacy on-disk file.
    prompts: list[str] = []
    prompts_source = "—"
    if t["prompt_set_id"] is not None:
        ps = conn.execute(
            "SELECT name, prompts_json FROM prompt_sets WHERE id = ?",
            (t["prompt_set_id"],),
        ).fetchone()
        if ps is not None:
            try:
                parsed = json.loads(ps["prompts_json"] or "[]")
                if isinstance(parsed, list):
                    prompts = [str(p) for p in parsed if p]
            except Exception:
                pass
            prompts_source = f"set:{ps['name']}"
    if not prompts and t["prompts_path"]:
        try:
            with open(t["prompts_path"], "r", encoding="utf-8") as f:
                prompts = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            prompts_source = f"file:{t['prompts_path']}"
        except OSError as e:
            raise RuntimeError(
                f"could not read prompts_path {t['prompts_path']}: {e}"
            ) from e

    if not prompts:
        raise RuntimeError(f"test {test_id} has no prompts")

    advanced: dict[str, Any] = {}
    try:
        parsed = json.loads(t["advanced_json"] or "{}")
        if isinstance(parsed, dict):
            advanced = parsed
    except Exception:
        pass

    tests_root_row = conn.execute(
        "SELECT value FROM settings WHERE key = 'tests_root'"
    ).fetchone()
    tests_root = tests_root_row["value"] if tests_root_row else None

    return {
        "id": t["id"],
        "name": t["name"],
        "lora_path": t["lora_path"],
        "dataset_path": t["dataset_path"],
        "dataset_group_id": t["dataset_group_id"],
        "trigger": t["trigger"] or "",
        "resolution": t["resolution"] or "1MP",
        "batch_size": int(t["batch_size"] or 0),
        "quant": t["quant"] or "auto",
        "compile_mode": t["compile_mode"] or "on",
        "advanced": advanced,
        "prompts": prompts,
        "prompts_source": prompts_source,
        "tests_root": tests_root,
    }


def load_face_cfg_for_test(
    conn: sqlite3.Connection, test_id: int,
) -> dict | None:
    """Build a face_recognition config for a test by pulling its dataset's
    centroid from the `centroids` table.

    Returns a `face_cfg` dict ready for `FaceScorer(cfg)`, or `None` when
    the test has no resolvable centroid (e.g. dataset not analyzed yet, or
    test has no dataset bound). Caller treats `None` as "skip scoring".

    Scope lookup:
      - dataset_path set     → centroids row WHERE scope_kind='folder' AND scope_key=path
      - dataset_group_id set → centroids row WHERE scope_kind='group'  AND scope_key=str(id)

    Defaults for model/providers/det_size/thresholds match what
    `config.json` used to ship; advanced users can still override via
    `advanced_json.face_model_*` keys but that's not exposed in the UI
    yet (current users only need the centroid).
    """
    t = conn.execute(
        "SELECT dataset_path, dataset_group_id FROM tests WHERE id = ?",
        (test_id,),
    ).fetchone()
    if t is None:
        return None

    if t["dataset_path"]:
        row = conn.execute(
            "SELECT centroid_b64 FROM centroids WHERE scope_kind='folder' AND scope_key=?",
            (t["dataset_path"],),
        ).fetchone()
    elif t["dataset_group_id"] is not None:
        row = conn.execute(
            "SELECT centroid_b64 FROM centroids WHERE scope_kind='group' AND scope_key=?",
            (str(t["dataset_group_id"]),),
        ).fetchone()
    else:
        return None

    if row is None or not row["centroid_b64"]:
        return None

    return {
        "centroid_b64": row["centroid_b64"],
        "model_name": "buffalo_l",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "det_size": [640, 640],
        "thresholds": {"good": 0.5, "ok": 0.35},
    }


# ---- Reads (used by HTML output) --------------------------------------

def fetch_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Pull the full run snapshot: meta + every row + every cell. Used by
    the HTML writer to embed the inline JSON manifest."""
    run = conn.execute(
        "SELECT * FROM test_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise LookupError(f"run {run_id} not found")
    rows = conn.execute(
        """
        SELECT lora_idx, lora_display, metrics_json
          FROM test_run_rows
         WHERE run_id = ?
         ORDER BY lora_idx ASC
        """,
        (run_id,),
    ).fetchall()
    cells = conn.execute(
        """
        SELECT lora_idx, prompt_idx, prompt_text, prompt_width, prompt_height,
               image_filename, face_score
          FROM test_run_cells
         WHERE run_id = ?
         ORDER BY lora_idx ASC, prompt_idx ASC
        """,
        (run_id,),
    ).fetchall()
    return {
        "id": run["id"],
        "test_id": run["test_id"],
        "status": run["status"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "config": json.loads(run["config_json"] or "{}"),
        "base_loras": json.loads(run["base_loras_json"] or "[]"),
        "face_meta": json.loads(run["face_meta_json"] or "null"),
        "error": run["error"],
        "rows": [
            {
                "lora_idx": r["lora_idx"],
                "lora_display": r["lora_display"],
                "metrics": json.loads(r["metrics_json"] or "null"),
            }
            for r in rows
        ],
        "cells": [
            {
                "lora_idx": c["lora_idx"],
                "prompt_idx": c["prompt_idx"],
                "prompt_text": c["prompt_text"],
                "prompt_width": c["prompt_width"],
                "prompt_height": c["prompt_height"],
                "image_filename": c["image_filename"],
                "face_score": c["face_score"],
            }
            for c in cells
        ],
    }
