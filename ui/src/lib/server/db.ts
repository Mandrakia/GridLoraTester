// SQLite client + schema bootstrap. better-sqlite3 is synchronous and fast —
// perfect for the dashboard's load patterns (settings read on every request,
// no concurrent writes).
import Database from 'better-sqlite3';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

import { DEFAULT_PROMPT_SETS } from './seeds/prompt-sets';

export const DB_PATH = process.env.GLT_DB_PATH
    ? resolve(process.env.GLT_DB_PATH)
    : resolve(process.cwd(), 'data', 'glt.db');

mkdirSync(dirname(DB_PATH), { recursive: true });

export const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// Schema. Additive only — never destructive. New tables / columns are added
// via CREATE … IF NOT EXISTS or ALTER TABLE checks below.
db.exec(`
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT
    );

    -- User-curated bundles of dataset folders. paths_json holds a JSON array
    -- of absolute paths (each pointing at a subfolder of settings.dataset_root,
    -- though we don't enforce that at the SQL level — the user can keep a
    -- group around even after re-rooting).
    CREATE TABLE IF NOT EXISTS dataset_groups (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        paths_json  TEXT NOT NULL DEFAULT '[]',
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );

    -- Named lists of prompts (one prompt = one line in prompts_json[]).
    -- Used by tests to drive the per-row prompt generation.
    CREATE TABLE IF NOT EXISTS prompt_sets (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL UNIQUE,
        prompts_json TEXT NOT NULL DEFAULT '[]',
        created_at   TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
    );

    -- Grid tests: a named recipe pointing at one LoRA folder + a dataset
    -- (single path or a group), with the resolution / batch / quant / offload
    -- knobs surfaced in the UI. All the rare flags (steps, seed, sage, ...)
    -- live in advanced_json to keep the schema flat.
    -- The on-disk output folder is settings.tests_root / name; presence of
    -- a manifest.json there drives the displayed status (not_started /
    -- in_progress / completed / out_of_sync), computed on read.
    CREATE TABLE IF NOT EXISTS tests (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        name              TEXT NOT NULL UNIQUE,
        lora_path         TEXT NOT NULL,
        dataset_path      TEXT,                -- mutually exclusive with dataset_group_id
        dataset_group_id  INTEGER REFERENCES dataset_groups(id) ON DELETE SET NULL,
        prompts_path      TEXT,                -- legacy: free-text path (kept for compat)
        prompt_set_id     INTEGER REFERENCES prompt_sets(id) ON DELETE SET NULL,
        -- LoRA trigger word substituted into [trigger] placeholders in
        -- prompts at run time (handled by the grid script -- empty here means
        -- the script leaves the placeholder as-is). Free text, no validation.
        trigger           TEXT NOT NULL DEFAULT '',
        -- Target image area in megapixels (e.g. '1MP' = 1024^2, '2MP' = ~1448^2).
        -- The grid script combines this with the per-prompt aspect-ratio tag
        -- (e.g. [3:4], [16:9]) to derive the actual width/height at run time.
        -- Stored as text so non-MP encodings stay possible later.
        resolution        TEXT NOT NULL DEFAULT '1MP',
        batch_size        INTEGER NOT NULL DEFAULT 0,
        -- Transformer quant: 'auto' (resolve per GPU at run time),
        -- 'none', 'fp8_weight', 'fp8_dynamic', 'fp8_quanto', 'int8_convrot'.
        -- Qwen3 is always loaded briefly during the encode-and-unload phase
        -- and never co-resides with the transformer, so there's no offload
        -- knob to expose. Custom Qwen file / dtype lives in advanced_json.
        quant             TEXT NOT NULL DEFAULT 'auto',
        -- torch.compile mode for the transformer:
        --   'on'   — always compile (default; rentable with the disk cache
        --             for any grid of ~3+ images on a single shape)
        --   'auto' — let glt decide based on n_loras × n_prompts vs
        --             n_shapes × 8 (warm-cache break-even heuristic)
        --   'off'  — never compile (useful when debugging the model or
        --             when a one-off single-image test isn't worth the
        --             per-shape warmup)
        compile_mode      TEXT NOT NULL DEFAULT 'on'
                            CHECK (compile_mode IN ('on','auto','off')),
        advanced_json     TEXT NOT NULL DEFAULT '{}',
        created_at        TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
    );
`);

// Additive migration: pre-existing DBs (created before the prompt_set_id
// column existed) won't have it, since CREATE TABLE IF NOT EXISTS is a no-op
// on an existing table. ALTER TABLE ADD COLUMN is the standard SQLite path —
// we try it and swallow the "duplicate column" error when it's already there.
try {
    db.exec(
        'ALTER TABLE tests ADD COLUMN prompt_set_id INTEGER REFERENCES prompt_sets(id) ON DELETE SET NULL'
    );
} catch (e) {
    const msg = (e as Error).message;
    if (!msg.includes('duplicate column name')) throw e;
}

// Schema cleanup for the pre-release quant overhaul: the `offload` column
// is gone (Qwen and the transformer are never co-resident — see
// pipeline/build.py's two-phase lifecycle) and legacy quant strings need
// to map to the canonical names the Python side now accepts.
// Both ops are idempotent: DROP COLUMN swallows "no such column", and
// UPDATEs match 0 rows on already-migrated DBs.
try {
    db.exec('ALTER TABLE tests DROP COLUMN offload');
} catch (e) {
    const msg = (e as Error).message;
    if (!msg.includes('no such column')) throw e;
}
db.exec("UPDATE tests SET quant = 'fp8_weight' WHERE quant = 'fp8_weight_only'");
db.exec("UPDATE tests SET quant = 'none'       WHERE quant = 'bf16'");
db.exec("UPDATE tests SET quant = 'auto'       WHERE quant = '' OR quant IS NULL");

// Compile mode column was added in the same pre-release polish pass.
// Pre-existing DBs need ALTER TABLE; new installs get it via CREATE.
// Migrate any leftover `advanced.compile_transformer` boolean → top
// level `compile_mode` so the new dropdown reflects the historical setting.
try {
    db.exec(
        "ALTER TABLE tests ADD COLUMN compile_mode TEXT NOT NULL DEFAULT 'on' " +
            "CHECK (compile_mode IN ('on','auto','off'))"
    );
} catch (e) {
    const msg = (e as Error).message;
    if (!msg.includes('duplicate column name')) throw e;
}
db.exec(
    "UPDATE tests " +
    "  SET compile_mode = CASE " +
    "          WHEN json_extract(advanced_json, '$.compile_transformer') = 1 THEN 'on' " +
    "          WHEN json_extract(advanced_json, '$.compile_transformer') = 0 THEN 'off' " +
    "          ELSE compile_mode " +
    "      END " +
    " WHERE json_extract(advanced_json, '$.compile_transformer') IS NOT NULL"
);

// Canonical "member of a dataset" table. Every image that lives in a dataset
// folder gets a row here — whether dropped in manually by the user OR
// imported via a connector. Holds image-level metadata (phash, dims) and
// the active/excluded status used by the prune/centroid pipeline.
//
//   status='active'   — counts toward centroid + coverage + targets
//   status='excluded' — kept around (history + reversible) but skipped
//                       everywhere it would otherwise be counted
//
// Connector-imported images carry (connector_id, picture_id, imported_at)
// so the suggestion engine can dedup against already-imported pictures.
// UNIQUE(connector_id, picture_id) prevents importing the same connector
// picture twice; (NULL, NULL) for manual images is fine — SQLite treats
// NULLs as distinct in UNIQUE constraints.
//
// folder_path is denormalized = dirname(image_path), indexed for fast
// per-folder coverage queries (replaces the prior `LIKE 'folder/%'`
// scans on face_embeddings). Writers must keep it consistent.
db.exec(`
    CREATE TABLE IF NOT EXISTS dataset_images (
        image_path      TEXT PRIMARY KEY,
        folder_path     TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'excluded')),
        excluded_at     TEXT,
        excluded_reason TEXT,
        phash           TEXT,
        image_width     INTEGER,
        image_height    INTEGER,
        source_kind     TEXT NOT NULL DEFAULT 'manual'
                            CHECK (source_kind IN ('manual', 'imported')),
        connector_id    TEXT,
        picture_id      TEXT,
        imported_at     TEXT,
        added_at        TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(connector_id, picture_id)
    );
    CREATE INDEX IF NOT EXISTS idx_dataset_images_folder ON dataset_images(folder_path);
    CREATE INDEX IF NOT EXISTS idx_dataset_images_status ON dataset_images(status);

    -- Per-scope dataset size cap. NULL or absent row = no cap (current
    -- behavior preserved for legacy scopes). When set, the dataset-targets
    -- math scales per-bucket targets to (ratio × max_size) instead of
    -- (ratio × actual_total), which is what turns prune from "always-off"
    -- into a real swap signal: once you hit the cap, adding requires
    -- removing from an over-rep bucket first.
    CREATE TABLE IF NOT EXISTS dataset_size_limits (
        scope_kind  TEXT NOT NULL CHECK (scope_kind IN ('folder', 'group')),
        scope_key   TEXT NOT NULL,
        max_size    INTEGER,
        updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (scope_kind, scope_key)
    );

    -- Face detection results + per-scope centroids. A "scope" is either a single
    -- dataset folder ('folder', scope_key = abs path) or a group ('group',
    -- scope_key = group id as text). is_target = 1 marks the face picked
    -- during the dataset-local two-pass.
    --
    -- image_path FKs into dataset_images so deleting a dataset image
    -- cascades its face rows (and so the recompute flow can join cheaply
    -- on status). Writers must upsert dataset_images BEFORE inserting
    -- face_embeddings (centroids.ts + dataset-import.ts both do).
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path    TEXT NOT NULL REFERENCES dataset_images(image_path) ON DELETE CASCADE,
        face_index    INTEGER NOT NULL,
        bbox_json     TEXT NOT NULL,
        det_score     REAL,
        embedding_b64 TEXT NOT NULL,
        is_target     INTEGER NOT NULL DEFAULT 0,
        similarity    REAL,
        pitch         REAL,
        yaw           REAL,
        roll          REAL,
        computed_at   TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(image_path, face_index)
    );
    CREATE INDEX IF NOT EXISTS idx_face_embeddings_image_path ON face_embeddings(image_path);

    -- Photo-DB connector credentials (Immich, Google Photos, …). One row per
    -- configured connector type. credentials_json shape is connector-specific
    -- (Immich: base_url + api_key; Google Photos: OAuth tokens).
    -- Plain-text on purpose — this is a local-only dashboard, the DB file
    -- itself is the trust boundary.
    CREATE TABLE IF NOT EXISTS connector_credentials (
        connector_id     TEXT PRIMARY KEY,
        credentials_json TEXT NOT NULL,
        status           TEXT,
        last_check_at    TEXT,
        last_error       TEXT,
        updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
    );

    -- Background jobs queue. Long-running tasks (face detection on connector
    -- pictures, future imports, …) live here so the UI can monitor + cancel.
    -- In-memory state is kept in lib/server/jobs/runner.ts, but the DB is
    -- the source of truth so a dashboard reload doesn't lose visibility.
    CREATE TABLE IF NOT EXISTS jobs (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        type             TEXT NOT NULL,
        params_json      TEXT NOT NULL DEFAULT '{}',
        status           TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued','running','cancelled','completed','failed')),
        progress_current INTEGER NOT NULL DEFAULT 0,
        progress_total   INTEGER,
        current_label    TEXT,
        error            TEXT,
        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
        started_at       TEXT,
        finished_at      TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);

    CREATE TABLE IF NOT EXISTS job_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id     INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        level      TEXT NOT NULL DEFAULT 'info'
                       CHECK (level IN ('info','warn','error')),
        message    TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_job_logs_job ON job_logs(job_id, id);

    -- Idempotence tracking for connector face detection. One row per picture
    -- we've already processed (regardless of face count) so re-runs skip
    -- the download + detection work for known pictures.
    CREATE TABLE IF NOT EXISTS connector_pictures (
        connector_id TEXT NOT NULL,
        picture_id   TEXT NOT NULL,
        person_id    TEXT,
        filename     TEXT,
        image_width  INTEGER,
        image_height INTEGER,
        n_faces      INTEGER NOT NULL DEFAULT 0,
        processed_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (connector_id, picture_id)
    );

    -- Face data for connector pictures. Schema mirrors face_embeddings but
    -- keyed on (connector, picture) instead of file path — keeps connector
    -- faces isolated from dataset faces so they don't pollute coverage
    -- counts. is_target is NOT filled here (no two-pass per connector).
    CREATE TABLE IF NOT EXISTS connector_faces (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        connector_id  TEXT NOT NULL,
        picture_id    TEXT NOT NULL,
        face_index    INTEGER NOT NULL,
        bbox_json     TEXT NOT NULL,
        det_score     REAL,
        embedding_b64 TEXT NOT NULL,
        pitch         REAL,
        yaw           REAL,
        roll          REAL,
        computed_at   TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(connector_id, picture_id, face_index),
        FOREIGN KEY (connector_id, picture_id)
            REFERENCES connector_pictures(connector_id, picture_id)
            ON DELETE CASCADE
    );

    -- Link a scope (a folder dataset or a dataset group) to a person inside
    -- one connector. At most one link per (scope, connector) — re-linking
    -- the same connector upserts. person_name and person_thumb_url are
    -- cached at link time so the UI keeps showing them even if the connector
    -- goes offline.
    CREATE TABLE IF NOT EXISTS connector_links (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_kind       TEXT NOT NULL CHECK (scope_kind IN ('folder', 'group')),
        scope_key        TEXT NOT NULL,
        connector_id     TEXT NOT NULL,
        person_id        TEXT NOT NULL,
        person_name      TEXT,
        person_thumb_url TEXT,
        linked_at        TEXT NOT NULL DEFAULT (datetime('now')),
        last_sync_at     TEXT,
        last_sync_count  INTEGER,
        UNIQUE(scope_kind, scope_key, connector_id)
    );

    CREATE TABLE IF NOT EXISTS centroids (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_kind      TEXT NOT NULL CHECK (scope_kind IN ('folder', 'group')),
        scope_key       TEXT NOT NULL,
        centroid_b64    TEXT NOT NULL,
        n_single_face   INTEGER NOT NULL DEFAULT 0,
        n_multi_face    INTEGER NOT NULL DEFAULT 0,
        n_no_face       INTEGER NOT NULL DEFAULT 0,
        median_sim      REAL,
        mean_sim        REAL,
        computed_at     TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(scope_kind, scope_key)
    );

    -- Test run history. Each invocation of glt --grid for a given test
    -- inserts ONE row here, then streams test_run_rows + test_run_cells
    -- as it produces images. status transitions running -> completed/failed.
    -- config_json is a snapshot of the test definition at run start so
    -- a later definition edit doesn't retroactively change run semantics.
    CREATE TABLE IF NOT EXISTS test_runs (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id          INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
        status           TEXT NOT NULL DEFAULT 'running'
                            CHECK (status IN ('running','completed','failed','cancelled')),
        started_at       TEXT NOT NULL DEFAULT (datetime('now')),
        finished_at      TEXT,
        config_json      TEXT NOT NULL,
        base_loras_json  TEXT,
        face_meta_json   TEXT,
        error            TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_test_runs_test
        ON test_runs(test_id, started_at DESC);

    -- One row per LoRA in the run grid, plus its aggregate face metrics.
    -- lora_idx preserves the grid ordering the script chose.
    CREATE TABLE IF NOT EXISTS test_run_rows (
        run_id        INTEGER NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
        lora_idx      INTEGER NOT NULL,
        lora_display  TEXT NOT NULL,
        metrics_json  TEXT,
        PRIMARY KEY (run_id, lora_idx)
    );

    -- One row per generated cell. image_filename NULL = not yet generated
    -- (the row is pre-populated when planning so the UI sees progress
    -- holes filling in). prompt_text is the FINAL post-substitution text
    -- (trigger expanded, [W:H] tag stripped) so a future SELECT WHERE
    -- prompt_text = 'Charlotte sitting' just works.
    CREATE TABLE IF NOT EXISTS test_run_cells (
        run_id         INTEGER NOT NULL,
        lora_idx       INTEGER NOT NULL,
        prompt_idx     INTEGER NOT NULL,
        prompt_text    TEXT NOT NULL,
        prompt_width   INTEGER NOT NULL,
        prompt_height  INTEGER NOT NULL,
        image_filename TEXT,
        face_score     REAL,
        PRIMARY KEY (run_id, lora_idx, prompt_idx),
        FOREIGN KEY (run_id, lora_idx)
            REFERENCES test_run_rows(run_id, lora_idx) ON DELETE CASCADE
    );
`);

// Additive migrations for columns added between commits on pre-existing DBs.
// Fresh installs already have these inline in the CREATE TABLE statements
// above; the ALTER swallows "duplicate column name" so the loop is a no-op
// for them.
for (const [table, column, type] of [
    ['centroids', 'median_sim', 'REAL'],
    ['centroids', 'mean_sim', 'REAL'],
    // Per-phase timing aggregates for long-running jobs (face detection
    // scans). Written debounced by the handler; consumed live by the UI
    // for the optim-decision panel (download vs python vs db).
    ['jobs', 'metrics_json', 'TEXT'],
    // Logical-key columns. (type, key_arg1, key_arg2) groups together
    // every run of the "same" job (e.g. face-detect on the same
    // connector+person), so the dashboard can show one row per logical
    // key in the Latest view and push older runs to Archives. NULLs are
    // allowed and grouped as '' by the read queries — old rows without
    // keys collapse to a single bucket per type, which is the desired
    // degraded behavior for legacy data.
    ['jobs', 'key_arg1', 'TEXT'],
    ['jobs', 'key_arg2', 'TEXT'],
    // OS pid of the process actually running the work. For in-process
    // Node handlers (face-detect, compute-hashes) this is process.pid;
    // for subprocess handlers (grid-test-run) it's the child's pid set
    // via setPid() once spawned. The orphan-reaper polls `kill -0 pid`
    // and finalizes any 'running' row whose pid is dead.
    ['jobs', 'pid', 'INTEGER'],
    // BlockHash 256-bit perceptual hash on the connector side. The dataset
    // side lives on dataset_images.phash (image-level by construction).
    ['connector_pictures', 'phash', 'TEXT'],
    // Sync metadata for connectors that snapshot photos into a local cache
    // (google-photos picker today). last_sync_at + last_sync_count surface
    // in the connector pill's tooltip; other connectors leave them NULL.
    ['connector_links', 'last_sync_at', 'TEXT'],
    ['connector_links', 'last_sync_count', 'INTEGER']
] as const) {
    try {
        db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${type}`);
    } catch (e) {
        const msg = (e as Error).message;
        if (!msg.includes('duplicate column name')) throw e;
    }
}

// Speeds up the latest-per-key query used by the Jobs page (Latest view).
// Without this we'd full-scan jobs on every poll once history grows.
db.exec(
    'CREATE INDEX IF NOT EXISTS idx_jobs_logical_key ON jobs(type, key_arg1, key_arg2, id DESC)'
);

// Seed the canonical default prompt set on fresh installs. We use "table
// is empty" as the gate (rather than "first boot" or "INSERT OR IGNORE")
// so a user who deletes the default won't have it come back on the next
// dashboard restart — but a brand-new install gets something runnable
// without poking around the UI first.
{
    const empty =
        (db.prepare('SELECT COUNT(*) AS n FROM prompt_sets').get() as { n: number }).n === 0;
    if (empty) {
        const insert = db.prepare('INSERT INTO prompt_sets(name, prompts_json) VALUES(?, ?)');
        const tx = db.transaction((sets: typeof DEFAULT_PROMPT_SETS) => {
            for (const s of sets) insert.run(s.name, JSON.stringify(s.prompts));
        });
        tx(DEFAULT_PROMPT_SETS);
    }
}
