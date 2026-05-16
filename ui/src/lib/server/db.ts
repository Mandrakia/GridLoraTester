// SQLite client + schema bootstrap. better-sqlite3 is synchronous and fast —
// perfect for the dashboard's load patterns (settings read on every request,
// no concurrent writes).
import Database from 'better-sqlite3';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const DB_PATH = process.env.GLT_DB_PATH
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
        width             INTEGER NOT NULL DEFAULT 1024,
        height            INTEGER NOT NULL DEFAULT 1024,
        batch_size        INTEGER NOT NULL DEFAULT 0,
        quant             TEXT NOT NULL DEFAULT 'fp8_weight_only',
        offload           TEXT NOT NULL DEFAULT 'text-encoder',
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

// Face detection results + per-scope centroids. A "scope" is either a single
// dataset folder ('folder', scope_key = abs path) or a group ('group',
// scope_key = group id as text). When the user clicks "Calculate centroid"
// on a group, we produce N+1 centroids: one per member folder, plus one
// global. is_target = 1 marks the face picked during the dataset-local
// two-pass (kept on the row so we can show which face won in the UI later).
db.exec(`
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path    TEXT NOT NULL,
        face_index    INTEGER NOT NULL,
        bbox_json     TEXT NOT NULL,
        det_score     REAL,
        embedding_b64 TEXT NOT NULL,
        is_target     INTEGER NOT NULL DEFAULT 0,
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

    -- Import lineage: which connector picture became which file on disk
    -- under a dataset scope. The UNIQUE constraint is what prevents the
    -- suggestion engine from re-proposing already-added pictures. We only
    -- need (scope_kind, scope_key, connector_id, picture_id) since a single
    -- connector picture is at most imported once per scope.
    CREATE TABLE IF NOT EXISTS dataset_imports (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_kind      TEXT NOT NULL CHECK (scope_kind IN ('folder', 'group')),
        scope_key       TEXT NOT NULL,
        connector_id    TEXT NOT NULL,
        picture_id      TEXT NOT NULL,
        target_folder   TEXT NOT NULL,
        dest_image_path TEXT NOT NULL,
        imported_at     TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(scope_kind, scope_key, connector_id, picture_id)
    );
    CREATE INDEX IF NOT EXISTS idx_dataset_imports_scope
        ON dataset_imports(scope_kind, scope_key);

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
`);

// Additive: per-face similarity to its dataset's centroid (winners only; the
// losing faces in a multi-face image keep similarity NULL — they didn't
// contribute to the centroid). Aggregate stats live on the centroids row.
// pitch/yaw/roll come from InsightFace's landmark_3d_68 head (degrees,
// signed). NULL when the detector didn't expose a pose for that face.
for (const [table, column, type] of [
    ['face_embeddings', 'similarity', 'REAL'],
    ['face_embeddings', 'pitch', 'REAL'],
    ['face_embeddings', 'yaw', 'REAL'],
    ['face_embeddings', 'roll', 'REAL'],
    // Source image dimensions (denormalized per face row — same value for
    // every face in the same image, but cheap and keeps the schema flat).
    // Used to compute the framing-distance proxy (bbox height / image height,
    // optionally roll-corrected).
    ['face_embeddings', 'image_width', 'INTEGER'],
    ['face_embeddings', 'image_height', 'INTEGER'],
    ['centroids', 'median_sim', 'REAL'],
    ['centroids', 'mean_sim', 'REAL']
] as const) {
    try {
        db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${type}`);
    } catch (e) {
        const msg = (e as Error).message;
        if (!msg.includes('duplicate column name')) throw e;
    }
}
