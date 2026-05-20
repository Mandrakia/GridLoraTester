// Boot-time settings seed for the Docker image.
//
// Writes the workspace-aware defaults into the SQLite `settings` table using
// INSERT OR IGNORE — any value already set by the user (via the dashboard's
// Settings page) is preserved. Idempotent: safe to run on every container
// start.
//
// Runs BEFORE the SvelteKit server boots so the very first page render sees
// usable paths. Reads/writes the same DB the UI will open (resolved via
// GLT_DB_PATH, which mirrors ui/src/lib/server/db.ts).
const Database = require('better-sqlite3');
const { mkdirSync } = require('node:fs');
const { dirname, resolve } = require('node:path');

const DB_PATH = process.env.GLT_DB_PATH
    ? resolve(process.env.GLT_DB_PATH)
    : '/workspace/data/glt.db';

mkdirSync(dirname(DB_PATH), { recursive: true });

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.exec(
    'CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);'
);

// Env overrides let a different image layout (e.g. the RunPod multi-service
// image, whose venv lives at /opt/venv) repoint a value without forking this
// script. Unset → the standalone image's defaults, unchanged.
const DEFAULTS = {
    dataset_root: process.env.GLT_DATASET_ROOT || '/workspace/datasets',
    lora_root: process.env.GLT_LORA_ROOT || '/workspace/outputs',
    tests_root: process.env.GLT_TESTS_ROOT || '/workspace/grids',
    python_bin: process.env.GLT_PYTHON_BIN || '/opt/glt-venv/bin/python'
};

const ins = db.prepare(
    'INSERT INTO settings(key, value) VALUES(?, ?) ' +
        'ON CONFLICT(key) DO NOTHING'
);
const tx = db.transaction(() => {
    for (const [k, v] of Object.entries(DEFAULTS)) ins.run(k, v);
});
tx();

const rows = db.prepare('SELECT key, value FROM settings').all();
db.close();

console.log(`[glt-docker] seeded settings at ${DB_PATH}`);
for (const r of rows) console.log(`[glt-docker]   ${r.key} = ${r.value}`);
