// Versioned, run-once migrations — distinct from the additive CREATE/ALTER
// bootstrap in db.ts (which is idempotent by construction and runs every
// boot). These are one-shot transformations that must run EXACTLY once per
// install and be recorded so they never re-run, even across restarts.
//
// db.ts calls `runMigrations(db)` after the schema is in place, at module
// load time — i.e. before the dashboard serves any HTTP request. Each
// migration runs in its own transaction; a throw rolls it back and is NOT
// recorded, so the next boot retries it.
//
// Rules for adding one:
//   - `id` is the primary key in `schema_migrations`. Pick a dated, unique
//     slug and NEVER rename or reuse it once shipped.
//   - Append to the END of MIGRATIONS; order is the array order.
//   - Only touch regenerable / derived data here without a heads-up. Anything
//     that destroys user-authored content needs explicit sign-off (see the
//     project's NEVER-wipe rule), not a silent boot migration.
import type Database from 'better-sqlite3';

type DB = Database.Database;

export interface Migration {
    /** Stable unique id, also the `schema_migrations` key. Dated prefix keeps
     * ordering obvious. Never rename/reuse once shipped. */
    id: string;
    /** One-shot transformation. Wrapped in a transaction by the runner. */
    run: (db: DB) => void;
}

const MIGRATIONS: Migration[] = [
    {
        // The perceptual hash now honors EXIF orientation (image-hash.ts
        // gained `.rotate()`). Hashes computed before that fix read the
        // stored-orientation pixels, so an EXIF-rotated photo and a copy with
        // the rotation baked in hashed ~150/256 bits apart and never deduped.
        // Clear every stored phash so the compute-image-hashes job recomputes
        // it correctly: dataset images re-hash from local disk on the next
        // Analyze; connector pictures re-hash on re-download. phash is derived
        // data, fully regenerable — safe to clear.
        id: '2026-05-20-exif-phash-reset',
        run: (db) => {
            db.exec('UPDATE dataset_images SET phash = NULL WHERE phash IS NOT NULL');
            db.exec('UPDATE connector_pictures SET phash = NULL WHERE phash IS NOT NULL');
        }
    }
];

/** Create the tracking table if needed, then run every migration not already
 * recorded, in order, each in its own transaction. */
export function runMigrations(db: DB): void {
    db.exec(`
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id         TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    `);
    const applied = new Set(
        (db.prepare('SELECT id FROM schema_migrations').all() as { id: string }[]).map(
            (r) => r.id
        )
    );
    const record = db.prepare('INSERT INTO schema_migrations(id) VALUES(?)');
    for (const m of MIGRATIONS) {
        if (applied.has(m.id)) continue;
        const tx = db.transaction(() => {
            m.run(db);
            record.run(m.id);
        });
        tx();
    }
}
