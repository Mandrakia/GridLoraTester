// Per-scope max dataset size. Set by the user on the folder/group page,
// consumed by the dataset-targets math to scale per-bucket targets to
// (ratio × max_size) instead of (ratio × current_total). That's what
// turns the prune signal from a no-op (everything is balanced by
// definition when targets follow actuals) into something meaningful —
// over-rep happens relative to the eventual cap, not the live count.
//
// NULL or absent row = no cap → fall back to current proportional behavior.
import type { LinkScope } from './connector-links';
import { db } from './db';

const getStmt = db.prepare(
    'SELECT max_size FROM dataset_size_limits WHERE scope_kind = ? AND scope_key = ?'
);

const upsertStmt = db.prepare(`
    INSERT INTO dataset_size_limits (scope_kind, scope_key, max_size)
    VALUES (?, ?, ?)
    ON CONFLICT(scope_kind, scope_key) DO UPDATE SET
        max_size = excluded.max_size,
        updated_at = datetime('now')
`);

const deleteStmt = db.prepare(
    'DELETE FROM dataset_size_limits WHERE scope_kind = ? AND scope_key = ?'
);

export function getMaxSize(scope: LinkScope, scopeKey: string): number | null {
    const row = getStmt.get(scope, scopeKey) as { max_size: number | null } | undefined;
    return row?.max_size ?? null;
}

/** Upsert. Passing null / 0 / negative clears the cap (no-cap semantics). */
export function setMaxSize(
    scope: LinkScope,
    scopeKey: string,
    maxSize: number | null
): void {
    if (maxSize == null || maxSize <= 0) {
        deleteStmt.run(scope, scopeKey);
        return;
    }
    upsertStmt.run(scope, scopeKey, Math.floor(maxSize));
}
