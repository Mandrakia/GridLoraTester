// Cache of (dataset_scope, connector, person) bindings. Stored independently
// per connector so a single dataset can be linked to e.g. Immich person X
// AND Google Photos person Y simultaneously. Re-linking upserts.
import type { ConnectorId } from '$lib/connectors/types';
import { db } from './db';

export type LinkScope = 'folder' | 'group';

export interface ConnectorLink {
    scope_kind: LinkScope;
    scope_key: string;
    connector_id: ConnectorId;
    person_id: string;
    person_name: string | null;
    person_thumb_url: string | null;
    linked_at: string;
    /** Set by connectors that snapshot photos into a local cache (e.g.
     * google-photos picker). NULL for live-source connectors. */
    last_sync_at: string | null;
    last_sync_count: number | null;
}

const upsertStmt = db.prepare(`
    INSERT INTO connector_links(scope_kind, scope_key, connector_id, person_id, person_name, person_thumb_url)
    VALUES(@scope_kind, @scope_key, @connector_id, @person_id, @person_name, @person_thumb_url)
    ON CONFLICT(scope_kind, scope_key, connector_id) DO UPDATE SET
        person_id = excluded.person_id,
        person_name = excluded.person_name,
        person_thumb_url = excluded.person_thumb_url,
        linked_at = datetime('now')
`);
const deleteStmt = db.prepare(
    'DELETE FROM connector_links WHERE scope_kind = ? AND scope_key = ? AND connector_id = ?'
);
const LINK_COLUMNS =
    'scope_kind, scope_key, connector_id, person_id, person_name, person_thumb_url, linked_at, last_sync_at, last_sync_count';

const listForScopeStmt = db.prepare(
    `SELECT ${LINK_COLUMNS} FROM connector_links WHERE scope_kind = ? AND scope_key = ?`
);
const listManyForScopeStmt = (placeholders: string) =>
    db.prepare(
        `SELECT ${LINK_COLUMNS}
         FROM connector_links
         WHERE scope_kind = ? AND scope_key IN (${placeholders})`
    );
const updateSyncStmt = db.prepare(
    `UPDATE connector_links
     SET last_sync_at = datetime('now'), last_sync_count = ?
     WHERE scope_kind = ? AND scope_key = ? AND connector_id = ?`
);

export function recordSync(
    scopeKind: LinkScope,
    scopeKey: string,
    connectorId: ConnectorId,
    count: number
): void {
    updateSyncStmt.run(count, scopeKind, scopeKey, connectorId);
}

export function listLinksForScope(scopeKind: LinkScope, scopeKey: string): ConnectorLink[] {
    return listForScopeStmt.all(scopeKind, scopeKey) as ConnectorLink[];
}

/** Bulk-load links for many scope_keys of the same scope_kind. Used by
 * /datasets so a single query covers every group + every available dataset
 * card on the page. */
export function listLinksForScopes(
    scopeKind: LinkScope,
    scopeKeys: string[]
): Map<string, ConnectorLink[]> {
    const out = new Map<string, ConnectorLink[]>();
    if (scopeKeys.length === 0) return out;
    const placeholders = scopeKeys.map(() => '?').join(',');
    const rows = listManyForScopeStmt(placeholders).all(scopeKind, ...scopeKeys) as ConnectorLink[];
    for (const r of rows) {
        const arr = out.get(r.scope_key) ?? [];
        arr.push(r);
        out.set(r.scope_key, arr);
    }
    return out;
}

export interface UpsertLinkInput {
    scope_kind: LinkScope;
    scope_key: string;
    connector_id: ConnectorId;
    person_id: string;
    person_name?: string | null;
    person_thumb_url?: string | null;
}

export function upsertLink(input: UpsertLinkInput): void {
    upsertStmt.run({
        scope_kind: input.scope_kind,
        scope_key: input.scope_key,
        connector_id: input.connector_id,
        person_id: input.person_id,
        person_name: input.person_name ?? null,
        person_thumb_url: input.person_thumb_url ?? null
    });
}

export function deleteLink(
    scopeKind: LinkScope,
    scopeKey: string,
    connectorId: ConnectorId
): void {
    deleteStmt.run(scopeKind, scopeKey, connectorId);
}
