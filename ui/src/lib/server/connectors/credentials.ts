// CRUD layer for connector_credentials. Stored as JSON per connector_id.
// Plain text — see db.ts comment.
import type { ConnectorId } from '$lib/connectors/types';
import { db } from '../db';

export interface CredentialsRow {
    connector_id: ConnectorId;
    credentials: Record<string, unknown>;
    status: string | null;
    last_check_at: string | null;
    last_error: string | null;
    updated_at: string;
}

const selectStmt = db.prepare(
    'SELECT connector_id, credentials_json, status, last_check_at, last_error, updated_at FROM connector_credentials WHERE connector_id = ?'
);
const listStmt = db.prepare(
    'SELECT connector_id, credentials_json, status, last_check_at, last_error, updated_at FROM connector_credentials ORDER BY connector_id ASC'
);
const upsertStmt = db.prepare(`
    INSERT INTO connector_credentials(connector_id, credentials_json, status, last_check_at, last_error, updated_at)
    VALUES(@connector_id, @credentials_json, @status, @last_check_at, @last_error, datetime('now'))
    ON CONFLICT(connector_id) DO UPDATE SET
        credentials_json = excluded.credentials_json,
        status = excluded.status,
        last_check_at = excluded.last_check_at,
        last_error = excluded.last_error,
        updated_at = datetime('now')
`);
const updateStatusStmt = db.prepare(`
    UPDATE connector_credentials
    SET status = ?, last_check_at = datetime('now'), last_error = ?
    WHERE connector_id = ?
`);
const deleteStmt = db.prepare('DELETE FROM connector_credentials WHERE connector_id = ?');

type RawRow = {
    connector_id: string;
    credentials_json: string;
    status: string | null;
    last_check_at: string | null;
    last_error: string | null;
    updated_at: string;
};

function parse(row: RawRow): CredentialsRow {
    let creds: Record<string, unknown> = {};
    try {
        const parsed = JSON.parse(row.credentials_json);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            creds = parsed as Record<string, unknown>;
        }
    } catch {
        // ignore — corrupt JSON degrades to empty creds; the UI will offer
        // to re-enter them.
    }
    return {
        connector_id: row.connector_id as ConnectorId,
        credentials: creds,
        status: row.status,
        last_check_at: row.last_check_at,
        last_error: row.last_error,
        updated_at: row.updated_at
    };
}

export function getCredentials(connectorId: ConnectorId): CredentialsRow | null {
    const row = selectStmt.get(connectorId) as RawRow | undefined;
    return row ? parse(row) : null;
}

export function listCredentials(): CredentialsRow[] {
    return (listStmt.all() as RawRow[]).map(parse);
}

export function saveCredentials(
    connectorId: ConnectorId,
    credentials: Record<string, unknown>,
    status: 'signed_in' | 'error' | 'unknown' = 'unknown',
    lastError: string | null = null
): void {
    upsertStmt.run({
        connector_id: connectorId,
        credentials_json: JSON.stringify(credentials ?? {}),
        status,
        last_check_at: status === 'unknown' ? null : new Date().toISOString(),
        last_error: lastError
    });
}

export function setStatus(
    connectorId: ConnectorId,
    status: 'signed_in' | 'error',
    lastError: string | null = null
): void {
    updateStatusStmt.run(status, lastError, connectorId);
}

export function deleteCredentials(connectorId: ConnectorId): void {
    deleteStmt.run(connectorId);
}
