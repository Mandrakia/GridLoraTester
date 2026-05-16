// POST: pull a connector picture, copy it into a dataset folder, lift the
// face data over, and recompute the folder's centroid. Idempotent at the
// dataset_imports table — a second call for the same (scope, connector,
// picture) returns the original dest_image_path without re-downloading.
import type { RequestHandler } from './$types';
import { error, json } from '@sveltejs/kit';

import type { ConnectorId } from '$lib/connectors/types';
import { addPictureToDataset } from '$lib/server/dataset-import';
import { ALL_CONNECTORS } from '$lib/server/connectors/registry';

export const POST: RequestHandler = async ({ request }) => {
    let body: Record<string, unknown>;
    try {
        body = (await request.json()) as Record<string, unknown>;
    } catch {
        throw error(400, 'Bad JSON body');
    }

    const scope_kind = body.scope_kind === 'folder' || body.scope_kind === 'group'
        ? body.scope_kind
        : null;
    const scope_key = typeof body.scope_key === 'string' ? body.scope_key.trim() : '';
    const target_folder = typeof body.target_folder === 'string' ? body.target_folder.trim() : '';
    const picture_id = typeof body.picture_id === 'string' ? body.picture_id.trim() : '';
    const connector_id = ALL_CONNECTORS.some((c) => c.id === body.connector_id)
        ? (body.connector_id as ConnectorId)
        : null;

    if (!scope_kind || !scope_key || !target_folder || !connector_id || !picture_id) {
        throw error(400, 'Missing fields');
    }

    try {
        const result = await addPictureToDataset({
            scope_kind,
            scope_key,
            target_folder,
            connector_id,
            picture_id
        });
        return json({ ok: true, ...result });
    } catch (e) {
        const err = e as Error & { code?: string };
        // Surface the unique-constraint violation as a clean "already
        // imported" rather than a 500. Match on the SQLite error code (not
        // the message text) — better-sqlite3 wording can change between
        // versions.
        if (err.code === 'SQLITE_CONSTRAINT_UNIQUE') {
            return json({ ok: false, error: 'Already imported' }, { status: 409 });
        }
        return json({ ok: false, error: err.message }, { status: 500 });
    }
};
