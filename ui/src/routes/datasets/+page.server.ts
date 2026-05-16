import type { Actions, PageServerLoad } from './$types';
import { fail } from '@sveltejs/kit';
import type { ConnectorId } from '$lib/connectors/types';
import {
    deleteLink,
    listLinksForScope,
    listLinksForScopes,
    upsertLink,
    type ConnectorLink,
    type LinkScope
} from '$lib/server/connector-links';
import { ALL_CONNECTORS, listAvailableConnectors } from '$lib/server/connectors/registry';
import { enqueue } from '$lib/server/jobs';
import { getSettings } from '$lib/server/settings';
import { listDatasets } from '$lib/server/datasets';
import {
    createDatasetGroup,
    deleteDatasetGroup,
    listDatasetGroups,
    updateDatasetGroup
} from '$lib/server/dataset-groups';

export const load: PageServerLoad = () => {
    const settings = getSettings();
    const datasets = listDatasets(settings.dataset_root);
    const groups = listDatasetGroups();

    // Bulk-load every link for the scopes shown on this page so the UI can
    // render link state per row without N+1 queries.
    const folderKeys = datasets.map((d) => d.path);
    const groupKeys = groups.map((g) => String(g.id));
    const folderLinks = listLinksForScopes('folder', folderKeys);
    const groupLinks = listLinksForScopes('group', groupKeys);

    // Convert Map → Record for JSON serialization to the client.
    const links_by_folder: Record<string, ConnectorLink[]> = {};
    for (const [k, v] of folderLinks) links_by_folder[k] = v;
    const links_by_group: Record<string, ConnectorLink[]> = {};
    for (const [k, v] of groupLinks) links_by_group[k] = v;

    return {
        dataset_root: settings.dataset_root,
        datasets,
        groups,
        // Every connector usable for linking: credentials-backed types that
        // signed in successfully, plus always-available types (hard-drive).
        connectors: listAvailableConnectors(),
        all_connector_types: ALL_CONNECTORS,
        links_by_folder,
        links_by_group
    };
};

function validKnownConnector(id: unknown): ConnectorId | null {
    if (typeof id !== 'string') return null;
    return ALL_CONNECTORS.some((c) => c.id === id) ? (id as ConnectorId) : null;
}

function validLinkScope(s: unknown): LinkScope | null {
    return s === 'folder' || s === 'group' ? s : null;
}

/** Pull a list of paths from a form: every `path[]` field plus, when the
 * caller sends a `paths` text field, one path per line. We accept both
 * shapes so the UI can use checkboxes today and free-form text later. */
function readPaths(data: FormData): string[] {
    const out = new Set<string>();
    for (const p of data.getAll('path')) {
        const s = String(p).trim();
        if (s) out.add(s);
    }
    const raw = data.get('paths');
    if (typeof raw === 'string') {
        for (const line of raw.split(/\r?\n/)) {
            const s = line.trim();
            if (s) out.add(s);
        }
    }
    return [...out];
}

export const actions: Actions = {
    create: async ({ request }) => {
        const data = await request.formData();
        const name = String(data.get('name') ?? '').trim();
        const paths = readPaths(data);
        if (!name) return fail(400, { error: 'Name is required', name, paths });
        if (paths.length === 0) {
            return fail(400, { error: 'Select at least one dataset', name, paths });
        }
        try {
            createDatasetGroup(name, paths);
            return { ok: true };
        } catch (e) {
            return fail(500, { error: (e as Error).message, name, paths });
        }
    },

    update: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        const name = String(data.get('name') ?? '').trim();
        const paths = readPaths(data);
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        if (!name) return fail(400, { error: 'Name is required', id, name, paths });
        if (paths.length === 0) {
            return fail(400, { error: 'Select at least one dataset', id, name, paths });
        }
        try {
            updateDatasetGroup(id, name, paths);
            return { ok: true };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    },

    delete: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        deleteDatasetGroup(id);
        return { ok: true };
    },

    'link-set': async ({ request }) => {
        const data = await request.formData();
        const scope_kind = validLinkScope(data.get('scope_kind'));
        const scope_key = String(data.get('scope_key') ?? '').trim();
        const connector_id = validKnownConnector(data.get('connector_id'));
        const person_id = String(data.get('person_id') ?? '').trim();
        const person_name = String(data.get('person_name') ?? '') || null;
        if (!scope_kind || !scope_key || !connector_id || !person_id) {
            return fail(400, { error: 'Missing fields for link-set' });
        }
        // Detect "same person re-linked": don't enqueue a redundant job in
        // that case (it would skip everything anyway thanks to the
        // idempotence guard but the user shouldn't see a noop job spawned).
        const existing = listLinksForScope(scope_kind, scope_key).find(
            (l) => l.connector_id === connector_id
        );
        const personChanged = existing?.person_id !== person_id;

        upsertLink({
            scope_kind,
            scope_key,
            connector_id,
            person_id,
            person_name,
            person_thumb_url: String(data.get('person_thumb_url') ?? '') || null
        });

        // Kick off face detection on every picture of the chosen person.
        // Idempotent at the handler level — re-linking the same person
        // re-enqueues a job that picks up only newly-added pictures since
        // the last run.
        let job_id: number | null = null;
        if (personChanged) {
            try {
                job_id = enqueue('connector-face-detect', {
                    connector_id,
                    person_id,
                    person_name
                });
            } catch (e) {
                // Don't fail the link itself — the user can re-enqueue
                // manually later if needed.
                return { ok: true, link_warn: (e as Error).message };
            }
        }
        return { ok: true, job_id };
    },

    'link-remove': async ({ request }) => {
        const data = await request.formData();
        const scope_kind = validLinkScope(data.get('scope_kind'));
        const scope_key = String(data.get('scope_key') ?? '').trim();
        const connector_id = validKnownConnector(data.get('connector_id'));
        if (!scope_kind || !scope_key || !connector_id) {
            return fail(400, { error: 'Missing fields for link-remove' });
        }
        deleteLink(scope_kind, scope_key, connector_id);
        return { ok: true };
    }
};
