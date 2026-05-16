import type { Actions, PageServerLoad } from './$types';
import { existsSync, statSync } from 'node:fs';
import { fail } from '@sveltejs/kit';
import type { ConnectorId } from '$lib/connectors/types';
import { deleteCredentials, getCredentials } from '$lib/server/connectors/credentials';
import {
    ALL_CONNECTORS,
    CREDENTIAL_BACKED_CONNECTORS,
    getConnector,
    listAllConnectorStatuses
} from '$lib/server/connectors/registry';
import { getSettings, updateSettings, type SettingKey } from '$lib/server/settings';

const FIELDS: SettingKey[] = [
    'dataset_root',
    'tests_root',
    'lora_root',
    'python_bin',
    'glt_root'
];

export const load: PageServerLoad = () => {
    return {
        settings: getSettings(),
        connector_types: CREDENTIAL_BACKED_CONNECTORS,
        connectors: listAllConnectorStatuses()
    };
};

function validKnownConnector(id: unknown): ConnectorId | null {
    if (typeof id !== 'string') return null;
    // Settings only manages credentials-backed connectors. The hard-drive
    // type is registered globally but has no business here.
    return CREDENTIAL_BACKED_CONNECTORS.some((c) => c.id === id)
        ? (id as ConnectorId)
        : null;
}

export const actions: Actions = {
    'save-paths': async ({ request }) => {
        const data = await request.formData();
        const patch: Record<string, string> = {};
        const warnings: Record<string, string> = {};

        const fileFields = new Set<SettingKey>(['python_bin']);
        for (const key of FIELDS) {
            const raw = (data.get(key) ?? '').toString();
            patch[key] = raw;

            if (raw) {
                try {
                    if (!existsSync(raw)) {
                        warnings[key] = 'Path does not exist.';
                    } else {
                        const st = statSync(raw);
                        if (fileFields.has(key)) {
                            if (!st.isFile()) warnings[key] = 'Not a file.';
                        } else if (!st.isDirectory()) {
                            warnings[key] = 'Not a directory.';
                        }
                    }
                } catch (e) {
                    warnings[key] = `Could not stat path: ${(e as Error).message}`;
                }
            }
        }

        const settings = updateSettings(patch);
        return { saved: true, settings, warnings };
    },

    'connector-save': async ({ request }) => {
        const data = await request.formData();
        const id = validKnownConnector(data.get('connector_id'));
        if (!id) return fail(400, { connector_error: 'Unknown connector type', connector_id: null });

        // Bundle every form field except `connector_id` into a credentials
        // record. The connector's signIn validates the shape.
        const creds: Record<string, unknown> = {};
        for (const [k, v] of data.entries()) {
            if (k === 'connector_id') continue;
            if (typeof v === 'string') creds[k] = v;
        }
        const res = await getConnector(id).signIn(creds);
        if (!res.ok) {
            return fail(400, {
                connector_error: res.error ?? 'Sign-in failed.',
                connector_id: id
            });
        }
        return { connector_saved: id };
    },

    'connector-test': async ({ request }) => {
        const data = await request.formData();
        const id = validKnownConnector(data.get('connector_id'));
        if (!id) return fail(400, { connector_error: 'Unknown connector type', connector_id: null });
        const row = getCredentials(id);
        if (!row) return fail(400, { connector_error: 'Not configured.', connector_id: id });
        try {
            // Re-run signIn with the saved credentials — this is the only
            // method that actually hits the network and validates auth.
            // It also refreshes status + last_error in the DB so the table
            // pill updates on the next reload.
            const res = await getConnector(id).signIn(row.credentials);
            if (res.ok) return { connector_tested: id, connector_test_ok: true };
            return fail(400, {
                connector_error: res.error ?? 'Validation failed.',
                connector_id: id
            });
        } catch (e) {
            return fail(500, { connector_error: (e as Error).message, connector_id: id });
        }
    },

    'connector-remove': async ({ request }) => {
        const data = await request.formData();
        const id = validKnownConnector(data.get('connector_id'));
        if (!id) return fail(400, { connector_error: 'Unknown connector type', connector_id: null });
        deleteCredentials(id);
        return { connector_removed: id };
    }
};
