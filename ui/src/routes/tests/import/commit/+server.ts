// POST /tests/import/commit — { token, dataset, lora }
// Applies the modal's resolved choices to the staged zip: create/match the
// test, insert run + rows + cells, extract images into tests_root, drop the
// staged zip. Returns { test_id, run_id, images }.
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { error, json } from '@sveltejs/kit';

import { commitImport, readManifestFromZip } from '$lib/server/test-import';

import type { RequestHandler } from './$types';

const STAGING = join(tmpdir(), 'glt-import');

export const POST: RequestHandler = async ({ request }) => {
    const body = (await request.json().catch(() => null)) as {
        token?: string;
        dataset?: string;
        lora?: string;
    } | null;
    // token is a UUID we minted in analyze — validate the shape so it can't be
    // used to path-traverse out of the staging dir.
    if (!body?.token || !/^[a-f0-9-]{36}$/i.test(body.token)) throw error(400, 'bad token');
    const zipPath = join(STAGING, `${body.token}.zip`);
    try {
        const manifest = await readManifestFromZip(zipPath);
        const res = await commitImport(
            manifest,
            { dataset: body.dataset ?? 'none', lora: body.lora ?? 'none' },
            zipPath
        );
        return json(res);
    } catch (e) {
        throw error(400, (e as Error).message);
    }
};
