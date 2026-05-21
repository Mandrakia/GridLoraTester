// POST /tests/import/analyze — body is the raw .zip (streamed to disk).
// Stages the upload, reads its manifest, and returns the proposed resolution
// (prompt match-or-create, dataset/lora match + options) for the import modal.
// The staged zip is reused by /tests/import/commit so the (large) upload only
// happens once.
import { randomUUID } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

import { error, json } from '@sveltejs/kit';

import { analyzeManifest, readManifestFromZip } from '$lib/server/test-import';

import type { RequestHandler } from './$types';

const STAGING = join(tmpdir(), 'glt-import');

export const POST: RequestHandler = async ({ request }) => {
    if (!request.body) throw error(400, 'empty body');
    await mkdir(STAGING, { recursive: true });
    const token = randomUUID();
    const zipPath = join(STAGING, `${token}.zip`);
    // Stream the upload straight to disk — never buffer a ~500 MB zip in RAM.
    await pipeline(
        Readable.fromWeb(request.body as unknown as Parameters<typeof Readable.fromWeb>[0]),
        createWriteStream(zipPath)
    );
    try {
        const manifest = await readManifestFromZip(zipPath);
        return json(analyzeManifest(manifest, token));
    } catch (e) {
        throw error(400, (e as Error).message);
    }
};
