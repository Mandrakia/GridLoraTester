// GET /tests/<id>/export — stream a .zip of the COMPOSITE grid (the best-of-
// all-runs view): manifest.json + the winning image for each cell, each pulled
// from the run folder it actually lives in. This is the default "Export" — for
// a single specific run use /tests/export/<run_id>.
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { Readable } from 'node:stream';

import { error } from '@sveltejs/kit';

import { getSettings } from '$lib/server/settings';
import { buildCompositeManifest } from '$lib/server/test-export';

import type { RequestHandler } from './$types';

// archiver is CJS; load via Node's require (see /tests/export/[run_id]).
const archiver = createRequire(import.meta.url)('archiver') as (
    format: 'zip',
    options?: import('archiver').ArchiverOptions
) => import('archiver').Archiver;

export const GET: RequestHandler = ({ params }) => {
    const testId = Number(params.id);
    if (!Number.isInteger(testId) || testId <= 0) throw error(400, 'invalid test id');

    let result: ReturnType<typeof buildCompositeManifest>;
    try {
        result = buildCompositeManifest(testId);
    } catch (e) {
        throw error(404, (e as Error).message);
    }
    const { manifest, images } = result;

    const s = getSettings();
    const archive = archiver('zip', { zlib: { level: 0 } });
    archive.append(JSON.stringify(manifest, null, 2), { name: 'manifest.json' });
    if (s.tests_root) {
        for (const { filename, runId } of images) {
            const abs = join(s.tests_root, manifest.test.name, `run_${runId}`, filename);
            if (existsSync(abs)) archive.file(abs, { name: `images/${filename}` });
        }
    }
    void archive.finalize();

    const fn = `${manifest.test.name}_composite.zip`.replace(/[^\w.\-]+/g, '_');
    return new Response(Readable.toWeb(archive) as unknown as ReadableStream, {
        headers: {
            'content-type': 'application/zip',
            'content-disposition': `attachment; filename="${fn}"`
        }
    });
};
