// GET /tests/export/<run_id> — stream a .zip of { manifest.json, images/* } for
// one run, so it can be drag-and-dropped into another install's import.
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { Readable } from 'node:stream';

import { error } from '@sveltejs/kit';

import { getSettings } from '$lib/server/settings';
import { buildRunManifest } from '$lib/server/test-export';

import type { RequestHandler } from './$types';

// archiver / unzipper are CJS; Vite's SSR ESM interop doesn't synthesize their
// `default`, so load via Node's require (works in dev SSR and prod). archiver's
// `export =` callable can't be referenced directly as a type, so type the
// require with the exact call signature we use.
const archiver = createRequire(import.meta.url)('archiver') as (
    format: 'zip',
    options?: import('archiver').ArchiverOptions
) => import('archiver').Archiver;

export const GET: RequestHandler = ({ params }) => {
    const runId = Number(params.run_id);
    if (!Number.isInteger(runId) || runId <= 0) throw error(400, 'invalid run_id');

    let manifest;
    try {
        manifest = buildRunManifest(runId);
    } catch (e) {
        throw error(404, (e as Error).message);
    }

    const s = getSettings();
    const runDir = s.tests_root ? join(s.tests_root, manifest.test.name, `run_${runId}`) : null;

    // level 0 (store): the images are already PNG/WebP/JPEG, so re-compressing
    // them only burns CPU for no size gain.
    const archive = archiver('zip', { zlib: { level: 0 } });
    archive.append(JSON.stringify(manifest, null, 2), { name: 'manifest.json' });
    if (runDir && existsSync(runDir)) archive.directory(runDir, 'images');
    void archive.finalize();

    const filename = `${manifest.test.name}_run${runId}.zip`.replace(/[^\w.\-]+/g, '_');
    return new Response(Readable.toWeb(archive) as unknown as ReadableStream, {
        headers: {
            'content-type': 'application/zip',
            'content-disposition': `attachment; filename="${filename}"`
        }
    });
};
