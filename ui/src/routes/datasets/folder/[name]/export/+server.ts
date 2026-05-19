import { error } from '@sveltejs/kit';
import { Readable } from 'node:stream';
import { statSync } from 'node:fs';
import { resolve } from 'node:path';

import { buildFolderZip, safeZipFilename } from '$lib/server/dataset-export';
import { isPathInside } from '$lib/server/path-utils';
import { getSettings } from '$lib/server/settings';

import type { RequestHandler } from './$types';

function resolveFolder(name: string): string {
    const { dataset_root } = getSettings();
    if (!dataset_root) throw error(404, 'dataset_root not configured');
    const root = resolve(dataset_root);
    const folder = resolve(root, name);
    if (!isPathInside(root, folder)) throw error(403, 'Forbidden');
    try {
        if (!statSync(folder).isDirectory()) throw error(404, 'Not a directory');
    } catch {
        throw error(404, 'Dataset not found');
    }
    return folder;
}

export const GET: RequestHandler = ({ params }) => {
    const folder = resolveFolder(params.name);
    const zip = buildFolderZip(folder);
    return new Response(Readable.toWeb(zip.outputStream) as ReadableStream, {
        headers: {
            'Content-Type': 'application/zip',
            'Content-Disposition': `attachment; filename="${safeZipFilename(params.name)}"`
        }
    });
};
