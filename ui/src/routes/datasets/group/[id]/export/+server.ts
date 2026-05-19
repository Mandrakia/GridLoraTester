import { error } from '@sveltejs/kit';
import { Readable } from 'node:stream';

import { buildGroupZip, safeZipFilename } from '$lib/server/dataset-export';
import { getDatasetGroup } from '$lib/server/dataset-groups';

import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ params }) => {
    const id = Number(params.id);
    if (!Number.isFinite(id) || id <= 0) throw error(404, 'Bad group id');
    const group = getDatasetGroup(id);
    if (!group) throw error(404, 'Group not found');

    // Skip members that no longer exist on disk; addFolderToZip would
    // silently no-op them anyway but pre-filtering keeps the slug map tight.
    const existing = group.paths.filter((p) => !group.missing_paths.includes(p));
    if (existing.length === 0) throw error(400, 'No existing dataset folders in this group');

    const zip = buildGroupZip(existing);
    return new Response(Readable.toWeb(zip.outputStream) as ReadableStream, {
        headers: {
            'Content-Type': 'application/zip',
            'Content-Disposition': `attachment; filename="${safeZipFilename(group.name)}"`
        }
    });
};
