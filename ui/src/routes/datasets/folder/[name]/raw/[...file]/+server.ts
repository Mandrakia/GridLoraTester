// Image proxy for /datasets/folder/<name>: streams files from
//   <dataset_root>/<name>/<...file>
// With `?w=<int>` returns a sharp-resized WebP thumbnail (cached on disk).
// Without it, streams the original file. Hardened against path traversal.
import type { RequestHandler } from './$types';
import { error } from '@sveltejs/kit';
import { readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

import { mimeFor } from '$lib/server/mime';
import { getSettings } from '$lib/server/settings';
import { bufferToBytes, clampWidth, getThumbnail } from '$lib/server/thumbs';

export const GET: RequestHandler = async ({ params, url }) => {
    const { dataset_root } = getSettings();
    if (!dataset_root) throw error(404, 'dataset_root not configured');

    const root = resolve(dataset_root);
    const folder = resolve(root, params.name);
    if (folder !== root && !folder.startsWith(root + '/')) throw error(403, 'Forbidden');

    const target = resolve(folder, params.file || '');
    if (target !== folder && !target.startsWith(folder + '/')) throw error(403, 'Forbidden');

    try {
        if (!statSync(target).isFile()) throw error(404);
    } catch {
        throw error(404, 'Not found');
    }

    const wRaw = url.searchParams.get('w');
    if (wRaw != null) {
        const width = clampWidth(Number(wRaw));
        const thumb = await getThumbnail(target, width);
        if (thumb) {
            return new Response(thumb.data, {
                headers: {
                    'content-type': thumb.mime,
                    // immutable: cache key already includes mtime, so the URL
                    // changes when the source does — the browser can hold on
                    // to this forever within a session.
                    'cache-control': 'public, max-age=31536000, immutable'
                }
            });
        }
        // sharp failed (corrupt / unsupported) → fall through to raw stream.
    }

    return new Response(bufferToBytes(readFileSync(target)), {
        headers: { 'content-type': mimeFor(target), 'cache-control': 'public, max-age=300' }
    });
};
