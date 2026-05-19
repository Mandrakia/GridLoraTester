// Streams files out of `<tests_root>/<name>/` so the Result/Grid iframe in
// the tests page can display the index.html the Python pipeline writes —
// and serve the images it references with their relative paths intact.
//
// With `?w=<int>` returns a sharp-resized WebP thumbnail (cached on disk),
// same proxy contract as /datasets/.../raw. The dashboard's live grid view
// uses this for the per-cell thumbs so the browser doesn't repaint full-
// MP outputs on every column-count change.
//
// Hardened against path traversal: the resolved absolute path must stay
// inside the test's own output dir.
import type { RequestHandler } from './$types';
import { error } from '@sveltejs/kit';
import { readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

import { mimeFor } from '$lib/server/mime';
import { isPathInside } from '$lib/server/path-utils';
import { getSettings } from '$lib/server/settings';
import { bufferToBytes, clampWidth, getThumbnail } from '$lib/server/thumbs';

export const GET: RequestHandler = async ({ params, url }) => {
    const { tests_root } = getSettings();
    if (!tests_root) throw error(404, 'tests_root not configured');

    const root = resolve(tests_root, params.name);
    const relative = params.file && params.file.length > 0 ? params.file : 'index.html';
    const requested = resolve(root, relative);

    // Path-traversal guard: the resolved path must live inside `root`.
    if (!isPathInside(root, requested)) {
        throw error(403, 'Forbidden');
    }

    try {
        const st = statSync(requested);
        if (!st.isFile()) throw error(404);
    } catch {
        throw error(404, 'Not found');
    }

    const wRaw = url.searchParams.get('w');
    if (wRaw != null) {
        const width = clampWidth(Number(wRaw));
        const thumb = await getThumbnail(requested, width);
        if (thumb) {
            return new Response(thumb.data, {
                headers: {
                    'content-type': thumb.mime,
                    // Cache key already includes mtime; the URL changes when
                    // the source does, so the browser can hold this forever.
                    'cache-control': 'public, max-age=31536000, immutable'
                }
            });
        }
        // sharp failed (corrupt / unsupported) → fall through to raw stream.
    }

    const data = readFileSync(requested);
    return new Response(bufferToBytes(data), {
        headers: {
            'content-type': mimeFor(requested),
            // The grid index.html is rewritten by the Python pipeline on every
            // row update; opt out of caching so a tab refresh always shows
            // the freshest state.
            'cache-control': 'no-cache'
        }
    });
};
