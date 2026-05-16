// Image proxy for /datasets/group/<id>. The `[...file]` segment is
// `<dataset_slug>/<filename>` where `dataset_slug` matches the basename of
// one of the group's stored paths (with a `__N` disambiguator for duplicates).
//
// Resolving via the in-DB path list keeps the proxy from leaking arbitrary
// filesystem access — only paths the user has explicitly registered in a
// group can be served, regardless of what the URL claims.
import type { RequestHandler } from './$types';
import { error } from '@sveltejs/kit';
import { readFileSync, statSync } from 'node:fs';
import { basename, resolve } from 'node:path';

import { getDatasetGroup } from '$lib/server/dataset-groups';
import { mimeFor } from '$lib/server/mime';
import { bufferToBytes, clampWidth, getThumbnail } from '$lib/server/thumbs';

export const GET: RequestHandler = async ({ params, url }) => {
    const id = Number(params.id);
    if (!Number.isFinite(id) || id <= 0) throw error(404, 'Bad group id');

    const group = getDatasetGroup(id);
    if (!group) throw error(404, 'Group not found');

    const segments = (params.file ?? '').split('/').filter(Boolean);
    if (segments.length < 2) throw error(404);

    const [slug, ...rest] = segments;
    if (rest.length === 0) throw error(404);

    // Rebuild the same slug map the +page.server.ts uses so URL paths match.
    const seen = new Map<string, number>();
    let matchedPath: string | null = null;
    for (const p of group.paths) {
        const base = basename(p);
        const count = (seen.get(base) ?? 0) + 1;
        seen.set(base, count);
        const s = count === 1 ? base : `${base}__${count - 1}`;
        if (s === slug) {
            matchedPath = p;
            break;
        }
    }
    if (!matchedPath) throw error(404, 'Unknown dataset slug');

    const folder = resolve(matchedPath);
    const target = resolve(folder, rest.join('/'));
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
                    'cache-control': 'public, max-age=31536000, immutable'
                }
            });
        }
    }

    return new Response(bufferToBytes(readFileSync(target)), {
        headers: { 'content-type': mimeFor(target), 'cache-control': 'public, max-age=300' }
    });
};
