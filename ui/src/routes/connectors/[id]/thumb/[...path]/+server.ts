// Stream-through proxy for connector-served assets that need auth headers
// (thumbnails, originals). The browser hits us at
//   /connectors/<id>/thumb/<path...>
// We forward to the connector with its own auth and pipe the response back.
import type { RequestHandler } from './$types';
import { error } from '@sveltejs/kit';
import { ALL_CONNECTORS, getConnector } from '$lib/server/connectors/registry';

export const GET: RequestHandler = async ({ params, request }) => {
    const id = params.id;
    if (!ALL_CONNECTORS.some((c) => c.id === id)) throw error(404, 'Unknown connector');

    const connector = getConnector(id as 'immich' | 'google-photos');

    let upstream: Response;
    try {
        upstream = await connector.proxyFetch(request.url.replace(/^https?:\/\/[^/]+/, ''));
    } catch (e) {
        throw error(502, `Upstream connector error: ${(e as Error).message}`);
    }

    // Pipe the body straight back. The browser controls caching via the
    // upstream headers we forward (Immich sets cache-control on its thumbs).
    const headers = new Headers();
    const passThrough = ['content-type', 'content-length', 'cache-control', 'etag', 'last-modified'];
    for (const k of passThrough) {
        const v = upstream.headers.get(k);
        if (v) headers.set(k, v);
    }
    if (!headers.has('cache-control')) headers.set('cache-control', 'private, max-age=300');
    return new Response(upstream.body, { status: upstream.status, headers });
};
