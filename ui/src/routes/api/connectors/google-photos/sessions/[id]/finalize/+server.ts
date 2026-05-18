// Finalize a completed picker session: list the items the user picked,
// download every one into our local cache, persist/refresh the link row
// (cache dir as person_id, sync timestamp + count), enqueue the standard
// face-detect job, then ask Google to forget the session.
//
// Downloads happen here (concurrent, capped) rather than inside the
// face-detect job because the bytes are only fetchable for 60 min after
// the user finished picking. Once cached, every downstream concern reads
// from disk via the local-folder pipeline.
//
// Response shape: NDJSON stream — the client reads one JSON event per
// line and updates a progress bar in real time. Events:
//   { phase: 'start', total: N }
//   { phase: 'progress', done: K, total: N }   (after each saved item)
//   { phase: 'done', count: K, total: N, job_id: number|null }
//   { phase: 'error', message: string }        (terminal)
import {
    cacheDirFor,
    deletePickerSession,
    ensureCacheDir,
    listPickedMediaItems,
    saveItemToCache,
    type PickedMediaItem
} from '$lib/server/connectors/google-photos';
import { recordSync, upsertLink } from '$lib/server/connector-links';
import { enqueue } from '$lib/server/jobs/runner';
import type { RequestHandler } from './$types';

interface Body {
    scope_kind?: 'folder' | 'group';
    scope_key?: string;
}

const CONCURRENCY = 6;

export const POST: RequestHandler = async ({ params, request }) => {
    const body = (await request.json().catch(() => ({}))) as Body;
    const scope_kind = body.scope_kind;
    const scope_key = body.scope_key;
    if (
        (scope_kind !== 'folder' && scope_kind !== 'group') ||
        typeof scope_key !== 'string' ||
        !scope_key
    ) {
        return new Response(JSON.stringify({ error: 'scope_kind + scope_key required' }), {
            status: 400,
            headers: { 'content-type': 'application/json' }
        });
    }

    const items = await listPickedMediaItems(params.id);
    if (items.length === 0) {
        return new Response(JSON.stringify({ error: 'no items picked in this session' }), {
            status: 400,
            headers: { 'content-type': 'application/json' }
        });
    }

    const cacheDir = cacheDirFor(scope_kind, scope_key);
    ensureCacheDir(cacheDir);

    // Upsert the link FIRST so listLinkedFolders() approves the cache dir
    // before downloads kick in (otherwise the proxy/face-detect would 403
    // on a freshly-created dir). The cache dir IS the link's person_id —
    // we don't surface that to users, but face-detect needs it.
    upsertLink({
        scope_kind,
        scope_key,
        connector_id: 'google-photos',
        person_id: cacheDir,
        person_name: null,
        person_thumb_url: null
    });

    const stream = new ReadableStream({
        async start(controller) {
            const enc = new TextEncoder();
            const emit = (obj: Record<string, unknown>) =>
                controller.enqueue(enc.encode(JSON.stringify(obj) + '\n'));

            const total = items.length;
            emit({ phase: 'start', total });

            let written = 0;
            let nextIndex = 0;
            async function worker() {
                while (nextIndex < items.length) {
                    const item: PickedMediaItem = items[nextIndex++];
                    try {
                        const dest = await saveItemToCache(item, cacheDir);
                        if (dest) {
                            written++;
                            emit({ phase: 'progress', done: written, total });
                        }
                    } catch {
                        // One bad item doesn't stop the batch; the count of
                        // 'written' just stays lower than 'total' — the UI
                        // shows the gap.
                    }
                }
            }
            const workers: Promise<void>[] = [];
            for (let k = 0; k < Math.min(CONCURRENCY, total); k++) workers.push(worker());
            try {
                await Promise.all(workers);
            } catch (e) {
                emit({ phase: 'error', message: (e as Error).message });
                controller.close();
                return;
            }

            recordSync(scope_kind, scope_key, 'google-photos', written);
            // Fire-and-forget cleanup — failing the DELETE is harmless.
            void deletePickerSession(params.id);

            let job_id: number | null = null;
            try {
                job_id = enqueue(
                    'connector-face-detect',
                    {
                        connector_id: 'google-photos',
                        person_id: cacheDir,
                        person_name: null
                    },
                    { key_arg1: 'google-photos', key_arg2: cacheDir }
                );
            } catch {
                // Don't fail the whole finalize: the user can retry later.
            }

            emit({ phase: 'done', count: written, total, job_id });
            controller.close();
        }
    });

    return new Response(stream, {
        // Use NDJSON so each newline-terminated line is a JSON object.
        // x-no-compression hints proxies (when present) not to buffer.
        headers: {
            'content-type': 'application/x-ndjson; charset=utf-8',
            'cache-control': 'no-store',
            'x-accel-buffering': 'no'
        }
    });
};
