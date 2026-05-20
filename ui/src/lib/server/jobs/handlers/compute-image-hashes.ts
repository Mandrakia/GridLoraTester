// Backfill the BlockHash perceptual hash for everything the suggestion
// engine needs to dedup against — both sides of the comparison:
//
//   Phase 1 — dataset images (dataset_images.phash IS NULL, status='active')
//     Sequential fs.readFile + Sharp+blockhash. Sharp is multi-threaded
//     internally so per-file workers don't help, and DB write contention
//     would serialize at commit anyway.
//
//   Phase 2 — connector pictures linked to this scope
//     (connector_pictures.phash IS NULL for the (connector_id, person_id)
//     of every connector_link pointing at this folder)
//     Adaptive pool over connector.downloadPicture — network-bound,
//     parallelism helps on cloud connectors (Google Photos, remote Immich).
//
// Both phases are idempotent (SQL `WHERE phash IS NULL` filter skips
// already-hashed rows), so a re-run only touches what's missing. Excluded
// images are skipped — the user already decided they're not part of the
// dataset, no point computing dedup material for them.
import { readFileSync } from 'node:fs';
import type { ConnectorId } from '$lib/connectors/types';
import { decodeEmbedding, dot } from '../../centroid-math';
import { getCentroid } from '../../centroids';
import { getConnector } from '../../connectors/registry';
import { updatePhash } from '../../dataset-images';
import { db } from '../../db';
import { computeImageHash } from '../../image-hash';
import { getSettings } from '../../settings';
import { runAdaptivePool } from '../adaptive-pool';
import { registerHandler, type JobContext, type JobHandler } from '../runner';

interface Params {
    /** Optional folder path. When set:
     *   - Phase 1 only hashes dataset images under this folder
     *   - Phase 2 hashes connector pictures linked to this folder
     * When absent, Phase 1 walks the whole dataset_images table; Phase 2
     * is skipped (no scope = no link table to consult). */
    folder_path?: string;
}

// ---- Phase 1: dataset images -------------------------------------------
const listMissingForFolderStmt = db.prepare<[string]>(`
    SELECT image_path
    FROM dataset_images
    WHERE phash IS NULL
      AND status = 'active'
      AND folder_path = ?
`);
const listMissingAllStmt = db.prepare(`
    SELECT image_path
    FROM dataset_images
    WHERE phash IS NULL
      AND status = 'active'
`);

// ---- Phase 2: connector pictures ---------------------------------------
const listLinksForFolderStmt = db.prepare<[string]>(`
    SELECT connector_id, person_id
    FROM connector_links
    WHERE scope_kind = 'folder' AND scope_key = ?
`);
const listMissingConnectorForPersonStmt = db.prepare<[string, string]>(`
    SELECT connector_id, picture_id, filename, image_width, image_height
    FROM connector_pictures
    WHERE connector_id = ? AND person_id = ? AND phash IS NULL
`);
const facesForConnectorPicStmt = db.prepare<[string, string]>(`
    SELECT embedding_b64, pitch, yaw
    FROM connector_faces
    WHERE connector_id = ? AND picture_id = ?
`);
const updateConnectorPhashStmt = db.prepare(
    'UPDATE connector_pictures SET phash = ? WHERE connector_id = ? AND picture_id = ?'
);

interface ConnectorPicRow {
    connector_id: string;
    picture_id: string;
    filename: string | null;
    image_width: number | null;
    image_height: number | null;
}

/** Per-connector full-resolution download URL. Format must match what each
 * connector's `proxyFetch` / `downloadPicture` recognizes — they don't
 * accept bare picture IDs. */
function downloadUrlFor(connector_id: ConnectorId, picture_id: string): string {
    if (connector_id === 'immich') {
        return `/connectors/immich/thumb/assets/${picture_id}/original`;
    }
    // hard-drive: downloadPicture reads picture.id (filesystem path)
    // directly and ignores download_url. Same string is fine.
    return picture_id;
}

/** Node's fetch wraps the actual network error in `e.cause`, leaving the
 * top-level message as the unhelpful "fetch failed". Pull the cause's
 * `code` (`UND_ERR_*`, `ECONNRESET`, etc.) and `message` so the user can
 * tell socket-reset apart from cert-failed apart from rate-limit. */
function formatFetchError(e: Error): string {
    const cause = (e as Error & { cause?: { code?: string; message?: string } }).cause;
    if (cause && (cause.code || cause.message)) {
        return `${e.message} [${cause.code ?? '?'}]${cause.message ? ' ' + cause.message : ''}`;
    }
    return e.message;
}

const handler: JobHandler = async (ctx: JobContext) => {
    const params = ctx.params as unknown as Params;

    // ===== Phase 1: dataset images =====
    const datasetRows = params.folder_path
        ? (listMissingForFolderStmt.all(params.folder_path) as { image_path: string }[])
        : (listMissingAllStmt.all() as { image_path: string }[]);

    // ===== Phase 2: connector pictures linked to this scope =====
    //
    // Pre-filter pass: we only hash pictures that COULD ever surface as a
    // suggestion. Hashing every picture from a person's library when many
    // will never appear as candidates is wasteful (download + Sharp + DB
    // write cost). The qualifying floor mirrors the suggestion engine's own:
    //   1. Resolution: image_width × image_height ≥ min_image_mp
    //   2. Identity: raw cosine to the scope centroid ≥ suggestion_identity_sim_min
    //
    // Requires the scope's centroid to compute (2). If there's no centroid
    // yet, Phase 2 falls back to MP-only since the suggestion engine couldn't
    // surface anything until the centroid is ready.
    const connectorPicsRaw: ConnectorPicRow[] = [];
    if (params.folder_path) {
        const links = listLinksForFolderStmt.all(params.folder_path) as {
            connector_id: string;
            person_id: string;
        }[];
        for (const l of links) {
            const pics = listMissingConnectorForPersonStmt.all(
                l.connector_id,
                l.person_id
            ) as ConnectorPicRow[];
            connectorPicsRaw.push(...pics);
        }
    }

    let connectorPics: ConnectorPicRow[] = connectorPicsRaw;
    let filteredOutMp = 0;
    let filteredOutIdentity = 0;
    let filteredOutNoCentroid = 0;
    if (params.folder_path && connectorPicsRaw.length > 0) {
        const settings = getSettings();
        const rawMinMp = Number(settings.suggestion_min_image_mp);
        const minMp = Number.isFinite(rawMinMp) && rawMinMp > 0 ? rawMinMp : 0;
        const minPixels = minMp * 1_000_000;
        const rawIdMin = Number(settings.suggestion_identity_sim_min);
        const identitySimMin = Number.isFinite(rawIdMin) && rawIdMin > 0 ? rawIdMin : 0;

        const centroidRow = getCentroid('folder', params.folder_path);
        let centroidVec: Float32Array | null = null;
        if (centroidRow) {
            try {
                centroidVec = decodeEmbedding(centroidRow.centroid_b64);
            } catch {
                centroidVec = null; // degrade to MP-only
            }
        }

        const kept: ConnectorPicRow[] = [];
        for (const pic of connectorPicsRaw) {
            // MP filter — cheap and applies always when set.
            if (
                minPixels > 0 &&
                pic.image_width != null &&
                pic.image_height != null &&
                pic.image_width * pic.image_height < minPixels
            ) {
                filteredOutMp++;
                continue;
            }
            // Identity filter — needs centroid. When missing, keep the row
            // (no way to score it) so we don't lose anything irrecoverably.
            if (!centroidVec) {
                filteredOutNoCentroid++;
                kept.push(pic);
                continue;
            }
            const faces = facesForConnectorPicStmt.all(
                pic.connector_id,
                pic.picture_id
            ) as { embedding_b64: string }[];
            let bestSim = -Infinity;
            for (const f of faces) {
                try {
                    const sim = dot(decodeEmbedding(f.embedding_b64), centroidVec);
                    if (sim > bestSim) bestSim = sim;
                } catch {
                    // skip undecodable face
                }
            }
            // Same absolute identity gate the suggestion engine applies, so
            // we hash exactly the set that can surface (no usable face → never
            // a suggestion → skip).
            if (bestSim < identitySimMin) {
                filteredOutIdentity++;
                continue;
            }
            kept.push(pic);
        }
        connectorPics = kept;
    }

    const total = datasetRows.length + connectorPics.length;
    const filterReport =
        filteredOutMp + filteredOutIdentity > 0
            ? ` (pre-filtered: -${filteredOutMp} below MP, -${filteredOutIdentity} below identity floor` +
              (filteredOutNoCentroid > 0
                  ? `, ${filteredOutNoCentroid} kept (no centroid to score)`
                  : '') +
              `)`
            : '';
    ctx.log(
        'info',
        `Backfill scope=${params.folder_path ?? '(all)'} — ` +
            `${datasetRows.length} dataset image(s) + ${connectorPics.length} connector picture(s) without phash${filterReport}.`
    );
    if (total === 0) {
        ctx.log('info', 'Nothing to do — everything already hashed.');
        return;
    }

    let done = 0;
    let hashed = 0;
    let failed = 0;

    /** Push the current counters as a metrics snapshot. The runner
     * debounces writes to ~1s, so this is safe to call on every item.
     * The UI uses `failed` to render an amber pill on completed jobs
     * that had non-trivial in-flight errors. */
    const pushCounters = (force = false): void => {
        ctx.metrics(
            { hashed, failed, total },
            force ? { force: true } : undefined
        );
    };

    // ---- Phase 1 execution: sequential fs read + hash ----
    for (const r of datasetRows) {
        if (ctx.shouldCancel()) {
            ctx.log('info', `Cancelled after ${done}/${total}.`);
            return;
        }
        try {
            const bytes = readFileSync(r.image_path);
            const hash = await computeImageHash(bytes);
            if (hash) {
                updatePhash(r.image_path, hash);
                hashed++;
            } else {
                // Sharp couldn't decode — leave phash NULL so a later run
                // doesn't keep retrying the same un-decodeable file.
                failed++;
            }
        } catch (e) {
            failed++;
            if (failed <= 10) {
                ctx.log('warn', `${r.image_path}: ${(e as Error).message}`);
            }
        }
        done++;
        ctx.progress(done, total, r.image_path);
        pushCounters();
    }

    // ---- Phase 2 execution: parallel connector downloads via pool ----
    if (connectorPics.length > 0) {
        ctx.log('info', `Phase 2: ${connectorPics.length} connector picture(s) to hash.`);
        interface PoolItem {
            row: ConnectorPicRow;
            bytes: Buffer | Uint8Array;
        }
        await runAdaptivePool<ConnectorPicRow, PoolItem>({
            inputs: connectorPics,
            maxProducers: 4,
            queueCapacity: 10,
            scaleWindow: 20,
            scaleUpThresholdMs: 50,
            shouldCancel: () => ctx.shouldCancel(),
            produce: async (row) => {
                const connector = getConnector(row.connector_id as ConnectorId);
                const bytes = await connector.downloadPicture({
                    id: row.picture_id,
                    filename: row.filename ?? row.picture_id,
                    // Immich's downloadPicture passes this through proxyFetch,
                    // which expects `/connectors/immich/thumb/<rest>` to map
                    // to `<base_url>/api/<rest>`. Passing just the picture_id
                    // (UUID) makes proxyFetch fall through to a bare fetch
                    // against `<base_url><uuid>` (no separator) →
                    // getaddrinfo ENOTFOUND. HD ignores this field entirely
                    // (it reads picture.id as a filesystem path).
                    download_url: downloadUrlFor(
                        row.connector_id as ConnectorId,
                        row.picture_id
                    ),
                    created_date: '',
                    width: row.image_width ?? 0,
                    height: row.image_height ?? 0
                });
                return { row, bytes };
            },
            consume: async ({ row, bytes }) => {
                const hash = await computeImageHash(bytes);
                if (hash) {
                    updateConnectorPhashStmt.run(hash, row.connector_id, row.picture_id);
                    hashed++;
                } else {
                    failed++;
                }
                done++;
                ctx.progress(done, total, row.filename ?? row.picture_id);
                pushCounters();
            },
            onProduceError: (row, e) => {
                failed++;
                done++;
                // Log the actual cause (undici code + message) instead of
                // the unhelpful generic "fetch failed" — Node fetch wraps
                // the real network error in `e.cause`.
                if (failed <= 10) {
                    ctx.log(
                        'warn',
                        `${row.filename ?? row.picture_id} (download): ${formatFetchError(e)}`
                    );
                }
                ctx.progress(done, total, row.filename ?? row.picture_id);
                pushCounters();
            },
            onConsumeError: ({ row }, e) => {
                failed++;
                done++;
                if (failed <= 10) {
                    ctx.log('warn', `${row.filename ?? row.picture_id} (hash): ${e.message}`);
                }
                ctx.progress(done, total, row.filename ?? row.picture_id);
                pushCounters();
            }
        });
    }

    // Force one last snapshot so the UI's amber-on-failure pill picks up
    // the final counter without waiting for the next debounce window.
    pushCounters(true);

    ctx.log(
        'info',
        `Done: ${hashed} hashed, ${failed} failed${failed > 10 ? ' (only first 10 logged)' : ''}.`
    );
};

registerHandler('compute-image-hashes', handler);
