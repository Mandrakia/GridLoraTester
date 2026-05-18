// Job handler: for a given (connector, person), walk every picture, skip
// the ones we've already processed, download the rest in-memory, send them
// to the Python worker for face detection, and persist into
// connector_pictures + connector_faces.
//
// Idempotent: a re-run only touches pictures whose `picture_id` isn't
// already in `connector_pictures` for this connector. Cancellation is
// checked between every picture.
import type { ConnectorId, ConnectorPicture } from '$lib/connectors/types';
import { db } from '../../db';
import { getConnector } from '../../connectors/registry';
import { computeImageHash } from '../../image-hash';
import { buildFaceWorkerHeaders, requestBytes } from '../../python-worker';
import { runAdaptivePool, type AdaptivePoolMetrics } from '../adaptive-pool';
import { PhaseStats } from '../phase-stats';
import { registerHandler, type JobContext, type JobHandler } from '../runner';

interface DetectFacesBlobResp {
    image_width: number | null;
    image_height: number | null;
    faces: {
        face_index: number;
        bbox: number[];
        det_score: number | null;
        embedding_b64: string;
        pitch: number | null;
        yaw: number | null;
        roll: number | null;
    }[];
    /** Per-phase timings reported by the Python worker (ms). Present on
     * every successful response since the OOM-fix patch — guard for
     * older callers regardless. */
    timing_ms?: {
        decode: number;
        detect: number;
        total: number;
    };
}

// ---- DB statements ------------------------------------------------------
const existsStmt = db.prepare(
    'SELECT 1 FROM connector_pictures WHERE connector_id = ? AND picture_id = ?'
);
const upsertPictureStmt = db.prepare(`
    INSERT INTO connector_pictures(connector_id, picture_id, person_id, filename, image_width, image_height, n_faces, phash)
    VALUES(@connector_id, @picture_id, @person_id, @filename, @image_width, @image_height, @n_faces, @phash)
    ON CONFLICT(connector_id, picture_id) DO UPDATE SET
        person_id = excluded.person_id,
        filename = excluded.filename,
        image_width = excluded.image_width,
        image_height = excluded.image_height,
        n_faces = excluded.n_faces,
        phash = excluded.phash,
        processed_at = datetime('now')
`);
const insertFaceStmt = db.prepare(`
    INSERT INTO connector_faces(connector_id, picture_id, face_index, bbox_json, det_score, embedding_b64, pitch, yaw, roll)
    VALUES(@connector_id, @picture_id, @face_index, @bbox_json, @det_score, @embedding_b64, @pitch, @yaw, @roll)
    ON CONFLICT(connector_id, picture_id, face_index) DO UPDATE SET
        bbox_json = excluded.bbox_json,
        det_score = excluded.det_score,
        embedding_b64 = excluded.embedding_b64,
        pitch = excluded.pitch,
        yaw = excluded.yaw,
        roll = excluded.roll,
        computed_at = datetime('now')
`);

interface Params {
    connector_id: ConnectorId;
    person_id: string;
    /** Caller-visible label; we use it in logs but it's optional. */
    person_name?: string;
}

const handler: JobHandler = async (ctx: JobContext) => {
    const params = ctx.params as unknown as Params;
    if (!params.connector_id || !params.person_id) {
        throw new Error('Missing connector_id or person_id');
    }
    const connector = getConnector(params.connector_id);

    ctx.log(
        'info',
        `Starting face detection for ${params.connector_id} person ${params.person_name ?? params.person_id}`
    );

    // ---- 1. Enumerate every picture (paginated). ----
    const all: ConnectorPicture[] = [];
    let cursor: string | null | undefined = null;
    while (true) {
        if (ctx.shouldCancel()) return;
        const page = await connector.listPictures(params.person_id, { cursor });
        all.push(...page.pictures);
        cursor = page.nextCursor ?? null;
        ctx.progress(0, undefined, `Enumerating pictures (${all.length}…)`);
        if (!cursor) break;
    }

    // ---- 2. Filter out pictures we've already processed for this connector. ----
    const toProcess: ConnectorPicture[] = [];
    for (const p of all) {
        const seen = existsStmt.get(params.connector_id, p.id);
        if (!seen) toProcess.push(p);
    }
    const skipped = all.length - toProcess.length;
    ctx.log(
        'info',
        `Found ${all.length} picture(s); ${skipped} already processed, ${toProcess.length} to detect.`
    );

    // ---- 3. Walk the new pictures via an adaptive pool. ----
    // Producers handle downloads (network/disk, parallelizable). The single
    // consumer talks to the Python InsightFace worker + writes DB rows
    // (sequential — that worker IS the GPU, one session, one queue). The
    // pool grows the producer count automatically when the consumer is
    // starved on `queue.get()` — self-tunes to whatever your HTTP latency
    // and connector throughput are.
    const total = toProcess.length;
    let withFaces = 0;
    let noFace = 0;
    let failed = 0;
    const stats = new PhaseStats();
    const runStartedAt = Date.now();
    /** Item flowing producers → consumer. We close over the picture so the
     * consumer can log on failure + write rows keyed on `pic.id`. */
    interface PoolItem {
        pic: ConnectorPicture;
        bytes: Buffer | Uint8Array;
    }
    let lastPipeline: AdaptivePoolMetrics | null = null;

    function pushMetrics(force = false): void {
        const elapsedSec = (Date.now() - runStartedAt) / 1000;
        const done = withFaces + noFace + failed;
        ctx.metrics(
            {
                ...stats.snapshot(),
                throughput_per_s: elapsedSec > 0 ? done / elapsedSec : 0,
                with_faces: withFaces,
                no_face: noFace,
                failed,
                pipeline: lastPipeline
                    ? {
                          active_producers: lastPipeline.active_producers,
                          queue_depth: lastPipeline.queue_depth,
                          queue_capacity: lastPipeline.queue_capacity,
                          consumer_wait_p50_ms: lastPipeline.consumer_wait_p50_ms,
                          scale_ups: lastPipeline.scale_ups
                      }
                    : null
            },
            force ? { force: true } : undefined
        );
    }

    await runAdaptivePool<ConnectorPicture, PoolItem>({
        inputs: toProcess,
        maxProducers: 4,
        queueCapacity: 10,
        scaleWindow: 20,
        scaleUpThresholdMs: 50,
        shouldCancel: () => ctx.shouldCancel(),
        produce: async (pic) => {
            const t0 = performance.now();
            const bytes = await connector.downloadPicture(pic);
            stats.record('download', performance.now() - t0);
            return { pic, bytes };
        },
        consume: async ({ pic, bytes }) => {
            const tHttpStart = performance.now();
            const result = await requestBytes<DetectFacesBlobResp>(
                '/detect-faces-blob',
                bytes,
                { timeoutMs: 60_000, headers: buildFaceWorkerHeaders() }
            );
            const tHttpEnd = performance.now();

            // Perceptual hash: computed on the bytes we already have. The
            // few ms of Sharp+blockhash cost amortizes nicely vs. having
            // to re-fetch the asset later for a separate hash pass.
            const tHashStart = performance.now();
            const phash = await computeImageHash(bytes);
            const tHashEnd = performance.now();
            stats.record('phash', tHashEnd - tHashStart);

            const imgW = result.image_width ?? pic.width ?? null;
            const imgH = result.image_height ?? pic.height ?? null;

            const tDbStart = performance.now();
            db.transaction(() => {
                upsertPictureStmt.run({
                    connector_id: params.connector_id,
                    picture_id: pic.id,
                    person_id: params.person_id,
                    filename: pic.filename,
                    image_width: imgW,
                    image_height: imgH,
                    n_faces: result.faces.length,
                    phash
                });
                for (const f of result.faces) {
                    insertFaceStmt.run({
                        connector_id: params.connector_id,
                        picture_id: pic.id,
                        face_index: f.face_index,
                        bbox_json: JSON.stringify(f.bbox),
                        det_score: f.det_score,
                        embedding_b64: f.embedding_b64,
                        pitch: f.pitch,
                        yaw: f.yaw,
                        roll: f.roll
                    });
                }
            })();
            const tDbEnd = performance.now();

            const httpRtt = tHttpEnd - tHttpStart;
            stats.record('detect_http', httpRtt);
            if (result.timing_ms) {
                stats.record('python_decode', result.timing_ms.decode);
                stats.record('python_detect', result.timing_ms.detect);
                stats.record(
                    'http_overhead',
                    Math.max(0, httpRtt - result.timing_ms.total)
                );
            }
            stats.record('db_write', tDbEnd - tDbStart);

            if (result.faces.length > 0) withFaces++;
            else noFace++;
        },
        onProduceError: (pic, e) => {
            failed++;
            ctx.log('warn', `${pic.filename} (download): ${e.message}`);
        },
        onConsumeError: ({ pic }, e) => {
            failed++;
            ctx.log('warn', `${pic.filename}: ${e.message}`);
        },
        onMetrics: (m) => {
            lastPipeline = m;
            // No per-item label — items don't have a stable "current"
            // when 4 are in flight. Progress count is enough.
            ctx.progress(withFaces + noFace + failed, total, undefined);
            pushMetrics();
        }
    });

    if (ctx.shouldCancel()) {
        ctx.log('info', `Cancelled after ${withFaces + noFace + failed}/${total} pictures.`);
        pushMetrics(true);
        return;
    }

    // Force a final snapshot so the UI shows the final aggregated p50/p95 +
    // throughput right after completion (not whatever the last debounced
    // sample happened to be).
    pushMetrics(true);

    ctx.log(
        'info',
        `Done: ${withFaces} with face(s), ${noFace} no-face, ${failed} failed, ${skipped} previously processed.`
    );
};

registerHandler('connector-face-detect', handler);
