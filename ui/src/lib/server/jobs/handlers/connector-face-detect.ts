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
import { requestBytes } from '../../python-worker';
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
}

// ---- DB statements ------------------------------------------------------
const existsStmt = db.prepare(
    'SELECT 1 FROM connector_pictures WHERE connector_id = ? AND picture_id = ?'
);
const upsertPictureStmt = db.prepare(`
    INSERT INTO connector_pictures(connector_id, picture_id, person_id, filename, image_width, image_height, n_faces)
    VALUES(@connector_id, @picture_id, @person_id, @filename, @image_width, @image_height, @n_faces)
    ON CONFLICT(connector_id, picture_id) DO UPDATE SET
        person_id = excluded.person_id,
        filename = excluded.filename,
        image_width = excluded.image_width,
        image_height = excluded.image_height,
        n_faces = excluded.n_faces,
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

    // ---- 3. Walk the new pictures. ----
    const total = toProcess.length;
    let done = 0;
    let withFaces = 0;
    let noFace = 0;
    let failed = 0;

    for (const pic of toProcess) {
        if (ctx.shouldCancel()) {
            ctx.log('info', `Cancelled after ${done}/${total} pictures.`);
            return;
        }
        try {
            const bytes = await connector.downloadPicture(pic);
            const result = await requestBytes<DetectFacesBlobResp>(
                '/detect-faces-blob',
                bytes,
                { timeoutMs: 60_000 }
            );

            // Image dims: prefer what the worker reports from the actual
            // decoded image; fall back to whatever the connector advertised.
            const imgW = result.image_width ?? pic.width ?? null;
            const imgH = result.image_height ?? pic.height ?? null;

            db.transaction(() => {
                upsertPictureStmt.run({
                    connector_id: params.connector_id,
                    picture_id: pic.id,
                    person_id: params.person_id,
                    filename: pic.filename,
                    image_width: imgW,
                    image_height: imgH,
                    n_faces: result.faces.length
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

            if (result.faces.length > 0) withFaces++;
            else noFace++;
        } catch (e) {
            failed++;
            ctx.log('warn', `${pic.filename}: ${(e as Error).message}`);
        }
        done++;
        ctx.progress(done, total, pic.filename);
    }

    ctx.log(
        'info',
        `Done: ${withFaces} with face(s), ${noFace} no-face, ${failed} failed, ${skipped} previously processed.`
    );
};

registerHandler('connector-face-detect', handler);
