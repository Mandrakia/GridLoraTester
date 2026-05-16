// Pipeline: ask the long-lived Python worker for face embeddings (only its
// model/ONNX work runs there), then do all the math + persistence in TS.
//
// For a single folder: one 'folder' centroid is written. For a group of N
// folders: N 'folder' centroids + one 'group' centroid (over the union of
// every face from every member, with its own two-pass).
import { encodeEmbedding, twoPassCentroid, type PyImage } from './centroid-math';
import { db } from './db';
import { request as workerRequest } from './python-worker';

interface PyDataset {
    path: string;
    images: PyImage[];
}
interface PyOutput {
    datasets: PyDataset[];
}

export interface CentroidStats {
    n_single_face: number;
    n_multi_face: number;
    n_no_face: number;
}

export interface RunResult {
    /** Per-folder centroid stats keyed by absolute folder path. */
    per_folder: Record<string, CentroidStats>;
    /** Stats of the union pass (only populated when `groupId` is non-null). */
    global: CentroidStats | null;
    persisted_faces: number;
}

const deleteFacesByImageStmt = db.prepare('DELETE FROM face_embeddings WHERE image_path = ?');
const insertFaceStmt = db.prepare(`
    INSERT INTO face_embeddings(
        image_path, face_index, bbox_json, det_score, embedding_b64,
        is_target, similarity, pitch, yaw, roll,
        image_width, image_height
    )
    VALUES(
        @image_path, @face_index, @bbox_json, @det_score, @embedding_b64,
        @is_target, @similarity, @pitch, @yaw, @roll,
        @image_width, @image_height
    )
    ON CONFLICT(image_path, face_index) DO UPDATE SET
        bbox_json = excluded.bbox_json,
        det_score = excluded.det_score,
        embedding_b64 = excluded.embedding_b64,
        is_target = excluded.is_target,
        similarity = excluded.similarity,
        pitch = excluded.pitch,
        yaw = excluded.yaw,
        roll = excluded.roll,
        image_width = excluded.image_width,
        image_height = excluded.image_height,
        computed_at = datetime('now')
`);
const upsertCentroidStmt = db.prepare(`
    INSERT INTO centroids(scope_kind, scope_key, centroid_b64, n_single_face, n_multi_face, n_no_face, median_sim, mean_sim)
    VALUES(@scope_kind, @scope_key, @centroid_b64, @n_single_face, @n_multi_face, @n_no_face, @median_sim, @mean_sim)
    ON CONFLICT(scope_kind, scope_key) DO UPDATE SET
        centroid_b64 = excluded.centroid_b64,
        n_single_face = excluded.n_single_face,
        n_multi_face = excluded.n_multi_face,
        n_no_face = excluded.n_no_face,
        median_sim = excluded.median_sim,
        mean_sim = excluded.mean_sim,
        computed_at = datetime('now')
`);
const getCentroidStmt = db.prepare(
    'SELECT centroid_b64, n_single_face, n_multi_face, n_no_face, median_sim, mean_sim, computed_at FROM centroids WHERE scope_kind = ? AND scope_key = ?'
);

export interface CentroidRow {
    centroid_b64: string;
    n_single_face: number;
    n_multi_face: number;
    n_no_face: number;
    median_sim: number | null;
    mean_sim: number | null;
    computed_at: string;
}

export function getCentroid(scopeKind: 'folder' | 'group', scopeKey: string): CentroidRow | null {
    const row = getCentroidStmt.get(scopeKind, scopeKey) as CentroidRow | undefined;
    return row ?? null;
}

// ---- TS-side centroid recompute (no Python required) --------------------
//
// Recompute a folder's centroid + per-row similarity/is_target from the
// face_embeddings rows already in DB. Used after an import: we just added
// new face rows, no need to re-detect anything.
const loadFolderFacesStmt = db.prepare(`
    SELECT image_path, face_index, bbox_json, det_score, embedding_b64,
           pitch, yaw, roll, image_width, image_height
    FROM face_embeddings
    WHERE image_path LIKE ? ESCAPE '\\'
`);
const clearOneImageStmt = db.prepare('DELETE FROM face_embeddings WHERE image_path = ?');

interface DbFaceRow {
    image_path: string;
    face_index: number;
    bbox_json: string;
    det_score: number | null;
    embedding_b64: string;
    pitch: number | null;
    yaw: number | null;
    roll: number | null;
    image_width: number | null;
    image_height: number | null;
}

function escapeForLike(s: string): string {
    return s.replace(/\\/g, '\\\\').replace(/%/g, '\\%').replace(/_/g, '\\_');
}

/** Read every face_embeddings row under `folderPath` and reshape into the
 * PyImage[] structure twoPassCentroid expects. Returns empty when nothing
 * lives under that folder. */
function loadImagesUnderFolder(folderPath: string): PyImage[] {
    const rows = loadFolderFacesStmt.all(escapeForLike(folderPath) + '/%') as DbFaceRow[];
    if (rows.length === 0) return [];

    type ImgAcc = {
        image_path: string;
        image_width: number | null;
        image_height: number | null;
        faces: PyImage['faces'];
    };
    const byImage = new Map<string, ImgAcc>();
    for (const r of rows) {
        let img = byImage.get(r.image_path);
        if (!img) {
            img = {
                image_path: r.image_path,
                image_width: r.image_width,
                image_height: r.image_height,
                faces: []
            };
            byImage.set(r.image_path, img);
        }
        let bbox: number[] = [];
        try {
            const parsed = JSON.parse(r.bbox_json);
            if (Array.isArray(parsed)) bbox = parsed.map(Number);
        } catch {
            // ignore — bbox stays empty, twoPassCentroid will still take
            // the embedding into account but framing class won't work
        }
        img.faces.push({
            face_index: r.face_index,
            bbox,
            det_score: r.det_score,
            embedding_b64: r.embedding_b64,
            pitch: r.pitch,
            yaw: r.yaw,
            roll: r.roll
        });
    }
    return [...byImage.values()];
}

/** Re-derive the centroid for a folder from the face_embeddings it already
 * holds, then re-upsert per-row similarity/is_target + the centroids row.
 * Returns the new CentroidRow, or null when the folder has no faces at all.
 *
 * Important properties:
 *  - The face_embeddings load + delete + reinsert all happen inside a single
 *    serialized better-sqlite3 transaction, so a concurrent `addPictureToDataset`
 *    cannot lose-write a face row.
 *  - Only image_paths we actually touch are deleted (per-image, not a
 *    `LIKE folder/%` wipe), so any tombstone rows for unrelated images stay.
 *  - `n_no_face` is preserved from the prior centroids row (we only see rows,
 *    so we have no idea how many no-face images exist).
 */
export function recomputeFolderCentroidFromDb(
    folderPath: string
): CentroidRow | null {
    const priorNoFace = getCentroid('folder', folderPath)?.n_no_face ?? 0;

    let row: CentroidRow | null = null;
    db.transaction(() => {
        // Reading inside the tx serializes against concurrent writers.
        const images = loadImagesUnderFolder(folderPath);
        if (images.length === 0) return;

        const result = twoPassCentroid(images);
        if (result == null) return;

        for (const img of images) {
            clearOneImageStmt.run(img.image_path);
            const winners = result.targets.get(img.image_path) ?? new Set<number>();
            const winnerSim = result.image_similarity.get(img.image_path);
            for (const f of img.faces) {
                const isTarget = winners.has(f.face_index);
                insertFaceStmt.run({
                    image_path: img.image_path,
                    face_index: f.face_index,
                    bbox_json: JSON.stringify(f.bbox),
                    det_score: f.det_score,
                    embedding_b64: f.embedding_b64,
                    is_target: isTarget ? 1 : 0,
                    similarity: isTarget && winnerSim != null ? winnerSim : null,
                    pitch: f.pitch ?? null,
                    yaw: f.yaw ?? null,
                    roll: f.roll ?? null,
                    image_width: img.image_width ?? null,
                    image_height: img.image_height ?? null
                });
            }
        }
        upsertCentroidStmt.run({
            scope_kind: 'folder',
            scope_key: folderPath,
            centroid_b64: encodeEmbedding(result.centroid),
            n_single_face: result.n_single_face,
            n_multi_face: result.n_multi_face,
            // Preserve the count from the last full detection run — we
            // have no view into images with zero face rows here.
            n_no_face: priorNoFace,
            median_sim: Number.isFinite(result.median_sim) ? result.median_sim : null,
            mean_sim: Number.isFinite(result.mean_sim) ? result.mean_sim : null
        });
        row = getCentroid('folder', folderPath);
    })();

    return row;
}

/** Re-derive a group's union centroid from face_embeddings already in DB.
 * Mirrors the union pass in computeAndPersist but skips per-folder writes
 * (those are owned by recomputeFolderCentroidFromDb / computeAndPersist).
 * Only upserts the 'group' centroid row.
 * Returns the new CentroidRow, or null when the union has no faces. */
export function recomputeGroupCentroidFromDb(
    groupId: number,
    memberPaths: string[]
): CentroidRow | null {
    const allImages: PyImage[] = [];
    for (const p of memberPaths) {
        for (const img of loadImagesUnderFolder(p)) allImages.push(img);
    }
    if (allImages.length === 0) return null;

    const result = twoPassCentroid(allImages);
    if (result == null) return null;

    upsertCentroidStmt.run({
        scope_kind: 'group',
        scope_key: String(groupId),
        centroid_b64: encodeEmbedding(result.centroid),
        n_single_face: result.n_single_face,
        n_multi_face: result.n_multi_face,
        // No-face images aren't in DB at all — leave at 0; the group view
        // only uses median/mean.
        n_no_face: 0,
        median_sim: Number.isFinite(result.median_sim) ? result.median_sim : null,
        mean_sim: Number.isFinite(result.mean_sim) ? result.mean_sim : null
    });

    return getCentroid('group', String(groupId));
}

async function runPython(paths: string[]): Promise<PyOutput> {
    // The worker keeps the InsightFace model warm across calls; the first
    // request on a fresh dashboard pays the ~3 s model-load cost, subsequent
    // ones are detection-only.
    return workerRequest<PyOutput>('/detect-faces', { paths });
}

/** Compute + persist centroids for one or more folder paths.
 * Always produces a per-folder centroid for each member; when `groupId` is
 * non-null, additionally produces a global centroid using the union of all
 * detected faces. */
export async function computeAndPersist(
    paths: string[],
    groupId: number | null = null
): Promise<RunResult> {
    const output = await runPython(paths);

    const writeAll = db.transaction((datasets: PyDataset[]) => {
        let persisted_faces = 0;
        const per_folder: Record<string, CentroidStats> = {};

        for (const ds of datasets) {
            // Refresh face rows for every image we touched (handles deletes
            // on disk cleanly: an image dropped between runs no longer appears
            // in the detector output, so its old row stays — call it a tombstone
            // that we ignore. The user can wipe the table manually if needed.)
            for (const img of ds.images) {
                deleteFacesByImageStmt.run(img.image_path);
            }

            const result = twoPassCentroid(ds.images);
            if (result == null) {
                per_folder[ds.path] = { n_single_face: 0, n_multi_face: 0, n_no_face: ds.images.length };
                continue;
            }

            // Insert ALL detected faces (including the losing faces from
            // multi-face images), with is_target + similarity set per the
            // two-pass. Losers keep similarity=NULL — they didn't contribute
            // to the centroid so a similarity score there is meaningless.
            const winnerSim = result.image_similarity;
            for (const img of ds.images) {
                const winners = result.targets.get(img.image_path) ?? new Set<number>();
                const simForWinner = winnerSim.get(img.image_path);
                for (const f of img.faces) {
                    const isTarget = winners.has(f.face_index);
                    insertFaceStmt.run({
                        image_path: img.image_path,
                        face_index: f.face_index,
                        bbox_json: JSON.stringify(f.bbox),
                        det_score: f.det_score,
                        embedding_b64: f.embedding_b64,
                        is_target: isTarget ? 1 : 0,
                        similarity: isTarget && simForWinner != null ? simForWinner : null,
                        pitch: f.pitch ?? null,
                        yaw: f.yaw ?? null,
                        roll: f.roll ?? null,
                        image_width: img.image_width ?? null,
                        image_height: img.image_height ?? null
                    });
                    persisted_faces++;
                }
            }

            upsertCentroidStmt.run({
                scope_kind: 'folder',
                scope_key: ds.path,
                centroid_b64: encodeEmbedding(result.centroid),
                n_single_face: result.n_single_face,
                n_multi_face: result.n_multi_face,
                n_no_face: result.n_no_face,
                median_sim: Number.isFinite(result.median_sim) ? result.median_sim : null,
                mean_sim: Number.isFinite(result.mean_sim) ? result.mean_sim : null
            });
            per_folder[ds.path] = {
                n_single_face: result.n_single_face,
                n_multi_face: result.n_multi_face,
                n_no_face: result.n_no_face
            };
        }

        let global: CentroidStats | null = null;
        if (groupId != null && datasets.length > 1) {
            // Union pass: feed every image from every dataset into one
            // two-pass run. Doesn't touch face_embeddings (already persisted
            // above) — only the group centroid row.
            const allImages: PyImage[] = datasets.flatMap((d) => d.images);
            const result = twoPassCentroid(allImages);
            if (result != null) {
                upsertCentroidStmt.run({
                    scope_kind: 'group',
                    scope_key: String(groupId),
                    centroid_b64: encodeEmbedding(result.centroid),
                    n_single_face: result.n_single_face,
                    n_multi_face: result.n_multi_face,
                    n_no_face: result.n_no_face,
                    median_sim: Number.isFinite(result.median_sim) ? result.median_sim : null,
                    mean_sim: Number.isFinite(result.mean_sim) ? result.mean_sim : null
                });
                global = {
                    n_single_face: result.n_single_face,
                    n_multi_face: result.n_multi_face,
                    n_no_face: result.n_no_face
                };
            }
        }

        return { per_folder, global, persisted_faces };
    });

    return writeAll(output.datasets);
}
