// Centroid + two-pass logic, kept in TypeScript so the Python side stays
// limited to its model/ONNX role. All embeddings round-trip as base64-encoded
// float32 (matches what InsightFace produces and what we want to store).
import { Buffer } from 'node:buffer';

export function decodeEmbedding(b64: string): Float32Array {
    const buf = Buffer.from(b64, 'base64');
    // Reuse the underlying ArrayBuffer slice; node's Buffer is backed by one.
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function encodeEmbedding(arr: Float32Array): string {
    return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64');
}

/** Cosine similarity. Inputs MUST already be L2-normalized (which is what
 * the Python side emits), so this collapses to a dot product. */
export function dot(a: Float32Array, b: Float32Array): number {
    let s = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/** L2-normalize in place. Returns the same array for chaining. */
export function normalize(a: Float32Array): Float32Array {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * a[i];
    const n = Math.sqrt(s);
    if (n === 0) return a;
    for (let i = 0; i < a.length; i++) a[i] /= n;
    return a;
}

/** Average a list of equal-length embeddings into a new normalized vector. */
export function meanNormalized(embs: Float32Array[]): Float32Array | null {
    if (embs.length === 0) return null;
    const dim = embs[0].length;
    const acc = new Float32Array(dim);
    for (const e of embs) {
        for (let i = 0; i < dim; i++) acc[i] += e[i];
    }
    for (let i = 0; i < dim; i++) acc[i] /= embs.length;
    return normalize(acc);
}

export interface PyFace {
    face_index: number;
    bbox: number[];
    det_score: number | null;
    embedding_b64: string;
    /** Head pose in degrees, from InsightFace's landmark_3d_68. Each field is
     * `null` when the underlying detector didn't expose a pose for this face. */
    pitch?: number | null;
    yaw?: number | null;
    roll?: number | null;
}

export interface PyImage {
    image_path: string;
    /** Dims of the source image; null when imread failed (no faces either). */
    image_width?: number | null;
    image_height?: number | null;
    faces: PyFace[];
}

/** Output of the two-pass centroid run. Carries the dataset-local decisions
 * (which face index won per multi-face image) so the persistence layer can
 * tag rows with is_target. */
export interface TwoPassResult {
    centroid: Float32Array;
    /** Map image_path → set of face_index values that contributed to the
     * final centroid (single-face images: the sole face; multi-face: the
     * pass-2 winner). */
    targets: Map<string, Set<number>>;
    /** Map image_path → cosine similarity of the winning face vs the FINAL
     * centroid. Used downstream to rank images by closeness to the cluster. */
    image_similarity: Map<string, number>;
    n_single_face: number;
    n_multi_face: number;
    n_no_face: number;
    /** Median / mean of `image_similarity` values, NaN if no faces. */
    median_sim: number;
    mean_sim: number;
}

function median(xs: number[]): number {
    if (xs.length === 0) return NaN;
    const sorted = [...xs].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Run the two-pass centroid algorithm on a list of per-image detections.
 *
 *   Pass 1 — initial centroid from single-face images only.
 *   Pass 2 — for each multi-face image, pick the face with the highest
 *            cosine-sim to the pass-1 centroid as the target.
 *   Final  — centroid over (every single-face face + every pass-2 winner).
 *
 * Returns `null` when no faces were detected at all (centroid would be
 * undefined). When the dataset has only multi-face images (no single-face
 * baseline available), pass 1 falls back to the largest-bbox face per
 * image as a seed.
 */
export function twoPassCentroid(images: PyImage[]): TwoPassResult | null {
    const single: { image_path: string; face: PyFace; emb: Float32Array }[] = [];
    const multi: { image_path: string; faces: { face: PyFace; emb: Float32Array }[] }[] = [];
    let n_no_face = 0;

    for (const img of images) {
        if (img.faces.length === 0) {
            n_no_face++;
            continue;
        }
        if (img.faces.length === 1) {
            const f = img.faces[0];
            single.push({
                image_path: img.image_path,
                face: f,
                emb: decodeEmbedding(f.embedding_b64)
            });
        } else {
            multi.push({
                image_path: img.image_path,
                faces: img.faces.map((f) => ({ face: f, emb: decodeEmbedding(f.embedding_b64) }))
            });
        }
    }

    if (single.length === 0 && multi.length === 0) return null;

    // Pass 1.
    let pass1: Float32Array;
    if (single.length > 0) {
        pass1 = meanNormalized(single.map((s) => s.emb))!;
    } else {
        // No single-face anchor — bootstrap from largest-bbox face per image.
        const seeds = multi.map((m) => {
            let best = m.faces[0];
            let bestArea = 0;
            for (const fe of m.faces) {
                const [x1, y1, x2, y2] = fe.face.bbox;
                const area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
                if (area > bestArea) {
                    bestArea = area;
                    best = fe;
                }
            }
            return best.emb;
        });
        pass1 = meanNormalized(seeds)!;
    }

    // Pass 2.
    const targets = new Map<string, Set<number>>();
    const contributing: Float32Array[] = [];
    for (const s of single) {
        targets.set(s.image_path, new Set([s.face.face_index]));
        contributing.push(s.emb);
    }
    for (const m of multi) {
        let bestI = 0;
        let bestSim = -Infinity;
        for (let i = 0; i < m.faces.length; i++) {
            const sim = dot(m.faces[i].emb, pass1);
            if (sim > bestSim) {
                bestSim = sim;
                bestI = i;
            }
        }
        const winner = m.faces[bestI];
        targets.set(m.image_path, new Set([winner.face.face_index]));
        contributing.push(winner.emb);
    }

    const finalCentroid = meanNormalized(contributing)!;

    // Score each winning face against the FINAL centroid (not pass1 — we want
    // similarity vs the same vector the UI will treat as ground truth).
    const image_similarity = new Map<string, number>();
    for (const s of single) {
        image_similarity.set(s.image_path, dot(s.emb, finalCentroid));
    }
    for (const m of multi) {
        // Re-find the winner by face_index — `targets` already has it.
        const winnerIdx = [...(targets.get(m.image_path) ?? [])][0];
        const winner = m.faces.find((fe) => fe.face.face_index === winnerIdx);
        if (winner) image_similarity.set(m.image_path, dot(winner.emb, finalCentroid));
    }

    const sims = [...image_similarity.values()];
    const mean_sim = sims.length === 0 ? NaN : sims.reduce((a, b) => a + b, 0) / sims.length;
    const median_sim = median(sims);

    return {
        centroid: finalCentroid,
        targets,
        image_similarity,
        n_single_face: single.length,
        n_multi_face: multi.length,
        n_no_face,
        median_sim,
        mean_sim
    };
}

// Threshold constants now live in `$lib/centroid-thresholds.ts` so the
// client-side components can import them without dragging server-only code
// into the browser bundle. Re-export for any server-side caller that
// already imports from this module.
export { DELTA_AMBER, DELTA_RED } from '../centroid-thresholds';
