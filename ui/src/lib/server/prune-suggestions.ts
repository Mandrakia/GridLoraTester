// Prune engine: for each over-represented framing/pose bucket, surface the
// dataset images most redundant with the rest of the bucket — the user
// removes those to free room for under-represented additions without
// hurting bucket diversity.
//
// Redundancy = average cosine similarity to the k=3 nearest neighbors of
// the same image WITHIN the same bucket. Picking the "most redundant"
// (highest mean) drops a near-twin first; picking the "most centroid-
// aligned" would drop the best exemplar instead. The neighbors-only score
// is more stable than "closest single pair" — a lone outlier can't dominate.
//
// Excluded images are filtered out via the dataset_images JOIN (status filter).
// The centroid argument is unused by the redundancy math but is plumbed in
// so callers can switch to centroid-distance ranking later without a
// signature change.
import {
    framingGroupForBand,
    poseGroupForBucket,
    type GroupGap
} from '$lib/dataset-targets';
import {
    classifyFraming,
    computeFramingRatio,
    type FramingBand
} from '$lib/framing-grid';
import { classifyPose, type PoseBucketId } from '$lib/pose-grid';
import { decodeEmbedding, dot } from './centroid-math';
import type { CentroidRow } from './centroids';
import { countActiveForFolders } from './dataset-images';
import { db } from './db';

/** Top candidates surfaced per over-rep bucket. Mirror of SUGGESTION_TOP_N. */
const PRUNE_TOP_N = 12;
/** k for k-NN redundancy. Stable across small buckets; for buckets with
 * fewer than (k+1) members, falls back to the available neighbor count. */
const KNN_K = 3;
/** How many nearest in-bucket neighbors we send to the UI per candidate.
 * The lightbox renders these alongside the zoomed candidate so the user
 * can compare and pick which one to drop. Bigger than KNN_K so the user
 * has visual options beyond the strict redundancy contributors. */
const NEIGHBOR_TOP_N = 6;

/** A near-neighbor of a prune candidate inside the same over-rep bucket.
 * The lightbox renders these so the user can compare and pick which
 * member of the redundancy cluster to drop. */
export interface PruneNeighbor {
    image_path: string;
    folder_path: string;
    filename: string;
    /** Cosine of this neighbor vs the candidate. The K nearest neighbors
     * driving the candidate's redundancy score are at the top of the list. */
    similarity: number;
}

export interface PruneCandidate {
    image_path: string;
    folder_path: string;
    filename: string;
    pose_bucket: PoseBucketId | null;
    framing_bucket: FramingBand | null;
    /** Cosine similarity of this image's target face vs the scope centroid.
     * Tooltip-only — not used for ranking. NULL when no centroid was
     * available when face_embeddings was last written. */
    similarity: number | null;
    yaw: number | null;
    pitch: number | null;
    /** Mean cosine vs the k=3 nearest neighbors in the same bucket. Higher =
     * more redundant. Range ~[0..1]. */
    redundancy_score: number;
    /** Which over-rep group keys this image belongs to — same shape as
     * SuggestionCandidate.gaps_filled, e.g. ['framing:close', 'pose:profile'].
     * Surfaced so a single excluded picture credits ALL the buckets it
     * unloaded (multi-bucket prune candidates are extra valuable). */
    over_rep_buckets: string[];
    /** Top nearest in-bucket neighbors, sorted by similarity desc. Up to
     * NEIGHBOR_TOP_N — the first KNN_K of these are the ones that drove
     * redundancy_score. The UI lightbox can offer to exclude any of them
     * instead of the candidate (the user picks within the cluster). */
    neighbors: PruneNeighbor[];
}

export interface PruneBucket {
    dimension: 'pose' | 'framing';
    group_key: string;
    label: string;
    target: number;
    actual: number;
    /** How many over target (positive integer-ish, can be fractional since
     * targets are floats). UI displays as count. */
    over: number;
    candidates: PruneCandidate[];
}

export interface PruneResult {
    /** Total active dataset images in scope. Denominator for the UI counter
     * ("N / max"). Excludes excluded — that's the whole point. */
    n_active: number;
    /** The scope's max_size if set, null otherwise. */
    max_size: number | null;
    /** One entry per over-rep bucket, sorted by "over" desc (biggest excess
     * first; framing before pose on tie — mirrors the add-suggestion engine
     * ordering for visual coherence). */
    buckets: PruneBucket[];
}

export interface PruneInput {
    folder_paths: string[];
    max_size: number | null;
    pose_gaps: GroupGap[];
    framing_gaps: GroupGap[];
    /** Plumbed but unused today. Reserved for a future centroid-distance
     * ranking variant when DinoV2 lands. */
    centroid: CentroidRow | null;
}

interface ActiveImageRow {
    image_path: string;
    folder_path: string;
    yaw: number | null;
    pitch: number | null;
    roll: number | null;
    similarity: number | null;
    bbox_json: string;
    image_height: number | null;
    embedding_b64: string;
}

interface ClassifiedImage {
    image_path: string;
    folder_path: string;
    yaw: number | null;
    pitch: number | null;
    similarity: number | null;
    framing_bucket: FramingBand | null;
    framing_group: string | null;
    pose_bucket: PoseBucketId | null;
    pose_group: string | null;
    embedding: Float32Array;
}

function loadActiveImages(folderPaths: string[]): ClassifiedImage[] {
    if (folderPaths.length === 0) return [];
    const placeholders = folderPaths.map(() => '?').join(',');
    const stmt = db.prepare(`
        SELECT fe.image_path, di.folder_path, fe.yaw, fe.pitch, fe.roll,
               fe.similarity, fe.bbox_json, di.image_height, fe.embedding_b64
          FROM face_embeddings fe
          JOIN dataset_images di ON di.image_path = fe.image_path
         WHERE di.folder_path IN (${placeholders})
           AND di.status = 'active'
           AND fe.is_target = 1
    `);
    const rows = stmt.all(...folderPaths) as ActiveImageRow[];
    return rows.map((r) => {
        let bbox: number[] | null = null;
        try {
            const parsed = JSON.parse(r.bbox_json);
            if (Array.isArray(parsed)) bbox = parsed.map(Number);
        } catch {
            // ignore — bbox stays null, framing classifier returns null
        }
        const framingRatio = computeFramingRatio(bbox, r.image_height, r.roll);
        const framingBand = classifyFraming(framingRatio);
        const poseBucket = classifyPose(r.yaw, r.pitch);
        return {
            image_path: r.image_path,
            folder_path: r.folder_path,
            yaw: r.yaw,
            pitch: r.pitch,
            similarity: r.similarity,
            framing_bucket: framingBand,
            framing_group: framingBand ? framingGroupForBand(framingBand) : null,
            pose_bucket: poseBucket,
            pose_group: poseBucket ? poseGroupForBucket(poseBucket) : null,
            embedding: decodeEmbedding(r.embedding_b64)
        };
    });
}

interface RankResult {
    score: number;
    neighbors: PruneNeighbor[];
}

/** Compute the mean-of-top-k cosine similarity for each image vs the others
 * in `bucket`, AND capture the top NEIGHBOR_TOP_N closest in-bucket peers
 * so the UI lightbox can render them for cluster comparison. O(n²) within
 * a bucket — fine since buckets cap at ~dataset size (tens, maybe low
 * hundreds). */
function rankByRedundancy(bucket: ClassifiedImage[]): Map<string, RankResult> {
    const out = new Map<string, RankResult>();
    if (bucket.length <= 1) {
        for (const m of bucket) out.set(m.image_path, { score: 0, neighbors: [] });
        return out;
    }
    for (let i = 0; i < bucket.length; i++) {
        const me = bucket[i];
        const pairs: { idx: number; sim: number }[] = new Array(bucket.length - 1);
        let w = 0;
        for (let j = 0; j < bucket.length; j++) {
            if (i === j) continue;
            pairs[w++] = { idx: j, sim: dot(me.embedding, bucket[j].embedding) };
        }
        pairs.sort((a, b) => b.sim - a.sim);
        const knnK = Math.min(KNN_K, pairs.length);
        let sum = 0;
        for (let t = 0; t < knnK; t++) sum += pairs[t].sim;
        const score = sum / knnK;
        const topPeers = pairs.slice(0, Math.min(NEIGHBOR_TOP_N, pairs.length));
        const neighbors: PruneNeighbor[] = topPeers.map((p) => {
            const peer = bucket[p.idx];
            return {
                image_path: peer.image_path,
                folder_path: peer.folder_path,
                filename: peer.image_path.split('/').pop() ?? peer.image_path,
                similarity: p.sim
            };
        });
        out.set(me.image_path, { score, neighbors });
    }
    return out;
}

export function suggestPruneCandidates(input: PruneInput): PruneResult {
    const n_active = countActiveForFolders(input.folder_paths);

    const overFraming = input.framing_gaps.filter((g) => g.signed_gap < 0);
    const overPose = input.pose_gaps.filter((g) => g.signed_gap < 0);
    if (overFraming.length === 0 && overPose.length === 0) {
        return { n_active, max_size: input.max_size, buckets: [] };
    }

    const classified = loadActiveImages(input.folder_paths);
    if (classified.length === 0) {
        return { n_active, max_size: input.max_size, buckets: [] };
    }

    const overFramingKeys = new Set(overFraming.map((g) => g.key));
    const overPoseKeys = new Set(overPose.map((g) => g.key));

    const byFramingGroup = new Map<string, ClassifiedImage[]>();
    const byPoseGroup = new Map<string, ClassifiedImage[]>();
    for (const img of classified) {
        if (img.framing_group && overFramingKeys.has(img.framing_group)) {
            const arr = byFramingGroup.get(img.framing_group) ?? [];
            arr.push(img);
            byFramingGroup.set(img.framing_group, arr);
        }
        if (img.pose_group && overPoseKeys.has(img.pose_group)) {
            const arr = byPoseGroup.get(img.pose_group) ?? [];
            arr.push(img);
            byPoseGroup.set(img.pose_group, arr);
        }
    }

    function gapsForImage(img: ClassifiedImage): string[] {
        const out: string[] = [];
        if (img.framing_group && overFramingKeys.has(img.framing_group)) {
            out.push(`framing:${img.framing_group}`);
        }
        if (img.pose_group && overPoseKeys.has(img.pose_group)) {
            out.push(`pose:${img.pose_group}`);
        }
        return out;
    }

    function buildCandidate(
        img: ClassifiedImage,
        rank: RankResult
    ): PruneCandidate {
        return {
            image_path: img.image_path,
            folder_path: img.folder_path,
            filename: img.image_path.split('/').pop() ?? img.image_path,
            pose_bucket: img.pose_bucket,
            framing_bucket: img.framing_bucket,
            similarity: img.similarity,
            yaw: img.yaw,
            pitch: img.pitch,
            redundancy_score: rank.score,
            over_rep_buckets: gapsForImage(img),
            neighbors: rank.neighbors
        };
    }

    const emptyRank: RankResult = { score: 0, neighbors: [] };
    const buckets: PruneBucket[] = [];
    for (const g of overFraming) {
        const members = byFramingGroup.get(g.key);
        if (!members || members.length === 0) continue;
        const ranks = rankByRedundancy(members);
        const sorted = [...members].sort(
            (a, b) =>
                (ranks.get(b.image_path)?.score ?? 0) -
                (ranks.get(a.image_path)?.score ?? 0)
        );
        buckets.push({
            dimension: 'framing',
            group_key: g.key,
            label: g.label,
            target: g.target,
            actual: g.actual,
            over: -g.signed_gap,
            candidates: sorted
                .slice(0, PRUNE_TOP_N)
                .map((img) => buildCandidate(img, ranks.get(img.image_path) ?? emptyRank))
        });
    }
    for (const g of overPose) {
        const members = byPoseGroup.get(g.key);
        if (!members || members.length === 0) continue;
        const ranks = rankByRedundancy(members);
        const sorted = [...members].sort(
            (a, b) =>
                (ranks.get(b.image_path)?.score ?? 0) -
                (ranks.get(a.image_path)?.score ?? 0)
        );
        buckets.push({
            dimension: 'pose',
            group_key: g.key,
            label: g.label,
            target: g.target,
            actual: g.actual,
            over: -g.signed_gap,
            candidates: sorted
                .slice(0, PRUNE_TOP_N)
                .map((img) => buildCandidate(img, ranks.get(img.image_path) ?? emptyRank))
        });
    }

    // Biggest over first, framing before pose on tie. Mirrors the
    // suggestion engine for visual coherence on the Stats tab.
    buckets.sort((a, b) => {
        if (b.over !== a.over) return b.over - a.over;
        return a.dimension === 'framing' ? -1 : 1;
    });

    return { n_active, max_size: input.max_size, buckets };
}
