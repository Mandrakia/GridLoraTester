// Suggestion engine: for a given scope's centroid + coverage, mine the
// connector_faces table for pictures of the linked persons that would help
// fill the dataset's under-represented pose/framing buckets — taking the
// pose-tempered similarity into account so a profile picture isn't dismissed
// just because its raw similarity is low.
//
// Per picture, the face with the highest similarity to the centroid is the
// "representative" — that's the one we score, classify, and surface.
import { poseSimOffset, type PoseOffsets } from '../centroid-thresholds';
import {
    framingGroupForBand,
    poseGroupForBucket,
    SUGGESTION_DELTA_MIN,
    SUGGESTION_TOP_N,
    type GroupGap
} from '../dataset-targets';
import {
    classifyFraming,
    computeFramingRatio,
    type FramingBand
} from '../framing-grid';
import { classifyPose, type PoseBucketId } from '../pose-grid';
import { decodeEmbedding, dot } from './centroid-math';
import type { CentroidRow } from './centroids';
import { listLinksForScope, type LinkScope } from './connector-links';
import { importedKeysForScope } from './dataset-import';
import { db } from './db';

interface ConnectorFaceRow {
    connector_id: string;
    picture_id: string;
    filename: string | null;
    image_width: number | null;
    image_height: number | null;
    face_index: number;
    bbox_json: string;
    embedding_b64: string;
    pitch: number | null;
    yaw: number | null;
    roll: number | null;
}

const facesForPersonStmt = db.prepare(`
    SELECT cp.connector_id, cp.picture_id, cp.filename, cp.image_width, cp.image_height,
           cf.face_index, cf.bbox_json, cf.embedding_b64, cf.pitch, cf.yaw, cf.roll
    FROM connector_pictures cp
    JOIN connector_faces cf USING (connector_id, picture_id)
    WHERE cp.connector_id = ? AND cp.person_id = ?
`);

export interface SuggestionCandidate {
    connector_id: string;
    picture_id: string;
    filename: string | null;
    thumbnail_url: string;
    /** Raw cosine similarity to the centroid (winning face). */
    similarity: number;
    /** sim - median + pose_offset — same number the cell pills use. */
    tempered_delta: number;
    pose_offset_applied: number;
    pose_bucket: PoseBucketId | null;
    framing_bucket: FramingBand | null;
    yaw: number | null;
    pitch: number | null;
}

export interface SuggestionGroup {
    /** 'pose' or 'framing'. */
    dimension: 'pose' | 'framing';
    /** Key from POSE_TARGETS / FRAMING_TARGETS (e.g. 'profile', 'close'). */
    group_key: string;
    label: string;
    target: number;
    actual: number;
    gap: number;
    candidates: SuggestionCandidate[];
}

export interface SuggestionResult {
    /** True when at least one connector link exists for the scope. */
    has_linked_connectors: boolean;
    /** True when we couldn't score anything (no centroid yet, or no faces
     * in the linked connectors). */
    no_data: boolean;
    /** Groups with a positive gap, each carrying top-N candidates. Sorted
     * with the biggest gaps first. */
    groups: SuggestionGroup[];
    /** Total number of distinct pictures we considered (deduped). */
    candidates_pool: number;
    /** How many of those passed the identity floor. */
    candidates_qualifying: number;
}

function thumbnailUrlFor(connector_id: string, picture_id: string): string {
    if (connector_id === 'immich') {
        return `/connectors/immich/thumb/assets/${picture_id}/thumbnail`;
    }
    if (connector_id === 'hard-drive') {
        // picture_id is the absolute file path; the HD proxy decodes the
        // same url-safe base64 we produce here. Kept inline rather than
        // imported to avoid pulling the hard-drive module into a hot path.
        const b64 = Buffer.from(picture_id, 'utf-8')
            .toString('base64')
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=+$/, '');
        return `/connectors/hard-drive/thumb/${b64}`;
    }
    return `/connectors/${connector_id}/thumb/asset/${picture_id}`;
}

/** Group rows by (connector_id, picture_id), keep the face with the highest
 * dot(emb, centroid) per picture. Returns one SuggestionCandidate per pic. */
function pickRepresentatives(
    rows: ConnectorFaceRow[],
    centroid: Float32Array,
    medianSim: number,
    poseOverrides: PoseOffsets | null
): SuggestionCandidate[] {
    type Best = {
        row: ConnectorFaceRow;
        embedding: Float32Array;
        similarity: number;
    };
    const byPic = new Map<string, Best>();
    for (const r of rows) {
        const emb = decodeEmbedding(r.embedding_b64);
        const sim = dot(emb, centroid);
        const key = `${r.connector_id}::${r.picture_id}`;
        const prev = byPic.get(key);
        if (!prev || sim > prev.similarity) {
            byPic.set(key, { row: r, embedding: emb, similarity: sim });
        }
    }

    const out: SuggestionCandidate[] = [];
    for (const best of byPic.values()) {
        const r = best.row;
        const offset = poseSimOffset(r.yaw, r.pitch, poseOverrides);
        const tempered = best.similarity - medianSim + offset;
        let bbox: number[] | null = null;
        try {
            const parsed = JSON.parse(r.bbox_json);
            if (Array.isArray(parsed)) bbox = parsed.map(Number);
        } catch {
            // ignore — framing bucket will be null
        }
        const framingRatio = computeFramingRatio(bbox, r.image_height, r.roll);
        out.push({
            connector_id: r.connector_id,
            picture_id: r.picture_id,
            filename: r.filename,
            thumbnail_url: thumbnailUrlFor(r.connector_id, r.picture_id),
            similarity: best.similarity,
            tempered_delta: tempered,
            pose_offset_applied: offset,
            pose_bucket: classifyPose(r.yaw, r.pitch),
            framing_bucket: classifyFraming(framingRatio),
            yaw: r.yaw,
            pitch: r.pitch
        });
    }
    return out;
}

interface SuggestInput {
    scope_kind: LinkScope;
    scope_key: string;
    centroid: CentroidRow | null;
    pose_overrides: PoseOffsets | null;
    pose_gaps: GroupGap[];
    framing_gaps: GroupGap[];
}

export function suggestExternalPictures(input: SuggestInput): SuggestionResult {
    const links = listLinksForScope(input.scope_kind, input.scope_key);
    const has_linked_connectors = links.length > 0;

    if (!has_linked_connectors || !input.centroid) {
        return {
            has_linked_connectors,
            no_data: true,
            groups: [],
            candidates_pool: 0,
            candidates_qualifying: 0
        };
    }

    const centroid = decodeEmbedding(input.centroid.centroid_b64);
    const medianSim = input.centroid.median_sim ?? 0;

    // Collect all face rows from every linked (connector, person),
    // skipping rows for pictures already imported into this scope — the
    // dedup pre-requisite for "Add to dataset".
    const alreadyImported = importedKeysForScope(input.scope_kind, input.scope_key);
    const rows: ConnectorFaceRow[] = [];
    for (const l of links) {
        const allRows = facesForPersonStmt.all(
            l.connector_id,
            l.person_id
        ) as ConnectorFaceRow[];
        for (const r of allRows) {
            if (alreadyImported.has(`${r.connector_id}::${r.picture_id}`)) continue;
            rows.push(r);
        }
    }
    if (rows.length === 0) {
        return {
            has_linked_connectors,
            no_data: true,
            groups: [],
            candidates_pool: 0,
            candidates_qualifying: 0
        };
    }

    const candidates = pickRepresentatives(
        rows,
        centroid,
        medianSim,
        input.pose_overrides
    );
    const qualifying = candidates.filter(
        (c) => c.tempered_delta >= SUGGESTION_DELTA_MIN
    );

    // For each under-represented framing group, gather candidates whose
    // framing bucket falls in it. Same for pose.
    const groups: SuggestionGroup[] = [];

    const framingByGroup = new Map<string, SuggestionCandidate[]>();
    const poseByGroup = new Map<string, SuggestionCandidate[]>();
    for (const c of qualifying) {
        if (c.framing_bucket) {
            const gk = framingGroupForBand(c.framing_bucket);
            if (gk) {
                const arr = framingByGroup.get(gk) ?? [];
                arr.push(c);
                framingByGroup.set(gk, arr);
            }
        }
        if (c.pose_bucket) {
            const gk = poseGroupForBucket(c.pose_bucket);
            if (gk) {
                const arr = poseByGroup.get(gk) ?? [];
                arr.push(c);
                poseByGroup.set(gk, arr);
            }
        }
    }

    const sortBest = (xs: SuggestionCandidate[]) =>
        [...xs].sort((a, b) => b.tempered_delta - a.tempered_delta).slice(0, SUGGESTION_TOP_N);

    for (const g of input.framing_gaps) {
        if (g.gap <= 0) continue;
        const pool = framingByGroup.get(g.key) ?? [];
        if (pool.length === 0) continue;
        groups.push({
            dimension: 'framing',
            group_key: g.key,
            label: g.label,
            target: g.target,
            actual: g.actual,
            gap: g.gap,
            candidates: sortBest(pool)
        });
    }
    for (const g of input.pose_gaps) {
        if (g.gap <= 0) continue;
        const pool = poseByGroup.get(g.key) ?? [];
        if (pool.length === 0) continue;
        groups.push({
            dimension: 'pose',
            group_key: g.key,
            label: g.label,
            target: g.target,
            actual: g.actual,
            gap: g.gap,
            candidates: sortBest(pool)
        });
    }
    // Bigger gap first; tie-break with dimension (framing first — usually
    // the more impactful axis to fix).
    groups.sort((a, b) => {
        if (b.gap !== a.gap) return b.gap - a.gap;
        return a.dimension === 'framing' ? -1 : 1;
    });

    return {
        has_linked_connectors,
        no_data: false,
        groups,
        candidates_pool: candidates.length,
        candidates_qualifying: qualifying.length
    };
}
