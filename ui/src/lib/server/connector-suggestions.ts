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
    POSE_TARGETS,
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
import {
    datasetHashesForFolder,
    datasetHashesForFolders
} from './dataset-images';
import { importedKeysForScope } from './dataset-import';
import { db } from './db';
import {
    DEDUP_HAMMING_THRESHOLD,
    hashHammingPacked,
    parseHashPacked,
    type PackedHash
} from './image-hash';
import { pathBasename } from './path-utils';
import { getSettings } from './settings';

/** Tolerance band: a non-target face whose bbox area falls within this
 * fraction of the target face's area is "comparably prominent" — the
 * picture probably depicts multiple subjects of similar focus, so identity
 * is ambiguous even when ArcFace assigns one face as the closest match. */
const AMBIGUOUS_FACE_RATIO_MIN = 0.8;
const AMBIGUOUS_FACE_RATIO_MAX = 1.2;

interface ConnectorFaceRow {
    connector_id: string;
    picture_id: string;
    filename: string | null;
    image_width: number | null;
    image_height: number | null;
    phash: string | null;
    face_index: number;
    bbox_json: string;
    embedding_b64: string;
    pitch: number | null;
    yaw: number | null;
    roll: number | null;
}

const facesForPersonStmt = db.prepare(`
    SELECT cp.connector_id, cp.picture_id, cp.filename, cp.image_width, cp.image_height, cp.phash,
           cf.face_index, cf.bbox_json, cf.embedding_b64, cf.pitch, cf.yaw, cf.roll
    FROM connector_pictures cp
    JOIN connector_faces cf USING (connector_id, picture_id)
    WHERE cp.connector_id = ? AND cp.person_id = ?
`);

/** Union of every dataset image hash under any of the scope's folders,
 * indexed by phash → an example image_path. For folder scope: just the
 * one folder. For group scope: every member folder. Excluded images are
 * deliberately omitted — the user has rejected them so a near-dup is fair
 * game to re-suggest. */
function loadDatasetHashesForScope(
    scope_kind: LinkScope,
    scope_key: string
): Map<string, string> {
    if (scope_kind === 'folder') {
        return datasetHashesForFolder(scope_key);
    }
    // Inline lookup to avoid pulling a circular dataset-groups dep —
    // we just need the paths_json of the matching group.
    const row = db
        .prepare('SELECT paths_json FROM dataset_groups WHERE id = ?')
        .get(Number(scope_key)) as { paths_json: string } | undefined;
    if (!row) return new Map();
    let folders: string[] = [];
    try {
        const parsed = JSON.parse(row.paths_json);
        if (Array.isArray(parsed)) {
            folders = parsed.filter((p): p is string => typeof p === 'string');
        }
    } catch {
        // ignore — group's paths corrupt, dedup just degrades
    }
    return datasetHashesForFolders(folders);
}

export interface SuggestionCandidate {
    connector_id: string;
    picture_id: string;
    filename: string | null;
    thumbnail_url: string;
    /** Browser-reachable URL for the original (non-resized) asset.
     * Right-click → open opens this; the lightbox click does too. */
    full_url: string;
    /** Source image dimensions in pixels (when reported by the worker /
     * connector). Used in tooltips so the user sees the resolution
     * before importing. */
    image_width: number | null;
    image_height: number | null;
    /** Raw cosine similarity to the centroid (winning face). */
    similarity: number;
    /** sim - median + pose_offset — same number the cell pills use. */
    tempered_delta: number;
    pose_offset_applied: number;
    pose_bucket: PoseBucketId | null;
    framing_bucket: FramingBand | null;
    yaw: number | null;
    pitch: number | null;
    /** Set when the picture has another face whose bbox area is within
     * AMBIGUOUS_FACE_RATIO_MIN..MAX of the winning face's area. The user
     * has to eyeball who's actually the subject before importing. */
    ambiguous_identity: boolean;
    /** Area of the runner-up face / target face — surfaced so the UI
     * tooltip can give the user the actual ratio when ambiguous. */
    runner_up_area_ratio: number | null;
    phash: string | null;
    dedup_match: {
        dataset_path: string;
        dataset_basename: string;
        hamming: number;
    } | null;
    /** Group keys this candidate's pose/framing buckets fall under, intersected
     * with the scope's under-represented set. e.g. ['framing:close',
     * 'pose:profile', 'pose:tilted']. */
    gaps_filled: string[];
    gap_score: number;
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
    has_linked_connectors: boolean;
    no_data: boolean;
    /** Candidates filling ≥ 2 gaps simultaneously, sorted by gap_score
     * desc then tempered_delta desc, capped at MULTI_GAP_TOP_N. Excluded
     * from `groups` to avoid duplication. */
    multi_gap: SuggestionCandidate[];
    /** Per-dimension groups, single-gap candidates only (multi-gap moved
     * to `multi_gap` above). */
    groups: SuggestionGroup[];
    /** Candidates that would have qualified but were flagged as group
     * photos (multiple comparably-prominent faces). Surfaced separately
     * so the user can audit identity before importing. */
    ambiguous: SuggestionCandidate[];
    /** Total number of distinct pictures we considered (deduped). */
    candidates_pool: number;
    /** How many of those passed the identity floor. */
    candidates_qualifying: number;
    /** Resolution floor applied (megapixels). Surfaced for the UI tooltip
     * "candidates rejected for resolution". */
    min_image_mp: number;
    /** Pictures dropped only because they were below `min_image_mp`. */
    rejected_low_res: number;
    /** Pictures dropped because their phash matched an existing dataset
     * image within DEDUP_HAMMING_THRESHOLD bits. */
    rejected_duplicates: number;
    /** Up to N rejected candidates, each annotated with the matched
     * dataset image + hamming distance. Lets the user visually verify
     * the dedup is identifying the right pairs (and spot false
     * positives — e.g. two unrelated photos with similar composition). */
    rejected_duplicate_samples: SuggestionCandidate[];
    /** True when at least one dataset image in scope has been hashed —
     * lets the UI distinguish "0 dups found" from "dedup not active yet,
     * run the backfill job". */
    dataset_hashes_indexed: boolean;
}

/** Max rejected-duplicate samples returned per call. Caps response size —
 * if 500 candidates were filtered, showing all of them in the UI is
 * useless; the first ~30 already give the user enough signal to validate
 * threshold + sanity-check the matches. */
const REJECTED_SAMPLE_CAP = 30;

const MULTI_GAP_TOP_N = 20;
const MULTI_GAP_MIN_SCORE = 2;

/** Connectors backed by a local folder (hard-drive + google-photos cache).
 * `picture_id` for these is the absolute file path, served by the per-
 * connector `/thumb/<base64>` proxy. */
const LOCAL_FOLDER_CONNECTORS = new Set(['hard-drive', 'google-photos']);

function encodeFsPath(picture_id: string): string {
    return Buffer.from(picture_id, 'utf-8')
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}

function thumbnailUrlFor(connector_id: string, picture_id: string): string {
    if (connector_id === 'immich') {
        return `/connectors/immich/thumb/assets/${picture_id}/thumbnail`;
    }
    if (LOCAL_FOLDER_CONNECTORS.has(connector_id)) {
        return `/connectors/${connector_id}/thumb/${encodeFsPath(picture_id)}`;
    }
    return `/connectors/${connector_id}/thumb/asset/${picture_id}`;
}

/** Browser-reachable URL for the FULL-RESOLUTION asset (no resize). Used
 * by the click target so the user can audit a candidate at real quality
 * before importing it. For local-folder-backed connectors the thumb
 * route already serves raw bytes (no Sharp resize), so the thumb URL
 * doubles as the full URL. */
function fullUrlFor(connector_id: string, picture_id: string): string {
    if (connector_id === 'immich') {
        return `/connectors/immich/thumb/assets/${picture_id}/original`;
    }
    if (LOCAL_FOLDER_CONNECTORS.has(connector_id)) {
        return `/connectors/${connector_id}/thumb/${encodeFsPath(picture_id)}`;
    }
    return `/connectors/${connector_id}/thumb/asset/${picture_id}`;
}

/** bbox JSON in connector_faces is `[x1, y1, x2, y2]`. Returns area in
 * pixels² or null when the bbox can't be parsed. */
function parseBboxArea(bbox_json: string): number | null {
    try {
        const a = JSON.parse(bbox_json);
        if (!Array.isArray(a) || a.length < 4) return null;
        const w = Number(a[2]) - Number(a[0]);
        const h = Number(a[3]) - Number(a[1]);
        if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
        return w * h;
    } catch {
        return null;
    }
}

/** Group rows by (connector_id, picture_id), keep the face with the highest
 * dot(emb, centroid) per picture. Tags pictures whose non-target faces are
 * comparably large to the target — those are "group photo" candidates that
 * pollute identity. Returns one SuggestionCandidate per picture. */
function pickRepresentatives(
    rows: ConnectorFaceRow[],
    centroid: Float32Array,
    medianSim: number,
    poseOverrides: PoseOffsets | null
): SuggestionCandidate[] {
    type FaceWithArea = { row: ConnectorFaceRow; sim: number; area: number | null };
    /** All faces of a picture, with their similarities + areas, so we can
     * compute the "target vs runner-up area ratio" once the target is
     * known. */
    const byPic = new Map<string, FaceWithArea[]>();
    for (const r of rows) {
        const emb = decodeEmbedding(r.embedding_b64);
        const sim = dot(emb, centroid);
        const area = parseBboxArea(r.bbox_json);
        const key = `${r.connector_id}::${r.picture_id}`;
        const arr = byPic.get(key);
        if (arr) arr.push({ row: r, sim, area });
        else byPic.set(key, [{ row: r, sim, area }]);
    }

    const out: SuggestionCandidate[] = [];
    for (const faces of byPic.values()) {
        // Target = the face whose embedding is closest to the centroid.
        let target = faces[0];
        for (const f of faces) if (f.sim > target.sim) target = f;
        const r = target.row;

        // Multi-face ambiguity: if another face's area lands in
        // [0.8, 1.2] × target.area, the picture likely contains another
        // subject of comparable prominence — flag it so the UI can put it
        // in a separate panel the user has to confirm before importing.
        let ambiguous = false;
        let runnerUpRatio: number | null = null;
        if (faces.length > 1 && target.area != null && target.area > 0) {
            for (const f of faces) {
                if (f === target) continue;
                if (f.area == null) continue;
                const ratio = f.area / target.area;
                if (
                    runnerUpRatio == null ||
                    Math.abs(ratio - 1) < Math.abs(runnerUpRatio - 1)
                ) {
                    runnerUpRatio = ratio;
                }
                if (
                    ratio >= AMBIGUOUS_FACE_RATIO_MIN &&
                    ratio <= AMBIGUOUS_FACE_RATIO_MAX
                ) {
                    ambiguous = true;
                }
            }
        }

        const offset = poseSimOffset(r.yaw, r.pitch, poseOverrides);
        const tempered = target.sim - medianSim + offset;
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
            full_url: fullUrlFor(r.connector_id, r.picture_id),
            image_width: r.image_width,
            image_height: r.image_height,
            similarity: target.sim,
            tempered_delta: tempered,
            pose_offset_applied: offset,
            pose_bucket: classifyPose(r.yaw, r.pitch),
            framing_bucket: classifyFraming(framingRatio),
            yaw: r.yaw,
            pitch: r.pitch,
            ambiguous_identity: ambiguous,
            runner_up_area_ratio: runnerUpRatio,
            phash: r.phash,
            dedup_match: null,
            gaps_filled: [],
            gap_score: 0
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
    // Resolution floor — drops low-res pictures before scoring. Stored as
    // a string setting; "0" disables. 1 MP default ≈ 1000×1000, the
    // comfortable LoRA / portrait training floor.
    const rawMinMp = Number(getSettings().suggestion_min_image_mp);
    const min_image_mp =
        Number.isFinite(rawMinMp) && rawMinMp > 0 ? rawMinMp : 0;
    const minPixels = min_image_mp * 1_000_000;

    const links = listLinksForScope(input.scope_kind, input.scope_key);
    const has_linked_connectors = links.length > 0;

    if (!has_linked_connectors || !input.centroid) {
        return {
            has_linked_connectors,
            no_data: true,
            multi_gap: [],
            groups: [],
            ambiguous: [],
            candidates_pool: 0,
            candidates_qualifying: 0,
            min_image_mp,
            rejected_low_res: 0,
            rejected_duplicates: 0,
            rejected_duplicate_samples: [],
            dataset_hashes_indexed: false
        };
    }

    const centroid = decodeEmbedding(input.centroid.centroid_b64);
    const medianSim = input.centroid.median_sim ?? 0;

    // Collect all face rows from every linked (connector, person),
    // skipping rows for pictures already imported into this scope — the
    // dedup pre-requisite for "Add to dataset".
    const alreadyImported = importedKeysForScope(input.scope_kind, input.scope_key);
    const rows: ConnectorFaceRow[] = [];
    /** Pictures filtered out by the resolution floor. Reported in the
     * result so the UI can show "X pictures hidden, below 1 MP". */
    let rejected_low_res = 0;
    const seenRejected = new Set<string>();
    for (const l of links) {
        const allRows = facesForPersonStmt.all(
            l.connector_id,
            l.person_id
        ) as ConnectorFaceRow[];
        for (const r of allRows) {
            if (alreadyImported.has(`${r.connector_id}::${r.picture_id}`)) continue;
            if (minPixels > 0 && r.image_width != null && r.image_height != null) {
                if (r.image_width * r.image_height < minPixels) {
                    const key = `${r.connector_id}::${r.picture_id}`;
                    if (!seenRejected.has(key)) {
                        seenRejected.add(key);
                        rejected_low_res++;
                    }
                    continue;
                }
            }
            rows.push(r);
        }
    }
    if (rows.length === 0) {
        return {
            has_linked_connectors,
            no_data: true,
            multi_gap: [],
            groups: [],
            ambiguous: [],
            candidates_pool: 0,
            candidates_qualifying: 0,
            min_image_mp,
            rejected_low_res,
            rejected_duplicates: 0,
            rejected_duplicate_samples: [],
            dataset_hashes_indexed: false
        };
    }

    const candidates = pickRepresentatives(
        rows,
        centroid,
        medianSim,
        input.pose_overrides
    );

    // pHash dedup: drop candidates whose perceptual hash matches any
    // existing dataset image within DEDUP_HAMMING_THRESHOLD bits. Both
    // sides must be hashed to participate — unhashed candidates pass
    // through silently (no false negatives), and an empty dataset hash
    // set degrades the filter to a no-op (caller can prompt for a backfill
    // job via dataset_hashes_indexed=false).
    //
    // For each rejected candidate we also record WHICH dataset image it
    // matched and at what Hamming distance, so the UI's "Duplicates
    // filtered" panel can show side-by-side validation evidence (helps
    // the user spot false positives or mistuned thresholds).
    const datasetHashes = loadDatasetHashesForScope(
        input.scope_kind,
        input.scope_key
    );
    const dataset_hashes_indexed = datasetHashes.size > 0;

    // Parse every phash once into Uint32 words. Both dedup passes below
    // run hashHammingPacked over these — the BigInt-parse string path
    // used to dominate the load time on multi-thousand-candidate scopes.
    const datasetPacked: { words: PackedHash; dpath: string }[] = [];
    for (const [dh, dpath] of datasetHashes) {
        const w = parseHashPacked(dh);
        if (w) datasetPacked.push({ words: w, dpath });
    }
    const candidatePacked = new Map<SuggestionCandidate, PackedHash>();
    for (const c of candidates) {
        if (c.phash) {
            const w = parseHashPacked(c.phash);
            if (w) candidatePacked.set(c, w);
        }
    }

    let rejected_duplicates = 0;
    const rejected_duplicate_samples: SuggestionCandidate[] = [];
    const dedupedCandidates: SuggestionCandidate[] = [];
    if (dataset_hashes_indexed) {
        for (const c of candidates) {
            let match: { dataset_path: string; hamming: number } | null = null;
            const cPacked = candidatePacked.get(c);
            if (cPacked) {
                let bestH = Infinity;
                let bestPath: string | null = null;
                for (const d of datasetPacked) {
                    const h = hashHammingPacked(cPacked, d.words);
                    if (h < bestH) {
                        bestH = h;
                        bestPath = d.dpath;
                        if (bestH === 0) break; // can't beat exact match
                    }
                }
                if (bestPath != null && bestH <= DEDUP_HAMMING_THRESHOLD) {
                    match = { dataset_path: bestPath, hamming: bestH };
                }
            }
            if (match) {
                rejected_duplicates++;
                if (rejected_duplicate_samples.length < REJECTED_SAMPLE_CAP) {
                    rejected_duplicate_samples.push({
                        ...c,
                        dedup_match: {
                            dataset_path: match.dataset_path,
                            dataset_basename: pathBasename(match.dataset_path),
                            hamming: match.hamming
                        }
                    });
                }
            } else {
                dedupedCandidates.push(c);
            }
        }
    } else {
        dedupedCandidates.push(...candidates);
    }

    // Second pass: dedup ACROSS surviving candidates. Two connector
    // pictures of the same scene (burst shots, edits in different albums)
    // would otherwise both surface as "fill the same gap" suggestions —
    // visually redundant for the user. Sort by pixel area desc (the LoRA
    // trainer benefits more from the higher-res take of the same scene)
    // with tempered_delta as the tiebreaker (when resolutions match, fall
    // back to the best identity score above median). Unhashed candidates
    // pass through silently (no false negatives). Drops bump the same
    // rejected_duplicates counter — samples aren't recorded because the
    // "matched" image is another suggestion the user is already seeing,
    // no extra evidence to surface.
    const interDeduped: SuggestionCandidate[] = [];
    const interDedupedPacked: PackedHash[] = [];
    const pixelArea = (c: SuggestionCandidate): number =>
        (c.image_width ?? 0) * (c.image_height ?? 0);
    const sortedForInterDedup = [...dedupedCandidates].sort(
        (a, b) =>
            pixelArea(b) - pixelArea(a) ||
            b.tempered_delta - a.tempered_delta
    );
    for (const c of sortedForInterDedup) {
        const cPacked = candidatePacked.get(c);
        if (!cPacked) {
            interDeduped.push(c);
            continue;
        }
        let dup = false;
        for (const keptWords of interDedupedPacked) {
            if (hashHammingPacked(cPacked, keptWords) <= DEDUP_HAMMING_THRESHOLD) {
                dup = true;
                break;
            }
        }
        if (dup) {
            rejected_duplicates++;
        } else {
            interDeduped.push(c);
            interDedupedPacked.push(cPacked);
        }
    }

    const qualifying = interDeduped.filter(
        (c) => c.tempered_delta >= SUGGESTION_DELTA_MIN
    );
    const cleanQualifying = qualifying.filter((c) => !c.ambiguous_identity);
    const ambiguous = qualifying.filter((c) => c.ambiguous_identity);

    const gappedFraming = new Set(input.framing_gaps.filter((g) => g.gap > 0).map((g) => g.key));
    const gappedPose = new Set(input.pose_gaps.filter((g) => g.gap > 0).map((g) => g.key));
    for (const c of qualifying) {
        const filled: string[] = [];
        if (c.framing_bucket) {
            const k = framingGroupForBand(c.framing_bucket);
            if (k && gappedFraming.has(k)) filled.push(`framing:${k}`);
        }
        if (c.pose_bucket) {
            for (const pt of POSE_TARGETS) {
                if (pt.match(c.pose_bucket) && gappedPose.has(pt.key)) {
                    filled.push(`pose:${pt.key}`);
                }
            }
        }
        c.gaps_filled = filled;
        c.gap_score = filled.length;
    }

    // Multi-gap candidates promoted out of per-dim groups (the per-dim
    // sections only show single-gap to avoid duplication).
    const multi_gap = cleanQualifying
        .filter((c) => c.gap_score >= MULTI_GAP_MIN_SCORE)
        .sort((a, b) => b.gap_score - a.gap_score || b.tempered_delta - a.tempered_delta)
        .slice(0, MULTI_GAP_TOP_N);
    const multiGapKeys = new Set(multi_gap.map((c) => `${c.connector_id}::${c.picture_id}`));
    const singleGapClean = cleanQualifying.filter(
        (c) => !multiGapKeys.has(`${c.connector_id}::${c.picture_id}`)
    );

    const groups: SuggestionGroup[] = [];
    const framingByGroup = new Map<string, SuggestionCandidate[]>();
    const poseByGroup = new Map<string, SuggestionCandidate[]>();
    for (const c of singleGapClean) {
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
        multi_gap,
        groups,
        ambiguous: sortBest(ambiguous),
        candidates_pool: candidates.length,
        candidates_qualifying: qualifying.length,
        min_image_mp,
        rejected_low_res,
        rejected_duplicates,
        // Sort samples best→worst match (lowest Hamming first) so the
        // user inspects the most-confident matches first.
        rejected_duplicate_samples: rejected_duplicate_samples.sort(
            (a, b) => (a.dedup_match!.hamming - b.dedup_match!.hamming)
        ),
        dataset_hashes_indexed
    };
}
