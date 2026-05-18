// Image + caption pair listing for a single dataset folder. Captions follow
// the kohya / SDXL convention: a sidecar `<basename>.txt` (or `.caption`)
// next to each image. Missing caption → empty string.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

import { countByFraming, type FramingCoverage, type FramingSample } from '../framing-grid';
import { countByBucket, type PoseCoverage } from '../pose-grid';
import { decodeEmbedding, dot } from './centroid-math';
import { db } from './db';
import type { DatasetImageStatus } from './dataset-images';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

export interface DatasetItem {
    /** File basename (with extension). */
    filename: string;
    /** Caption from the sidecar `.txt` / `.caption`, or `''`. */
    caption: string;
    /** Active vs excluded. Drives the grid badge — excluded items are still
     * shown (so the user can restore them) but greyed out. `null` when the
     * image has no dataset_images row yet (not analyzed). */
    status: DatasetImageStatus | null;
    /** Cos-sim of the winning face vs the folder's centroid, or `null` when
     * no centroid has been computed yet OR no face was detected. */
    similarity: number | null;
    /** Cos-sim of the winning face vs the GROUP centroid (when the dataset
     * is being rendered as part of a group view). Computed on the fly from
     * the cached winner embedding. `null` when no group centroid is known. */
    similarity_group: number | null;
    /** Number of faces detected for this image during the last centroid run.
     * `null` when this image isn't in face_embeddings (no run yet). */
    face_count: number | null;
    /** Head pose of the winning face (degrees, signed). NULL when the
     * detector didn't expose pose or no face was picked. Used client-side
     * to temper the per-cell delta badge for off-axis shots. */
    yaw: number | null;
    pitch: number | null;
}

export interface DatasetDetail {
    /** Resolved absolute path on disk. */
    path: string;
    /** Display label (basename of `path`). */
    name: string;
    items: DatasetItem[];
}

function lowerExt(fname: string): string {
    const i = fname.lastIndexOf('.');
    return i < 0 ? '' : fname.slice(i).toLowerCase();
}

function readCaption(folder: string, imageName: string): string {
    const stem = imageName.replace(/\.[^.]+$/, '');
    for (const ext of ['.txt', '.caption']) {
        const candidate = join(folder, stem + ext);
        if (existsSync(candidate)) {
            try {
                return readFileSync(candidate, 'utf-8').trim();
            } catch {
                return '';
            }
        }
    }
    return '';
}

// Per-image face stats loaded in one shot per folder. Driven from the
// canonical dataset_images table so we get status + dim metadata in the
// same row, then LEFT JOIN face_embeddings for the per-face aggregates
// (winning face = is_target=1). All face-side fields are NULL when no
// detection has run yet OR no face was detected — listDatasetItems still
// surfaces the image with status='active' so the UI grid shows it.
const imageStatsStmt = db.prepare(`
    SELECT di.image_path,
           di.status,
           COUNT(fe.id) AS face_count,
           MAX(CASE WHEN fe.is_target = 1 THEN fe.similarity END)    AS target_similarity,
           MAX(CASE WHEN fe.is_target = 1 THEN fe.embedding_b64 END) AS target_embedding_b64,
           MAX(CASE WHEN fe.is_target = 1 THEN fe.yaw END)           AS target_yaw,
           MAX(CASE WHEN fe.is_target = 1 THEN fe.pitch END)         AS target_pitch
      FROM dataset_images di
      LEFT JOIN face_embeddings fe ON fe.image_path = di.image_path
     WHERE di.folder_path = ?
     GROUP BY di.image_path
`);

interface ImageStatsRow {
    image_path: string;
    status: DatasetImageStatus;
    face_count: number;
    target_similarity: number | null;
    target_embedding_b64: string | null;
    target_yaw: number | null;
    target_pitch: number | null;
}

function loadImageStats(folderPath: string): Map<string, ImageStatsRow> {
    const rows = imageStatsStmt.all(folderPath) as ImageStatsRow[];
    const map = new Map<string, ImageStatsRow>();
    for (const r of rows) map.set(r.image_path, r);
    return map;
}

/**
 * @param folderPath dataset folder being listed
 * @param groupCentroidB64 optional — when set, every item gets a second
 *        similarity computed vs this centroid (used for "vs group" toggles).
 */
export function listDatasetItems(
    folderPath: string,
    groupCentroidB64: string | null = null
): DatasetItem[] {
    let entries: string[];
    try {
        entries = readdirSync(folderPath);
    } catch {
        return [];
    }
    const stats = loadImageStats(folderPath);
    // Decode the group centroid once — every item dot-products against it.
    const groupCentroid = groupCentroidB64 ? decodeEmbedding(groupCentroidB64) : null;
    const items: DatasetItem[] = [];
    for (const f of entries) {
        if (!IMAGE_EXTS.has(lowerExt(f))) continue;
        const abs = join(folderPath, f);
        try {
            if (!statSync(abs).isFile()) continue;
        } catch {
            continue;
        }
        const s = stats.get(abs);
        let simGroup: number | null = null;
        if (groupCentroid && s?.target_embedding_b64) {
            simGroup = dot(decodeEmbedding(s.target_embedding_b64), groupCentroid);
        }
        items.push({
            filename: f,
            caption: readCaption(folderPath, f),
            status: s?.status ?? null,
            similarity: s?.target_similarity ?? null,
            similarity_group: simGroup,
            face_count: s?.face_count ?? null,
            yaw: s?.target_yaw ?? null,
            pitch: s?.target_pitch ?? null
        });
    }
    items.sort((a, b) => a.filename.localeCompare(b.filename));
    return items;
}

export function readDatasetDetail(
    folderPath: string,
    groupCentroidB64: string | null = null
): DatasetDetail {
    const abs = resolve(folderPath);
    return {
        path: abs,
        name: basename(abs),
        items: listDatasetItems(abs, groupCentroidB64)
    };
}

// Pull yaw/pitch for every winning face in this folder and bucket them into
// the 15-cell pose grid (see lib/pose-grid.ts). Useful for dataset coverage
// audit. NOT lazily evaluated — server-cheap: <= one row per image.
// Excluded images are filtered out via the JOIN — their pose shouldn't
// count toward coverage targets.
const poseStmt = db.prepare(`
    SELECT fe.yaw, fe.pitch
      FROM face_embeddings fe
      JOIN dataset_images di ON di.image_path = fe.image_path
     WHERE di.folder_path = ?
       AND di.status = 'active'
       AND fe.is_target = 1
`);

export function loadPoseCoverage(folderPath: string): PoseCoverage {
    const rows = poseStmt.all(folderPath) as { yaw: number | null; pitch: number | null }[];
    return countByBucket(rows);
}

// ---- Data-driven pose calibration --------------------------------------
//
// The hardcoded SIM_OFFSET_* constants in centroid-thresholds.ts are
// fallbacks. Whenever the dataset has enough samples in a pose band, we
// derive the offset directly: `median(front baseline) - median(group)`.
// Datasets vary a lot — a controlled studio set will have profiles drop
// 0.05 vs the centroid, a candid mobile-camera set can drop 0.15-0.20.
import {
    PITCH_TILT_DEG,
    YAW_PROFILE_DEG,
    YAW_QUARTER_DEG,
    YAW_SIGN_FLIP
} from '../pose-grid';

const calibStmt = db.prepare(`
    SELECT fe.yaw, fe.pitch, fe.similarity
      FROM face_embeddings fe
      JOIN dataset_images di ON di.image_path = fe.image_path
     WHERE di.folder_path = ?
       AND di.status = 'active'
       AND fe.is_target = 1
       AND fe.similarity IS NOT NULL
`);

interface CalibRow {
    yaw: number | null;
    pitch: number | null;
    similarity: number;
}

export interface PoseCalibration {
    /** Sample count + median sim per group, plus the derived offsets. When
     * a group has <`MIN_N` samples, its offset is `null` — the client falls
     * back to the constant for that band. */
    front: { n: number; median: number | null };
    threequarter: { n: number; median: number | null };
    profile: { n: number; median: number | null };
    tilted: { n: number; median: number | null };
    /** Offsets in "subtract this much because of pose" semantics. Same units
     * as the raw cosine similarity. Null when the corresponding group has
     * insufficient samples — the client falls back to defaults. */
    offset_threequarter: number | null;
    offset_profile: number | null;
    offset_tilted: number | null;
    /** Which pool was used as the front baseline. Surfaced in the UI tooltip
     * so the user knows whether the offsets are data-derived or defaults. */
    baseline_kind: 'front-level' | 'front-any' | 'all' | 'none';
}

const MIN_CALIB_N = 3;

function median(xs: number[]): number | null {
    if (xs.length === 0) return null;
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function yawAbs(yaw: number): number {
    return Math.abs(YAW_SIGN_FLIP ? -yaw : yaw);
}

/** Aggregate calibration over one or more folders. Pass a single folder for
 * the dataset-level view, pass every member path for the group-level view. */
export function loadPoseCalibration(folderPaths: string[]): PoseCalibration {
    const rows: CalibRow[] = [];
    for (const p of folderPaths) {
        rows.push(...(calibStmt.all(p) as CalibRow[]));
    }

    // Buckets we care about for calibration.
    const frontLevel: number[] = [];
    const frontAny: number[] = [];
    const threequarter: number[] = [];
    const profile: number[] = [];
    const tilted: number[] = [];
    const all: number[] = [];

    for (const r of rows) {
        if (r.similarity == null || !Number.isFinite(r.similarity)) continue;
        all.push(r.similarity);
        const hasYaw = r.yaw != null && Number.isFinite(r.yaw);
        const hasPitch = r.pitch != null && Number.isFinite(r.pitch);
        if (!hasYaw || !hasPitch) continue;
        const ay = yawAbs(r.yaw!);
        const ap = Math.abs(r.pitch!);

        const isFront = ay < YAW_QUARTER_DEG;
        const isTq = ay >= YAW_QUARTER_DEG && ay < YAW_PROFILE_DEG;
        const isProfile = ay >= YAW_PROFILE_DEG;
        const isTilted = ap >= PITCH_TILT_DEG;

        if (isFront) {
            frontAny.push(r.similarity);
            if (!isTilted) frontLevel.push(r.similarity);
        }
        if (isTq && !isTilted) threequarter.push(r.similarity);
        if (isProfile && !isTilted) profile.push(r.similarity);
        if (isFront && isTilted) tilted.push(r.similarity);
    }

    // Pick the cleanest baseline available. The cascade keeps the offsets
    // meaningful on lopsided datasets (e.g. one with mostly profiles).
    let baseline: number | null = null;
    let baseline_kind: PoseCalibration['baseline_kind'] = 'none';
    if (frontLevel.length >= MIN_CALIB_N) {
        baseline = median(frontLevel);
        baseline_kind = 'front-level';
    } else if (frontAny.length >= MIN_CALIB_N) {
        baseline = median(frontAny);
        baseline_kind = 'front-any';
    } else if (all.length >= MIN_CALIB_N) {
        baseline = median(all);
        baseline_kind = 'all';
    }

    const deriveOffset = (samples: number[]): number | null => {
        if (baseline == null || samples.length < MIN_CALIB_N) return null;
        const m = median(samples);
        if (m == null) return null;
        // Floor at 0: if a group genuinely scores HIGHER than baseline, we
        // don't want to subtract a negative (would over-credit those cells).
        return Math.max(0, baseline - m);
    };

    return {
        front: { n: frontAny.length, median: median(frontAny) },
        threequarter: { n: threequarter.length, median: median(threequarter) },
        profile: { n: profile.length, median: median(profile) },
        tilted: { n: tilted.length, median: median(tilted) },
        offset_threequarter: deriveOffset(threequarter),
        offset_profile: deriveOffset(profile),
        offset_tilted: deriveOffset(tilted),
        baseline_kind
    };
}

// Framing distance proxy: bbox HEIGHT / image_height, optionally roll-
// corrected. Thresholds + classifier live in framing-grid.ts; we just pull
// the raw inputs here. image_height lives on dataset_images now (image-level
// property); join in to fetch it alongside the per-face bbox + roll.
const framingStmt = db.prepare(`
    SELECT fe.bbox_json, di.image_height, fe.roll
      FROM face_embeddings fe
      JOIN dataset_images di ON di.image_path = fe.image_path
     WHERE di.folder_path = ?
       AND di.status = 'active'
       AND fe.is_target = 1
`);

export function loadFramingCoverage(folderPath: string): FramingCoverage {
    const rows = framingStmt.all(folderPath) as {
        bbox_json: string | null;
        image_height: number | null;
        roll: number | null;
    }[];
    const samples: FramingSample[] = rows.map((r) => {
        let bbox: number[] | null = null;
        if (r.bbox_json) {
            try {
                const parsed = JSON.parse(r.bbox_json);
                if (Array.isArray(parsed)) bbox = parsed.map(Number);
            } catch {
                // ignore — bbox stays null, sample becomes "unknown"
            }
        }
        return { bbox, image_height: r.image_height, roll: r.roll };
    });
    return countByFraming(samples);
}
