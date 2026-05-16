// Canonical 3 × 5 pose grid — 15 head-pose "buckets" we want a balanced
// dataset to cover. Used for coverage audit / outlier detection / display
// later on. Pure constants + classifier, no DB, no UI side-effects.
//
// Geometry (degrees, InsightFace convention: pitch up = positive, yaw right
// = positive — to verify on first real samples; flip YAW_SIGN_FLIP if your
// data says otherwise):
//
//   pitch  →  up (≥30°)         eye level         down (≤-30°)
//   yaw    →  profile-L  3q-L  front  3q-R  profile-R
//               (≤-60)  (-60..-30) (-30..30) (30..60)  (≥60)
//
// Thresholds match the user's spec: ±30° boundary between front/3-quarter,
// ±60° between 3-quarter/profile, ±30° between level/up-down.

/** Yaw band thresholds, in degrees. Inclusive on the "outer" side. */
export const YAW_PROFILE_DEG = 60;
export const YAW_QUARTER_DEG = 30;

/** Pitch band thresholds, in degrees. */
export const PITCH_TILT_DEG = 30;

/** If your detector reports the opposite sign for yaw (subject's-right vs
 * camera's-right), flip this. InsightFace's `landmark_3d_68` head returns
 * yaw positive when the face is rotated towards the viewer's right —
 * leaving this `false` matches that. */
export const YAW_SIGN_FLIP = false;

export type YawBand = 'profile-l' | 'tq-l' | 'front' | 'tq-r' | 'profile-r';
export type PitchBand = 'up' | 'level' | 'down';
export type PoseBucketId = `${PitchBand}-${YawBand}`;

export interface PoseBucket {
    id: PoseBucketId;
    pitch_band: PitchBand;
    yaw_band: YawBand;
    /** Short label for badges / column headers. */
    label: string;
    /** Longer human-readable description for tooltips. */
    description: string;
}

const YAW_BANDS: { band: YawBand; label: string; desc: string }[] = [
    { band: 'profile-l', label: 'profile L', desc: 'profile, looking far left' },
    { band: 'tq-l', label: '¾ L', desc: 'three-quarter, looking left' },
    { band: 'front', label: 'front', desc: 'facing camera' },
    { band: 'tq-r', label: '¾ R', desc: 'three-quarter, looking right' },
    { band: 'profile-r', label: 'profile R', desc: 'profile, looking far right' }
];

const PITCH_BANDS: { band: PitchBand; label: string; desc: string }[] = [
    { band: 'up', label: 'up', desc: 'tilted up (>+30°)' },
    { band: 'level', label: 'level', desc: 'eye-level' },
    { band: 'down', label: 'down', desc: 'tilted down (<-30°)' }
];

/** All 15 buckets in stable row-major order (pitch outer, yaw inner). The
 * UI can lay these out as a 3-row × 5-col table — the iteration order is
 * already correct. */
export const POSE_BUCKETS: readonly PoseBucket[] = PITCH_BANDS.flatMap((pb) =>
    YAW_BANDS.map(
        (yb): PoseBucket => ({
            id: `${pb.band}-${yb.band}` as PoseBucketId,
            pitch_band: pb.band,
            yaw_band: yb.band,
            label: pb.band === 'level' ? yb.label : `${pb.label} · ${yb.label}`,
            description: `${pb.desc}, ${yb.desc}`
        })
    )
);

function yawBand(yaw: number): YawBand {
    const y = YAW_SIGN_FLIP ? -yaw : yaw;
    if (y <= -YAW_PROFILE_DEG) return 'profile-l';
    if (y <= -YAW_QUARTER_DEG) return 'tq-l';
    if (y < YAW_QUARTER_DEG) return 'front';
    if (y < YAW_PROFILE_DEG) return 'tq-r';
    return 'profile-r';
}

function pitchBand(pitch: number): PitchBand {
    if (pitch >= PITCH_TILT_DEG) return 'up';
    if (pitch <= -PITCH_TILT_DEG) return 'down';
    return 'level';
}

/** Classify a (yaw, pitch) sample into one of the 15 buckets. Returns
 * `null` when either value is missing — the caller should treat that as
 * "pose unknown" (e.g. detector didn't expose head pose for this face). */
export function classifyPose(
    yaw: number | null | undefined,
    pitch: number | null | undefined
): PoseBucketId | null {
    if (yaw == null || pitch == null || !Number.isFinite(yaw) || !Number.isFinite(pitch))
        return null;
    return `${pitchBand(pitch)}-${yawBand(yaw)}` as PoseBucketId;
}

/** Per-bucket count, plus the number of samples we couldn't classify
 * (pose unknown). Total over all buckets + `unknown` == input length. */
export interface PoseCoverage {
    counts: Record<PoseBucketId, number>;
    unknown: number;
    total: number;
}

/** Aggregate a list of {yaw, pitch} samples into per-bucket counts. Useful
 * for audit ("which buckets are under-represented in this dataset"). */
export function countByBucket(
    samples: { yaw: number | null; pitch: number | null }[]
): PoseCoverage {
    const counts = {} as Record<PoseBucketId, number>;
    for (const b of POSE_BUCKETS) counts[b.id] = 0;
    let unknown = 0;
    for (const s of samples) {
        const id = classifyPose(s.yaw, s.pitch);
        if (id == null) unknown++;
        else counts[id]++;
    }
    return { counts, unknown, total: samples.length };
}
