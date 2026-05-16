// Threshold constants shared by the server-side math and the client-side
// rendering. Kept outside `$lib/server` so the UI can import them safely.
//
// `delta = sim - median` per scope: 0 = exactly the dataset's median,
// positive = closer to the cluster center, negative = drifting away.

/** Delta cutoff below which a cell turns AMBER. */
export const DELTA_AMBER = -0.03;

/** Delta cutoff below which a cell turns RED — candidate outlier. */
export const DELTA_RED = -0.08;

// ---- Pose-induced similarity drop (calibration offsets) ----
//
// ArcFace cosine similarity drops with pose even for the SAME identity:
// the embedding has residual pose dependence that averaging into a centroid
// doesn't fully cancel. Without correcting for this, a clean profile shot
// looks like an outlier and a slightly-off front shot looks fine.
//
// We subtract an *expected* drop from `raw_delta` so the displayed
// "tempered" delta reflects "is this image worse than other shots taken
// from a similar angle?" instead of "is this image worse than the average?"
//
// Defaults are calibrated from typical ArcFace-on-buffalo_l behavior on
// curated identity datasets — tune from one place when you have your own
// numbers.
import { PITCH_TILT_DEG, YAW_PROFILE_DEG, YAW_QUARTER_DEG, YAW_SIGN_FLIP } from './pose-grid';

/** Expected sim drop for a 3-quarter (30°–60° yaw) shot. */
export const SIM_OFFSET_THREEQUARTER = 0.025;
/** Expected sim drop for a profile (>60° yaw) shot. */
export const SIM_OFFSET_PROFILE = 0.07;
/** Expected sim drop for a vertically-tilted (|pitch| >= 30°) shot. */
export const SIM_OFFSET_TILTED = 0.025;

/** Source for the per-group offsets passed to `poseSimOffset`. The keys
 * mirror `PoseCalibration` from the server: each can be `null` to fall
 * back on the hardcoded default. */
export interface PoseOffsets {
    threequarter: number | null;
    profile: number | null;
    tilted: number | null;
}

/** How much "below the median" we expect this image to legitimately fall,
 * purely because of its head pose. Returns a non-negative number; 0 = no
 * expected drop (front, eye-level, or pose unknown). Add this back into the
 * raw delta to get a pose-tempered delta.
 *
 * `overrides` is the data-derived calibration — when a band has enough
 * samples to derive its own offset, we use that; otherwise we fall back to
 * the hardcoded `SIM_OFFSET_*` constants above. */
export function poseSimOffset(
    yaw: number | null | undefined,
    pitch: number | null | undefined,
    overrides: PoseOffsets | null = null
): number {
    let off = 0;
    if (yaw != null && Number.isFinite(yaw)) {
        const a = Math.abs(YAW_SIGN_FLIP ? -yaw : yaw);
        if (a >= YAW_PROFILE_DEG) {
            off += overrides?.profile ?? SIM_OFFSET_PROFILE;
        } else if (a >= YAW_QUARTER_DEG) {
            off += overrides?.threequarter ?? SIM_OFFSET_THREEQUARTER;
        }
    }
    if (pitch != null && Number.isFinite(pitch) && Math.abs(pitch) >= PITCH_TILT_DEG) {
        off += overrides?.tilted ?? SIM_OFFSET_TILTED;
    }
    return off;
}
