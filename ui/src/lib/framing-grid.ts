// "Framing distance" proxy = how big the face is relative to the frame.
// Cinematography convention adapted for dataset audit: 5 bins from extreme
// close-up to extreme wide. Uses bbox HEIGHT instead of area because height
// is mostly yaw-invariant (a profile shot still spans top-of-head to chin),
// with an optional roll correction folded in via the bbox diagonal model.
//
//   head_size = h*|cos(roll)| + w*|sin(roll)|   (in pixels)
//   ratio     = head_size / image_height
//
// Pure module — no DB, no UI side-effects. Imported from both the server
// aggregator and the client display.

export type FramingBand =
    | 'extreme-close'
    | 'close-up'
    | 'medium'
    | 'wide'
    | 'extreme-wide';

export interface FramingBucket {
    id: FramingBand;
    label: string;
    description: string;
    /** Inclusive lower bound on `ratio`. The bucket runs from this value up
     * to the next bucket's lower bound. */
    min_ratio: number;
}

/** In stable order from "biggest face in frame" to "smallest". The UI can
 * iterate this list directly to render the columns of a coverage table. */
export const FRAMING_BUCKETS: readonly FramingBucket[] = [
    {
        id: 'extreme-close',
        label: 'extreme close',
        description: 'Face fills most of the frame (head only)',
        min_ratio: 0.55
    },
    {
        id: 'close-up',
        label: 'close-up',
        description: 'Head & shoulders portrait',
        min_ratio: 0.3
    },
    {
        id: 'medium',
        label: 'medium',
        description: 'Chest / waist up',
        min_ratio: 0.12
    },
    { id: 'wide', label: 'wide', description: 'Full body or wider', min_ratio: 0.04 },
    {
        id: 'extreme-wide',
        label: 'extreme wide',
        description: 'Subject is small in the frame (group / scene)',
        min_ratio: 0
    }
];

const DEG_TO_RAD = Math.PI / 180;

/** Compute the framing ratio with optional roll correction. `bbox` is the
 * `[x1, y1, x2, y2]` face bbox in pixel coordinates. Returns `null` when
 * we don't have enough info to classify (missing image_height etc.). */
export function computeFramingRatio(
    bbox: number[] | null | undefined,
    image_height: number | null | undefined,
    roll_deg: number | null | undefined = null
): number | null {
    if (!bbox || bbox.length < 4 || !image_height || image_height <= 0) return null;
    const [x1, y1, x2, y2] = bbox;
    const h = y2 - y1;
    const w = x2 - x1;
    if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0) return null;

    // Roll correction: a head rotated 90° turns its anatomical height into
    // the bbox width and vice versa. h*cos + w*sin is the projection of the
    // true vertical onto the image's vertical axis.
    let head_size = h;
    if (roll_deg != null && Number.isFinite(roll_deg)) {
        const r = roll_deg * DEG_TO_RAD;
        head_size = h * Math.abs(Math.cos(r)) + w * Math.abs(Math.sin(r));
    }
    return head_size / image_height;
}

export function classifyFraming(ratio: number | null): FramingBand | null {
    if (ratio == null || !Number.isFinite(ratio) || ratio < 0) return null;
    for (const b of FRAMING_BUCKETS) {
        if (ratio >= b.min_ratio) return b.id;
    }
    return 'extreme-wide';
}

export interface FramingCoverage {
    counts: Record<FramingBand, number>;
    /** Images we couldn't classify (no face / missing image dims). */
    unknown: number;
    total: number;
}

export interface FramingSample {
    bbox: number[] | null;
    image_height: number | null;
    roll: number | null;
}

export function countByFraming(samples: FramingSample[]): FramingCoverage {
    const counts = {} as Record<FramingBand, number>;
    for (const b of FRAMING_BUCKETS) counts[b.id] = 0;
    let unknown = 0;
    for (const s of samples) {
        const r = computeFramingRatio(s.bbox, s.image_height, s.roll);
        const id = classifyFraming(r);
        if (id == null) unknown++;
        else counts[id]++;
    }
    return { counts, unknown, total: samples.length };
}
