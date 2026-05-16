// Community-derived recommendations for a balanced real-subject LoRA dataset,
// encoded as machine-checkable targets so the Stats tab can show actionable
// "what to add / what's over-represented" bullets.
//
// Baseline: ~24 images, ~42% close-ups, ~33% medium, ~25% wide; lateral
// pose spread (front / 3-quarter / profile) ~40/30/20%; one in ten with
// vertical tilt (looking up or down). Tunable defaults — see SOURCES below.
import { FRAMING_BUCKETS, type FramingBand, type FramingCoverage } from './framing-grid';
import { POSE_BUCKETS, type PoseBucketId, type PoseCoverage } from './pose-grid';

export const TARGET_TOTAL_MIN = 18;
export const TARGET_TOTAL_IDEAL = 24;

/** Framing groups + their target share of the total dataset. */
export const FRAMING_TARGETS = [
    {
        key: 'close',
        label: 'close-ups',
        ids: ['extreme-close', 'close-up'] as FramingBand[],
        target_pct: 10 / 24, // ≈ 0.42
        ideal_n: 10
    },
    {
        key: 'medium',
        label: 'portraits / busts',
        ids: ['medium'] as FramingBand[],
        target_pct: 8 / 24, // ≈ 0.33
        ideal_n: 8
    },
    {
        key: 'wide',
        label: 'wide shots',
        ids: ['wide', 'extreme-wide'] as FramingBand[],
        target_pct: 6 / 24, // ≈ 0.25
        ideal_n: 6
    }
] as const;

/** Pose groups (across all pitch bands) + their target share of the total. */
export const POSE_TARGETS = [
    {
        key: 'front',
        label: 'front / face',
        target_pct: 0.4,
        match: (id: PoseBucketId) => id.endsWith('-front')
    },
    {
        key: 'tq',
        label: 'three-quarter',
        target_pct: 0.3,
        match: (id: PoseBucketId) => id.includes('-tq-')
    },
    {
        key: 'profile',
        label: 'profile',
        target_pct: 0.2,
        match: (id: PoseBucketId) => id.includes('-profile-')
    },
    {
        key: 'tilted',
        label: 'tilted (up / down)',
        target_pct: 0.1,
        match: (id: PoseBucketId) => id.startsWith('up-') || id.startsWith('down-')
    }
] as const;

export type Severity = 'good' | 'warn' | 'bad' | 'info';

/** Tempered-delta floor for a connector picture to be considered "plausibly
 * the right person". Below this, we drop it from suggestions to keep the
 * noise out — even if Immich clustered it under the same person. */
export const SUGGESTION_DELTA_MIN = -0.05;

/** How many candidates to surface per under-represented group in the UI. */
export const SUGGESTION_TOP_N = 12;

/** Per-group breakdown used by the suggestions panel: which framing/pose
 * groups are under-represented and by how much. */
export interface GroupGap {
    key: string;
    label: string;
    actual: number;
    target: number;
    /** Positive when we need MORE of this group. 0 = at/above target. */
    gap: number;
}

export function framingGroupGaps(coverage: import('./framing-grid').FramingCoverage): GroupGap[] {
    const total = coverage.total - coverage.unknown;
    return FRAMING_TARGETS.map((g) => {
        const actual = g.ids.reduce((s, id) => s + (coverage.counts[id] ?? 0), 0);
        const target = total * g.target_pct;
        return { key: g.key, label: g.label, actual, target, gap: Math.max(0, target - actual) };
    });
}

export function poseGroupGaps(coverage: import('./pose-grid').PoseCoverage): GroupGap[] {
    const total = coverage.total - coverage.unknown;
    return POSE_TARGETS.map((g) => {
        const actual = POSE_BUCKETS.filter((b) => g.match(b.id)).reduce(
            (s, b) => s + (coverage.counts[b.id] ?? 0),
            0
        );
        const target = total * g.target_pct;
        return { key: g.key, label: g.label, actual, target, gap: Math.max(0, target - actual) };
    });
}

/** Quick lookup helpers used by the suggestion pipeline to bucket candidates
 * into the same framing/pose groups we audit against. */
export function framingGroupForBand(
    bandId: import('./framing-grid').FramingBand
): string | null {
    for (const g of FRAMING_TARGETS) if (g.ids.includes(bandId)) return g.key;
    return null;
}
export function poseGroupForBucket(
    bucketId: import('./pose-grid').PoseBucketId
): string | null {
    for (const g of POSE_TARGETS) if (g.match(bucketId)) return g.key;
    return null;
}

export interface Recommendation {
    severity: Severity;
    text: string;
}

interface GroupCount {
    label: string;
    actual: number;
    target: number;
    target_pct: number;
}

function classify(actual: number, target: number, total: number): {
    severity: Severity;
    ratio: number;
} {
    if (target === 0) return { severity: 'info', ratio: 0 };
    const ratio = actual / target;
    // Tighter bands when the dataset is small — a single missing image
    // shouldn't trip "bad" on a 6-image dataset.
    const tol = total < TARGET_TOTAL_MIN ? 0.45 : 0.6;
    if (ratio >= tol && ratio <= 1 / tol) return { severity: 'good', ratio };
    if (ratio < tol * 0.6 || ratio > 1 / (tol * 0.6)) return { severity: 'bad', ratio };
    return { severity: 'warn', ratio };
}

function fmtCount(n: number) {
    return n === Math.floor(n) ? `${n}` : n.toFixed(1);
}

function framingCounts(coverage: FramingCoverage): GroupCount[] {
    const total = coverage.total - coverage.unknown;
    return FRAMING_TARGETS.map((g) => ({
        label: g.label,
        actual: g.ids.reduce((s, id) => s + (coverage.counts[id] ?? 0), 0),
        target: total * g.target_pct,
        target_pct: g.target_pct
    }));
}

function poseCounts(coverage: PoseCoverage): GroupCount[] {
    const total = coverage.total - coverage.unknown;
    return POSE_TARGETS.map((g) => ({
        label: g.label,
        actual: POSE_BUCKETS.filter((b) => g.match(b.id)).reduce(
            (s, b) => s + (coverage.counts[b.id] ?? 0),
            0
        ),
        target: total * g.target_pct,
        target_pct: g.target_pct
    }));
}

export interface DatasetAssessment {
    total: number;
    classified: number;
    /** 0..1 — fraction of recommendations passing as good (excl. info). */
    health: number;
    recommendations: Recommendation[];
}

export function assessDataset(
    framing: FramingCoverage | null,
    pose: PoseCoverage | null
): DatasetAssessment {
    const recs: Recommendation[] = [];
    const total = framing?.total ?? pose?.total ?? 0;
    const classified = framing
        ? framing.total - framing.unknown
        : pose
          ? pose.total - pose.unknown
          : 0;

    if (total === 0) {
        recs.push({
            severity: 'info',
            text: 'No images yet — add some to the folder, then run "Calculate centroid".'
        });
        return { total: 0, classified: 0, health: 0, recommendations: recs };
    }

    if (!framing || !pose || classified === 0) {
        recs.push({
            severity: 'info',
            text: `${total} image${total === 1 ? '' : 's'} on disk · click "Calculate centroid" to score them.`
        });
        return { total, classified: 0, health: 0, recommendations: recs };
    }

    // ---- Dataset size ----
    if (total < TARGET_TOTAL_MIN) {
        recs.push({
            severity: 'bad',
            text: `Dataset is small (${total} images). Aim for at least ${TARGET_TOTAL_MIN}, ideally ${TARGET_TOTAL_IDEAL}, for stable training.`
        });
    } else if (total < TARGET_TOTAL_IDEAL) {
        recs.push({
            severity: 'warn',
            text: `${total}/${TARGET_TOTAL_IDEAL} images — usable, but a few more (especially in under-covered buckets) won't hurt.`
        });
    } else {
        recs.push({
            severity: 'good',
            text: `${total} images — size is comfortable.`
        });
    }

    if (classified < total) {
        recs.push({
            severity: 'warn',
            text: `${total - classified}/${total} images have no detected face — they're excluded from the analysis.`
        });
    }

    // ---- Framing ----
    for (const g of framingCounts(framing)) {
        const c = classify(g.actual, g.target, classified);
        if (c.severity === 'good') {
            recs.push({
                severity: 'good',
                text: `${capitalize(g.label)} coverage is balanced (${fmtCount(g.actual)} ≈ target ${fmtCount(g.target)}).`
            });
        } else if (c.severity === 'bad' && c.ratio < 1) {
            recs.push({
                severity: 'bad',
                text: `Add more ${g.label} — currently ${fmtCount(g.actual)}, ideal ~${fmtCount(g.target)} (${Math.round(g.target_pct * 100)}% of dataset).`
            });
        } else if (c.severity === 'bad' && c.ratio > 1) {
            recs.push({
                severity: 'warn',
                text: `${capitalize(g.label)} are over-represented (${fmtCount(g.actual)}, ideal ~${fmtCount(g.target)}) — diversify with other shot types.`
            });
        } else if (c.severity === 'warn' && c.ratio < 1) {
            recs.push({
                severity: 'warn',
                text: `${capitalize(g.label)}: ${fmtCount(g.actual)}/${fmtCount(g.target)} ideal — could use a few more.`
            });
        } else if (c.severity === 'warn' && c.ratio > 1) {
            recs.push({
                severity: 'warn',
                text: `${capitalize(g.label)}: ${fmtCount(g.actual)} (a bit above the ${Math.round(g.target_pct * 100)}% target).`
            });
        }
    }

    // ---- Pose ----
    for (const g of poseCounts(pose)) {
        const c = classify(g.actual, g.target, classified);
        if (c.severity === 'good') {
            recs.push({
                severity: 'good',
                text: `${capitalize(g.label)} shots are well covered (${fmtCount(g.actual)}).`
            });
        } else if (c.severity === 'bad' && g.actual === 0) {
            recs.push({
                severity: 'bad',
                text: `No ${g.label} shots — add at least ${Math.max(1, Math.round(g.target))} for pose diversity.`
            });
        } else if (c.severity === 'bad' && c.ratio < 1) {
            recs.push({
                severity: 'bad',
                text: `Few ${g.label} shots — ${fmtCount(g.actual)}, ideal ~${fmtCount(g.target)}.`
            });
        } else if (c.severity === 'bad' && c.ratio > 1) {
            recs.push({
                severity: 'warn',
                text: `Lots of ${g.label} shots (${fmtCount(g.actual)}, ideal ~${fmtCount(g.target)}) — verify the rest of the pose spectrum is there.`
            });
        } else if (c.severity === 'warn' && c.ratio < 1) {
            recs.push({
                severity: 'warn',
                text: `${capitalize(g.label)} shots: ${fmtCount(g.actual)} (target ${fmtCount(g.target)}).`
            });
        }
    }

    // ---- Health score ----
    const goods = recs.filter((r) => r.severity === 'good').length;
    const bads = recs.filter((r) => r.severity === 'bad').length;
    const warns = recs.filter((r) => r.severity === 'warn').length;
    const denom = goods + bads + warns;
    const health = denom === 0 ? 1 : (goods + warns * 0.5) / denom;

    return { total, classified, health, recommendations: recs };
}

function capitalize(s: string) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Sources used to define the targets above — kept here so the UI can link
 * to them and the user can revisit / tune. */
export const TARGET_SOURCES: { title: string; url: string }[] = [
    {
        title: 'Civitai — Detailed Flux Training Guide: Dataset Preparation',
        url: 'https://civitai.com/articles/7777/detailed-flux-training-guide-dataset-preparation'
    },
    {
        title: 'RunComfy — Z-Image Character LoRA Dataset Guide',
        url: 'https://www.runcomfy.com/trainer/ai-toolkit/z-image-character-lora-dataset-guide'
    },
    {
        title: 'RunComfy — LoKr vs LoRA Training for FLUX Klein',
        url: 'https://www.runcomfy.com/assets/trainer/ai-toolkit/flux-2-klein-lokr-vs-lora-character-training'
    },
    {
        title: 'Apatero — Qwen-Image-2512 Character LoRA Training Guide 2025',
        url: 'https://apatero.com/blog/qwen-image-2512-character-lora-training-real-people-guide-2025'
    },
    {
        title: 'Apatero — FLUX 2 Pro LoRA Training: Character Consistency Guide 2026',
        url: 'https://apatero.com/blog/flux-2-pro-lora-training-character-consistency-2026'
    },
    {
        title: 'MyAIForce — How to Train a Highly Convincing Real-Life LoRA Model',
        url: 'https://myaiforce.com/real-life-lora-training/'
    },
    {
        title: 'HuggingFace — Perfect LoRA Training parameters human character',
        url: 'https://discuss.huggingface.co/t/perfect-lora-training-parameters-human-character/147211'
    },
    {
        title: 'modl.run — Train a Character LoRA — From 24 Photos to Infinite Scenes',
        url: 'https://modl.run/guides/train-character-lora/'
    },
    {
        title: 'Aimensa — Creating AI Influencers Through LoRA for Z-Image Training',
        url: 'https://aimensa.com/creating-ai-influencers-lora-z-image-training'
    }
];
