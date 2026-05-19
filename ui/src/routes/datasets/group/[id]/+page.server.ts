import type { Actions, PageServerLoad } from './$types';
import { error, fail } from '@sveltejs/kit';
import { basename } from 'node:path';

import { framingGroupGaps, poseGroupGaps } from '$lib/dataset-targets';
import {
    computeAndPersist,
    getCentroid,
    recomputeFolderCentroidFromDb,
    recomputeGroupCentroidFromDb
} from '$lib/server/centroids';
import { suggestExternalPictures } from '$lib/server/connector-suggestions';
import {
    countActiveForFolders,
    listExcludedByFolder,
    markExcluded,
    restoreActive,
    type DatasetImageRow
} from '$lib/server/dataset-images';
import { defer } from '$lib/server/defer';
import { getDatasetGroup } from '$lib/server/dataset-groups';
import {
    loadFramingCoverage,
    loadPoseCalibration,
    loadPoseCoverage,
    readDatasetDetail
} from '$lib/server/dataset-detail';
import { getMaxSize, setMaxSize } from '$lib/server/dataset-size-limits';
import { enqueue } from '$lib/server/jobs';
import { isPathInside, pathBasename } from '$lib/server/path-utils';
import { suggestPruneCandidates, type PruneCandidate } from '$lib/server/prune-suggestions';

function resolveGroup(rawId: string) {
    const id = Number(rawId);
    if (!Number.isFinite(id) || id <= 0) throw error(404, 'Bad group id');
    const group = getDatasetGroup(id);
    if (!group) throw error(404, 'Group not found');
    return { id, group };
}

/** Same slug algorithm the group's raw-image route uses (basename + __N for
 * duplicates). Returns folder_path → slug, so prune candidates can resolve
 * their member URL piece by looking up `folder_path`. */
function buildSlugMap(paths: string[]): Map<string, string> {
    const seen = new Map<string, number>();
    const out = new Map<string, string>();
    for (const p of paths) {
        const base = basename(p);
        const count = (seen.get(base) ?? 0) + 1;
        seen.set(base, count);
        out.set(p, count === 1 ? base : `${base}__${count - 1}`);
    }
    return out;
}

function decoratePruneForGroup(
    id: number,
    slugMap: Map<string, string>,
    c: PruneCandidate
) {
    const urlFor = (folder_path: string, filename: string) => {
        const slug = slugMap.get(folder_path) ?? basename(folder_path);
        const base = `/datasets/group/${id}/raw/${encodeURIComponent(slug)}/${encodeURIComponent(filename)}`;
        return { thumbnail_url: `${base}?w=400`, full_url: base };
    };
    const own = urlFor(c.folder_path, c.filename);
    return {
        ...c,
        thumbnail_url: own.thumbnail_url,
        full_url: own.full_url,
        neighbors: c.neighbors.map((n) => ({
            ...n,
            ...urlFor(n.folder_path, n.filename)
        }))
    };
}

function decorateExcludedForGroup(
    id: number,
    slugMap: Map<string, string>,
    row: DatasetImageRow
) {
    const filename = pathBasename(row.image_path);
    const slug = slugMap.get(row.folder_path) ?? basename(row.folder_path);
    return {
        image_path: row.image_path,
        filename,
        folder_path: row.folder_path,
        slug,
        excluded_at: row.excluded_at,
        excluded_reason: row.excluded_reason,
        thumbnail_url: `/datasets/group/${id}/raw/${encodeURIComponent(slug)}/${encodeURIComponent(filename)}?w=400`
    };
}

function assertUnderGroupMember(group: { paths: string[] }, imagePath: string): void {
    for (const p of group.paths) {
        if (isPathInside(p, imagePath)) return;
    }
    throw error(403, 'image_path is not under any member of this group');
}

export const load: PageServerLoad = ({ params }) => {
    const { id, group } = resolveGroup(params.id);

    // Look up the group centroid first so we can ask listDatasetItems to
    // double-score every image against it on the way out (cheap dot product
    // per image, single decode of the centroid).
    const global_centroid = getCentroid('group', String(id));

    // Resolve each path. Duplicates resolving to the same basename get a
    // disambiguating suffix so the image proxy can still find them.
    const seen = new Map<string, number>();
    const datasets = group.paths.map((p) => {
        const detail = readDatasetDetail(p, global_centroid?.centroid_b64 ?? null);
        const base = basename(p);
        const count = (seen.get(base) ?? 0) + 1;
        seen.set(base, count);
        const slug = count === 1 ? base : `${base}__${count - 1}`;
        const ds_pose = loadPoseCoverage(detail.path);
        const ds_framing = loadFramingCoverage(detail.path);
        const ds_centroid = getCentroid('folder', detail.path);
        const ds_calib = loadPoseCalibration([detail.path]);
        const ds_max_size = getMaxSize('folder', detail.path);
        return {
            slug,
            ...detail,
            centroid: ds_centroid,
            pose_coverage: ds_pose,
            framing_coverage: ds_framing,
            pose_calibration: ds_calib,
            // Streamed: each member's suggestions resolves on its own I/O
            // turn. With N members, the page renders with N placeholders
            // and they fill in one by one — better perceived latency than
            // a single N×130ms sync block.
            suggestions: defer(() =>
                suggestExternalPictures({
                    scope_kind: 'folder',
                    scope_key: detail.path,
                    centroid: ds_centroid,
                    pose_overrides: {
                        threequarter: ds_calib.offset_threequarter,
                        profile: ds_calib.offset_profile,
                        tilted: ds_calib.offset_tilted
                    },
                    pose_gaps: poseGroupGaps(ds_pose, ds_max_size),
                    framing_gaps: framingGroupGaps(ds_framing, ds_max_size)
                })
            )
        };
    });

    const group_pose_calibration = loadPoseCalibration(group.paths);

    // Global (group-wide) suggestions: aggregate coverage across members,
    // use the group centroid + group-level calibration.
    const total_pose = datasets.reduce<
        ReturnType<typeof loadPoseCoverage>
    >((acc, ds) => {
        for (const k of Object.keys(ds.pose_coverage.counts)) {
            acc.counts[k as keyof typeof acc.counts] =
                (acc.counts[k as keyof typeof acc.counts] ?? 0) +
                (ds.pose_coverage.counts[k as keyof typeof ds.pose_coverage.counts] ?? 0);
        }
        acc.unknown += ds.pose_coverage.unknown;
        acc.total += ds.pose_coverage.total;
        return acc;
    }, { counts: {} as ReturnType<typeof loadPoseCoverage>['counts'], unknown: 0, total: 0 });

    const total_framing = datasets.reduce<
        ReturnType<typeof loadFramingCoverage>
    >((acc, ds) => {
        for (const k of Object.keys(ds.framing_coverage.counts)) {
            acc.counts[k as keyof typeof acc.counts] =
                (acc.counts[k as keyof typeof acc.counts] ?? 0) +
                (ds.framing_coverage.counts[k as keyof typeof ds.framing_coverage.counts] ?? 0);
        }
        acc.unknown += ds.framing_coverage.unknown;
        acc.total += ds.framing_coverage.total;
        return acc;
    }, { counts: {} as ReturnType<typeof loadFramingCoverage>['counts'], unknown: 0, total: 0 });

    const max_size = getMaxSize('group', String(id));
    const group_pose_gaps = poseGroupGaps(total_pose, max_size);
    const group_framing_gaps = framingGroupGaps(total_framing, max_size);

    const slugMap = buildSlugMap(group.paths);
    // Header counter — cheap COUNT across member folders. Kept sync so
    // the "N / max" pill renders without a placeholder.
    const n_active = countActiveForFolders(group.paths);

    // Excluded list spans every member folder.
    const excluded: ReturnType<typeof decorateExcludedForGroup>[] = [];
    for (const p of group.paths) {
        for (const row of listExcludedByFolder(p)) {
            excluded.push(decorateExcludedForGroup(id, slugMap, row));
        }
    }

    return {
        group,
        datasets,
        global_centroid,
        // Group-level calibration is computed from the union of every
        // member's winners — used in the "vs group" badge mode.
        group_pose_calibration,
        max_size,
        n_active,
        // Streamed: the group-level suggestion + prune compute is the
        // heaviest block here (full pHash dedup across all members).
        group_suggestions: defer(() =>
            suggestExternalPictures({
                scope_kind: 'group',
                scope_key: String(id),
                centroid: global_centroid,
                pose_overrides: {
                    threequarter: group_pose_calibration.offset_threequarter,
                    profile: group_pose_calibration.offset_profile,
                    tilted: group_pose_calibration.offset_tilted
                },
                pose_gaps: group_pose_gaps,
                framing_gaps: group_framing_gaps
            })
        ),
        prune: defer(() => {
            const p = suggestPruneCandidates({
                folder_paths: group.paths,
                max_size,
                pose_gaps: group_pose_gaps,
                framing_gaps: group_framing_gaps,
                centroid: global_centroid
            });
            return {
                n_active: p.n_active,
                max_size: p.max_size,
                buckets: p.buckets.map((b) => ({
                    ...b,
                    candidates: b.candidates.map((c) =>
                        decoratePruneForGroup(id, slugMap, c)
                    )
                }))
            };
        }),
        excluded
    };
};

/** Resync every member centroid + the group centroid after a status flip
 * affected one member folder. */
function resyncAfterStatusChange(
    groupId: number,
    groupPaths: string[],
    affectedFolder: string
): void {
    recomputeFolderCentroidFromDb(affectedFolder);
    recomputeGroupCentroidFromDb(groupId, groupPaths);
}

export const actions: Actions = {
    // Per-member centroid + global group centroid (sync) PLUS one
    // background image-hash job per member folder. See the folder action
    // for the rationale on merging into a single user-facing action.
    analyze: async ({ params }) => {
        const { id, group } = resolveGroup(params.id);
        // Only feed paths that actually exist on disk — skipping missing
        // members keeps the Python detector from erroring on an unknown dir.
        const existing = group.paths.filter((p) => !group.missing_paths.includes(p));
        if (existing.length === 0) {
            return fail(400, { error: 'No existing dataset folders in this group.' });
        }
        try {
            const result = await computeAndPersist(existing, id);
            const hash_job_ids: number[] = [];
            for (const p of existing) {
                try {
                    hash_job_ids.push(
                        enqueue(
                            'compute-image-hashes',
                            { folder_path: p },
                            { key_arg1: p }
                        )
                    );
                } catch {
                    // ignore per-folder enqueue failures; the centroid is
                    // the user-visible deliverable
                }
            }
            return {
                ok: true,
                per_folder: result.per_folder,
                global: result.global,
                hash_job_ids
            };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    },
    setMaxSize: async ({ params, request }) => {
        const { id } = resolveGroup(params.id);
        const form = await request.formData();
        const raw = String(form.get('max_size') ?? '').trim();
        const n = raw === '' ? null : Number(raw);
        if (n != null && (!Number.isFinite(n) || n < 0)) {
            return fail(400, { error: 'max_size must be a positive integer or empty' });
        }
        setMaxSize('group', String(id), n);
        return { ok: true };
    },
    exclude: async ({ params, request }) => {
        const { id, group } = resolveGroup(params.id);
        const form = await request.formData();
        const imagePath = String(form.get('image_path') ?? '');
        const reason = String(form.get('reason') ?? '') || null;
        if (!imagePath) return fail(400, { error: 'image_path is required' });
        try {
            assertUnderGroupMember(group, imagePath);
        } catch (e) {
            return fail(403, { error: (e as Error).message });
        }
        const member = group.paths.find((p) => {
            return isPathInside(p, imagePath);
        });
        if (!member) return fail(403, { error: 'image_path is not under any member' });
        markExcluded(imagePath, reason);
        resyncAfterStatusChange(id, group.paths, member);
        return { ok: true };
    },
    restore: async ({ params, request }) => {
        const { id, group } = resolveGroup(params.id);
        const form = await request.formData();
        const imagePath = String(form.get('image_path') ?? '');
        if (!imagePath) return fail(400, { error: 'image_path is required' });
        try {
            assertUnderGroupMember(group, imagePath);
        } catch (e) {
            return fail(403, { error: (e as Error).message });
        }
        const member = group.paths.find((p) => {
            return isPathInside(p, imagePath);
        });
        if (!member) return fail(403, { error: 'image_path is not under any member' });
        restoreActive(imagePath);
        resyncAfterStatusChange(id, group.paths, member);
        return { ok: true };
    }
};
