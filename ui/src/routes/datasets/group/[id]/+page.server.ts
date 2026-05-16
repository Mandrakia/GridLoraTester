import type { Actions, PageServerLoad } from './$types';
import { error, fail } from '@sveltejs/kit';
import { basename } from 'node:path';

import { framingGroupGaps, poseGroupGaps } from '$lib/dataset-targets';
import { computeAndPersist, getCentroid } from '$lib/server/centroids';
import { suggestExternalPictures } from '$lib/server/connector-suggestions';
import { getDatasetGroup } from '$lib/server/dataset-groups';
import {
    loadFramingCoverage,
    loadPoseCalibration,
    loadPoseCoverage,
    readDatasetDetail
} from '$lib/server/dataset-detail';

function resolveGroup(rawId: string) {
    const id = Number(rawId);
    if (!Number.isFinite(id) || id <= 0) throw error(404, 'Bad group id');
    const group = getDatasetGroup(id);
    if (!group) throw error(404, 'Group not found');
    return { id, group };
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
        return {
            slug,
            ...detail,
            centroid: ds_centroid,
            pose_coverage: ds_pose,
            framing_coverage: ds_framing,
            pose_calibration: ds_calib,
            suggestions: suggestExternalPictures({
                scope_kind: 'folder',
                scope_key: detail.path,
                centroid: ds_centroid,
                pose_overrides: {
                    threequarter: ds_calib.offset_threequarter,
                    profile: ds_calib.offset_profile,
                    tilted: ds_calib.offset_tilted
                },
                pose_gaps: poseGroupGaps(ds_pose),
                framing_gaps: framingGroupGaps(ds_framing)
            })
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

    return {
        group,
        datasets,
        global_centroid,
        // Group-level calibration is computed from the union of every
        // member's winners — used in the "vs group" badge mode.
        group_pose_calibration,
        group_suggestions: suggestExternalPictures({
            scope_kind: 'group',
            scope_key: String(id),
            centroid: global_centroid,
            pose_overrides: {
                threequarter: group_pose_calibration.offset_threequarter,
                profile: group_pose_calibration.offset_profile,
                tilted: group_pose_calibration.offset_tilted
            },
            pose_gaps: poseGroupGaps(total_pose),
            framing_gaps: framingGroupGaps(total_framing)
        })
    };
};

export const actions: Actions = {
    'compute-centroid': async ({ params }) => {
        const { id, group } = resolveGroup(params.id);
        // Only feed paths that actually exist on disk — skipping missing
        // members keeps the Python detector from erroring on an unknown dir.
        const existing = group.paths.filter((p) => !group.missing_paths.includes(p));
        if (existing.length === 0) {
            return fail(400, { error: 'No existing dataset folders in this group.' });
        }
        try {
            const result = await computeAndPersist(existing, id);
            return { ok: true, per_folder: result.per_folder, global: result.global };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    }
};
