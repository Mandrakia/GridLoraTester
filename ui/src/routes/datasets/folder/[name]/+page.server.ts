import type { Actions, PageServerLoad } from './$types';
import { error, fail } from '@sveltejs/kit';
import { statSync } from 'node:fs';
import { resolve } from 'node:path';

import { framingGroupGaps, poseGroupGaps } from '$lib/dataset-targets';
import { computeAndPersist, getCentroid } from '$lib/server/centroids';
import { suggestExternalPictures } from '$lib/server/connector-suggestions';
import {
    loadFramingCoverage,
    loadPoseCalibration,
    loadPoseCoverage,
    readDatasetDetail
} from '$lib/server/dataset-detail';
import { getSettings } from '$lib/server/settings';

function resolveFolder(name: string): string {
    const { dataset_root } = getSettings();
    if (!dataset_root) throw error(404, 'dataset_root not configured');
    const root = resolve(dataset_root);
    const folder = resolve(root, name);
    if (folder !== root && !folder.startsWith(root + '/')) throw error(403, 'Forbidden');
    try {
        if (!statSync(folder).isDirectory()) throw error(404, 'Not a directory');
    } catch {
        throw error(404, 'Dataset not found');
    }
    return folder;
}

export const load: PageServerLoad = ({ params }) => {
    const folder = resolveFolder(params.name);
    const centroid = getCentroid('folder', folder);
    const pose_coverage = loadPoseCoverage(folder);
    const framing_coverage = loadFramingCoverage(folder);
    const pose_calibration = loadPoseCalibration([folder]);
    return {
        dataset: readDatasetDetail(folder),
        centroid,
        pose_coverage,
        framing_coverage,
        pose_calibration,
        suggestions: suggestExternalPictures({
            scope_kind: 'folder',
            scope_key: folder,
            centroid,
            pose_overrides: {
                threequarter: pose_calibration.offset_threequarter,
                profile: pose_calibration.offset_profile,
                tilted: pose_calibration.offset_tilted
            },
            pose_gaps: poseGroupGaps(pose_coverage),
            framing_gaps: framingGroupGaps(framing_coverage)
        })
    };
};

export const actions: Actions = {
    'compute-centroid': async ({ params }) => {
        const folder = resolveFolder(params.name);
        try {
            const result = await computeAndPersist([folder]);
            return { ok: true, summary: result.per_folder[folder] ?? null };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    }
};
