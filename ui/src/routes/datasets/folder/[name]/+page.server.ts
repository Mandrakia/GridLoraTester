import type { Actions, PageServerLoad } from './$types';
import { error, fail } from '@sveltejs/kit';
import { statSync } from 'node:fs';
import { resolve } from 'node:path';

import { framingGroupGaps, poseGroupGaps } from '$lib/dataset-targets';
import {
    computeAndPersist,
    getCentroid,
    recomputeFolderCentroidFromDb,
    recomputeGroupCentroidFromDb
} from '$lib/server/centroids';
import { suggestExternalPictures } from '$lib/server/connector-suggestions';
import {
    listExcludedByFolder,
    markExcluded,
    restoreActive
} from '$lib/server/dataset-images';
import {
    loadFramingCoverage,
    loadPoseCalibration,
    loadPoseCoverage,
    readDatasetDetail
} from '$lib/server/dataset-detail';
import { findGroupsContaining } from '$lib/server/dataset-groups';
import { getMaxSize, setMaxSize } from '$lib/server/dataset-size-limits';
import { enqueue } from '$lib/server/jobs';
import { suggestPruneCandidates, type PruneCandidate } from '$lib/server/prune-suggestions';
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

/** Decorate a prune candidate with the dataset-image URLs the UI needs.
 * Folder scope: every image lives directly under the URL slug `name`, so
 * candidates AND their in-bucket neighbors share the same URL pattern. */
function decoratePruneForFolder(name: string, c: PruneCandidate) {
    const urlFor = (filename: string) => {
        const base = `/datasets/folder/${encodeURIComponent(name)}/raw/${encodeURIComponent(filename)}`;
        return { thumbnail_url: `${base}?w=400`, full_url: base };
    };
    const own = urlFor(c.filename);
    return {
        ...c,
        thumbnail_url: own.thumbnail_url,
        full_url: own.full_url,
        neighbors: c.neighbors.map((n) => ({
            ...n,
            ...urlFor(n.filename)
        }))
    };
}

/** Assert that the image_path is actually under the folder scope. Guards
 * the markExcluded / restoreActive actions against IDs from a different
 * dataset being passed through this scope's action endpoint. */
function assertUnderFolder(folder: string, imagePath: string): void {
    const resolved = resolve(imagePath);
    if (resolved !== folder && !resolved.startsWith(folder + '/')) {
        throw error(403, 'image_path is not under this folder');
    }
}

export const load: PageServerLoad = ({ params }) => {
    const folder = resolveFolder(params.name);
    const centroid = getCentroid('folder', folder);
    const pose_coverage = loadPoseCoverage(folder);
    const framing_coverage = loadFramingCoverage(folder);
    const pose_calibration = loadPoseCalibration([folder]);
    const max_size = getMaxSize('folder', folder);
    const pose_gaps = poseGroupGaps(pose_coverage, max_size);
    const framing_gaps = framingGroupGaps(framing_coverage, max_size);
    const prune = suggestPruneCandidates({
        folder_paths: [folder],
        max_size,
        pose_gaps,
        framing_gaps,
        centroid
    });
    return {
        dataset: readDatasetDetail(folder),
        centroid,
        pose_coverage,
        framing_coverage,
        pose_calibration,
        max_size,
        suggestions: suggestExternalPictures({
            scope_kind: 'folder',
            scope_key: folder,
            centroid,
            pose_overrides: {
                threequarter: pose_calibration.offset_threequarter,
                profile: pose_calibration.offset_profile,
                tilted: pose_calibration.offset_tilted
            },
            pose_gaps,
            framing_gaps
        }),
        prune: {
            n_active: prune.n_active,
            max_size: prune.max_size,
            buckets: prune.buckets.map((b) => ({
                ...b,
                candidates: b.candidates.map((c) => decoratePruneForFolder(params.name, c))
            }))
        },
        excluded: listExcludedByFolder(folder).map((row) => ({
            image_path: row.image_path,
            filename: row.image_path.split('/').pop() ?? row.image_path,
            excluded_at: row.excluded_at,
            excluded_reason: row.excluded_reason,
            thumbnail_url: `/datasets/folder/${encodeURIComponent(params.name)}/raw/${encodeURIComponent(
                row.image_path.split('/').pop() ?? ''
            )}?w=400`
        }))
    };
};

/** Re-sync the folder's centroid + every group containing it after a
 * status flip. Doing this server-side keeps the page load consistent on
 * the next invalidate. */
function resyncAfterStatusChange(folder: string): number[] {
    recomputeFolderCentroidFromDb(folder);
    const affected = findGroupsContaining(folder);
    const resynced: number[] = [];
    for (const g of affected) {
        const row = recomputeGroupCentroidFromDb(g.id, g.paths);
        if (row != null) resynced.push(g.id);
    }
    return resynced;
}

export const actions: Actions = {
    // Single end-to-end action: detect faces + compute centroid (sync,
    // returns the summary right away) THEN fire-and-forget enqueue a
    // background image-hash job for the same folder. One click, one
    // mental model — both outputs are consumed by the same downstream
    // suggestion engine so splitting them was just decision fatigue for
    // the user.
    analyze: async ({ params }) => {
        const folder = resolveFolder(params.name);
        try {
            const result = await computeAndPersist([folder]);
            let hash_job_id: number | null = null;
            try {
                hash_job_id = enqueue(
                    'compute-image-hashes',
                    { folder_path: folder },
                    { key_arg1: folder }
                );
            } catch {
                // Don't fail analyze if the hash job enqueue hiccups —
                // the centroid is the user-visible deliverable. They can
                // retry the hash separately if needed.
            }
            return {
                ok: true,
                summary: result.per_folder[folder] ?? null,
                hash_job_id
            };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    },
    setMaxSize: async ({ params, request }) => {
        const folder = resolveFolder(params.name);
        const form = await request.formData();
        const raw = String(form.get('max_size') ?? '').trim();
        const n = raw === '' ? null : Number(raw);
        if (n != null && (!Number.isFinite(n) || n < 0)) {
            return fail(400, { error: 'max_size must be a positive integer or empty' });
        }
        setMaxSize('folder', folder, n);
        return { ok: true };
    },
    exclude: async ({ params, request }) => {
        const folder = resolveFolder(params.name);
        const form = await request.formData();
        const imagePath = String(form.get('image_path') ?? '');
        const reason = String(form.get('reason') ?? '') || null;
        if (!imagePath) return fail(400, { error: 'image_path is required' });
        try {
            assertUnderFolder(folder, imagePath);
        } catch (e) {
            return fail(403, { error: (e as Error).message });
        }
        markExcluded(imagePath, reason);
        const resynced = resyncAfterStatusChange(folder);
        return { ok: true, resynced_group_ids: resynced };
    },
    restore: async ({ params, request }) => {
        const folder = resolveFolder(params.name);
        const form = await request.formData();
        const imagePath = String(form.get('image_path') ?? '');
        if (!imagePath) return fail(400, { error: 'image_path is required' });
        try {
            assertUnderFolder(folder, imagePath);
        } catch (e) {
            return fail(403, { error: (e as Error).message });
        }
        restoreActive(imagePath);
        const resynced = resyncAfterStatusChange(folder);
        return { ok: true, resynced_group_ids: resynced };
    }
};
