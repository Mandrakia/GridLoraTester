// Detect faces + compute the ArcFace centroid for one or more dataset
// folders as a background job, so the analysis surfaces in /jobs and the
// Analyze button can spin on the job's lifetime instead of blocking the POST
// (computeAndPersist used to run inline in the form action — no job row, the
// spinner was just the request awaiting a synchronous Python round-trip).
//
// One job covers every path in `paths`: computeAndPersist spawns a single
// `glt --detect-folders` subprocess for the whole set (its stderr → this
// job's log), then one DB transaction. When `group_id` is set it additionally
// writes the union 'group' centroid.
//
// The dedup-hash backfill stays a separate compute-image-hashes job, fired
// once the centroid lands: its phase-2 connector pass is network-bound (much
// longer than the centroid) so we don't want it on the Analyze spinner, and
// its identity pre-filter reads the centroid we just wrote — it has to run
// after, not alongside.
import { computeAndPersist } from '../../centroids';
import { enqueue, registerHandler, type JobContext, type JobHandler } from '../runner';

interface Params {
    paths?: string[];
    group_id?: number | null;
}

const handler: JobHandler = async (ctx: JobContext) => {
    const params = ctx.params as unknown as Params;
    const paths = Array.isArray(params.paths)
        ? params.paths.filter((p): p is string => typeof p === 'string' && p.length > 0)
        : [];
    const groupId =
        typeof params.group_id === 'number' && Number.isFinite(params.group_id)
            ? params.group_id
            : null;

    if (paths.length === 0) {
        ctx.log('warn', 'No folders to analyze.');
        return;
    }

    const scope = paths.length === 1 ? paths[0] : `${paths.length} folders`;
    // computeAndPersist is atomic (one Python call + one transaction), so
    // there's no per-image granularity to report — 0→1 just marks the job
    // as having started for the /jobs progress column.
    ctx.progress(0, 1, scope);
    ctx.log('info', `Detecting faces + computing centroid for ${scope}…`);

    const result = await computeAndPersist(paths, groupId, { log: ctx.log });

    for (const [folder, stats] of Object.entries(result.per_folder)) {
        ctx.log(
            'info',
            `${folder}: ${stats.n_single_face} single · ${stats.n_multi_face} multi · ${stats.n_no_face} no-face`
        );
    }
    ctx.log('info', `Persisted ${result.persisted_faces} face embedding(s).`);
    ctx.progress(1, 1, scope);

    for (const p of paths) {
        try {
            enqueue('compute-image-hashes', { folder_path: p }, { key_arg1: p });
        } catch {
            // Don't fail the centroid job if the hash enqueue hiccups —
            // the centroid is the user-visible deliverable.
        }
    }
};

registerHandler('compute-centroid', handler);
