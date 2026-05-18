// Re-enqueue a terminal-state job (failed/cancelled/completed) with its
// original params. Returns the new job id so the UI can highlight it.
import type { RequestHandler } from './$types';
import { error, json } from '@sveltejs/kit';
import { retry } from '$lib/server/jobs';

export const POST: RequestHandler = ({ params }) => {
    const id = Number(params.id);
    if (!Number.isFinite(id) || id <= 0) throw error(400, 'Bad id');
    const new_job_id = retry(id);
    if (new_job_id == null) {
        return json(
            { ok: false, error: 'Job cannot be retried (still active, unknown id, or handler missing).' },
            { status: 409 }
        );
    }
    return json({ ok: true, new_job_id });
};
