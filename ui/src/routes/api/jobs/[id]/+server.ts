// Single job + recent log lines.
import type { RequestHandler } from './$types';
import { error, json } from '@sveltejs/kit';
import { getJob, getJobLogs } from '$lib/server/jobs';

export const GET: RequestHandler = ({ params }) => {
    const id = Number(params.id);
    if (!Number.isFinite(id) || id <= 0) throw error(400, 'Bad id');
    const job = getJob(id);
    if (!job) throw error(404, 'Job not found');
    return json({ job, logs: getJobLogs(id, 500) });
};
