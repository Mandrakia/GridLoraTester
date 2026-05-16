// List jobs. `?active=1` narrows to queued+running for the floating badge.
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { listActiveJobs, listJobs } from '$lib/server/jobs';

export const GET: RequestHandler = ({ url }) => {
    const activeOnly = url.searchParams.get('active') === '1';
    const limit = Math.min(Math.max(Number(url.searchParams.get('limit') ?? 100), 1), 500);
    const jobs = activeOnly ? listActiveJobs() : listJobs(limit);
    return json({ jobs });
};
