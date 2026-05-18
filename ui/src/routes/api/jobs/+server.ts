// List jobs. `?active=1` narrows to queued+running for the floating badge.
// `?view=latest` → most recent per (type, key_arg1, key_arg2); `?view=archive`
// → older runs of the same logical job. Default = `latest` for /jobs page.
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import {
    listActiveJobs,
    listArchiveJobs,
    listJobs,
    listLatestJobs
} from '$lib/server/jobs';

export const GET: RequestHandler = ({ url }) => {
    const activeOnly = url.searchParams.get('active') === '1';
    const view = url.searchParams.get('view'); // 'latest' | 'archive' | null
    const limit = Math.min(Math.max(Number(url.searchParams.get('limit') ?? 100), 1), 500);
    let jobs;
    if (activeOnly) {
        jobs = listActiveJobs();
    } else if (view === 'archive') {
        jobs = listArchiveJobs(limit);
    } else if (view === 'latest') {
        jobs = listLatestJobs(limit);
    } else {
        jobs = listJobs(limit);
    }
    return json({ jobs });
};
