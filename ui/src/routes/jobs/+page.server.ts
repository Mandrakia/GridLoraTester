import type { PageServerLoad } from './$types';
import { listJobs } from '$lib/server/jobs';

export const load: PageServerLoad = () => {
    return { jobs: listJobs(200) };
};
