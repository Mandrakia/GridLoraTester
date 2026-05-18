import type { PageServerLoad } from './$types';
import { listArchiveJobs, listLatestJobs } from '$lib/server/jobs';

// SSR seeds the Latest tab. Switching to Archives in the UI triggers a
// fetch to /api/jobs?view=archive — no need to ship both lists at once.
export const load: PageServerLoad = () => {
    return {
        latest: listLatestJobs(200),
        // Pre-load a small slice of archives so the tab feels instant on
        // first switch. Polling will refresh both on cadence.
        archive: listArchiveJobs(50)
    };
};
