// Barrel that pulls in every job handler so they're registered with the
// runner at the first import of this module. Server code that needs the
// runner should import from here (not from `./runner` directly) to make
// sure handlers are registered before they enqueue.
import './handlers/connector-face-detect';
import './handlers/compute-image-hashes';
import './handlers/grid-test-run';
import './handlers/test-rescore';

export {
    cancel,
    enqueue,
    getJob,
    getJobLogs,
    listActiveJobs,
    listArchiveJobs,
    listJobs,
    listLatestJobs,
    retry,
    setPid,
    type EnqueueOpts,
    type JobLogRow,
    type JobRow,
    type JobStatus
} from './runner';
