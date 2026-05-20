// In-memory job runner. DB is the source of truth (status + progress); the
// runtime here adds the "currently executing" handle so we can cooperatively
// cancel, batch progress writes, and avoid hammering SQLite on every tick.
//
// One worker per job type, sequential within a type. Different types can
// run in parallel — useful later when we add downloads + tests alongside
// face detection.
import { db } from '../db';
import {
    finalizeStaleRunsForTest,
    latestTestRunStatus,
    resurrectStaleRunsOnBoot
} from '../test-runs';

export type JobStatus = 'queued' | 'running' | 'cancelled' | 'completed' | 'failed';

export interface JobRow {
    id: number;
    type: string;
    params_json: string;
    status: JobStatus;
    progress_current: number;
    progress_total: number | null;
    current_label: string | null;
    error: string | null;
    created_at: string;
    started_at: string | null;
    finished_at: string | null;
    /** JSON-encoded PhaseSnapshot — null while no metrics have been written
     * yet (queued jobs, or handlers that don't use ctx.metrics()). */
    metrics_json: string | null;
    /** Logical-key columns. (type, key_arg1, key_arg2) groups runs of the
     * same job (e.g. face-detect on the same connector+person). NULL for
     * pre-migration rows. */
    key_arg1: string | null;
    key_arg2: string | null;
    /** OS pid of the process doing the work. For in-process Node handlers
     * it's process.pid (set on execute start); for subprocess handlers
     * (grid-test-run) it's the child's pid, set via setPid() once spawn
     * returns. The orphan reaper uses this to detect dead runs. */
    pid: number | null;
}

export interface JobLogRow {
    id: number;
    job_id: number;
    level: 'info' | 'warn' | 'error';
    message: string;
    created_at: string;
}

export interface JobContext {
    job_id: number;
    params: Record<string, unknown>;
    /** True the moment the user (or system) requested a cancel. Handlers
     * should check at every iteration boundary and bail early. */
    shouldCancel(): boolean;
    /** Update progress. Persisted in DB on a debounce — safe to call once
     * per item without flooding the DB. */
    progress(current: number, total?: number, label?: string): void;
    /** Append a log line (always persisted immediately, since logs are
     * usually low-frequency). */
    log(level: 'info' | 'warn' | 'error', message: string): void;
    /** Persist a JSON-serializable metrics snapshot (per-phase p50/p95
     * etc.) to `jobs.metrics_json`. Debounced like progress so handlers
     * can call it once per item without DB pressure. The UI reads this
     * field through the existing poll. Pass `{force: true}` to bypass the
     * debounce — useful for the final snapshot so the last sample isn't
     * lost. */
    metrics(snapshot: unknown, opts?: { force?: boolean }): void;
}

export type JobHandler = (ctx: JobContext) => Promise<void>;

// ---- DB statements --------------------------------------------------------
const insertJobStmt = db.prepare(
    "INSERT INTO jobs(type, params_json, key_arg1, key_arg2, status) VALUES(?, ?, ?, ?, 'queued') RETURNING id"
);
const updateProgressStmt = db.prepare(
    'UPDATE jobs SET progress_current = ?, progress_total = ?, current_label = ? WHERE id = ?'
);
const updateMetricsStmt = db.prepare('UPDATE jobs SET metrics_json = ? WHERE id = ?');
const startJobStmt = db.prepare(
    "UPDATE jobs SET status = 'running', started_at = datetime('now'), pid = ? WHERE id = ?"
);
const updatePidStmt = db.prepare('UPDATE jobs SET pid = ? WHERE id = ?');
const finishJobStmt = db.prepare(
    "UPDATE jobs SET status = ?, error = ?, finished_at = datetime('now'), pid = NULL WHERE id = ?"
);
const getJobStmt = db.prepare('SELECT * FROM jobs WHERE id = ?');
const listJobsStmt = db.prepare('SELECT * FROM jobs ORDER BY id DESC LIMIT ?');
// Latest run per logical key. NULL keys collapse to '' so old rows (no
// key set) group into a single bucket per type — acceptable degradation
// for pre-migration history; new runs of any handler always set the key.
// COALESCE-with-id-as-string in the GROUP BY for ungrouped rows would
// preserve every legacy row, but it'd also balloon the Latest view back
// to "everything", defeating the point.
const listLatestStmt = db.prepare(`
    SELECT j.* FROM jobs j
    JOIN (
        SELECT MAX(id) AS max_id
          FROM jobs
         GROUP BY type, IFNULL(key_arg1, ''), IFNULL(key_arg2, '')
    ) latest ON j.id = latest.max_id
    ORDER BY j.id DESC
    LIMIT ?
`);
const listArchiveStmt = db.prepare(`
    SELECT j.* FROM jobs j
    WHERE j.id NOT IN (
        SELECT MAX(id) FROM jobs
         GROUP BY type, IFNULL(key_arg1, ''), IFNULL(key_arg2, '')
    )
    ORDER BY j.id DESC
    LIMIT ?
`);
const listActiveStmt = db.prepare(
    "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY id ASC"
);
// Rows the orphan reaper needs to inspect — running with a pid we can
// poll. Excludes pid IS NULL (handler hasn't reported yet — give it a
// grace window via execute()).
const listRunningWithPidStmt = db.prepare(
    "SELECT id, pid, type FROM jobs WHERE status = 'running' AND pid IS NOT NULL"
);
const insertLogStmt = db.prepare(
    'INSERT INTO job_logs(job_id, level, message) VALUES(?, ?, ?)'
);
// Tail the most recent `limit` lines, then re-sort ascending for display.
// A plain "ORDER BY id ASC LIMIT ?" returns the OLDEST window, so a long
// run (>limit log lines) freezes the UI at the timestamp of line #limit
// while the live tail piles up unseen.
const listLogsStmt = db.prepare(`
    SELECT * FROM (
        SELECT * FROM job_logs WHERE job_id = ? ORDER BY id DESC LIMIT ?
    ) ORDER BY id ASC
`);
// Boot-time job recovery. We can't blindly fail every 'running' row —
// subprocess handlers (grid-test-run) spawn a detached-ish Python child
// that often outlives a Node restart (HMR, npm-run-dev quick restart),
// and that Python finishes its own test_runs row independently. So we
// check each pid: NULL or dead → fail; alive → leave 'running' and let
// the orphan reaper finalize it once the pid eventually goes away.
const listRunningForRecoveryStmt = db.prepare(
    "SELECT id, pid FROM jobs WHERE status = 'running'"
);
const failJobBootStmt = db.prepare(
    "UPDATE jobs SET status = 'failed', error = ?, finished_at = datetime('now'), pid = NULL WHERE id = ?"
);

// ---- Runtime state --------------------------------------------------------
const handlers = new Map<string, JobHandler>();
/** Pending queues per type — we run them sequentially per type. */
const queues = new Map<string, number[]>();
/** Per-type running flag so we don't double-execute. */
const running = new Map<string, boolean>();
/** Per-job cancel flag, cleared when the job terminates. */
const cancelFlags = new Map<number, boolean>();

/** Progress write debouncer: per-job last write timestamp. */
const lastProgressWrite = new Map<number, number>();
const PROGRESS_MIN_INTERVAL_MS = 500;
/** Metrics write debouncer — same idea, slightly slacker cadence (the UI
 * polls every 1.5s on active jobs so 1s is the right floor). */
const lastMetricsWrite = new Map<number, number>();
const METRICS_MIN_INTERVAL_MS = 1000;

export function registerHandler(type: string, handler: JobHandler): void {
    handlers.set(type, handler);
}

export function listHandlers(): string[] {
    return [...handlers.keys()];
}

export interface EnqueueOpts {
    /** Logical key part 1 (e.g. connector_id for face-detect, folder_path
     * for compute-image-hashes, test_id string for grid-test-run). Used by
     * the Jobs page to fold older runs of the same logical job into the
     * Archives tab. */
    key_arg1?: string | null;
    /** Logical key part 2 (e.g. person_id for face-detect). NULL when the
     * job has only one key dimension. */
    key_arg2?: string | null;
}

export function enqueue(
    type: string,
    params: Record<string, unknown> = {},
    opts: EnqueueOpts = {}
): number {
    if (!handlers.has(type)) {
        throw new Error(`No handler registered for job type "${type}".`);
    }
    const row = insertJobStmt.get(
        type,
        JSON.stringify(params),
        opts.key_arg1 ?? null,
        opts.key_arg2 ?? null
    ) as { id: number };
    const id = row.id;
    const q = queues.get(type) ?? [];
    q.push(id);
    queues.set(type, q);
    void drain(type);
    return id;
}

/** Update the pid recorded for a running job. Handlers that spawn a
 * subprocess (grid-test-run) call this once `spawn()` returns, so the
 * orphan reaper can detect a child killed externally before the parent
 * notices. In-process handlers don't need this — execute() records
 * process.pid for them at start. */
export function setPid(jobId: number, pid: number | null): void {
    updatePidStmt.run(pid, jobId);
}

/** Re-enqueue a terminal-state job with its original params. Creates a
 * NEW job (rather than reviving the old row) so the audit trail keeps both
 * runs. Handler idempotency (per-picture `existsStmt` skip etc.) ensures
 * the retry only does the work the first run didn't complete.
 *
 * Returns the new job id on success, or null when the source job can't be
 * retried (unknown id, still active, or handler no longer registered). */
export function retry(jobId: number): number | null {
    const job = getJob(jobId);
    if (!job) return null;
    // Only retry terminal-state jobs — re-enqueuing a queued/running job
    // would just race the existing run.
    if (job.status !== 'failed' && job.status !== 'cancelled' && job.status !== 'completed') {
        return null;
    }
    if (!handlers.has(job.type)) return null;
    const params = safeJson(job.params_json);
    return enqueue(job.type, params, {
        key_arg1: job.key_arg1,
        key_arg2: job.key_arg2
    });
}

export function cancel(jobId: number): boolean {
    const job = getJob(jobId);
    if (!job) return false;
    if (job.status === 'running' || job.status === 'queued') {
        cancelFlags.set(jobId, true);
        if (job.status === 'queued') {
            // For queued jobs we can finalize immediately.
            finishJobStmt.run('cancelled', null, jobId);
            // Remove from its type queue.
            const q = queues.get(job.type);
            if (q) queues.set(job.type, q.filter((id) => id !== jobId));
        }
        return true;
    }
    return false;
}

export function getJob(id: number): JobRow | null {
    return (getJobStmt.get(id) as JobRow | undefined) ?? null;
}

export function listJobs(limit = 100): JobRow[] {
    return listJobsStmt.all(limit) as JobRow[];
}

/** Most-recent run per (type, key_arg1, key_arg2). This is what the Jobs
 * page's Latest tab shows — older runs of the same logical job are
 * available via `listArchiveJobs`. */
export function listLatestJobs(limit = 200): JobRow[] {
    return listLatestStmt.all(limit) as JobRow[];
}

/** Every run that is NOT the latest for its (type, key_arg1, key_arg2)
 * bucket. Powers the Archives tab. */
export function listArchiveJobs(limit = 200): JobRow[] {
    return listArchiveStmt.all(limit) as JobRow[];
}

export function listActiveJobs(): JobRow[] {
    return listActiveStmt.all() as JobRow[];
}

export function getJobLogs(id: number, limit = 500): JobLogRow[] {
    return listLogsStmt.all(id, limit) as JobLogRow[];
}

// ---- Internals ------------------------------------------------------------
async function drain(type: string): Promise<void> {
    if (running.get(type)) return;
    running.set(type, true);
    try {
        while (true) {
            const q = queues.get(type) ?? [];
            const next = q.shift();
            queues.set(type, q);
            if (next == null) return;

            const job = getJob(next);
            if (!job) continue;
            if (job.status !== 'queued') continue; // cancelled while queued

            await execute(job);
        }
    } finally {
        running.set(type, false);
    }
}

async function execute(job: JobRow): Promise<void> {
    const handler = handlers.get(job.type);
    if (!handler) {
        finishJobStmt.run('failed', `No handler for type "${job.type}"`, job.id);
        return;
    }
    // Default the pid to this Node process — covers every in-process
    // handler. Subprocess handlers overwrite it via setPid() after spawn.
    startJobStmt.run(process.pid, job.id);

    const params = safeJson(job.params_json);
    const ctx: JobContext = {
        job_id: job.id,
        params,
        shouldCancel: () => cancelFlags.get(job.id) === true,
        progress: (current, total, label) => {
            // Debounce DB writes — face detection can fire dozens per second.
            const now = Date.now();
            const last = lastProgressWrite.get(job.id) ?? 0;
            if (now - last < PROGRESS_MIN_INTERVAL_MS) return;
            lastProgressWrite.set(job.id, now);
            updateProgressStmt.run(
                current,
                total ?? null,
                label ?? null,
                job.id
            );
        },
        log: (level, message) => {
            insertLogStmt.run(job.id, level, message.slice(0, 4000));
        },
        metrics: (snapshot, opts) => {
            const now = Date.now();
            if (!opts?.force) {
                const last = lastMetricsWrite.get(job.id) ?? 0;
                if (now - last < METRICS_MIN_INTERVAL_MS) return;
            }
            lastMetricsWrite.set(job.id, now);
            try {
                updateMetricsStmt.run(JSON.stringify(snapshot), job.id);
            } catch {
                // ignore — never let a metrics write break the handler
            }
        }
    };

    try {
        await handler(ctx);
        if (cancelFlags.get(job.id)) {
            // Handler honored the cancel and returned cleanly.
            finishJobStmt.run('cancelled', null, job.id);
            ctx.log('info', 'cancelled by user');
        } else {
            finishJobStmt.run('completed', null, job.id);
        }
    } catch (e) {
        const msg = (e as Error).message ?? String(e);
        finishJobStmt.run('failed', msg, job.id);
        try {
            insertLogStmt.run(job.id, 'error', msg.slice(0, 4000));
        } catch {
            // ignore
        }
    } finally {
        cancelFlags.delete(job.id);
        lastProgressWrite.delete(job.id);
        lastMetricsWrite.delete(job.id);
    }
}

function safeJson(s: string): Record<string, unknown> {
    try {
        const o = JSON.parse(s);
        if (o && typeof o === 'object' && !Array.isArray(o)) {
            return o as Record<string, unknown>;
        }
    } catch {
        // fall through
    }
    return {};
}

// ---- Boot recovery -------------------------------------------------------
// Walk every 'running' job and decide its fate based on the recorded pid.
// Live pids stay 'running' — the orphan reaper will finalize them when
// the child eventually exits (and inherit the test_run row's terminal
// status for grid-test-run, see reapOrphans below).
{
    const rows = listRunningForRecoveryStmt.all() as { id: number; pid: number | null }[];
    let reapedDead = 0;
    let leftAlive = 0;
    for (const r of rows) {
        if (r.pid == null || r.pid === process.pid || !isProcessAlive(r.pid)) {
            failJobBootStmt.run('Server restarted while job was running', r.id);
            reapedDead++;
        } else {
            leftAlive++;
        }
    }
    if (reapedDead > 0 || leftAlive > 0) {
        console.log(
            `[boot] job recovery: ${reapedDead} marked failed, ${leftAlive} left running (pid still alive)`
        );
    }
}
// For test_runs: a Python subprocess that owned the row may still be
// alive (see above), in which case it'll finalize the row itself. We
// only reap test_run rows that have no corresponding live job — handled
// inside resurrectStaleRunsOnBoot.
{
    const n = resurrectStaleRunsOnBoot();
    if (n > 0) {
        console.log(`[boot] reaped ${n} stale 'running' test_run(s) from previous session`);
    }
}

// ---- Orphan reaper -------------------------------------------------------
// Periodically check every 'running' job whose pid we know about. If the
// pid belongs to a process that no longer exists (kill -0 → ESRCH), mark
// the job failed. Three orphan flavors this catches:
//
//   1. Subprocess (grid-test-run): child Python killed externally
//      while the handler is still awaiting child.once('exit') for some
//      reason (in normal flow Node would notice on its own — this is a
//      safety net for hangs).
//   2. Stale pid from before a Node restart: resurrectStmt already
//      sweeps these at boot, but the reaper covers any narrow window
//      where a handler hadn't yet recorded its pid when the crash
//      happened.
//   3. Subprocess that crashed without our exit listener firing
//      (extremely rare; covered for free).
//
// For in-process handlers, the pid IS this Node process, so the check
// is a no-op while we're alive — that's intentional. If Node itself
// crashes, the resurrect-at-boot path takes over.
const ORPHAN_POLL_INTERVAL_MS = 10_000;
function isProcessAlive(pid: number): boolean {
    try {
        // Signal 0 doesn't deliver anything — it just probes for the
        // process's existence (and our permission to signal it). Throws
        // ESRCH if the pid is unused; we treat any throw other than EPERM
        // as "dead". EPERM means the process exists but is owned by
        // another user — still alive.
        process.kill(pid, 0);
        return true;
    } catch (e) {
        const code = (e as NodeJS.ErrnoException).code;
        if (code === 'EPERM') return true;
        return false;
    }
}

function reapOrphans(): void {
    try {
        const rows = listRunningWithPidStmt.all() as {
            id: number;
            pid: number;
            type: string;
        }[];
        for (const r of rows) {
            if (r.pid === process.pid) continue; // in-process — we're it
            if (isProcessAlive(r.pid)) continue;
            // Race window: a subprocess that just exited may still be in
            // the middle of the handler's cleanup (between exit event and
            // finishJobStmt). Re-read the row to confirm it's still
            // 'running' before clobbering it.
            const fresh = getJob(r.id);
            if (!fresh || fresh.status !== 'running') continue;

            // For grid-test-run: the Python subprocess writes its own
            // terminal status into test_runs before exiting. Inherit it
            // so a successful run that survived a Node restart doesn't
            // come back to the user as "failed".
            let inheritedStatus: 'completed' | 'failed' | 'cancelled' = 'failed';
            let inheritedError = `Orphan: pid ${r.pid} no longer exists`;
            let tid: number | null = null;
            if (r.type === 'grid-test-run') {
                try {
                    const params = safeJson(fresh.params_json);
                    const t = Number((params as { test_id?: unknown }).test_id);
                    if (Number.isFinite(t) && t > 0) tid = t;
                } catch {
                    // params corrupt — fall through with default failure
                }
                if (tid != null) {
                    const tr = latestTestRunStatus(tid);
                    if (tr && (tr.status === 'completed' || tr.status === 'cancelled')) {
                        inheritedStatus = tr.status;
                        inheritedError = tr.error ?? '';
                    } else if (tr && tr.status === 'failed') {
                        inheritedStatus = 'failed';
                        inheritedError = tr.error ?? `Run failed (pid ${r.pid} gone)`;
                    }
                }
            }

            finishJobStmt.run(inheritedStatus, inheritedError, r.id);
            try {
                insertLogStmt.run(
                    r.id,
                    inheritedStatus === 'completed' ? 'info' : 'error',
                    inheritedStatus === 'completed'
                        ? `Orphan reaper: pid ${r.pid} is gone; companion test run already completed — inherited 'completed'`
                        : `Orphan reaper: pid ${r.pid} is gone — marked ${inheritedStatus}`
                );
            } catch {
                // ignore log write errors
            }
            // Companion reconciliation: if no terminal test_runs row
            // existed yet (Python died mid-run without writing), force-
            // finalize the row so the dashboard doesn't show a forever-
            // "running" test. When the row already finalized itself this
            // is a no-op (status filter inside finalizeStaleRunsForTest).
            if (r.type === 'grid-test-run' && tid != null && inheritedStatus !== 'completed') {
                try {
                    const n = finalizeStaleRunsForTest(
                        tid,
                        inheritedStatus,
                        `Orphan: pid ${r.pid} no longer exists`
                    );
                    if (n > 0) {
                        insertLogStmt.run(
                            r.id,
                            'warn',
                            `Orphan reaper: finalized ${n} stale test_run row(s) for test #${tid}`
                        );
                    }
                } catch {
                    // never let the reaper crash the runner
                }
            }
        }
    } catch {
        // never let the reaper crash the runner
    }
}

// Run once at module load to catch anything the boot resurrectStmt
// missed (e.g. a row that became orphaned between the resurrect call and
// the first poll tick), then on a slow cadence.
reapOrphans();
setInterval(reapOrphans, ORPHAN_POLL_INTERVAL_MS).unref();
