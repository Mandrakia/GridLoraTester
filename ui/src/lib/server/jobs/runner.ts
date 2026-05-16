// In-memory job runner. DB is the source of truth (status + progress); the
// runtime here adds the "currently executing" handle so we can cooperatively
// cancel, batch progress writes, and avoid hammering SQLite on every tick.
//
// One worker per job type, sequential within a type. Different types can
// run in parallel — useful later when we add downloads + tests alongside
// face detection.
import { db } from '../db';

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
}

export type JobHandler = (ctx: JobContext) => Promise<void>;

// ---- DB statements --------------------------------------------------------
const insertJobStmt = db.prepare(
    "INSERT INTO jobs(type, params_json, status) VALUES(?, ?, 'queued') RETURNING id"
);
const updateProgressStmt = db.prepare(
    'UPDATE jobs SET progress_current = ?, progress_total = ?, current_label = ? WHERE id = ?'
);
const startJobStmt = db.prepare(
    "UPDATE jobs SET status = 'running', started_at = datetime('now') WHERE id = ?"
);
const finishJobStmt = db.prepare(
    "UPDATE jobs SET status = ?, error = ?, finished_at = datetime('now') WHERE id = ?"
);
const getJobStmt = db.prepare('SELECT * FROM jobs WHERE id = ?');
const listJobsStmt = db.prepare('SELECT * FROM jobs ORDER BY id DESC LIMIT ?');
const listActiveStmt = db.prepare(
    "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY id ASC"
);
const insertLogStmt = db.prepare(
    'INSERT INTO job_logs(job_id, level, message) VALUES(?, ?, ?)'
);
const listLogsStmt = db.prepare(
    'SELECT * FROM job_logs WHERE job_id = ? ORDER BY id ASC LIMIT ?'
);
// On startup we need to recover from a crash that left jobs marked running.
const resurrectStmt = db.prepare(
    "UPDATE jobs SET status = 'failed', error = 'Server restarted while job was running', finished_at = datetime('now') WHERE status IN ('running')"
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

export function registerHandler(type: string, handler: JobHandler): void {
    handlers.set(type, handler);
}

export function listHandlers(): string[] {
    return [...handlers.keys()];
}

export function enqueue(type: string, params: Record<string, unknown> = {}): number {
    if (!handlers.has(type)) {
        throw new Error(`No handler registered for job type "${type}".`);
    }
    const row = insertJobStmt.get(type, JSON.stringify(params)) as { id: number };
    const id = row.id;
    const q = queues.get(type) ?? [];
    q.push(id);
    queues.set(type, q);
    void drain(type);
    return id;
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
    startJobStmt.run(job.id);

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
// Anything left 'running' in DB across a restart is dead — mark it failed
// so the UI doesn't show forever-running ghosts.
resurrectStmt.run();
