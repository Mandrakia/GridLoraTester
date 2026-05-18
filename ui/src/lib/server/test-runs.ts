// Helpers for `test_runs` finalization. Python (`glt --grid`) creates a
// row with status='running' on start and flips it to completed/failed/
// cancelled on normal exit. When the subprocess crashes hard (SIGKILL,
// OOM, segfault, parent Node died) the flip never happens — the dashboard
// then shows a stale "running" test indefinitely. These helpers let the
// JOB layer (which DOES know when a process is dead, via the orphan
// reaper) reconcile the test_runs row.
import { db } from './db';

// Mark every running test_runs row for `test_id` as finalized with the
// given status + error. Returns the count of rows reconciled.
const finalizeForTestStmt = db.prepare(`
    UPDATE test_runs
       SET status      = @status,
           error       = @error,
           finished_at = datetime('now')
     WHERE test_id = @test_id
       AND status  = 'running'
`);

export function finalizeStaleRunsForTest(
    test_id: number,
    status: 'failed' | 'cancelled',
    error: string,
): number {
    const info = finalizeForTestStmt.run({ test_id, status, error });
    return info.changes ?? 0;
}

// Sweep across every test on Node startup. We can't blindly fail every
// 'running' row: a previous-session grid-test-run might still have its
// Python subprocess alive (Vite HMR / quick npm-run-dev restart) and
// that subprocess writes its own terminal status to test_runs when it
// exits. We only reap rows whose owning Python process is gone.
const listRunningTestRunsStmt = db.prepare(
    "SELECT id, test_id FROM test_runs WHERE status = 'running'"
);
const listRunningGridJobsStmt = db.prepare(
    "SELECT pid, params_json FROM jobs WHERE status = 'running' AND type = 'grid-test-run' AND pid IS NOT NULL"
);
const reapTestRunStmt = db.prepare(`
    UPDATE test_runs
       SET status      = 'failed',
           error       = 'Dashboard restarted while run was active',
           finished_at = datetime('now')
     WHERE id = ?
`);

function isPidAlive(pid: number): boolean {
    try {
        process.kill(pid, 0);
        return true;
    } catch (e) {
        return (e as NodeJS.ErrnoException).code === 'EPERM';
    }
}

const latestRunForTestStmt = db.prepare(
    "SELECT status, error FROM test_runs WHERE test_id = ? ORDER BY id DESC LIMIT 1"
);

/** Read the most recent test_run row for a given test. Used by the
 * orphan reaper to inherit a final status when the Python subprocess
 * finished (writing test_runs) but the Node-side handler is gone. */
export function latestTestRunStatus(
    testId: number
): { status: string; error: string | null } | null {
    const row = latestRunForTestStmt.get(testId) as
        | { status: string; error: string | null }
        | undefined;
    return row ?? null;
}

export function resurrectStaleRunsOnBoot(): number {
    // Resolve which test_ids currently have a live Python child. Those
    // test_runs rows stay 'running' — the Python process will write the
    // final status itself.
    const liveTestIds = new Set<number>();
    const jobs = listRunningGridJobsStmt.all() as { pid: number; params_json: string }[];
    for (const j of jobs) {
        if (!isPidAlive(j.pid)) continue;
        try {
            const p = JSON.parse(j.params_json) as { test_id?: unknown };
            const tid = Number(p.test_id);
            if (Number.isFinite(tid) && tid > 0) liveTestIds.add(tid);
        } catch {
            // ignore corrupt params; we'll just reap this row
        }
    }

    const runs = listRunningTestRunsStmt.all() as { id: number; test_id: number }[];
    let reaped = 0;
    for (const r of runs) {
        if (liveTestIds.has(r.test_id)) continue;
        reapTestRunStmt.run(r.id);
        reaped++;
    }
    return reaped;
}
