// Job handler: spawn `python -m glt --grid --test-id N --db ... --tests-root ...`
// as a subprocess, pipe stdout/stderr into job_logs, and poll the
// test_run_cells table for live progress (the Python side increments
// cells as images land; the dashboard just watches the count).
//
// One handler per `grid-test-run` job → sequential per-type ensures we
// don't double-occupy the GPU. Multiple queued runs serialize naturally.
import { spawn, type ChildProcess } from 'node:child_process';

import { db, DB_PATH } from '../../db';
import { getSettings, gltRoot } from '../../settings';
import { finalizeStaleRunsForTest } from '../../test-runs';
import { registerHandler, setPid, type JobContext, type JobHandler } from '../runner';

interface Params {
    test_id: number;
    /** Optional overrides — when absent, the Python script falls back to
     * the test row's stored values. */
    trigger?: string;
    resolution?: string;
    seed?: number;
}

const cellsAggStmt = db.prepare<[number, number]>(`
    SELECT COUNT(*) AS total,
           SUM(CASE WHEN image_filename IS NOT NULL THEN 1 ELSE 0 END) AS done
      FROM test_run_cells trc
      JOIN test_runs tr ON tr.id = trc.run_id
     WHERE tr.test_id = ?
       AND tr.id = (SELECT MAX(id) FROM test_runs WHERE test_id = ?)
`);

const latestRunStmt = db.prepare<[number]>(`
    SELECT id, status, error FROM test_runs
     WHERE test_id = ?
     ORDER BY id DESC
     LIMIT 1
`);

const testNameStmt = db.prepare<[number]>(
    'SELECT name FROM tests WHERE id = ?'
);

const handler: JobHandler = async (ctx: JobContext) => {
    const params = ctx.params as unknown as Params;
    if (!params.test_id) throw new Error('Missing test_id');
    const testRow = testNameStmt.get(params.test_id) as { name: string } | undefined;
    if (!testRow) throw new Error(`Test #${params.test_id} not found`);

    const settings = getSettings();
    if (!settings.python_bin) throw new Error('settings.python_bin not set');
    if (!settings.tests_root) throw new Error('settings.tests_root not set');
    const glt_root = gltRoot();

    // `-u` forces stdout/stderr to be unbuffered. Without it, Python block-
    // buffers stdout when stdio is a pipe (~4-8 KB chunks) — log lines pile
    // up invisible to us for minutes during quiet phases of a run, then
    // arrive in a burst, looking exactly like a hung handler from /jobs.
    const args = [
        '-u',
        '-m', 'glt',
        '--grid',
        '--test-id', String(params.test_id),
        '--db', DB_PATH,
        '--tests-root', settings.tests_root,
    ];
    if (params.trigger !== undefined) {
        args.push('--trigger', params.trigger);
    }
    if (params.resolution) {
        args.push('--resolution', params.resolution);
    }
    if (params.seed !== undefined) {
        args.push('-s', String(params.seed));
    }

    ctx.log(
        'info',
        `Spawning ${settings.python_bin} ${args.join(' ')} (cwd=${glt_root})`
    );

    const child: ChildProcess = spawn(settings.python_bin, args, {
        cwd: glt_root,
        // The Python side prints to stdout/stderr. We pipe both so we can
        // surface progress in the dashboard's log pane.
        stdio: ['ignore', 'pipe', 'pipe'],
        // PYTHONUNBUFFERED belt-and-suspenders to the `-u` CLI flag.
        // Either alone would do; both guarantees stdout is unbuffered even
        // if a downstream Python that ignores `-u` is in the loop.
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    // Record the child pid on the job row so the orphan reaper can detect
    // a Python process killed externally. execute() already set the row's
    // pid to this Node's pid as a default — overwrite with the child's.
    if (child.pid != null) {
        setPid(ctx.job_id, child.pid);
    }

    // Buffer line-fragmented output. Python's print uses '\n'; some libs
    // emit '\r\n' or CR-overwrites (\r). We split on either newline and
    // strip ANSI/CR remainders so logs read cleanly in the UI.
    const ANSI_RE = /\x1b\[[0-9;]*m/g;
    function streamLines(
        stream: NodeJS.ReadableStream,
        level: 'info' | 'warn' | 'error',
    ): void {
        let buf = '';
        stream.setEncoding('utf-8');
        stream.on('data', (chunk: string) => {
            buf += chunk;
            let idx: number;
            while ((idx = buf.indexOf('\n')) >= 0) {
                let line = buf.slice(0, idx);
                buf = buf.slice(idx + 1);
                line = line.replace(/\r/g, '').replace(ANSI_RE, '').trimEnd();
                if (!line) continue;
                // Heuristic: lines starting with [error]/[fatal]/[warn] get
                // promoted so the Jobs UI's amber/red pill activates.
                let lvl = level;
                if (/^\[(error|fatal)\]/i.test(line)) lvl = 'error';
                else if (/^\[warn\]/i.test(line)) lvl = 'warn';
                ctx.log(lvl, line);
            }
        });
        stream.on('end', () => {
            const tail = buf.replace(/\r/g, '').replace(ANSI_RE, '').trimEnd();
            if (tail) ctx.log(level, tail);
        });
    }

    if (child.stdout) streamLines(child.stdout, 'info');
    if (child.stderr) streamLines(child.stderr, 'warn');

    // Progress poller — reads test_run_cells.done / total for the latest
    // run of this test and feeds ctx.progress so the dashboard gets a
    // live percentage. The Python side writes incrementally so the count
    // climbs as cells fill in.
    let pollerTimer: NodeJS.Timeout | null = null;
    function startPoller(): void {
        if (pollerTimer != null) return;
        pollerTimer = setInterval(() => {
            try {
                const agg = cellsAggStmt.get(params.test_id, params.test_id) as
                    | { total: number | null; done: number | null }
                    | undefined;
                if (agg && agg.total != null) {
                    const total = Number(agg.total);
                    const done = Number(agg.done ?? 0);
                    ctx.progress(done, total, `${done}/${total} cells`);
                }
            } catch {
                // ignore — DB read can race with WAL checkpoints
            }
        }, 1000);
    }
    function stopPoller(): void {
        if (pollerTimer != null) {
            clearInterval(pollerTimer);
            pollerTimer = null;
        }
    }
    startPoller();

    // Cancel watcher — when the user cancels the job, SIGTERM the
    // subprocess. Python's try/except in grid.py catches and finalizes
    // the test_runs row as 'cancelled'.
    let cancelTimer: NodeJS.Timeout | null = null;
    let cancelled = false;
    function watchForCancel(): void {
        cancelTimer = setInterval(() => {
            if (ctx.shouldCancel() && !cancelled && !child.killed) {
                cancelled = true;
                ctx.log('warn', 'cancellation requested — sending SIGTERM to subprocess');
                try {
                    child.kill('SIGTERM');
                } catch (e) {
                    ctx.log('error', `failed to SIGTERM: ${(e as Error).message}`);
                }
                // Escalate to SIGKILL after 10s if still alive.
                setTimeout(() => {
                    if (!child.killed && child.exitCode === null) {
                        ctx.log('warn', 'subprocess still alive after SIGTERM — SIGKILL');
                        try { child.kill('SIGKILL'); } catch { /* ignore */ }
                    }
                }, 10_000);
            }
        }, 500);
    }
    watchForCancel();

    try {
        // Wait for the subprocess to exit.
        const exitCode: number | null = await new Promise((resolve, reject) => {
            child.once('exit', (code) => resolve(code));
            child.once('error', (err) => reject(err));
        });

        // Final progress snapshot — the poller might have missed the last
        // increment between its tick and exit.
        try {
            const agg = cellsAggStmt.get(params.test_id, params.test_id) as
                | { total: number | null; done: number | null }
                | undefined;
            if (agg && agg.total != null) {
                ctx.progress(
                    Number(agg.done ?? 0),
                    Number(agg.total),
                    `${Number(agg.done ?? 0)}/${Number(agg.total)} cells`,
                );
            }
        } catch {
            // ignore
        }

        // The Python side already marked the test_runs row on a normal
        // exit; on abnormal exit (crash, SIGKILL, OOM) it may not have.
        // We force-finalize any stale 'running' row here so the
        // dashboard never shows a forever-running test for this id.
        const latest = latestRunStmt.get(params.test_id) as
            | { id: number; status: string; error: string | null }
            | undefined;

        if (exitCode === 0) {
            ctx.log('info', 'subprocess exited cleanly (rc=0)');
            if (latest && latest.status === 'running') {
                // Python should have flipped it; treat as a Python bug.
                finalizeStaleRunsForTest(
                    params.test_id, 'failed',
                    'subprocess exited 0 but test_runs row was still running'
                );
                throw new Error(
                    'subprocess exited 0 but test_runs row still status=running'
                );
            }
            if (latest && latest.status === 'failed') {
                throw new Error(
                    `test run #${latest.id} marked failed: ${latest.error ?? 'unknown'}`
                );
            }
        } else {
            // Non-zero exit / signal: the subprocess didn't reach its
            // finally block. Finalize stale runs ourselves so the UI
            // doesn't lie.
            const finalStatus = cancelled ? 'cancelled' : 'failed';
            const msg = cancelled
                ? 'cancelled by user (job)'
                : `subprocess exited with code ${exitCode}`;
            const n = finalizeStaleRunsForTest(params.test_id, finalStatus, msg);
            if (n > 0) {
                ctx.log('warn', `force-finalized ${n} stale 'running' test_run(s) (${msg})`);
            }
            throw new Error(msg);
        }
    } finally {
        stopPoller();
        if (cancelTimer != null) clearInterval(cancelTimer);
    }
};

registerHandler('grid-test-run', handler);
