// Job handler: spawn `python -m glt --rescore --test-id N --db ... --tests-root ...`
// to re-run face scoring on the latest test_run for a given test, without
// regenerating any images. Mirrors grid-test-run's spawn/PID/stream pattern.
import { spawn, type ChildProcess } from 'node:child_process';

import { db, DB_PATH } from '../../db';
import { getSettings, gltRoot } from '../../settings';
import { registerHandler, setPid, type JobContext, type JobHandler } from '../runner';

interface Params {
    test_id: number;
    /** Optional: rescore a specific run instead of the latest. */
    run_id?: number;
    /** Optional: pass --force to recompute every cell instead of only NULL ones. */
    force?: boolean;
}

const testNameStmt = db.prepare<[number]>('SELECT name FROM tests WHERE id = ?');

const handler: JobHandler = async (ctx: JobContext) => {
    const params = ctx.params as unknown as Params;
    if (!params.test_id) throw new Error('Missing test_id');
    const testRow = testNameStmt.get(params.test_id) as { name: string } | undefined;
    if (!testRow) throw new Error(`Test #${params.test_id} not found`);

    const settings = getSettings();
    if (!settings.python_bin) throw new Error('settings.python_bin not set');
    if (!settings.tests_root) throw new Error('settings.tests_root not set');
    const glt_root = gltRoot();

    const args = [
        '-u',
        '-m', 'glt',
        '--rescore',
        '--test-id', String(params.test_id),
        '--db', DB_PATH,
        '--tests-root', settings.tests_root,
    ];
    if (params.run_id != null) args.push('--run-id', String(params.run_id));
    if (params.force) args.push('--force-rescore');

    ctx.log('info', `Spawning ${settings.python_bin} ${args.join(' ')} (cwd=${glt_root})`);

    const child: ChildProcess = spawn(settings.python_bin, args, {
        cwd: glt_root,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    if (child.pid != null) setPid(ctx.job_id, child.pid);

    // Same streamLines as grid-test-run, split on \n and strip ANSI/\r.
    const ANSI_RE = /\x1b\[[0-9;]*m/g;
    function streamLines(stream: NodeJS.ReadableStream, level: 'info' | 'warn' | 'error'): void {
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

    // Cancel watcher — same SIGTERM-then-SIGKILL escalation as grid-test-run.
    let cancelTimer: NodeJS.Timeout | null = null;
    let cancelled = false;
    cancelTimer = setInterval(() => {
        if (ctx.shouldCancel() && !cancelled && !child.killed) {
            cancelled = true;
            ctx.log('warn', 'cancellation requested — SIGTERM');
            try { child.kill('SIGTERM'); } catch (e) {
                ctx.log('error', `SIGTERM failed: ${(e as Error).message}`);
            }
            setTimeout(() => {
                if (!child.killed && child.exitCode === null) {
                    ctx.log('warn', 'still alive after SIGTERM — SIGKILL');
                    try { child.kill('SIGKILL'); } catch { /* ignore */ }
                }
            }, 10_000);
        }
    }, 500);

    try {
        const exitCode: number | null = await new Promise((resolve, reject) => {
            child.once('exit', (code) => resolve(code));
            child.once('error', (err) => reject(err));
        });
        if (exitCode === 0) {
            ctx.log('info', 'rescore exited cleanly (rc=0)');
        } else if (cancelled) {
            throw new Error('cancelled by user');
        } else {
            throw new Error(`rescore subprocess exited with code ${exitCode}`);
        }
    } finally {
        if (cancelTimer != null) clearInterval(cancelTimer);
    }
};

registerHandler('test-rescore', handler);
