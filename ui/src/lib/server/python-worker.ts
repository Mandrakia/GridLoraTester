// Long-lived Python worker. The dashboard talks to it via HTTP/JSON on
// 127.0.0.1:<port>. We spawn one on first demand, parse the single
// `READY <port>` line off its stdout, and from then on never go near
// stdout again — every request goes through `fetch`.
//
// Lifecycle:
//   - lazy spawn, cached singleton per Node process
//   - if the child dies (crash / SIGKILL), the next request re-spawns it
//   - on Node process exit, we POST /shutdown and terminate the child
import { spawn, type ChildProcessByStdio } from 'node:child_process';
import type { Readable } from 'node:stream';

import { getSettings } from './settings';

// stdio = ['ignore', 'pipe', 'pipe'] → stdin is null, stdout/stderr are
// readable. Reflect that exactly so neither end of the pipeline needs a cast.
type WorkerChild = ChildProcessByStdio<null, Readable, Readable>;

interface WorkerHandle {
    child: WorkerChild;
    port: number;
    /** Resolves when the READY line is parsed. */
    ready: Promise<void>;
}

let cached: WorkerHandle | null = null;
let exitHookInstalled = false;

function installExitHook() {
    if (exitHookInstalled) return;
    exitHookInstalled = true;
    const stop = () => {
        if (cached) {
            try {
                cached.child.kill('SIGTERM');
            } catch {
                // ignore
            }
            cached = null;
        }
    };
    process.once('exit', stop);
    process.once('SIGINT', () => {
        stop();
        process.exit(130);
    });
    process.once('SIGTERM', () => {
        stop();
        process.exit(143);
    });
}

function spawnWorker(): WorkerHandle {
    const { python_bin, glt_root } = getSettings();
    if (!python_bin) throw new Error('python_bin is not set — configure it in Settings.');
    if (!glt_root) throw new Error('glt_root is not set — configure it in Settings.');

    const child = spawn(python_bin, ['-m', 'glt', '--serve'], {
        cwd: glt_root,
        stdio: ['ignore', 'pipe', 'pipe']
    });

    let portResolved: number | null = null;
    let stdoutBuf = '';
    let stderrTail = '';

    const ready = new Promise<void>((resolve, reject) => {
        const onStdout = (chunk: Buffer) => {
            stdoutBuf += chunk.toString('utf-8');
            const newlineIdx = stdoutBuf.indexOf('\n');
            if (newlineIdx >= 0) {
                const first = stdoutBuf.slice(0, newlineIdx).trim();
                const m = /^READY\s+(\d+)$/.exec(first);
                if (m) {
                    portResolved = parseInt(m[1], 10);
                    child.stdout.off('data', onStdout);
                    resolve();
                } else {
                    reject(
                        new Error(
                            `Expected "READY <port>" from worker, got: ${first || '<empty>'}`
                        )
                    );
                }
            }
        };
        child.stdout.on('data', onStdout);
        child.on('error', (err) => reject(err));
        child.on('exit', (code, signal) => {
            // If we haven't resolved yet, the worker died before announcing.
            if (portResolved == null) {
                reject(
                    new Error(
                        `Worker exited before ready (code=${code} signal=${signal}).\n` +
                            `stderr tail:\n${stderrTail.slice(-2000)}`
                    )
                );
            }
        });
    });

    child.stderr.on('data', (chunk) => {
        const s = chunk.toString('utf-8');
        // Mirror to our own stderr so the user sees what the worker is doing
        // during dev. Capped to avoid log explosions.
        process.stderr.write(`[glt-worker] ${s}`);
        stderrTail = (stderrTail + s).slice(-4000);
    });

    child.on('exit', () => {
        // Clear the cache so the next request spawns a fresh worker.
        if (cached?.child === child) cached = null;
    });

    return {
        child,
        // Filled in once the READY line is parsed; `request()` awaits `ready`
        // before reading this.
        get port() {
            return portResolved!;
        },
        ready
    };
}

async function getWorker(): Promise<WorkerHandle> {
    installExitHook();
    if (cached) return cached;
    cached = spawnWorker();
    try {
        await cached.ready;
    } catch (e) {
        cached = null;
        throw e;
    }
    return cached;
}

/** Send a JSON request to the Python worker. Spawns the worker on first call. */
export async function request<T = unknown>(
    path: string,
    body: unknown,
    init: { method?: 'GET' | 'POST'; timeoutMs?: number } = {}
): Promise<T> {
    const worker = await getWorker();
    const method = init.method ?? 'POST';
    const ac = new AbortController();
    const timeout = init.timeoutMs ?? 5 * 60 * 1000;
    const timer = setTimeout(() => ac.abort(), timeout);
    try {
        const res = await fetch(`http://127.0.0.1:${worker.port}${path}`, {
            method,
            headers: { 'content-type': 'application/json' },
            body: method === 'GET' ? undefined : JSON.stringify(body ?? {}),
            signal: ac.signal
        });
        const text = await res.text();
        let parsed: unknown = null;
        try {
            parsed = text ? JSON.parse(text) : null;
        } catch {
            // body isn't JSON — fall through with `parsed = null`
        }
        if (!res.ok) {
            const err =
                (parsed && typeof parsed === 'object' && 'error' in parsed
                    ? String((parsed as { error: unknown }).error)
                    : text) || `HTTP ${res.status}`;
            throw new Error(`Worker ${path} failed: ${err}`);
        }
        return parsed as T;
    } finally {
        clearTimeout(timer);
    }
}

/** Send raw bytes to the Python worker. Used for /detect-faces-blob so we
 * don't have to base64-encode every connector picture. */
export async function requestBytes<T = unknown>(
    path: string,
    body: Buffer | Uint8Array,
    init: { timeoutMs?: number; headers?: Record<string, string> } = {}
): Promise<T> {
    const worker = await getWorker();
    const ac = new AbortController();
    const timeout = init.timeoutMs ?? 60 * 1000;
    const timer = setTimeout(() => ac.abort(), timeout);
    try {
        const res = await fetch(`http://127.0.0.1:${worker.port}${path}`, {
            method: 'POST',
            headers: {
                'content-type': 'application/octet-stream',
                ...(init.headers ?? {})
            },
            body: body as unknown as BodyInit,
            signal: ac.signal
        });
        const text = await res.text();
        let parsed: unknown = null;
        try {
            parsed = text ? JSON.parse(text) : null;
        } catch {
            // not JSON
        }
        if (!res.ok) {
            const err =
                (parsed && typeof parsed === 'object' && 'error' in parsed
                    ? String((parsed as { error: unknown }).error)
                    : text) || `HTTP ${res.status}`;
            throw new Error(`Worker ${path} failed: ${err}`);
        }
        return parsed as T;
    } finally {
        clearTimeout(timer);
    }
}

/** Best-effort graceful shutdown. Used by tests / scripts; the exit hook
 * also covers normal process termination. */
export async function shutdownWorker(): Promise<void> {
    if (!cached) return;
    try {
        await request('/shutdown', {}, { timeoutMs: 5000 });
    } catch {
        // ignore — we'll SIGTERM below
    }
    if (cached) {
        try {
            cached.child.kill('SIGTERM');
        } catch {
            // ignore
        }
        cached = null;
    }
}
