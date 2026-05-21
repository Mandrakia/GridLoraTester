// One short-lived `python -m glt --detect-stream` subprocess, owned by a
// single job, replacing the old long-lived shared HTTP worker. The dashboard
// owns connector credentials and downloads the bytes; this streams those bytes
// to Python over a framed stdin/stdout pipe (no HTTP, no shared state) and the
// subprocess's stderr flows straight into the job's own log.
//
// Frame format (both directions): [4-byte big-endian length][payload].
// Request payload = raw image bytes; reply payload = UTF-8 JSON.
//
// Lifecycle is bounded by the job: spawn at job start, `close()` (EOF on
// stdin) at job end. We deliberately do NOT register the child's pid with the
// job runner — unlike grid-test-run's child (which owns its job and outlives a
// Node restart), this is a helper of an in-process handler. If it dies the
// handler sees the rejection synchronously and fails the job; letting the
// orphan reaper watch the child would only race the handler's own completion.
import { spawn, type ChildProcess } from 'node:child_process';

import { streamLogLines, type LogFn } from './jobs/log-stream';
import { getSettings, gltRoot } from './settings';

export interface FaceStreamResult {
    image_width: number | null;
    image_height: number | null;
    faces: {
        face_index: number;
        bbox: number[];
        det_score: number | null;
        embedding_b64: string;
        pitch: number | null;
        yaw: number | null;
        roll: number | null;
    }[];
    timing_ms?: { decode: number; detect: number; total: number };
    /** Set when the Python side failed to process this one image — the
     * caller should treat it as a per-image failure, not a fatal error. */
    error?: string;
}

export interface FaceStream {
    /** Detect faces in one image. Resolves with the result frame, or rejects
     * if the subprocess died. A per-image decode/detect failure resolves with
     * `result.error` set rather than rejecting. */
    detect(bytes: Buffer | Uint8Array): Promise<FaceStreamResult>;
    /** Close stdin and wait for the subprocess to exit. Idempotent. */
    close(): Promise<void>;
}

interface SpawnOpts {
    /** Where the subprocess's stderr lines land (the job's log). */
    log: LogFn;
    /** ONNX CUDA arena cap in GiB. <= 0 / undefined = no cap. */
    gpuMemGb?: number;
    /** Optional config.json overriding face_recognition settings. */
    configPath?: string;
}

export function spawnFaceStream(opts: SpawnOpts): FaceStream {
    const settings = getSettings();
    if (!settings.python_bin) throw new Error('settings.python_bin not set');

    const args = ['-u', '-m', 'glt', '--detect-stream'];
    if (opts.gpuMemGb && opts.gpuMemGb > 0) args.push('--gpu-mem-limit', String(opts.gpuMemGb));
    if (opts.configPath) args.push('--config', opts.configPath);

    const child: ChildProcess = spawn(settings.python_bin, args, {
        cwd: gltRoot(),
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    // stderr → job log, line-buffered + de-garbled.
    if (child.stderr) streamLogLines(child.stderr, opts.log, 'info');

    // Pending request resolvers, FIFO. The pool drives one detect() at a time,
    // but a FIFO queue stays correct even if several are in flight: Python
    // replies in request order, so frame N answers request N.
    const pending: {
        resolve: (r: FaceStreamResult) => void;
        reject: (e: Error) => void;
    }[] = [];
    let exited = false;
    let exitErr: Error | null = null;

    // stdout frame parser.
    let buf: Buffer = Buffer.alloc(0);
    child.stdout?.on('data', (chunk: Buffer) => {
        buf = buf.length ? Buffer.concat([buf, chunk]) : chunk;
        while (buf.length >= 4) {
            const len = buf.readUInt32BE(0);
            if (buf.length < 4 + len) break;
            const payload = buf.subarray(4, 4 + len);
            buf = buf.subarray(4 + len);
            const waiter = pending.shift();
            if (!waiter) continue; // no one waiting — shouldn't happen
            try {
                waiter.resolve(JSON.parse(payload.toString('utf-8')) as FaceStreamResult);
            } catch (e) {
                waiter.reject(new Error(`face-stream: bad reply JSON: ${(e as Error).message}`));
            }
        }
    });

    function failAll(err: Error): void {
        exited = true;
        exitErr = err;
        while (pending.length) pending.shift()!.reject(err);
    }
    child.on('error', (e) => failAll(new Error(`face-stream spawn error: ${e.message}`)));
    child.on('exit', (code, signal) => {
        if (!exited) {
            failAll(new Error(`face-stream worker exited early (code=${code} signal=${signal})`));
        }
    });

    return {
        detect(bytes) {
            if (exited) return Promise.reject(exitErr ?? new Error('face-stream closed'));
            return new Promise<FaceStreamResult>((resolve, reject) => {
                pending.push({ resolve, reject });
                const body = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
                const header = Buffer.allocUnsafe(4);
                header.writeUInt32BE(body.length, 0);
                // One write keeps the frame contiguous on the wire.
                child.stdin?.write(Buffer.concat([header, body]));
            });
        },
        close() {
            return new Promise<void>((resolve) => {
                if (exited || !child.stdin) {
                    resolve();
                    return;
                }
                const done = () => {
                    clearTimeout(t);
                    resolve();
                };
                child.once('exit', done);
                // Don't wedge a job shutdown on a stuck child.
                const t = setTimeout(() => {
                    try {
                        child.kill('SIGTERM');
                    } catch {
                        /* ignore */
                    }
                }, 5000);
                try {
                    child.stdin.end();
                } catch {
                    done();
                }
            });
        }
    };
}
