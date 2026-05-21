// Pipe a child process's stdout/stderr into a job's log, one clean line at a
// time. Python `print` uses '\n'; some libs emit '\r\n' or CR-overwrites ('\r',
// e.g. tqdm progress bars). We buffer until a newline, strip ANSI + stray CRs,
// and drop empty lines — so logs read cleanly in the Jobs UI instead of the
// per-chunk garble you get from writing each raw read straight through.
//
// Lines beginning with [error]/[fatal] or [warn] are promoted so the Jobs UI's
// red/amber pill activates regardless of the stream's base level.

const ANSI_RE = /\x1b\[[0-9;]*m/g;

export type LogLevel = 'info' | 'warn' | 'error';
export type LogFn = (level: LogLevel, message: string) => void;

export function streamLogLines(
    stream: NodeJS.ReadableStream,
    log: LogFn,
    baseLevel: LogLevel = 'info'
): void {
    let buf = '';
    stream.setEncoding('utf-8');
    stream.on('data', (chunk: string) => {
        buf += chunk;
        let idx: number;
        while ((idx = buf.indexOf('\n')) >= 0) {
            const line = clean(buf.slice(0, idx));
            buf = buf.slice(idx + 1);
            if (line) log(levelFor(line, baseLevel), line);
        }
    });
    stream.on('end', () => {
        const tail = clean(buf);
        if (tail) log(baseLevel, tail);
    });
}

function clean(line: string): string {
    return line.replace(/\r/g, '').replace(ANSI_RE, '').trimEnd();
}

function levelFor(line: string, fallback: LogLevel): LogLevel {
    if (/^\[(error|fatal)\]/i.test(line)) return 'error';
    if (/^\[warn(ing)?\]/i.test(line)) return 'warn';
    return fallback;
}
