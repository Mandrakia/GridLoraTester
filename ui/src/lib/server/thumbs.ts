// On-disk thumbnail cache backed by sharp. Browsers used to redraw full-res
// images on every layout reflow (slider on the dataset detail page),
// crippling frame rate. Serving cached small WebPs collapses paint cost to
// ~nothing and lets the browser CSS-scale the same image as the column count
// changes — no extra HTTP, no re-decode of multi-megapixel files.
//
// Cache key: SHA256(absPath + ':' + mtime + ':' + width). mtime invalidates
// the cache when the source image is replaced on disk; width keys per size.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

import sharp from 'sharp';

const CACHE_DIR = resolve(process.cwd(), 'data', 'thumbs');
mkdirSync(CACHE_DIR, { recursive: true });

const ALLOWED_WIDTHS = new Set([128, 256, 384, 512, 768, 1024]);
export const DEFAULT_THUMB_WIDTH = 512;

export function clampWidth(raw: number | null): number {
    if (!raw || !Number.isFinite(raw)) return DEFAULT_THUMB_WIDTH;
    if (ALLOWED_WIDTHS.has(raw)) return raw;
    // Snap to the nearest allowed step — keeps the cache from exploding.
    let best = DEFAULT_THUMB_WIDTH;
    let bestDiff = Infinity;
    for (const w of ALLOWED_WIDTHS) {
        const d = Math.abs(w - raw);
        if (d < bestDiff) {
            bestDiff = d;
            best = w;
        }
    }
    return best;
}

function cacheKey(absPath: string, mtimeMs: number, width: number): string {
    return createHash('sha256').update(`${absPath}:${mtimeMs}:${width}`).digest('hex');
}

export interface Thumb {
    /** Bytes as a Uint8Array<ArrayBuffer> — accepted by `new Response(...)`
     * directly. Node's Buffer extends Uint8Array at runtime so no copy
     * happens, but BodyInit demands the ArrayBuffer-parameterized variant
     * (vs ArrayBufferLike). bufferToBytes asserts this safely. */
    data: Uint8Array<ArrayBuffer>;
    mime: string;
}

/** Wrap a Node Buffer as a Uint8Array view (zero-copy) so it can be passed
 * to `new Response(...)`. The `ArrayBuffer` assertion on `.buffer` is
 * runtime-correct under normal Node (we never use SharedArrayBuffer here)
 * but required because @types/node ≥ 22.10 widens it to `ArrayBufferLike`,
 * which isn't a BodyInit-compatible Uint8Array. */
export function bufferToBytes(buf: Buffer): Uint8Array<ArrayBuffer> {
    return new Uint8Array(buf.buffer as ArrayBuffer, buf.byteOffset, buf.byteLength);
}

/** Return a WebP-encoded thumbnail of `absPath` resized to fit `width` (no
 * upscaling). Reads from the disk cache when available; otherwise renders
 * with sharp and writes to the cache. Returns `null` if the source can't be
 * decoded (corrupt file, unsupported format). */
export async function getThumbnail(absPath: string, width: number): Promise<Thumb | null> {
    let mtimeMs: number;
    try {
        mtimeMs = Math.floor(statSync(absPath).mtimeMs);
    } catch {
        return null;
    }

    const key = cacheKey(absPath, mtimeMs, width);
    const cached = resolve(CACHE_DIR, `${key}.webp`);

    if (existsSync(cached)) {
        return { data: bufferToBytes(readFileSync(cached)), mime: 'image/webp' };
    }

    try {
        const buf = await sharp(absPath, { failOn: 'none' })
            .rotate() // honor EXIF orientation
            .resize({ width, withoutEnlargement: true })
            .webp({ quality: 78, effort: 4 })
            .toBuffer();
        // Atomic-ish write: tmp then rename.
        const tmp = cached + '.tmp';
        writeFileSync(tmp, buf);
        try {
            // fs.renameSync is synchronous and atomic on POSIX same-fs.
            // (We're not importing it explicitly to keep this file lean —
            // writeFileSync + rename via require would mix styles; rely on
            // an inline import.)
            const { renameSync } = await import('node:fs');
            renameSync(tmp, cached);
        } catch {
            // best-effort; the next request will retry
        }
        return { data: bufferToBytes(buf), mime: 'image/webp' };
    } catch {
        return null;
    }
}
