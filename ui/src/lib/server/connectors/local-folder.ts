// Shared dir-scanning + thumb-serving primitives for connectors whose
// "source" is a real folder on disk. HardDriveConnector points its
// person_id at a user-chosen path; GooglePhotosConnector points it at a
// cache dir it owns. Both flow through these helpers so the listing /
// download / thumb-proxy logic lives in one place.
//
// Security: callers pass an `allowed(absPath)` predicate that the helpers
// consult before touching the filesystem. The proxy route never trusts
// the URL — it decodes the base64 path and re-checks against `allowed`.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';

import type { ConnectorPicture, ListPicturesPage } from '$lib/connectors/types';
import { db } from '../db';
import { mimeFor } from '../mime';
import { isPathInsideAnyRoot } from '../path-utils';
import { bufferToBytes } from '../thumbs';

export const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

/** True when `target` lives inside (or equals) any of the roots. */
export function isInsideAnyRoot(target: string, roots: string[]): boolean {
    return isPathInsideAnyRoot(target, roots);
}

/** SQL helper: every `person_id` that's currently linked under one of the
 * named connector ids — used as the local-folder allowlist. Recomputed
 * on each call (cheap; the table is small). */
export function listLinkedFolders(connectorIds: string[]): string[] {
    if (connectorIds.length === 0) return [];
    const placeholders = connectorIds.map(() => '?').join(',');
    const rows = db
        .prepare(
            `SELECT DISTINCT person_id FROM connector_links WHERE connector_id IN (${placeholders})`
        )
        .all(...connectorIds) as { person_id: string }[];
    return rows.map((r) => resolve(r.person_id));
}

/** URL-safe base64 of an absolute path. Reversible via `decodePath`. */
export function encodePath(p: string): string {
    return Buffer.from(p, 'utf-8')
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}

export function decodePath(enc: string): string {
    const b64 = enc.replace(/-/g, '+').replace(/_/g, '/');
    return Buffer.from(b64, 'base64').toString('utf-8');
}

interface ListOpts {
    /** Predicate that gates the root folder. Throws if false. */
    allowed: (absPath: string) => boolean;
    /** Builder for the thumb URL exposed to the browser. Receives the
     * encoded path; returns the full URL to be served by the connector's
     * own proxy route. */
    thumbUrlFor: (encodedPath: string) => string;
}

/** Reads `folderPath` and turns every image file into a ConnectorPicture.
 * The picture's `id` is its absolute path — so `downloadFromDir` can read
 * it back without extra lookup. */
export function listPicturesInDir(folderPath: string, opts: ListOpts): ListPicturesPage {
    const root = resolve(folderPath);
    if (!opts.allowed(root)) {
        throw new Error(`Folder ${root} is not in any current connector link.`);
    }
    let entries: string[];
    try {
        entries = readdirSync(root);
    } catch (e) {
        throw new Error(`Could not read ${root}: ${(e as Error).message}`);
    }
    const pictures: ConnectorPicture[] = [];
    for (const name of entries) {
        const ext = extname(name).toLowerCase();
        if (!IMAGE_EXTS.has(ext)) continue;
        const abs = join(root, name);
        let st: ReturnType<typeof statSync>;
        try {
            st = statSync(abs);
        } catch {
            continue;
        }
        if (!st.isFile()) continue;
        pictures.push({
            id: abs,
            filename: name,
            download_url: abs,
            thumbnail_url: opts.thumbUrlFor(encodePath(abs)),
            created_date: st.birthtime?.toISOString?.() ?? st.mtime.toISOString(),
            width: 0,
            height: 0,
            mime_type: mimeFor(abs)
        });
    }
    return { pictures, nextCursor: null };
}

/** Reads the raw bytes of `picture.id` (its absolute path), guarded by
 * the same allowed predicate. */
export function downloadFromDir(
    picture: ConnectorPicture,
    allowed: (absPath: string) => boolean
): Buffer {
    const abs = resolve(picture.id);
    if (!allowed(abs)) throw new Error(`File ${abs} is not in any linked folder.`);
    return readFileSync(abs);
}

/** Serves a single image file behind a /connectors/<id>/thumb/<base64> URL.
 * `urlPrefix` is the route path including trailing slash (e.g.
 * `/connectors/hard-drive/thumb/`). */
export function proxyServeFromDir(
    upstreamUrl: string,
    urlPrefix: string,
    allowed: (absPath: string) => boolean
): Response {
    if (!upstreamUrl.startsWith(urlPrefix)) {
        return new Response('Bad path', { status: 400 });
    }
    let abs: string;
    try {
        abs = resolve(decodePath(upstreamUrl.slice(urlPrefix.length)));
    } catch {
        return new Response('Bad path', { status: 400 });
    }
    if (!allowed(abs)) {
        return new Response('Forbidden', { status: 403 });
    }
    // Even though the parent folder is allowlisted, gate on image
    // extension so a user who links ~/Pictures doesn't accidentally
    // serve passwords.txt as octet-stream.
    if (!IMAGE_EXTS.has(extname(abs).toLowerCase())) {
        return new Response('Not an image', { status: 403 });
    }
    try {
        if (!statSync(abs).isFile()) {
            return new Response('Not a file', { status: 404 });
        }
    } catch {
        return new Response('Not found', { status: 404 });
    }
    return new Response(bufferToBytes(readFileSync(abs)), {
        status: 200,
        headers: {
            'content-type': mimeFor(abs),
            'cache-control': 'private, max-age=300'
        }
    });
}
