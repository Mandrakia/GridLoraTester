// "Hard drive" connector — link a dataset to a plain folder on disk. No
// per-instance setup: each link carries its own folder path (stored as the
// link's person_id). isSignedIn() is always true; signIn/signOut are no-ops.
//
// Downstream code is connector-agnostic, so the face-detect job, the
// suggestion engine, and the proxy route Just Work — we only had to make
// the link modal branch on linker_kind to swap the picker UI.
//
// Security: the proxy route can be hit with any path encoded in the URL.
// We allowlist against `connector_links` — only folders the user has
// explicitly linked are readable. No arbitrary FS reads possible.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';

import type {
    ConnectorPicture,
    ConnectorSignInResult,
    ConnectorTypeInfo,
    ListPicturesOpts,
    ListPicturesPage,
    PhotoConnector
} from '$lib/connectors/types';
import { db } from '../db';
import { mimeFor } from '../mime';
import { bufferToBytes } from '../thumbs';

export const HARD_DRIVE_TYPE_INFO: ConnectorTypeInfo = {
    id: 'hard-drive',
    label: 'Folder on disk',
    linker_kind: 'folder',
    needs_credentials: false
};

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

/** All folders currently linked under the hard-drive connector — used as the
 * allowlist for the proxy route. Refreshed on every check (cheap SQL). */
const allowedRootsStmt = db.prepare(
    "SELECT DISTINCT person_id FROM connector_links WHERE connector_id = 'hard-drive'"
);

function listAllowedRoots(): string[] {
    return (allowedRootsStmt.all() as { person_id: string }[]).map((r) =>
        resolve(r.person_id)
    );
}

/** True when `target` lives inside (or equals) one of the linked folders. */
function isAllowed(target: string): boolean {
    const abs = resolve(target);
    for (const root of listAllowedRoots()) {
        if (abs === root) return true;
        if (abs.startsWith(root + '/')) return true;
    }
    return false;
}

/** URL-safe base64 of an absolute path. Reversible via the inverse below. */
function encodePath(p: string): string {
    return Buffer.from(p, 'utf-8')
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}

function decodePath(enc: string): string {
    const b64 = enc.replace(/-/g, '+').replace(/_/g, '/');
    return Buffer.from(b64, 'base64').toString('utf-8');
}

export class HardDriveConnector implements PhotoConnector {
    readonly id = 'hard-drive' as const;
    readonly label = 'Folder on disk';

    async isSignedIn(): Promise<boolean> {
        return true;
    }

    async signIn(): Promise<ConnectorSignInResult> {
        // No-op — kept on the contract so the Test button on Settings doesn't
        // need to special-case this connector if it ever shows up there.
        return { ok: true };
    }

    async signOut(): Promise<void> {
        // No persisted credentials to clear.
    }

    async listPersons() {
        // The link UI doesn't call this for folder-kind connectors — we
        // return [] in case anyone does.
        return [];
    }

    async listPictures(folderPath: string, _opts: ListPicturesOpts = {}): Promise<ListPicturesPage> {
        const root = resolve(folderPath);
        if (!isAllowed(root)) {
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
                id: abs, // path itself is the stable id — picture_id in the DB
                filename: name,
                download_url: abs, // not browser-reachable; consumed only by downloadPicture
                thumbnail_url: `/connectors/hard-drive/thumb/${encodePath(abs)}`,
                created_date: st.birthtime?.toISOString?.() ?? st.mtime.toISOString(),
                // Dims are unknown without decoding — the face-detect worker
                // fills them in when it reads the file.
                width: 0,
                height: 0,
                mime_type: mimeFor(abs)
            });
        }
        // Single page — no pagination on disk listings.
        return { pictures, nextCursor: null };
    }

    async downloadPicture(picture: ConnectorPicture): Promise<Buffer> {
        const abs = resolve(picture.id);
        if (!isAllowed(abs)) throw new Error(`File ${abs} is not in any linked folder.`);
        return readFileSync(abs);
    }

    async proxyFetch(upstreamUrl: string): Promise<Response> {
        const prefix = '/connectors/hard-drive/thumb/';
        if (!upstreamUrl.startsWith(prefix)) {
            return new Response('Bad path', { status: 400 });
        }
        let abs: string;
        try {
            abs = resolve(decodePath(upstreamUrl.slice(prefix.length)));
        } catch {
            return new Response('Bad path', { status: 400 });
        }
        if (!isAllowed(abs)) {
            return new Response('Forbidden', { status: 403 });
        }
        // Gate on image extensions even though `isAllowed` covers the parent
        // folder: a user who links `~/Pictures` doesn't want this proxy
        // serving e.g. `~/Pictures/passwords.txt` as octet-stream.
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
}
