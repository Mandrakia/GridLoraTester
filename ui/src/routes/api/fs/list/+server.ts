// Folder listing for the in-browser FolderPicker. Returns directories only,
// with hidden (dot-prefixed) entries flagged so the UI can render them at a
// lower opacity. Symlinks are resolved one level via stat() so they show up
// when they actually point at a directory.
//
// Safety: two checks. (1) The `root` query must match one of the server's
// allowed roots (see lib/server/fs-roots.ts) — the client can't pick an
// arbitrary cage. (2) The resolved `path` must equal or be a descendant of
// that root. Defense-in-depth: the UI also blocks up-navigation past root,
// but never trust the client.
import { error, json } from '@sveltejs/kit';
import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { getAllowedRoots } from '$lib/server/fs-roots';
import { isPathInside } from '$lib/server/path-utils';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
    const rawRoot = url.searchParams.get('root');
    if (!rawRoot) error(400, 'root query parameter required');
    const rawTarget = url.searchParams.get('path');

    const rootAbs = path.resolve(rawRoot);
    const allowed = getAllowedRoots().map((r) => path.resolve(r.path));
    if (!allowed.includes(rootAbs)) {
        error(403, 'root not in allowlist');
    }

    const targetAbs = rawTarget ? path.resolve(rawTarget) : rootAbs;
    if (!isPathInside(rootAbs, targetAbs)) {
        error(403, 'path escapes root');
    }

    let raw;
    try {
        raw = await readdir(targetAbs, { withFileTypes: true });
    } catch (e) {
        const err = e as NodeJS.ErrnoException;
        if ((err.code === 'EPERM' || err.code === 'EACCES') && process.platform === 'darwin') {
            error(
                500,
                `${err.message}. macOS may be blocking Terminal/Node from this folder; grant Full Disk Access or choose another folder.`
            );
        }
        error(500, err.message);
    }

    const folders: { name: string; path: string; isHidden: boolean }[] = [];
    await Promise.all(
        raw.map(async (e) => {
            const childPath = path.join(targetAbs, e.name);
            let isDir = e.isDirectory();
            if (!isDir && e.isSymbolicLink()) {
                try {
                    const s = await stat(childPath);
                    isDir = s.isDirectory();
                } catch {
                    // dangling symlink — skip
                }
            }
            if (isDir) {
                folders.push({ name: e.name, path: childPath, isHidden: e.name.startsWith('.') });
            }
        })
    );

    folders.sort((a, b) => a.name.localeCompare(b.name));

    const parent = targetAbs === rootAbs ? null : path.dirname(targetAbs);

    return json({
        root: rootAbs,
        path: targetAbs,
        parent,
        separator: path.sep,
        entries: folders
    });
};
