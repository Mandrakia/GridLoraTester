// Server-side allowlist of paths the in-browser FolderPicker is allowed
// to open. Both /api/fs/roots (which advertises the list to the UI) and
// /api/fs/list (which gates path access) consult this helper so they
// agree on what's reachable.
//
// The user's home directory is always exposed. Beyond that, we add OS-
// conventional mount-point bases when they actually exist on disk:
//   - Linux: /mnt, /media
//   - macOS: /Volumes
//   - Windows: existing drive roots (C:\, D:\, …)
import { existsSync } from 'node:fs';
import { homedir, platform } from 'node:os';

export interface AllowedRoot {
    label: string;
    path: string;
}

export function getAllowedRoots(): AllowedRoot[] {
    const out: AllowedRoot[] = [{ label: 'Home', path: homedir() }];
    const plat = platform();
    const candidates: string[] =
        plat === 'darwin'
            ? ['/Volumes']
            : plat === 'linux'
              ? ['/mnt', '/media']
              : plat === 'win32'
                ? 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map((d) => `${d}:\\`)
                : [];
    for (const p of candidates) {
        if (existsSync(p) && !out.some((r) => r.path.toLowerCase() === p.toLowerCase())) {
            out.push({ label: p, path: p });
        }
    }
    return out;
}
