// Server-side allowlist of paths the in-browser FolderPicker is allowed
// to open. Both /api/fs/roots (which advertises the list to the UI) and
// /api/fs/list (which gates path access) consult this helper so they
// agree on what's reachable.
//
// The user's home directory is always exposed. Beyond that, we add OS-
// conventional mount-point bases when they actually exist on disk:
//   - Linux: /mnt, /media
//   - macOS: /Volumes
//   - Windows: nothing for now — drive letters need a different model;
//     users can still type a path manually in the calling form.
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
        plat === 'darwin' ? ['/Volumes'] : plat === 'linux' ? ['/mnt', '/media'] : [];
    for (const p of candidates) {
        if (existsSync(p)) out.push({ label: p, path: p });
    }
    return out;
}
