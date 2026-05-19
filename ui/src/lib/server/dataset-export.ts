// ZIP export of a dataset folder (or every member of a group). Streams a
// kohya-style flat layout: each image + its caption sidecar (`.txt` or
// `.caption`) under a top-level prefix. Excluded images (status='excluded'
// in dataset_images) are skipped — the user already decided they don't
// belong in the training set. Images on disk without a dataset_images row
// are still included (they haven't been analyzed yet but the user dropped
// them in the folder on purpose).
import { existsSync, readdirSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';

import yazl from 'yazl';

import { listExcludedByFolder } from './dataset-images';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);
const CAPTION_EXTS = ['.txt', '.caption'];

function lowerExt(fname: string): string {
    const i = fname.lastIndexOf('.');
    return i < 0 ? '' : fname.slice(i).toLowerCase();
}

/** Add every image + caption sidecar from `folderPath` to `zip`, namespaced
 * under `prefix` (empty string for flat layout). Returns the number of
 * images added — caller can decide whether an empty result is an error. */
export function addFolderToZip(
    zip: yazl.ZipFile,
    folderPath: string,
    prefix: string
): number {
    let entries: string[];
    try {
        entries = readdirSync(folderPath);
    } catch {
        return 0;
    }

    const excluded = new Set(
        listExcludedByFolder(folderPath).map((r) => r.image_path)
    );

    let count = 0;
    for (const f of entries) {
        if (!IMAGE_EXTS.has(lowerExt(f))) continue;
        const abs = join(folderPath, f);
        try {
            if (!statSync(abs).isFile()) continue;
        } catch {
            continue;
        }
        if (excluded.has(abs)) continue;

        const dest = prefix ? `${prefix}/${f}` : f;
        zip.addFile(abs, dest);
        count++;

        // Caption sidecar: kohya convention is `<stem>.txt`; we also accept
        // `.caption`. Add whichever exists, preserving original extension.
        const stem = f.replace(/\.[^.]+$/, '');
        for (const ext of CAPTION_EXTS) {
            const capAbs = join(folderPath, stem + ext);
            if (existsSync(capAbs)) {
                const capDest = prefix ? `${prefix}/${stem}${ext}` : `${stem}${ext}`;
                zip.addFile(capAbs, capDest);
                break;
            }
        }
    }
    return count;
}

/** Stream a zip of a single folder, flat layout. */
export function buildFolderZip(folderPath: string): yazl.ZipFile {
    const zip = new yazl.ZipFile();
    addFolderToZip(zip, folderPath, '');
    zip.end();
    return zip;
}

/** Stream a zip of every folder in `folderPaths`, each under its own
 * subdirectory. Duplicate basenames are disambiguated with `__N` so the
 * archive is always extractable. Matches the slug scheme used by the
 * group's `raw` route. */
export function buildGroupZip(folderPaths: string[]): yazl.ZipFile {
    const zip = new yazl.ZipFile();
    const seen = new Map<string, number>();
    for (const p of folderPaths) {
        const base = basename(p);
        const count = (seen.get(base) ?? 0) + 1;
        seen.set(base, count);
        const slug = count === 1 ? base : `${base}__${count - 1}`;
        addFolderToZip(zip, p, slug);
    }
    zip.end();
    return zip;
}

/** Sanitize a label into something safe for a `Content-Disposition` filename
 * without depending on RFC 5987 encoding. ASCII-only, no path separators. */
export function safeZipFilename(label: string): string {
    const cleaned = label.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');
    return (cleaned || 'dataset') + '.zip';
}
