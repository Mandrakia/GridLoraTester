// Read the configured `dataset_root` and present each immediate subfolder as
// a dataset. Cheap-counts the images so the list view shows useful metadata
// without scanning recursively.
import { readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

export interface Dataset {
    name: string;
    path: string;
    image_count: number;
    sample: string | null; // file:// path to a sample image (for thumbnail later)
}

export function listDatasets(rootPath: string): Dataset[] {
    if (!rootPath) return [];
    const root = resolve(rootPath);
    let entries: string[];
    try {
        entries = readdirSync(root);
    } catch {
        return [];
    }

    const out: Dataset[] = [];
    for (const name of entries) {
        const full = join(root, name);
        let st: ReturnType<typeof statSync>;
        try {
            st = statSync(full);
        } catch {
            continue;
        }
        if (!st.isDirectory()) continue;

        let images: string[] = [];
        try {
            images = readdirSync(full).filter((f) => {
                const i = f.lastIndexOf('.');
                if (i < 0) return false;
                return IMAGE_EXTS.has(f.slice(i).toLowerCase());
            });
        } catch {
            // unreadable dir — surface it with 0 images
        }
        out.push({
            name,
            path: full,
            image_count: images.length,
            sample: images[0] ? join(full, images[0]) : null
        });
    }
    out.sort((a, b) => a.name.localeCompare(b.name));
    return out;
}
