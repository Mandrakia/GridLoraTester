// Dataset groups = user-curated bundles of dataset folders.
// Persisted in `dataset_groups`; paths are stored as a JSON array of strings
// in `paths_json` to keep the schema flat (small N per group, no queries on
// individual paths needed).
import { basename, resolve } from 'node:path';
import { readdirSync, statSync } from 'node:fs';
import { db } from './db';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

export interface DatasetGroup {
    id: number;
    name: string;
    paths: string[];
    created_at: string;
    updated_at: string;
}

export interface DatasetGroupRow extends DatasetGroup {
    /** Basenames joined for display (`"setA, setB, setC"`). */
    dataset_names: string;
    /** Sum of image counts across all paths (cheap non-recursive count). */
    total_images: number;
    /** Paths from `paths_json` that don't exist or aren't directories. */
    missing_paths: string[];
}

const listStmt = db.prepare(
    'SELECT id, name, paths_json, created_at, updated_at FROM dataset_groups ORDER BY name ASC'
);
const getStmt = db.prepare(
    'SELECT id, name, paths_json, created_at, updated_at FROM dataset_groups WHERE id = ?'
);
const insertStmt = db.prepare(
    'INSERT INTO dataset_groups(name, paths_json) VALUES(?, ?) RETURNING id'
);
const updateStmt = db.prepare(
    "UPDATE dataset_groups SET name = ?, paths_json = ?, updated_at = datetime('now') WHERE id = ?"
);
const deleteStmt = db.prepare('DELETE FROM dataset_groups WHERE id = ?');

function parseRow(row: {
    id: number;
    name: string;
    paths_json: string;
    created_at: string;
    updated_at: string;
}): DatasetGroup {
    let paths: string[] = [];
    try {
        const parsed = JSON.parse(row.paths_json);
        if (Array.isArray(parsed)) paths = parsed.filter((p) => typeof p === 'string');
    } catch {
        // ignore — corrupt JSON degrades to an empty path list, the user
        // can re-edit the group from the UI.
    }
    return {
        id: row.id,
        name: row.name,
        paths,
        created_at: row.created_at,
        updated_at: row.updated_at
    };
}

function countImages(dir: string): number {
    try {
        const entries = readdirSync(dir);
        let n = 0;
        for (const f of entries) {
            const i = f.lastIndexOf('.');
            if (i < 0) continue;
            if (IMAGE_EXTS.has(f.slice(i).toLowerCase())) n++;
        }
        return n;
    } catch {
        return 0;
    }
}

function isDir(p: string): boolean {
    try {
        return statSync(p).isDirectory();
    } catch {
        return false;
    }
}

/** Decorate a stored group with display fields (names, image count, missing). */
function decorate(g: DatasetGroup): DatasetGroupRow {
    const missing: string[] = [];
    let total = 0;
    for (const p of g.paths) {
        if (!isDir(p)) {
            missing.push(p);
            continue;
        }
        total += countImages(p);
    }
    return {
        ...g,
        dataset_names: g.paths.map((p) => basename(p)).join(', '),
        total_images: total,
        missing_paths: missing
    };
}

export function listDatasetGroups(): DatasetGroupRow[] {
    return (listStmt.all() as Parameters<typeof parseRow>[0][])
        .map(parseRow)
        .map(decorate);
}

export function getDatasetGroup(id: number): DatasetGroupRow | null {
    const row = getStmt.get(id) as Parameters<typeof parseRow>[0] | undefined;
    return row ? decorate(parseRow(row)) : null;
}

export function createDatasetGroup(name: string, paths: string[]): DatasetGroupRow {
    const trimmed = name.trim();
    if (!trimmed) throw new Error('Group name is required');
    const result = insertStmt.get(trimmed, JSON.stringify(paths)) as { id: number };
    return getDatasetGroup(result.id)!;
}

export function updateDatasetGroup(
    id: number,
    name: string,
    paths: string[]
): DatasetGroupRow {
    const trimmed = name.trim();
    if (!trimmed) throw new Error('Group name is required');
    updateStmt.run(trimmed, JSON.stringify(paths), id);
    const updated = getDatasetGroup(id);
    if (!updated) throw new Error(`Dataset group ${id} not found after update`);
    return updated;
}

export function deleteDatasetGroup(id: number): void {
    deleteStmt.run(id);
}

/** Return every group whose paths_json contains `folderPath`.
 * Comparison is done on resolved absolute paths so it's robust to harmless
 * variants (trailing slash, `./` prefix). Used after a folder mutation to
 * know which group centroids need resync. N is small (groups are user-
 * curated), so we just JSON.parse them all. */
export function findGroupsContaining(folderPath: string): DatasetGroup[] {
    const needle = resolve(folderPath);
    const rows = listStmt.all() as Parameters<typeof parseRow>[0][];
    const hits: DatasetGroup[] = [];
    for (const row of rows) {
        const g = parseRow(row);
        if (g.paths.some((p) => resolve(p) === needle)) hits.push(g);
    }
    return hits;
}
