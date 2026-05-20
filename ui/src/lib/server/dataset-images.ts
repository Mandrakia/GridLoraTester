// Canonical access layer for `dataset_images`. Every read/write of the table
// goes through here so the rest of the codebase doesn't sprinkle raw SQL.
//
// Two write paths feed this table:
//   - Manual: a file appears under a dataset folder (user dropped it, or an
//     existing-on-disk image surfaced via face detection). `upsertManual`.
//   - Imported: addPictureToDataset wrote a connector picture to disk and
//     wants the lineage recorded. `upsertImported`.
//
// Status flips (markExcluded / restoreActive) are the prune feature's hook.
// Excluded images stay in DB (reversible) but are filtered out of centroid
// recompute, coverage stats, and the suggestion engine's dedup pool — so a
// near-duplicate of an excluded image can re-surface, since the user already
// decided the bucket has enough.
import { dirname } from 'node:path';

import type { ConnectorId } from '$lib/connectors/types';
import { db } from './db';

export type DatasetImageStatus = 'active' | 'excluded';
export type DatasetImageSource = 'manual' | 'imported';

export interface DatasetImageRow {
    image_path: string;
    folder_path: string;
    status: DatasetImageStatus;
    excluded_at: string | null;
    excluded_reason: string | null;
    phash: string | null;
    image_width: number | null;
    image_height: number | null;
    source_kind: DatasetImageSource;
    connector_id: string | null;
    picture_id: string | null;
    imported_at: string | null;
    added_at: string;
}

const getStmt = db.prepare(
    'SELECT * FROM dataset_images WHERE image_path = ?'
);

export function getDatasetImage(image_path: string): DatasetImageRow | null {
    return (getStmt.get(image_path) as DatasetImageRow | undefined) ?? null;
}

// Upsert preserving any non-NULL existing field. Used by face detection /
// manual-discovery paths where we learn dims/phash incrementally and don't
// want to clobber the connector_id/picture_id of an imported image that
// was just face-detected for the first time.
const upsertManualStmt = db.prepare(`
    INSERT INTO dataset_images
        (image_path, folder_path, phash, image_width, image_height, source_kind)
    VALUES (@image_path, @folder_path, @phash, @image_width, @image_height, 'manual')
    ON CONFLICT(image_path) DO UPDATE SET
        phash = COALESCE(@phash, dataset_images.phash),
        image_width = COALESCE(@image_width, dataset_images.image_width),
        image_height = COALESCE(@image_height, dataset_images.image_height)
`);

export interface UpsertManualInput {
    image_path: string;
    phash?: string | null;
    image_width?: number | null;
    image_height?: number | null;
}

/** Upsert a dataset image with `source_kind='manual'` on insert. On conflict,
 * leaves the existing source_kind untouched and only fills NULL fields —
 * so a face-detection sweep over an imported image doesn't clobber its
 * lineage. */
export function upsertManual(input: UpsertManualInput): void {
    upsertManualStmt.run({
        image_path: input.image_path,
        folder_path: dirname(input.image_path),
        phash: input.phash ?? null,
        image_width: input.image_width ?? null,
        image_height: input.image_height ?? null
    });
}

const upsertImportedStmt = db.prepare(`
    INSERT INTO dataset_images
        (image_path, folder_path, phash, image_width, image_height,
         source_kind, connector_id, picture_id, imported_at)
    VALUES (@image_path, @folder_path, @phash, @image_width, @image_height,
            'imported', @connector_id, @picture_id, datetime('now'))
    ON CONFLICT(image_path) DO UPDATE SET
        source_kind = 'imported',
        connector_id = @connector_id,
        picture_id = @picture_id,
        imported_at = datetime('now'),
        phash = COALESCE(@phash, dataset_images.phash),
        image_width = COALESCE(@image_width, dataset_images.image_width),
        image_height = COALESCE(@image_height, dataset_images.image_height)
`);

export interface UpsertImportedInput {
    image_path: string;
    connector_id: ConnectorId;
    picture_id: string;
    phash?: string | null;
    image_width?: number | null;
    image_height?: number | null;
}

/** Upsert a dataset image flagging it as imported from a connector. The
 * UNIQUE(connector_id, picture_id) constraint guarantees the same connector
 * picture can't end up in two places via concurrent imports. */
export function upsertImported(input: UpsertImportedInput): void {
    upsertImportedStmt.run({
        image_path: input.image_path,
        folder_path: dirname(input.image_path),
        connector_id: input.connector_id,
        picture_id: input.picture_id,
        phash: input.phash ?? null,
        image_width: input.image_width ?? null,
        image_height: input.image_height ?? null
    });
}

const updatePhashStmt = db.prepare(
    'UPDATE dataset_images SET phash = ? WHERE image_path = ?'
);

/** Used by the compute-image-hashes background job to backfill phashes
 * computed from disk reads. */
export function updatePhash(image_path: string, phash: string): void {
    updatePhashStmt.run(phash, image_path);
}

const markExcludedStmt = db.prepare(`
    UPDATE dataset_images
       SET status = 'excluded',
           excluded_at = datetime('now'),
           excluded_reason = ?
     WHERE image_path = ?
`);

const restoreActiveStmt = db.prepare(`
    UPDATE dataset_images
       SET status = 'active',
           excluded_at = NULL,
           excluded_reason = NULL
     WHERE image_path = ?
`);

/** Flip status to excluded. Caller is responsible for triggering centroid
 * recompute on the affected folder (and any group that contains it). */
export function markExcluded(image_path: string, reason: string | null = null): void {
    markExcludedStmt.run(reason, image_path);
}

export function restoreActive(image_path: string): void {
    restoreActiveStmt.run(image_path);
}

const listActiveByFolderStmt = db.prepare(`
    SELECT * FROM dataset_images
     WHERE folder_path = ? AND status = 'active'
     ORDER BY image_path ASC
`);

const listExcludedByFolderStmt = db.prepare(`
    SELECT * FROM dataset_images
     WHERE folder_path = ? AND status = 'excluded'
     ORDER BY excluded_at DESC
`);

export function listActiveByFolder(folder_path: string): DatasetImageRow[] {
    return listActiveByFolderStmt.all(folder_path) as DatasetImageRow[];
}

export function listExcludedByFolder(folder_path: string): DatasetImageRow[] {
    return listExcludedByFolderStmt.all(folder_path) as DatasetImageRow[];
}

// ---- "Already imported" lookups for the suggestion engine ----------------
//
// Replaces the old `importedKeysForScope` that read from dataset_imports.
// Now derived directly from dataset_images: any row with source_kind='imported'
// and a (connector_id, picture_id) under the queried folder set counts as
// already imported. Excluded imports STILL count — we don't want to re-suggest
// a picture the user explicitly excluded.

const importedKeysForFolderStmt = db.prepare(`
    SELECT connector_id, picture_id FROM dataset_images
     WHERE folder_path = ?
       AND source_kind = 'imported'
       AND connector_id IS NOT NULL
       AND picture_id IS NOT NULL
`);

/** All connector pictures imported under this folder. Returns a Set keyed
 * `${connector_id}::${picture_id}` to match the suggestion engine's lookup. */
export function importedKeysForFolder(folder_path: string): Set<string> {
    const rows = importedKeysForFolderStmt.all(folder_path) as {
        connector_id: string;
        picture_id: string;
    }[];
    return new Set(rows.map((r) => `${r.connector_id}::${r.picture_id}`));
}

/** Union across multiple folders — used for group-scope suggestions. */
export function importedKeysForFolders(folder_paths: string[]): Set<string> {
    const out = new Set<string>();
    for (const f of folder_paths) {
        for (const k of importedKeysForFolder(f)) out.add(k);
    }
    return out;
}

// ---- pHash + image-level metadata lookups -------------------------------

const datasetHashesForFolderStmt = db.prepare(`
    SELECT phash, image_path
      FROM dataset_images
     WHERE folder_path = ?
       AND status = 'active'
       AND phash IS NOT NULL
`);

/** Map phash → an example active image_path under this folder. Used by the
 * suggestion engine's dedup. Excluded images deliberately omitted: the user
 * has already rejected them, so a near-dup of an excluded image is fair game
 * to re-suggest. */
export function datasetHashesForFolder(folder_path: string): Map<string, string> {
    const out = new Map<string, string>();
    const rows = datasetHashesForFolderStmt.all(folder_path) as {
        phash: string;
        image_path: string;
    }[];
    for (const r of rows) {
        if (!out.has(r.phash)) out.set(r.phash, r.image_path);
    }
    return out;
}

export function datasetHashesForFolders(folder_paths: string[]): Map<string, string> {
    const out = new Map<string, string>();
    for (const f of folder_paths) {
        for (const [h, p] of datasetHashesForFolder(f)) {
            if (!out.has(h)) out.set(h, p);
        }
    }
    return out;
}

export interface DatasetHashRow {
    image_path: string;
    folder_path: string;
    image_width: number | null;
    image_height: number | null;
    phash: string;
}

const activeHashedByFolderStmt = db.prepare(`
    SELECT image_path, folder_path, image_width, image_height, phash
      FROM dataset_images
     WHERE folder_path = ?
       AND status = 'active'
       AND phash IS NOT NULL
`);

/** Every active, hashed image under a folder — full row (with dims) rather
 * than the phash→path Map. Used by the duplicate-clustering pass, which
 * needs each individual image (not deduped by phash) plus its resolution
 * to pick which member of a cluster to keep. */
export function listActiveHashedByFolder(folder_path: string): DatasetHashRow[] {
    return activeHashedByFolderStmt.all(folder_path) as DatasetHashRow[];
}

// ---- Active counts (for max_size enforcement + UI counter) --------------

const countActiveForFolderStmt = db.prepare(
    `SELECT COUNT(*) AS n FROM dataset_images WHERE folder_path = ? AND status = 'active'`
);

export function countActiveForFolder(folder_path: string): number {
    return (countActiveForFolderStmt.get(folder_path) as { n: number }).n;
}

export function countActiveForFolders(folder_paths: string[]): number {
    let n = 0;
    for (const f of folder_paths) n += countActiveForFolder(f);
    return n;
}
