// Add a connector picture to a dataset folder:
//   1. Read the bytes from the connector (via its proxyFetch — same path
//      the worker uses)
//   2. Write to disk under <target_folder>/<safe filename>, avoiding
//      collisions with a numeric suffix
//   3. Upsert a `dataset_images` row tagged source_kind='imported' so the
//      suggestion engine's "already imported" dedup picks it up
//   4. Copy the connector_faces rows into face_embeddings, keyed on the
//      new on-disk path (FK enforces dataset_images was inserted first)
//   5. Re-derive the folder's centroid + per-row similarity
import { existsSync, mkdirSync, renameSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';

import type { ConnectorId } from '$lib/connectors/types';
import {
    recomputeFolderCentroidFromDb,
    recomputeGroupCentroidFromDb
} from './centroids';
import { getConnector } from './connectors/registry';
import type { LinkScope } from './connector-links';
import { findGroupsContaining, getDatasetGroup } from './dataset-groups';
import {
    importedKeysForFolder,
    importedKeysForFolders,
    upsertImported
} from './dataset-images';
import { db } from './db';
import { isPathInside } from './path-utils';

const insertFaceStmt = db.prepare(`
    INSERT INTO face_embeddings(
        image_path, face_index, bbox_json, det_score, embedding_b64,
        is_target, similarity, pitch, yaw, roll
    )
    VALUES(
        @image_path, @face_index, @bbox_json, @det_score, @embedding_b64,
        0, NULL, @pitch, @yaw, @roll
    )
    ON CONFLICT(image_path, face_index) DO UPDATE SET
        bbox_json = excluded.bbox_json,
        det_score = excluded.det_score,
        embedding_b64 = excluded.embedding_b64,
        pitch = excluded.pitch,
        yaw = excluded.yaw,
        roll = excluded.roll,
        computed_at = datetime('now')
`);

const loadConnectorPicStmt = db.prepare(
    'SELECT filename, image_width, image_height, phash FROM connector_pictures WHERE connector_id = ? AND picture_id = ?'
);
const loadConnectorFacesStmt = db.prepare(`
    SELECT face_index, bbox_json, det_score, embedding_b64, pitch, yaw, roll
    FROM connector_faces
    WHERE connector_id = ? AND picture_id = ?
    ORDER BY face_index ASC
`);

interface ConnectorPicMeta {
    filename: string | null;
    image_width: number | null;
    image_height: number | null;
    /** Carried over to dataset_images.phash so the imported image is
     * immediately dedup-aware — saves running a compute-image-hashes
     * pass just to backfill the one row we already know. */
    phash: string | null;
}
interface ConnectorFaceRow {
    face_index: number;
    bbox_json: string;
    det_score: number | null;
    embedding_b64: string;
    pitch: number | null;
    yaw: number | null;
    roll: number | null;
}

const SAFE_NAME_RE = /[^A-Za-z0-9._-]+/g;
function safeFilename(name: string, fallback: string): string {
    const trimmed = name.trim();
    if (!trimmed) return fallback;
    // Collapse `..` to `_` first so a single dot remains permitted (`foo.jpg`)
    // but parent-dir refs never survive into `join()`.
    const cleaned = trimmed
        .replace(/\.\.+/g, '_')
        .replace(SAFE_NAME_RE, '_')
        .replace(/^_+|_+$/g, '');
    return cleaned || fallback;
}

/** Verify `target_folder` is an authorized destination for `scope`.
 *  - folder scope: target must equal scope_key (the dataset's path)
 *  - group scope:  target must be one of the group's member paths
 * Comparison is on resolved absolute paths. Without this, the import endpoint
 * would accept any writable directory passed by the client. */
function assertTargetAllowed(
    scope_kind: LinkScope,
    scope_key: string,
    target_folder: string
): void {
    const target = resolve(target_folder);
    if (scope_kind === 'folder') {
        if (resolve(scope_key) !== target) {
            throw new Error(
                `target_folder does not match folder scope_key (${scope_key} vs ${target_folder})`
            );
        }
        return;
    }
    // group scope
    const id = Number(scope_key);
    if (!Number.isFinite(id) || id <= 0) {
        throw new Error(`Bad group scope_key: ${scope_key}`);
    }
    const group = getDatasetGroup(id);
    if (!group) throw new Error(`Group ${id} not found`);
    if (!group.paths.some((p) => resolve(p) === target)) {
        throw new Error(
            `target_folder ${target_folder} is not a member of group ${id}`
        );
    }
}

/** Per-connector full-resolution download URL — must match the format
 * each connector's proxyFetch / downloadPicture recognizes. */
function downloadUrlFor(connector_id: ConnectorId, picture_id: string): string {
    if (connector_id === 'immich') {
        return `/connectors/immich/thumb/assets/${picture_id}/original`;
    }
    // hard-drive: picture.id is the absolute path, download_url unused.
    return picture_id;
}

function nonCollidingPath(folder: string, baseName: string): string {
    let candidate = join(folder, baseName);
    if (!existsSync(candidate)) return candidate;
    const ext = extname(baseName);
    const stem = ext ? baseName.slice(0, -ext.length) : baseName;
    for (let i = 1; i < 10_000; i++) {
        candidate = join(folder, `${stem}_${i}${ext}`);
        if (!existsSync(candidate)) return candidate;
    }
    throw new Error(`Too many collisions for ${baseName} in ${folder}`);
}

export interface ImportInput {
    scope_kind: LinkScope;
    scope_key: string;
    target_folder: string; // for folder scope = scope_key; for group = chosen member
    connector_id: ConnectorId;
    picture_id: string;
}

export interface ImportResult {
    dest_image_path: string;
    new_faces: number;
    recomputed: boolean;
    resynced_group_ids: number[];
}

export async function addPictureToDataset(input: ImportInput): Promise<ImportResult> {
    // Trust-boundary check FIRST: prove the caller is allowed to write to
    // this folder under this scope. Without this any client could redirect
    // the import to e.g. ~/.ssh or /etc.
    assertTargetAllowed(input.scope_kind, input.scope_key, input.target_folder);

    const targetFolder = resolve(input.target_folder);

    // Sanity: target folder must exist and be a directory. We do NOT create
    // it — datasets are user-managed; auto-mkdir would mask configuration
    // mistakes (e.g. typo in dataset_root).
    try {
        if (!statSync(targetFolder).isDirectory()) {
            throw new Error(`Target is not a directory: ${targetFolder}`);
        }
    } catch (e) {
        throw new Error(`Cannot import: ${(e as Error).message}`);
    }

    // Fetch ML metadata from connector_pictures + connector_faces.
    const meta = loadConnectorPicStmt.get(input.connector_id, input.picture_id) as
        | ConnectorPicMeta
        | undefined;
    if (!meta) {
        throw new Error(
            `Picture ${input.picture_id} not found in connector_pictures (run the face-detect job first).`
        );
    }
    const faces = loadConnectorFacesStmt.all(
        input.connector_id,
        input.picture_id
    ) as ConnectorFaceRow[];

    // Download the bytes via the connector. For hard-drive this reads from
    // disk; for Immich/etc it makes the auth'd HTTP call.
    //
    // `download_url` MUST be the connector-specific proxy URL Immich's
    // proxyFetch recognizes (`/connectors/immich/thumb/<rest>`). Passing
    // just the picture_id (UUID) lets proxyFetch's prefix check fall
    // through to a bare fetch against `<base_url><uuid>` (no separator)
    // → DNS ENOTFOUND. HD downloadPicture ignores this field (reads
    // picture.id as a filesystem path).
    const connector = getConnector(input.connector_id);
    const bytes = await connector.downloadPicture({
        id: input.picture_id,
        filename: meta.filename ?? input.picture_id,
        download_url: downloadUrlFor(input.connector_id, input.picture_id),
        created_date: '',
        width: meta.image_width ?? 0,
        height: meta.image_height ?? 0
    });

    // Pick a safe, non-colliding filename on disk.
    const fallback = `${input.connector_id}_${input.picture_id}`.replace(SAFE_NAME_RE, '_');
    const safe = safeFilename(meta.filename ?? fallback, fallback);
    const dest = nonCollidingPath(targetFolder, safe);

    // Defense in depth: even with safeFilename + nonCollidingPath, assert
    // the final resolved path lives under the target folder. Catches any
    // future regression in the filename sanitizer.
    if (!isPathInside(targetFolder, dest)) {
        throw new Error(`Refusing to write outside target folder: ${dest}`);
    }

    mkdirSync(targetFolder, { recursive: true });

    // Write to a .tmp sibling first; only commit DB rows after the final
    // destination exists. If the DB transaction throws, delete the new file.
    // This prioritizes avoiding DB rows that point at a missing image.
    const tmp = dest + '.tmp';
    writeFileSync(tmp, bytes, { flag: 'wx' });

    let moved = false;
    try {
        renameSync(tmp, dest);
        moved = true;
        // dataset_images first (FK target), THEN face rows. Single tx so a
        // DB failure leaves no committed rows. Phash + dims carried over from the
        // connector record — no need to wait for compute-image-hashes to
        // backfill, the imported file participates in dedup immediately.
        db.transaction(() => {
            upsertImported({
                image_path: dest,
                connector_id: input.connector_id,
                picture_id: input.picture_id,
                phash: meta.phash,
                image_width: meta.image_width,
                image_height: meta.image_height
            });
            for (const f of faces) {
                insertFaceStmt.run({
                    image_path: dest,
                    face_index: f.face_index,
                    bbox_json: f.bbox_json,
                    det_score: f.det_score,
                    embedding_b64: f.embedding_b64,
                    pitch: f.pitch,
                    yaw: f.yaw,
                    roll: f.roll
                });
            }
        })();
    } catch (e) {
        // Best-effort cleanup — don't shadow the original error.
        try {
            unlinkSync(moved ? dest : tmp);
        } catch {
            // ignore
        }
        throw e;
    }

    // Recompute the folder's centroid + per-row sim/is_target. Pure TS
    // since we already have every embedding in DB (no Python needed).
    const after = recomputeFolderCentroidFromDb(targetFolder);

    // Propagate to every group that includes this folder so their union
    // centroid stays in sync (median/mean used by group view + cards).
    const affectedGroups = findGroupsContaining(targetFolder);
    const resynced_group_ids: number[] = [];
    for (const g of affectedGroups) {
        const row = recomputeGroupCentroidFromDb(g.id, g.paths);
        if (row != null) resynced_group_ids.push(g.id);
    }

    return {
        dest_image_path: dest,
        new_faces: faces.length,
        recomputed: after != null,
        resynced_group_ids
    };
}

/** Set of "connector_id::picture_id" already imported into this scope.
 * Used by the suggestion engine to dedupe candidates.
 *
 * Derived from dataset_images (the canonical "what's in the dataset" table).
 * For folder scope: walks the one folder. For group scope: unions every
 * member folder of the group. This is a behavior change vs. the pre-redesign
 * `dataset_imports` table, which keyed lineage by the SCOPE that initiated
 * the import — folder-scoped suggestions would miss imports done via the
 * parent group. The new path correctly excludes those. */
export function importedKeysForScope(
    scope_kind: LinkScope,
    scope_key: string
): Set<string> {
    if (scope_kind === 'folder') {
        return importedKeysForFolder(scope_key);
    }
    const id = Number(scope_key);
    if (!Number.isFinite(id) || id <= 0) return new Set();
    const group = getDatasetGroup(id);
    if (!group) return new Set();
    return importedKeysForFolders(group.paths);
}
