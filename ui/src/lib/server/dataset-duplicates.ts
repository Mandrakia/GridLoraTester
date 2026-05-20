// Duplicate detection for a dataset (or group): cluster the ACTIVE images
// into groups of near-identical photos by perceptual-hash Hamming distance,
// so the user can spot burst frames / re-edits / re-imports and drop all but
// the best take.
//
// Clustering = connected components (union-find) over pairwise Hamming
// distance ≤ the `dedup_hamming_threshold` setting — the same knob the
// connector-suggestion dedup uses, so "what counts as a duplicate" is
// consistent across the app. Transitive on purpose: a 3-frame burst where
// A~B and B~C but A and C drift just past the threshold still reads as the
// one cluster the user thinks of as "the same shot".
//
// O(n²) over the active hashed images in scope. Fine for the few-hundred to
// low-thousand range we deal with; a BK-tree would cut it down if a single
// scope ever holds tens of thousands.
import {
    countActiveForFolders,
    listActiveHashedByFolder,
    type DatasetHashRow
} from './dataset-images';
import { hashHammingPacked, parseHashPacked, type PackedHash } from './image-hash';
import { pathBasename } from './path-utils';
import { dedupHammingThreshold } from './settings';

export interface DuplicateMember {
    image_path: string;
    folder_path: string;
    filename: string;
    image_width: number | null;
    image_height: number | null;
    /** Resolution in megapixels — surfaced so the user sees why one take is
     * the suggested keep. null when dims are unknown. */
    megapixels: number | null;
    /** Hamming distance (bits) from this member to the cluster's suggested
     * keep. 0 for the keep itself. Lets the UI show how tight each twin is. */
    hamming_to_keep: number;
    /** The one member we suggest keeping: highest resolution, ties broken
     * lexicographically by path for determinism. Mirrors the connector
     * dedup's "prefer the higher-res take" rule. */
    suggested_keep: boolean;
}

export interface DuplicateCluster {
    /** Stable id = the suggested-keep image's path. */
    id: string;
    /** Members sorted keep-first, then closest twins first. Always ≥ 2. */
    members: DuplicateMember[];
    /** Largest pairwise Hamming inside the cluster — a tightness indicator
     * (0 = byte-identical set, near the threshold = looser same-scene). */
    max_hamming: number;
}

export interface DuplicatesResult {
    /** Threshold (bits) the clustering ran at — echoed for the UI subtitle. */
    threshold: number;
    /** Clusters of ≥ 2 near-identical images, tightest first (lowest
     * max_hamming), then larger clusters first. */
    clusters: DuplicateCluster[];
    /** Active images in scope that carry a phash (the clustering pool). */
    hashed_images: number;
    /** Total active images in scope. When hashed_images < this, some images
     * have no phash yet — the UI nudges the user to run Analyze / the hash
     * job so dedup sees the whole set. */
    total_active: number;
}

export interface DuplicatesInput {
    folder_paths: string[];
}

interface ParsedItem {
    row: DatasetHashRow;
    words: PackedHash;
}

function megapixels(r: DatasetHashRow): number | null {
    if (r.image_width == null || r.image_height == null) return null;
    return (r.image_width * r.image_height) / 1_000_000;
}

export function findDuplicateClusters(input: DuplicatesInput): DuplicatesResult {
    const threshold = dedupHammingThreshold();
    const total_active = countActiveForFolders(input.folder_paths);

    const rows: DatasetHashRow[] = [];
    for (const f of input.folder_paths) rows.push(...listActiveHashedByFolder(f));

    // Parse hashes once; a malformed hash is treated as unhashed (dropped
    // from the pool, never silently mismatched).
    const items: ParsedItem[] = [];
    for (const row of rows) {
        const words = parseHashPacked(row.phash);
        if (words) items.push({ row, words });
    }
    const n = items.length;
    if (n < 2) {
        return { threshold, clusters: [], hashed_images: n, total_active };
    }

    // Union-find over every near-duplicate pair.
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => {
        let r = x;
        while (parent[r] !== r) r = parent[r];
        // Path-halving compression.
        while (parent[x] !== r) {
            const next = parent[x];
            parent[x] = r;
            x = next;
        }
        return r;
    };
    const union = (a: number, b: number) => {
        const ra = find(a);
        const rb = find(b);
        if (ra !== rb) parent[ra] = rb;
    };
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (hashHammingPacked(items[i].words, items[j].words) <= threshold) {
                union(i, j);
            }
        }
    }

    const groups = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
        const root = find(i);
        const arr = groups.get(root);
        if (arr) arr.push(i);
        else groups.set(root, [i]);
    }

    const clusters: DuplicateCluster[] = [];
    for (const idxs of groups.values()) {
        if (idxs.length < 2) continue;

        // Suggested keep = highest megapixels; tie → lexicographically first
        // path (deterministic across reloads).
        let keep = idxs[0];
        for (const i of idxs) {
            const a = megapixels(items[i].row) ?? -1;
            const b = megapixels(items[keep].row) ?? -1;
            if (a > b || (a === b && items[i].row.image_path < items[keep].row.image_path)) {
                keep = i;
            }
        }
        const keepWords = items[keep].words;

        let max_hamming = 0;
        for (let a = 0; a < idxs.length; a++) {
            for (let b = a + 1; b < idxs.length; b++) {
                const h = hashHammingPacked(items[idxs[a]].words, items[idxs[b]].words);
                if (h > max_hamming) max_hamming = h;
            }
        }

        const members: DuplicateMember[] = idxs.map((i) => {
            const r = items[i].row;
            return {
                image_path: r.image_path,
                folder_path: r.folder_path,
                filename: pathBasename(r.image_path),
                image_width: r.image_width,
                image_height: r.image_height,
                megapixels: megapixels(r),
                hamming_to_keep: hashHammingPacked(items[i].words, keepWords),
                suggested_keep: i === keep
            };
        });
        members.sort((a, b) => {
            if (a.suggested_keep !== b.suggested_keep) return a.suggested_keep ? -1 : 1;
            return a.hamming_to_keep - b.hamming_to_keep;
        });

        clusters.push({ id: items[keep].row.image_path, members, max_hamming });
    }

    clusters.sort(
        (a, b) => a.max_hamming - b.max_hamming || b.members.length - a.members.length
    );

    return { threshold, clusters, hashed_images: n, total_active };
}
