// Grid test definitions: stored in the `tests` table, with a status that is
// COMPUTED from the filesystem state of the configured tests_root every time
// the page is loaded (no stale DB flag to keep in sync).
//
// Status semantics:
//   not_started  → no output dir, or no manifest.json yet
//   in_progress  → manifest exists but at least one cell is still missing
//                  (this is "running" in the absence of a real job-runner —
//                  once we have process tracking we can split the two)
//   completed    → every (lora × prompt) cell has an image on disk, and
//                  the source LoRA folder contains no new files since
//   out_of_sync  → completed, BUT the LoRA folder has gained files since
//                  the manifest was last written (need a rerun to cover them)
import { existsSync, readFileSync, statSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

import { db } from './db';
import { getDatasetGroup } from './dataset-groups';
import { listLoraSafetensorNames } from './loras';
import { getPromptSet } from './prompts';

export type TestStatus = 'not_started' | 'in_progress' | 'completed' | 'out_of_sync';

export type DatasetSelector =
    | { kind: 'path'; path: string }
    | { kind: 'group'; id: number };

export interface Test {
    id: number;
    name: string;
    lora_path: string;
    dataset: DatasetSelector | null;
    /** Legacy free-text path. New tests prefer prompt_set_id. */
    prompts_path: string | null;
    /** FK → prompt_sets.id. Takes precedence over prompts_path when set. */
    prompt_set_id: number | null;
    width: number;
    height: number;
    batch_size: number;
    quant: string;
    offload: string;
    advanced: Record<string, unknown>;
    created_at: string;
    updated_at: string;
}

export interface TestRow extends Test {
    /** Resolved label for the dataset selector (group name or basename). */
    dataset_label: string;
    /** Resolved label for the prompts source (set name, basename of path, or "—"). */
    prompts_label: string;
    /** Prompt count: from the prompt set (if set) or the prompts file. */
    prompt_count: number;
    /** Number of images currently on disk for this test. */
    images_generated: number;
    /** Total target images = prompts × LoRAs the manifest knows about. */
    images_target: number;
    status: TestStatus;
    /** Highest per-row median score in the manifest, or null. */
    best_median_score: number | null;
    /** Files in lora_path that aren't in the manifest's rows (out_of_sync hint). */
    new_loras_count: number;
}

type TestRecord = {
    id: number;
    name: string;
    lora_path: string;
    dataset_path: string | null;
    dataset_group_id: number | null;
    prompts_path: string | null;
    prompt_set_id: number | null;
    width: number;
    height: number;
    batch_size: number;
    quant: string;
    offload: string;
    advanced_json: string;
    created_at: string;
    updated_at: string;
};

const SELECT_COLS =
    'id, name, lora_path, dataset_path, dataset_group_id, prompts_path, prompt_set_id, ' +
    'width, height, batch_size, quant, offload, advanced_json, ' +
    'created_at, updated_at';

const listStmt = db.prepare(`SELECT ${SELECT_COLS} FROM tests ORDER BY name ASC`);
const getStmt = db.prepare(`SELECT ${SELECT_COLS} FROM tests WHERE id = ?`);
const insertStmt = db.prepare(`
    INSERT INTO tests(name, lora_path, dataset_path, dataset_group_id, prompts_path, prompt_set_id,
                      width, height, batch_size, quant, offload, advanced_json)
    VALUES(@name, @lora_path, @dataset_path, @dataset_group_id, @prompts_path, @prompt_set_id,
           @width, @height, @batch_size, @quant, @offload, @advanced_json)
    RETURNING id
`);
const updateStmt = db.prepare(`
    UPDATE tests SET
        name = @name,
        lora_path = @lora_path,
        dataset_path = @dataset_path,
        dataset_group_id = @dataset_group_id,
        prompts_path = @prompts_path,
        prompt_set_id = @prompt_set_id,
        width = @width, height = @height,
        batch_size = @batch_size,
        quant = @quant, offload = @offload,
        advanced_json = @advanced_json,
        updated_at = datetime('now')
    WHERE id = @id
`);
const deleteStmt = db.prepare('DELETE FROM tests WHERE id = ?');

function parseRow(r: TestRecord): Test {
    let dataset: DatasetSelector | null = null;
    if (r.dataset_group_id != null) dataset = { kind: 'group', id: r.dataset_group_id };
    else if (r.dataset_path) dataset = { kind: 'path', path: r.dataset_path };
    let advanced: Record<string, unknown> = {};
    try {
        const parsed = JSON.parse(r.advanced_json);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            advanced = parsed as Record<string, unknown>;
        }
    } catch {
        // ignore — corrupt advanced JSON degrades to {} (user can re-save).
    }
    return {
        id: r.id,
        name: r.name,
        lora_path: r.lora_path,
        dataset,
        prompts_path: r.prompts_path,
        prompt_set_id: r.prompt_set_id,
        width: r.width,
        height: r.height,
        batch_size: r.batch_size,
        quant: r.quant,
        offload: r.offload,
        advanced,
        created_at: r.created_at,
        updated_at: r.updated_at
    };
}

function countLines(path: string | null): number {
    if (!path) return 0;
    try {
        const txt = readFileSync(path, 'utf-8');
        return txt
            .split(/\r?\n/)
            .filter((ln) => ln.trim() && !ln.trim().startsWith('#')).length;
    } catch {
        return 0;
    }
}

function resolveDatasetLabel(d: DatasetSelector | null): string {
    if (!d) return '—';
    if (d.kind === 'path') return basename(d.path);
    const g = getDatasetGroup(d.id);
    return g ? `group: ${g.name}` : `group #${d.id} (deleted)`;
}

interface ManifestRow {
    lora?: string;
    images?: (string | null)[];
    metrics?: { median?: number } & Record<string, unknown>;
}
interface Manifest {
    rows?: ManifestRow[];
    prompts?: unknown[];
}

function readManifest(testsRoot: string, testName: string): Manifest | null {
    const path = resolve(testsRoot, testName, 'manifest.json');
    if (!existsSync(path)) return null;
    try {
        return JSON.parse(readFileSync(path, 'utf-8')) as Manifest;
    } catch {
        return null;
    }
}

function decorate(t: Test, testsRoot: string): TestRow {
    // Prompts come from a DB-backed prompt set when present; the legacy
    // free-text path is the fallback for older rows.
    let prompt_count = 0;
    let prompts_label = '—';
    if (t.prompt_set_id != null) {
        const ps = getPromptSet(t.prompt_set_id);
        if (ps) {
            prompt_count = ps.prompt_count;
            prompts_label = `set: ${ps.name}`;
        } else {
            prompts_label = `set #${t.prompt_set_id} (deleted)`;
        }
    } else if (t.prompts_path) {
        prompt_count = countLines(t.prompts_path);
        prompts_label = t.prompts_path.split('/').pop() || t.prompts_path;
    }
    const manifest = testsRoot ? readManifest(testsRoot, t.name) : null;
    const outDirExists = (() => {
        if (!testsRoot) return false;
        try {
            return statSync(resolve(testsRoot, t.name)).isDirectory();
        } catch {
            return false;
        }
    })();

    let images_generated = 0;
    let images_target = 0;
    let best_median_score: number | null = null;
    let new_loras_count = 0;
    let status: TestStatus = 'not_started';

    if (manifest) {
        const rows = manifest.rows ?? [];
        const promptsInManifest = (manifest.prompts ?? []).length;
        images_target = rows.length * promptsInManifest;
        for (const r of rows) {
            for (const img of r.images ?? []) {
                if (img) images_generated++;
            }
            const med = r.metrics?.median;
            if (typeof med === 'number' && (best_median_score === null || med > best_median_score)) {
                best_median_score = med;
            }
        }

        const complete = images_target > 0 && images_generated >= images_target;
        if (complete) {
            // Check if the source lora folder grew since the manifest was written.
            const onDisk = listLoraSafetensorNames(t.lora_path);
            const known = new Set(rows.map((r) => r.lora ?? ''));
            new_loras_count = onDisk.filter((n) => !known.has(n)).length;
            status = new_loras_count > 0 ? 'out_of_sync' : 'completed';
        } else {
            status = 'in_progress';
        }
    } else if (outDirExists) {
        status = 'in_progress'; // dir present, no manifest yet
    } else {
        status = 'not_started';
    }

    return {
        ...t,
        dataset_label: resolveDatasetLabel(t.dataset),
        prompts_label,
        prompt_count,
        images_generated,
        images_target,
        status,
        best_median_score,
        new_loras_count
    };
}

export function listTests(testsRoot: string): TestRow[] {
    return (listStmt.all() as TestRecord[]).map(parseRow).map((t) => decorate(t, testsRoot));
}

export function getTest(id: number): Test | null {
    const row = getStmt.get(id) as TestRecord | undefined;
    return row ? parseRow(row) : null;
}

export interface TestInput {
    name: string;
    lora_path: string;
    dataset: DatasetSelector | null;
    /** Legacy free-text path — kept for backward compat. Cleared automatically
     * when prompt_set_id is set. */
    prompts_path: string | null;
    /** Preferred way to attach prompts to a test. */
    prompt_set_id: number | null;
    width: number;
    height: number;
    batch_size: number;
    quant: string;
    offload: string;
    advanced: Record<string, unknown>;
}

function toRecordParams(t: TestInput) {
    // FK and legacy path are mutually exclusive — picking a set clears the
    // path so we never display both as the prompts source for one test.
    const usingSet = t.prompt_set_id != null;
    return {
        name: t.name.trim(),
        lora_path: t.lora_path,
        dataset_path: t.dataset?.kind === 'path' ? t.dataset.path : null,
        dataset_group_id: t.dataset?.kind === 'group' ? t.dataset.id : null,
        prompts_path: usingSet ? null : t.prompts_path,
        prompt_set_id: t.prompt_set_id,
        width: t.width,
        height: t.height,
        batch_size: t.batch_size,
        quant: t.quant,
        offload: t.offload,
        advanced_json: JSON.stringify(t.advanced ?? {})
    };
}

export function createTest(t: TestInput): Test {
    if (!t.name.trim()) throw new Error('Test name is required');
    if (!t.lora_path) throw new Error('LoRA folder is required');
    const res = insertStmt.get(toRecordParams(t)) as { id: number };
    return getTest(res.id)!;
}

export function updateTest(id: number, t: TestInput): Test {
    if (!t.name.trim()) throw new Error('Test name is required');
    if (!t.lora_path) throw new Error('LoRA folder is required');
    updateStmt.run({ id, ...toRecordParams(t) });
    const out = getTest(id);
    if (!out) throw new Error(`Test ${id} not found after update`);
    return out;
}

export function deleteTest(id: number): void {
    deleteStmt.run(id);
}
