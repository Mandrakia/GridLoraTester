// Import a run exported from another install (a zip of { manifest.json, images/ }).
//
// Two phases so the user can resolve cross-machine references in a modal before
// anything is written:
//   analyzeManifest() — match the manifest's prompt set / dataset / LoRA against
//     what exists locally, and offer the local options for the ones that don't.
//   commitImport()    — apply the user's choices, create/match the test, insert
//     the run + rows + cells, and extract the images into tests_root.
//
// Matching rules (per design):
//   - prompts: exact equality on prompts_json → reuse, else AUTO-CREATE a set.
//   - dataset: by name (group) / basename (folder) → reuse, else the user picks
//     from a dropdown of their datasets, or "none". Never auto-created.
//   - lora:    by folder name → reuse, else dropdown of their LoRAs, or "none".
import { createWriteStream } from 'node:fs';
import { mkdir, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { basename, join } from 'node:path';
import { pipeline } from 'node:stream/promises';

import { db } from './db';
import { listDatasets } from './datasets';
import { listDatasetGroups } from './dataset-groups';
import { listLoraFamilies } from './loras';
import { createPromptSet } from './prompts';
import { getSettings } from './settings';

import type * as UnzipperModule from 'unzipper';
import type { RunManifest } from './test-export';

// unzipper is CJS; load via Node's require so Vite's SSR ESM interop doesn't
// trip on the missing `default` export (same reason as archiver in the export
// endpoint).
const unzipper = createRequire(import.meta.url)('unzipper') as typeof UnzipperModule;

export interface Option {
    value: string; // 'none' | 'group:<id>' | 'dataset:<path>' | '<lora_path>'
    label: string;
}
export interface ImportAnalysis {
    token: string;
    test: { name: string; exists: boolean };
    prompt: { mode: 'match' | 'create' | 'none'; name: string | null };
    dataset: { from: string | null; matched: string | null; options: Option[] };
    lora: { from: string | null; matched: string | null; options: Option[] };
    counts: { rows: number; cells: number };
}
export interface ImportChoices {
    token: string;
    dataset: string; // one of dataset.options[].value
    lora: string; // one of lora.options[].value
}

const NONE: Option = { value: 'none', label: '— none —' };

function datasetOptions(): Option[] {
    const s = getSettings();
    const groups = listDatasetGroups().map((g) => ({
        value: `group:${g.id}`,
        label: `${g.name} (group)`
    }));
    const folders = listDatasets(s.dataset_root).map((d) => ({
        value: `dataset:${d.path}`,
        label: d.name
    }));
    return [NONE, ...groups, ...folders];
}
function loraOptions(): Option[] {
    const s = getSettings();
    const loras = listLoraFamilies(s.lora_root).map((l) => ({ value: l.path, label: l.name }));
    return [NONE, ...loras];
}

/** Compute the proposed resolution for a manifest against the local install. */
export function analyzeManifest(manifest: RunManifest, token: string): ImportAnalysis {
    const t = manifest.test;

    const testExists =
        db.prepare('SELECT 1 FROM tests WHERE name = ?').get(t.name) !== undefined;

    // prompts: exact content match → reuse, else will auto-create.
    let prompt: ImportAnalysis['prompt'] = { mode: 'none', name: null };
    if (t.prompt_set) {
        const hit = db
            .prepare('SELECT name FROM prompt_sets WHERE prompts_json = ?')
            .get(t.prompt_set.prompts_json) as { name: string } | undefined;
        prompt = hit
            ? { mode: 'match', name: hit.name }
            : { mode: 'create', name: t.prompt_set.name };
    }

    // dataset: match by name (group) / basename (folder).
    const dOpts = datasetOptions();
    let dMatched: string | null = null;
    if (t.dataset) {
        if (t.dataset.kind === 'group') {
            dMatched = dOpts.find((o) => o.label === `${t.dataset!.name} (group)`)?.value ?? null;
        } else {
            dMatched = dOpts.find((o) => o.label === t.dataset!.name)?.value ?? null;
        }
    }

    // lora: match by folder name.
    const lOpts = loraOptions();
    const lMatched = t.lora ? (lOpts.find((o) => o.label === t.lora!.name)?.value ?? null) : null;

    return {
        token,
        test: { name: t.name, exists: testExists },
        prompt,
        dataset: { from: t.dataset?.name ?? null, matched: dMatched, options: dOpts },
        lora: { from: t.lora?.name ?? null, matched: lMatched, options: lOpts },
        counts: { rows: manifest.rows.length, cells: manifest.cells.length }
    };
}

// ---- staged zip on disk: read manifest / extract images ------------------
export async function readManifestFromZip(zipPath: string): Promise<RunManifest> {
    const dir = await unzipper.Open.file(zipPath);
    const entry = dir.files.find((f) => f.path === 'manifest.json');
    if (!entry) throw new Error('zip has no manifest.json — not a GLT run export');
    const buf = await entry.buffer();
    return JSON.parse(buf.toString('utf-8')) as RunManifest;
}

const insertTest = db.prepare(`
    INSERT INTO tests(name, lora_path, dataset_path, dataset_group_id, prompts_path,
                      prompt_set_id, trigger, resolution, batch_size, quant,
                      model_family, compile_mode, advanced_json, created_at, updated_at)
    VALUES(@name, @lora_path, @dataset_path, @dataset_group_id, NULL,
           @prompt_set_id, @trigger, @resolution, @batch_size, @quant,
           @model_family, @compile_mode, @advanced_json, datetime('now'), datetime('now'))
`);
const insertRun = db.prepare(`
    INSERT INTO test_runs(test_id, status, started_at, finished_at, config_json, base_loras_json, face_meta_json, error)
    VALUES(@test_id, @status, @started_at, @finished_at, @config_json, @base_loras_json, @face_meta_json, NULL)
`);
const insertRow = db.prepare(
    'INSERT INTO test_run_rows(run_id, lora_idx, lora_display, metrics_json) VALUES(?,?,?,?)'
);
const insertCell = db.prepare(`
    INSERT INTO test_run_cells(run_id, lora_idx, prompt_idx, prompt_text, prompt_width, prompt_height, image_filename, face_score)
    VALUES(?,?,?,?,?,?,?,?)
`);

/** Validate a user choice is one we actually offered (no path injection). */
function requireValid(value: string, options: Option[], what: string): void {
    if (!options.some((o) => o.value === value)) {
        throw new Error(`invalid ${what} choice: ${value}`);
    }
}

export interface ImportResult {
    test_id: number;
    run_id: number;
    images: number;
}

/** Apply the resolved choices: create/match test, insert run+rows+cells, and
 * extract the images into <tests_root>/<test.name>/run_<new_id>/. */
export async function commitImport(
    manifest: RunManifest,
    choices: { dataset: string; lora: string },
    stagedZipPath: string
): Promise<ImportResult> {
    const t = manifest.test;
    const s = getSettings();
    if (!s.tests_root) throw new Error('tests_root not configured');

    requireValid(choices.dataset, datasetOptions(), 'dataset');
    requireValid(choices.lora, loraOptions(), 'lora');

    // resolve dataset choice → dataset_path / dataset_group_id
    let dataset_path: string | null = null;
    let dataset_group_id: number | null = null;
    if (choices.dataset.startsWith('group:')) {
        dataset_group_id = Number(choices.dataset.slice('group:'.length));
    } else if (choices.dataset.startsWith('dataset:')) {
        dataset_path = choices.dataset.slice('dataset:'.length);
    }
    const lora_path = choices.lora === 'none' ? null : choices.lora;

    // resolve prompt set: reuse exact-content match, else auto-create.
    let prompt_set_id: number | null = null;
    if (t.prompt_set) {
        const hit = db
            .prepare('SELECT id FROM prompt_sets WHERE prompts_json = ?')
            .get(t.prompt_set.prompts_json) as { id: number } | undefined;
        prompt_set_id = hit
            ? hit.id
            : createPromptSet(t.prompt_set.name, JSON.parse(t.prompt_set.prompts_json) as string[]).id;
    }

    // create-or-attach the test, then insert the run + rows + cells in one tx.
    const { test_id, run_id } = db.transaction(() => {
        const existing = db.prepare('SELECT id FROM tests WHERE name = ?').get(t.name) as
            | { id: number }
            | undefined;
        let tid: number;
        if (existing) {
            tid = existing.id; // attach the run to the existing test (refs unchanged)
        } else {
            tid = Number(
                insertTest.run({
                    name: t.name,
                    lora_path,
                    dataset_path,
                    dataset_group_id,
                    prompt_set_id,
                    trigger: t.trigger,
                    resolution: t.resolution,
                    batch_size: t.batch_size ?? 0,
                    quant: t.quant,
                    model_family: t.model_family,
                    compile_mode: t.compile_mode,
                    advanced_json: t.advanced_json
                }).lastInsertRowid
            );
        }
        const rid = Number(
            insertRun.run({
                test_id: tid,
                status: manifest.run.status,
                started_at: manifest.run.started_at,
                finished_at: manifest.run.finished_at,
                config_json: manifest.run.config_json,
                base_loras_json: manifest.run.base_loras_json,
                face_meta_json: manifest.run.face_meta_json
            }).lastInsertRowid
        );
        for (const r of manifest.rows) insertRow.run(rid, r.lora_idx, r.lora_display, r.metrics_json);
        for (const c of manifest.cells)
            insertCell.run(
                rid,
                c.lora_idx,
                c.prompt_idx,
                c.prompt_text,
                c.prompt_width,
                c.prompt_height,
                c.image_filename,
                c.face_score
            );
        return { test_id: tid, run_id: rid };
    })();

    // extract images/ → <tests_root>/<name>/run_<run_id>/  (stream each file)
    const destDir = join(s.tests_root, t.name, `run_${run_id}`);
    await mkdir(destDir, { recursive: true });
    const dir = await unzipper.Open.file(stagedZipPath);
    let images = 0;
    for (const f of dir.files) {
        if (f.type !== 'File' || !f.path.startsWith('images/')) continue;
        const name = basename(f.path);
        if (!name) continue;
        await pipeline(f.stream(), createWriteStream(join(destDir, name)));
        images++;
    }

    await rm(stagedZipPath, { force: true });
    return { test_id, run_id, images };
}
