// Build a self-sufficient manifest for one test run, so a run produced on one
// machine (e.g. a RunPod pod) can be re-created on another with no shared DB.
//
// The run's images travel alongside (the export zips them in); this manifest
// is the data half. The cross-machine-fragile bits are the test's *references*
// — prompt set, dataset, LoRA folder — whose local ids/paths won't match the
// target. So we carry enough to RESOLVE them on import:
//   - prompt_set: name + the full prompts_json → import matches by exact
//     content (string equality) or auto-creates a set.
//   - dataset: name + kind (+ group paths) + source id → import matches by
//     name/id, else the user picks from a dropdown (or "none").
//   - lora: folder name + original path → same match-or-pick.
import { basename } from 'node:path';

import { db } from './db';
import { buildComposite } from './test-composite';

export const MANIFEST_SCHEMA = 1;

export interface ManifestPromptSet {
    name: string;
    /** JSON array of prompt strings — the import match key (exact equality). */
    prompts_json: string;
}
export interface ManifestDataset {
    kind: 'folder' | 'group';
    name: string;
    /** folder: the dataset path; group: null (members live in paths_json). */
    path: string | null;
    /** group: JSON array of member paths; folder: null. */
    paths_json: string | null;
    /** Source-DB id, reference only — import matches by name/content, not id. */
    source_id: number | null;
}
export interface ManifestLora {
    name: string;
    /** Original (source-machine) path, reference only. */
    path: string;
}
export interface ManifestRow {
    lora_idx: number;
    lora_display: string;
    metrics_json: string | null;
}
export interface ManifestCell {
    lora_idx: number;
    prompt_idx: number;
    prompt_text: string;
    prompt_width: number;
    prompt_height: number;
    image_filename: string | null;
    face_score: number | null;
}
export interface RunManifest {
    schema: number;
    exported_at: string;
    test: {
        name: string;
        trigger: string | null;
        resolution: string | null;
        model_family: string | null;
        quant: string | null;
        compile_mode: string | null;
        batch_size: number | null;
        advanced_json: string | null;
        prompt_set: ManifestPromptSet | null;
        dataset: ManifestDataset | null;
        lora: ManifestLora | null;
    };
    run: {
        status: string;
        started_at: string | null;
        finished_at: string | null;
        config_json: string;
        base_loras_json: string | null;
        face_meta_json: string | null;
    };
    rows: ManifestRow[];
    cells: ManifestCell[];
}

interface TestRow {
    id: number;
    name: string;
    lora_path: string | null;
    dataset_path: string | null;
    dataset_group_id: number | null;
    prompt_set_id: number | null;
    trigger: string | null;
    resolution: string | null;
    model_family: string | null;
    quant: string | null;
    compile_mode: string | null;
    batch_size: number | null;
    advanced_json: string | null;
}
interface RunRow {
    id: number;
    test_id: number;
    status: string;
    started_at: string | null;
    finished_at: string | null;
    config_json: string;
    base_loras_json: string | null;
    face_meta_json: string | null;
}

const runStmt = db.prepare('SELECT * FROM test_runs WHERE id = ?');
const testStmt = db.prepare('SELECT * FROM tests WHERE id = ?');
const promptSetStmt = db.prepare('SELECT name, prompts_json FROM prompt_sets WHERE id = ?');
const groupStmt = db.prepare('SELECT id, name, paths_json FROM dataset_groups WHERE id = ?');
const rowsStmt = db.prepare(
    'SELECT lora_idx, lora_display, metrics_json FROM test_run_rows WHERE run_id = ? ORDER BY lora_idx'
);
const cellsStmt = db.prepare(
    `SELECT lora_idx, prompt_idx, prompt_text, prompt_width, prompt_height, image_filename, face_score
       FROM test_run_cells WHERE run_id = ? ORDER BY lora_idx, prompt_idx`
);

/** The test-definition section of a manifest — shared by the single-run and
 * composite exporters: name + knobs + the cross-machine references (prompt set
 * content, dataset name/id, LoRA folder name) the import modal resolves. */
function manifestTest(test: TestRow): RunManifest['test'] {
    let prompt_set: ManifestPromptSet | null = null;
    if (test.prompt_set_id != null) {
        const ps = promptSetStmt.get(test.prompt_set_id) as
            | { name: string; prompts_json: string }
            | undefined;
        if (ps) prompt_set = { name: ps.name, prompts_json: ps.prompts_json };
    }

    let dataset: ManifestDataset | null = null;
    if (test.dataset_group_id != null) {
        const g = groupStmt.get(test.dataset_group_id) as
            | { id: number; name: string; paths_json: string }
            | undefined;
        if (g) {
            dataset = { kind: 'group', name: g.name, path: null, paths_json: g.paths_json, source_id: g.id };
        }
    } else if (test.dataset_path) {
        dataset = {
            kind: 'folder',
            name: basename(test.dataset_path),
            path: test.dataset_path,
            paths_json: null,
            source_id: null
        };
    }

    const lora: ManifestLora | null = test.lora_path
        ? { name: basename(test.lora_path), path: test.lora_path }
        : null;

    return {
        name: test.name,
        trigger: test.trigger,
        resolution: test.resolution,
        model_family: test.model_family,
        quant: test.quant,
        compile_mode: test.compile_mode,
        batch_size: test.batch_size,
        advanced_json: test.advanced_json,
        prompt_set,
        dataset,
        lora
    };
}

/** Manifest for one specific run. */
export function buildRunManifest(runId: number): RunManifest {
    const run = runStmt.get(runId) as RunRow | undefined;
    if (!run) throw new Error(`run ${runId} not found`);
    const test = testStmt.get(run.test_id) as TestRow | undefined;
    if (!test) throw new Error(`test ${run.test_id} not found`);

    return {
        schema: MANIFEST_SCHEMA,
        exported_at: new Date().toISOString(),
        test: manifestTest(test),
        run: {
            status: run.status,
            started_at: run.started_at,
            finished_at: run.finished_at,
            config_json: run.config_json,
            base_loras_json: run.base_loras_json,
            face_meta_json: run.face_meta_json
        },
        rows: rowsStmt.all(runId) as ManifestRow[],
        cells: cellsStmt.all(runId) as ManifestCell[]
    };
}

/** Manifest for the COMPOSITE grid the test page shows: one virtual run where
 * each LoRA's cells come from the run that wins for it (see test-composite.ts —
 * same selection as the view). Also returns the images to pack, each tagged
 * with the run folder it actually lives in. */
export function buildCompositeManifest(testId: number): {
    manifest: RunManifest;
    images: { filename: string; runId: number }[];
} {
    const test = testStmt.get(testId) as TestRow | undefined;
    if (!test) throw new Error(`test ${testId} not found`);

    const comp = buildComposite(testId);
    const rows: ManifestRow[] = [];
    const cells: ManifestCell[] = [];
    const images: { filename: string; runId: number }[] = [];

    comp.loraDisplays.forEach((lora, lora_idx) => {
        const winRun = comp.defaultRunByLora[lora];
        if (winRun == null) return;
        const entry = comp.loraRuns.find((e) => e.lora_display === lora && e.run_id === winRun);
        if (!entry) return;
        rows.push({
            lora_idx,
            lora_display: lora,
            metrics_json: entry.metrics ? JSON.stringify(entry.metrics) : null
        });
        for (const c of entry.cells) {
            cells.push({
                lora_idx,
                prompt_idx: c.prompt_idx,
                prompt_text: c.prompt_text,
                prompt_width: c.prompt_width,
                prompt_height: c.prompt_height,
                image_filename: c.image_filename,
                face_score: c.face_score
            });
            if (c.image_filename) images.push({ filename: c.image_filename, runId: entry.run_id });
        }
    });

    const latest = comp.allRuns[0];
    const manifest: RunManifest = {
        schema: MANIFEST_SCHEMA,
        exported_at: new Date().toISOString(),
        test: manifestTest(test),
        run: {
            // A virtual run standing in for the composite; config/face_meta from
            // the latest run (matches the "current" view).
            status: 'completed',
            started_at: latest?.started_at ?? null,
            finished_at: latest?.finished_at ?? null,
            config_json: latest?.config_json ?? '{}',
            base_loras_json: latest?.base_loras_json ?? null,
            face_meta_json: latest?.face_meta_json ?? null
        },
        rows,
        cells
    };
    return { manifest, images };
}
