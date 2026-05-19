// Grid test definitions: stored in the `tests` table. Run state lives in
// `test_runs` (one row per invocation of `glt --grid`), with per-row +
// per-cell rollups in `test_run_rows` and `test_run_cells`. The decorate
// step pulls the LATEST run for each test to compute status / progress.
//
// Status semantics:
//   not_started  → no test_runs row
//   in_progress  → latest run is status='running' OR completed but not
//                  every (lora × prompt) cell has an image filled in
//   completed    → latest run is status='completed' AND every cell has
//                  an image AND the LoRA folder hasn't gained new files
//   out_of_sync  → completed BUT the LoRA folder grew since (rerun needed)
//   failed       → latest run is status='failed' or 'cancelled'
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

import { db } from './db';
import { getDatasetGroup } from './dataset-groups';
import { listLoraSafetensorNames, parseStep } from './loras';
import { pathBasename } from './path-utils';
import { getPromptSet } from './prompts';

export type TestStatus =
    | 'not_started'
    | 'in_progress'
    | 'completed'
    | 'out_of_sync'
    | 'failed';

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
    /** LoRA trigger word substituted into `[trigger]` placeholders by the
     * grid script at run time. Empty string = leave placeholder as-is.
     * Not interpreted by the dashboard. */
    trigger: string;
    /** Free-form resolution descriptor — replaces the prior width/height
     * pair. The grid script parses it (e.g., "1024x1024", "1024", an AR
     * hint); the dashboard only round-trips the string. */
    resolution: string;
    batch_size: number;
    quant: string;
    compile_mode: 'on' | 'auto' | 'off';
    advanced: Record<string, unknown>;
    created_at: string;
    updated_at: string;
}

export interface ActiveJobRef {
    id: number;
    status: 'queued' | 'running';
    progress_current: number;
    progress_total: number | null;
    current_label: string | null;
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
    /** The most recent queued-or-running grid-test-run job for this test,
     * or null. Links the test row to the live job so the UI can show
     * progress + a deep link to /jobs without waiting for the Python side
     * to create its test_runs row (which only happens after pipeline build,
     * ~30s into the run). */
    active_job: ActiveJobRef | null;
}

type TestRecord = {
    id: number;
    name: string;
    lora_path: string;
    dataset_path: string | null;
    dataset_group_id: number | null;
    prompts_path: string | null;
    prompt_set_id: number | null;
    trigger: string;
    resolution: string;
    batch_size: number;
    quant: string;
    compile_mode: 'on' | 'auto' | 'off';
    advanced_json: string;
    created_at: string;
    updated_at: string;
};

const SELECT_COLS =
    'id, name, lora_path, dataset_path, dataset_group_id, prompts_path, prompt_set_id, ' +
    'trigger, resolution, batch_size, quant, compile_mode, advanced_json, ' +
    'created_at, updated_at';

const listStmt = db.prepare(`SELECT ${SELECT_COLS} FROM tests ORDER BY name ASC`);
const getStmt = db.prepare(`SELECT ${SELECT_COLS} FROM tests WHERE id = ?`);
const insertStmt = db.prepare(`
    INSERT INTO tests(name, lora_path, dataset_path, dataset_group_id, prompts_path, prompt_set_id,
                      trigger, resolution, batch_size, quant, compile_mode, advanced_json)
    VALUES(@name, @lora_path, @dataset_path, @dataset_group_id, @prompts_path, @prompt_set_id,
           @trigger, @resolution, @batch_size, @quant, @compile_mode, @advanced_json)
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
        trigger = @trigger,
        resolution = @resolution,
        batch_size = @batch_size,
        quant = @quant,
        compile_mode = @compile_mode,
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
        trigger: r.trigger ?? '',
        resolution: r.resolution ?? '',
        batch_size: r.batch_size,
        quant: r.quant,
        compile_mode: r.compile_mode,
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
            .filter((ln: string) => ln.trim() && !ln.trim().startsWith('#')).length;
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

// Latest-run lookup. Each test gets at most one row from this query —
// the most recent `test_runs` entry, or NULL when the test has never
// been launched.
const latestRunStmt = db.prepare(`
    SELECT id, status
      FROM test_runs
     WHERE test_id = ?
     ORDER BY started_at DESC
     LIMIT 1
`);

const runCellsAggStmt = db.prepare(`
    SELECT COUNT(*) AS total,
           SUM(CASE WHEN image_filename IS NOT NULL THEN 1 ELSE 0 END) AS done
      FROM test_run_cells
     WHERE run_id = ?
`);

const runBestMedianStmt = db.prepare(`
    SELECT MAX(CAST(json_extract(metrics_json, '$.median') AS REAL)) AS best
      FROM test_run_rows
     WHERE run_id = ?
       AND metrics_json IS NOT NULL
`);

// Distinct LoRA display names known to a run — used to detect new files
// added to the LoRA folder since (out_of_sync).
const runLoraDisplaysStmt = db.prepare(`
    SELECT lora_display FROM test_run_rows WHERE run_id = ?
`);

// Find the most recent queued-or-running grid-test-run job for this test.
// json_extract reads params_json.test_id — type-coerces TEXT/INT either
// way (sqlite is loose). We filter to active states only; a completed/
// failed job from earlier in the day doesn't count as "live".
const activeJobForTestStmt = db.prepare(`
    SELECT id, status, progress_current, progress_total, current_label
      FROM jobs
     WHERE type = 'grid-test-run'
       AND status IN ('queued','running')
       AND CAST(json_extract(params_json, '$.test_id') AS INTEGER) = ?
     ORDER BY id DESC
     LIMIT 1
`);

interface LatestRun {
    id: number;
    status: string;
}

function decorate(t: Test): TestRow {
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
        prompts_label = pathBasename(t.prompts_path);
    }

    let images_generated = 0;
    let images_target = 0;
    let best_median_score: number | null = null;
    let new_loras_count = 0;
    let status: TestStatus = 'not_started';

    // Active job lookup FIRST — if a job is queued or running, the test is
    // effectively in_progress even before Python's test_runs row appears.
    // Without this the UI sits frozen for ~30s after clicking Run while
    // the pipeline boots.
    const activeJobRow = activeJobForTestStmt.get(t.id) as
        | {
              id: number;
              status: 'queued' | 'running';
              progress_current: number;
              progress_total: number | null;
              current_label: string | null;
          }
        | undefined;
    const active_job: ActiveJobRef | null = activeJobRow
        ? {
              id: activeJobRow.id,
              status: activeJobRow.status,
              progress_current: activeJobRow.progress_current,
              progress_total: activeJobRow.progress_total,
              current_label: activeJobRow.current_label
          }
        : null;

    const latest = latestRunStmt.get(t.id) as LatestRun | undefined;
    if (!latest) {
        // No run row exists yet → not_started.
    } else {
        const agg = runCellsAggStmt.get(latest.id) as
            | { total: number | null; done: number | null }
            | undefined;
        images_target = Number(agg?.total ?? 0);
        images_generated = Number(agg?.done ?? 0);

        const best = runBestMedianStmt.get(latest.id) as { best: number | null } | undefined;
        if (best && best.best != null) {
            best_median_score = Number(best.best);
        }

        if (latest.status === 'failed' || latest.status === 'cancelled') {
            status = 'failed';
        } else if (latest.status === 'running') {
            status = 'in_progress';
        } else if (latest.status === 'completed') {
            // Compare on-disk LoRAs vs what the run knows about, gated by
            // the test's min_step: a snapshot under min_step is by design
            // not eligible for this test (the user said "ignore early
            // checkpoints"), so its arrival on disk shouldn't trip
            // "out_of_sync". "final" snapshots (no step → parseStep
            // returns null) always count.
            const knownRows = runLoraDisplaysStmt.all(latest.id) as { lora_display: string }[];
            const known = new Set(knownRows.map((r) => r.lora_display));
            const rawMinStep = Number(t.advanced?.min_step ?? 0);
            const minStep = Number.isFinite(rawMinStep) && rawMinStep > 0 ? rawMinStep : 0;
            const onDisk = listLoraSafetensorNames(t.lora_path);
            new_loras_count = onDisk.filter((n) => {
                // Filter A: already covered by the latest run (basename
                // match; tolerate optional `<dir>/` prefix from multi-
                // dir runs).
                for (const k of known) {
                    if (k === n || k.endsWith('/' + n)) return false;
                }
                // Filter B: below the test's min_step. parseStep returns
                // null for "final" (no step in filename) — those always
                // count as new.
                if (minStep > 0) {
                    const step = parseStep(n);
                    if (step != null && step < minStep) return false;
                }
                return true;
            }).length;
            const allCellsDone = images_target > 0 && images_generated >= images_target;
            if (!allCellsDone) {
                status = 'in_progress';
            } else if (new_loras_count > 0) {
                status = 'out_of_sync';
            } else {
                status = 'completed';
            }
        }
    }

    // Active job overrides whatever the test_runs lookup would say. This
    // covers the gap between job-enqueued and Python-created-test_runs.
    if (active_job) status = 'in_progress';

    return {
        ...t,
        dataset_label: resolveDatasetLabel(t.dataset),
        prompts_label,
        prompt_count,
        images_generated,
        images_target,
        status,
        best_median_score,
        new_loras_count,
        active_job
    };
}

export function listTests(_testsRoot?: string): TestRow[] {
    // testsRoot is unused now — run state lives in the DB. Kept on the
    // signature so the caller can stop passing it on its own schedule.
    return (listStmt.all() as TestRecord[]).map(parseRow).map((t) => decorate(t));
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
    trigger: string;
    resolution: string;
    batch_size: number;
    quant: string;
    compile_mode: 'on' | 'auto' | 'off';
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
        trigger: t.trigger,
        resolution: t.resolution,
        batch_size: t.batch_size,
        quant: t.quant,
        compile_mode: t.compile_mode,
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
