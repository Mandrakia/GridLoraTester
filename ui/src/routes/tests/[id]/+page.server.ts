import type { Actions, PageServerLoad } from './$types';
import { error, fail } from '@sveltejs/kit';

import { db } from '$lib/server/db';
import { enqueue } from '$lib/server/jobs/runner';
import { getSettings } from '$lib/server/settings';

interface TestDef {
    id: number;
    name: string;
    lora_path: string;
    trigger: string;
    resolution: string;
}

interface RunRow {
    id: number;
    test_id: number;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    started_at: string;
    finished_at: string | null;
    config_json: string;
    error: string | null;
}

interface DbRow {
    run_id: number;
    lora_idx: number;
    lora_display: string;
    metrics_json: string | null;
}

interface DbCell {
    run_id: number;
    lora_idx: number;
    prompt_idx: number;
    prompt_text: string;
    prompt_width: number;
    prompt_height: number;
    image_filename: string | null;
    face_score: number | null;
}

interface ActiveJob {
    id: number;
    status: 'queued' | 'running';
    progress_current: number;
    progress_total: number | null;
    current_label: string | null;
}

const testStmt = db.prepare<[number]>(
    'SELECT id, name, lora_path, trigger, resolution FROM tests WHERE id = ?'
);
// All runs for this test, newest first. We use them to (1) decorate the
// page header (latest = "current"), (2) populate per-row run selectors,
// (3) gather a unified pool of (lora_display, run_id) -> cells/metrics.
const allRunsStmt = db.prepare<[number]>(
    'SELECT * FROM test_runs WHERE test_id = ? ORDER BY id DESC'
);
const rowsForRunStmt = db.prepare<[number]>(
    'SELECT * FROM test_run_rows WHERE run_id = ? ORDER BY lora_idx ASC'
);
const cellsForRunStmt = db.prepare<[number]>(
    'SELECT * FROM test_run_cells WHERE run_id = ? ORDER BY lora_idx ASC, prompt_idx ASC'
);
const activeJobStmt = db.prepare<[number]>(`
    SELECT id, status, progress_current, progress_total, current_label
      FROM jobs
     WHERE type = 'grid-test-run'
       AND status IN ('queued','running')
       AND CAST(json_extract(params_json, '$.test_id') AS INTEGER) = ?
     ORDER BY id DESC
     LIMIT 1
`);

export const load: PageServerLoad = ({ params }) => {
    const id = Number(params.id);
    if (!Number.isFinite(id) || id <= 0) throw error(404, 'Bad test id');

    const test = testStmt.get(id) as TestDef | undefined;
    if (!test) throw error(404, 'Test not found');

    const allRuns = (allRunsStmt.all(id) as RunRow[]) ?? [];
    const latestRun = allRuns[0] ?? null;
    const active_job = (activeJobStmt.get(id) as ActiveJob | undefined) ?? null;

    // For each run, load its rows + cells. The grid is a composite: for
    // each (lora_display, chosen run_id) pick that run's cells. So we
    // need everything indexed by both axes. Storage cost is bounded —
    // typical test history is ~5-30 runs, each with ~5-15 LoRAs ×
    // 3-10 prompts = a few hundred rows total per page load.
    const runConfigs = allRuns.map((r) => {
        let cfg: Record<string, unknown> = {};
        try {
            cfg = JSON.parse(r.config_json) as Record<string, unknown>;
        } catch {
            // ignore corrupt config_json; row keeps an empty cfg
        }
        return {
            id: r.id,
            status: r.status,
            started_at: r.started_at,
            finished_at: r.finished_at,
            error: r.error,
            // Surface the most-watched config fields at top level for
            // the row-selector pill (no need for client-side JSON parse).
            quant: typeof cfg.quant === 'string' ? cfg.quant : null,
            compile_mode: typeof cfg.compile_mode === 'string' ? cfg.compile_mode : null,
            seed: typeof cfg.seed === 'number' ? cfg.seed : null,
            steps: typeof cfg.steps === 'number' ? cfg.steps : null,
            guidance: typeof cfg.guidance === 'number' ? cfg.guidance : null,
            lora_scale: typeof cfg.lora_scale === 'number' ? cfg.lora_scale : null,
            min_step: typeof cfg.min_step === 'number' ? cfg.min_step : null,
            qwen_dtype: typeof cfg.qwen_dtype === 'string' ? cfg.qwen_dtype : null,
            sage_attention: typeof cfg.sage_attention === 'boolean' ? cfg.sage_attention : null,
            // The raw object is also exposed for the (i) tooltip — the
            // pill only shows the curated fields above.
            config: cfg
        };
    });

    // Aggregate (lora_display, run_id) → {metrics, cells[]}. lora_idx is
    // per-run (a 5800 LoRA could be lora_idx=0 in run A and lora_idx=3 in
    // run B if min_step changed), so we match by display name.
    type LoraRunData = {
        lora_display: string;
        run_id: number;
        lora_idx: number;
        metrics: Record<string, number> | null;
        cells: {
            prompt_idx: number;
            prompt_text: string;
            prompt_width: number;
            prompt_height: number;
            image_filename: string | null;
            face_score: number | null;
            image_url: string | null;
        }[];
    };
    const loraRuns: LoraRunData[] = [];
    // Distinct prompts per run, used to derive the canonical column set.
    const promptsByRun = new Map<
        number,
        Map<number, { idx: number; text: string; width: number; height: number }>
    >();

    for (const run of allRuns) {
        const rows = rowsForRunStmt.all(run.id) as DbRow[];
        const cells = cellsForRunStmt.all(run.id) as DbCell[];

        // Build per-run prompt set.
        const promptMap = new Map<number, { idx: number; text: string; width: number; height: number }>();
        for (const c of cells) {
            if (!promptMap.has(c.prompt_idx)) {
                promptMap.set(c.prompt_idx, {
                    idx: c.prompt_idx,
                    text: c.prompt_text,
                    width: c.prompt_width,
                    height: c.prompt_height
                });
            }
        }
        promptsByRun.set(run.id, promptMap);

        // Index cells by lora_idx for this run so we can attach them to
        // the matching row entry.
        const cellsByLora = new Map<number, DbCell[]>();
        for (const c of cells) {
            const list = cellsByLora.get(c.lora_idx);
            if (list) list.push(c);
            else cellsByLora.set(c.lora_idx, [c]);
        }

        for (const r of rows) {
            const myCells = cellsByLora.get(r.lora_idx) ?? [];
            loraRuns.push({
                lora_display: r.lora_display,
                run_id: run.id,
                lora_idx: r.lora_idx,
                metrics: r.metrics_json
                    ? (JSON.parse(r.metrics_json) as Record<string, number>)
                    : null,
                cells: myCells.map((c) => ({
                    prompt_idx: c.prompt_idx,
                    prompt_text: c.prompt_text,
                    prompt_width: c.prompt_width,
                    prompt_height: c.prompt_height,
                    image_filename: c.image_filename,
                    face_score: c.face_score,
                    image_url: c.image_filename
                        ? `/tests/output/${encodeURIComponent(test.name)}/run_${run.id}/${encodeURIComponent(c.image_filename)}`
                        : null
                }))
            });
        }
    }

    // Distinct lora_displays across all runs, preserving "newest first"
    // discovery order so a freshly-trained LoRA appears at the top.
    // Within that, ties are broken by alphabetical order so the layout
    // is stable across reloads.
    const seenLoras = new Set<string>();
    const loraDisplays: string[] = [];
    for (const entry of loraRuns) {
        if (!seenLoras.has(entry.lora_display)) {
            seenLoras.add(entry.lora_display);
            loraDisplays.push(entry.lora_display);
        }
    }
    loraDisplays.sort();

    // Canonical prompts = latest run's prompts. If the latest run hasn't
    // pre-populated its cells yet (queued / just started), fall back to
    // the most recent run that has at least one prompt. Empty list when
    // the test was never run.
    let prompts: { idx: number; text: string; width: number; height: number }[] = [];
    for (const run of allRuns) {
        const m = promptsByRun.get(run.id);
        if (m && m.size > 0) {
            prompts = [...m.values()].sort((a, b) => a.idx - b.idx);
            break;
        }
    }

    // Default per-row run selection: latest run that has at least one
    // GENERATED image for this LoRA. A run can pre-populate `test_run_rows`
    // and `test_run_cells` skeleton entries before any image lands on
    // disk (queued/running, or failed mid-generation) — picking it as
    // default would show empty placeholders for a LoRA that has perfectly
    // good results in a previous run. So we prefer the latest run with
    // actual images and fall back to "latest with the LoRA at all" only
    // when no run ever produced an image (e.g. the test never completed
    // anywhere — fine to show placeholders in that case).
    const defaultRunByLora: Record<string, number> = {};
    for (const lora of loraDisplays) {
        // loraRuns is in newest-first run order (allRuns is ORDER BY id DESC).
        let chosen: number | null = null;
        for (const entry of loraRuns) {
            if (entry.lora_display !== lora) continue;
            if (entry.cells.some((c) => c.image_filename != null)) {
                chosen = entry.run_id;
                break;
            }
        }
        if (chosen == null) {
            // No run ever generated an image for this LoRA — fall back
            // to latest run containing the LoRA so the row at least
            // shows the "generating…" placeholders coherently.
            for (const entry of loraRuns) {
                if (entry.lora_display === lora) {
                    chosen = entry.run_id;
                    break;
                }
            }
        }
        if (chosen != null) defaultRunByLora[lora] = chosen;
    }

    return {
        test,
        run: latestRun, // page-level "current run" for header status
        active_job,
        prompts,
        runs: runConfigs,
        loraDisplays,
        loraRuns,
        defaultRunByLora,
        // Legacy: keep `config` pointing to the latest run's snapshot so
        // any UI that hadn't migrated to per-row picks still sees it.
        config: runConfigs[0]?.config ?? null
    };
};

/** Queue a rescore-only job for this test: re-runs face scoring on the
 * latest test_run's images without regenerating. Useful after a centroid
 * recompute, or when face_score cells were NULL because face detection
 * lagged behind generation. Reuses the standard jobs pipeline so cancel,
 * orphan reaping, and log streaming all work the same way. */
export const actions: Actions = {
    rescore: async ({ params, request }) => {
        const id = Number(params.id);
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        const data = await request.formData();
        const force = String(data.get('force') ?? '') === '1';
        const settings = getSettings();
        if (!settings.python_bin || !settings.tests_root) {
            return fail(400, {
                error: 'Settings incomplete: set python_bin and tests_root in /settings.'
            });
        }
        try {
            const job_id = enqueue(
                'test-rescore',
                { test_id: id, force },
                { key_arg1: String(id) }
            );
            return { ok: true, job_id };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    }
};
