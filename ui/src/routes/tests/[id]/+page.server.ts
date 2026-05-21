import type { Actions, PageServerLoad } from './$types';
import { error, fail } from '@sveltejs/kit';

import { db } from '$lib/server/db';
import { enqueue } from '$lib/server/jobs/runner';
import { getSettings } from '$lib/server/settings';
import { buildComposite } from '$lib/server/test-composite';

interface TestDef {
    id: number;
    name: string;
    lora_path: string;
    trigger: string;
    resolution: string;
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

    // The composite selection (which run/cells win per LoRA) lives in
    // test-composite.ts — shared with the run exporter so the rule is defined
    // once. Here we add only presentation: config pills + served image URLs.
    const comp = buildComposite(id);
    const { prompts, loraDisplays, defaultRunByLora } = comp;
    const latestRun = comp.allRuns[0] ?? null;
    const active_job = (activeJobStmt.get(id) as ActiveJob | undefined) ?? null;

    // Per-run config pills — surface the most-watched fields at top level so
    // the row-selector doesn't need a client-side JSON parse.
    const runConfigs = comp.allRuns.map((r) => {
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
            quant: typeof cfg.quant === 'string' ? cfg.quant : null,
            compile_mode: typeof cfg.compile_mode === 'string' ? cfg.compile_mode : null,
            seed: typeof cfg.seed === 'number' ? cfg.seed : null,
            steps: typeof cfg.steps === 'number' ? cfg.steps : null,
            guidance: typeof cfg.guidance === 'number' ? cfg.guidance : null,
            lora_scale: typeof cfg.lora_scale === 'number' ? cfg.lora_scale : null,
            min_step: typeof cfg.min_step === 'number' ? cfg.min_step : null,
            qwen_dtype: typeof cfg.qwen_dtype === 'string' ? cfg.qwen_dtype : null,
            sage_attention: typeof cfg.sage_attention === 'boolean' ? cfg.sage_attention : null,
            config: cfg
        };
    });

    // Attach the served image URL to each composite cell (uses the run the
    // cell actually came from, so multi-run composites resolve correctly).
    const loraRuns = comp.loraRuns.map((e) => ({
        ...e,
        cells: e.cells.map((c) => ({
            ...c,
            image_url: c.image_filename
                ? `/tests/output/${encodeURIComponent(test.name)}/run_${e.run_id}/${encodeURIComponent(c.image_filename)}`
                : null
        }))
    }));

    return {
        test,
        run: latestRun, // page-level "current run" for header status
        active_job,
        prompts,
        runs: runConfigs,
        loraDisplays,
        loraRuns,
        defaultRunByLora,
        // Legacy: keep `config` pointing to the latest run's snapshot so any UI
        // that hadn't migrated to per-row picks still sees it.
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
            const job_id = enqueue('test-rescore', { test_id: id, force }, { key_arg1: String(id) });
            return { ok: true, job_id };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    }
};
