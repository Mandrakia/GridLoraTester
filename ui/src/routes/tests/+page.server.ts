import type { Actions, PageServerLoad } from './$types';
import { fail } from '@sveltejs/kit';

import { listDatasets } from '$lib/server/datasets';
import { listDatasetGroups } from '$lib/server/dataset-groups';
import { enqueue } from '$lib/server/jobs';
import { listLoraFamilies } from '$lib/server/loras';
import { listPromptSets } from '$lib/server/prompts';
import { getSettings } from '$lib/server/settings';
import {
    createTest,
    deleteTest,
    listTests,
    updateTest,
    type DatasetSelector,
    type TestInput
} from '$lib/server/tests';
import { finalizeStaleRunsForTest } from '$lib/server/test-runs';

export const load: PageServerLoad = () => {
    const settings = getSettings();
    return {
        tests_root: settings.tests_root,
        tests: listTests(settings.tests_root),
        loras: listLoraFamilies(settings.lora_root),
        lora_root: settings.lora_root,
        datasets: listDatasets(settings.dataset_root),
        dataset_groups: listDatasetGroups(),
        dataset_root: settings.dataset_root,
        prompt_sets: listPromptSets()
    };
};

/** The dataset combobox sends `dataset:<path>` or `group:<id>`. Decode it. */
function readDatasetSelector(data: FormData): DatasetSelector | null {
    const raw = String(data.get('dataset') ?? '').trim();
    if (!raw) return null;
    if (raw.startsWith('dataset:')) return { kind: 'path', path: raw.slice('dataset:'.length) };
    if (raw.startsWith('group:')) {
        const id = Number(raw.slice('group:'.length));
        return Number.isFinite(id) && id > 0 ? { kind: 'group', id } : null;
    }
    return null;
}

function readNumber(data: FormData, key: string, fallback: number): number {
    const v = data.get(key);
    if (v == null) return fallback;
    const n = Number(String(v));
    return Number.isFinite(n) ? n : fallback;
}

function readAdvanced(data: FormData): Record<string, unknown> {
    // Anything with the `advanced.` prefix flows through to advanced_json
    // verbatim. Booleans come in as 'on' from <input type=checkbox>.
    const out: Record<string, unknown> = {};
    for (const [k, v] of data.entries()) {
        if (!k.startsWith('advanced.')) continue;
        const key = k.slice('advanced.'.length);
        if (typeof v !== 'string') continue;
        if (v === 'on') {
            out[key] = true;
        } else if (v === '' || v === 'off') {
            // skip — represents unchecked / empty
        } else if (/^-?\d+$/.test(v)) {
            out[key] = parseInt(v, 10);
        } else if (/^-?\d*\.\d+$/.test(v)) {
            out[key] = parseFloat(v);
        } else {
            out[key] = v;
        }
    }
    return out;
}

function readInput(data: FormData): TestInput {
    const psRaw = String(data.get('prompt_set_id') ?? '').trim();
    const psId = psRaw && Number.isFinite(Number(psRaw)) ? Number(psRaw) : null;
    return {
        name: String(data.get('name') ?? '').trim(),
        lora_path: String(data.get('lora_path') ?? '').trim(),
        dataset: readDatasetSelector(data),
        prompts_path: String(data.get('prompts_path') ?? '').trim() || null,
        prompt_set_id: psId && psId > 0 ? psId : null,
        trigger: String(data.get('trigger') ?? '').trim(),
        resolution: String(data.get('resolution') ?? '').trim() || '1MP',
        batch_size: Math.max(0, readNumber(data, 'batch_size', 0)),
        quant: String(data.get('quant') ?? 'auto'),
        compile_mode: parseCompileMode(data.get('compile_mode')),
        advanced: readAdvanced(data)
    };
}

function parseCompileMode(raw: FormDataEntryValue | null): 'on' | 'auto' | 'off' {
    const v = String(raw ?? '').trim().toLowerCase();
    if (v === 'auto') return 'auto';
    if (v === 'off' || v === 'false' || v === '0') return 'off';
    // Default for blank / unknown values matches the schema default:
    // compile pays for itself from ~3 images (warm cache), so on-by-default.
    return 'on';
}

export const actions: Actions = {
    create: async ({ request }) => {
        const data = await request.formData();
        const input = readInput(data);
        if (!input.name) return fail(400, { error: 'Test name is required', input });
        if (!input.lora_path) return fail(400, { error: 'Pick a LoRA folder', input });
        try {
            createTest(input);
            return { ok: true };
        } catch (e) {
            return fail(500, { error: (e as Error).message, input });
        }
    },

    update: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        const input = readInput(data);
        if (!input.name) return fail(400, { error: 'Test name is required', input });
        if (!input.lora_path) return fail(400, { error: 'Pick a LoRA folder', input });
        try {
            updateTest(id, input);
            return { ok: true };
        } catch (e) {
            return fail(500, { error: (e as Error).message, input });
        }
    },

    delete: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        deleteTest(id);
        return { ok: true };
    },

    /** Spawn `python -m glt --grid --test-id N` as a background job.
     * Sequential per type — multiple Run clicks queue rather than racing
     * for the GPU. */
    run: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        const settings = getSettings();
        if (!settings.python_bin || !settings.tests_root) {
            return fail(400, {
                error:
                    'Settings incomplete: set python_bin and tests_root in /settings before running tests.'
            });
        }
        try {
            const job_id = enqueue(
                'grid-test-run',
                { test_id: id },
                { key_arg1: String(id) }
            );
            return { ok: true, job_id };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    },

    /** Force-finalize any stuck 'running' test_run for this test. Used
     * when the dashboard knows the job is gone (orphan-reaped or just
     * stale) but the test_runs row never got flipped. Manual escape
     * hatch in addition to the automatic reaper. */
    'reconcile-runs': async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        const n = finalizeStaleRunsForTest(
            id, 'failed',
            'Manually reconciled by user (was stuck in running)',
        );
        return { ok: true, reconciled: n };
    }
};
