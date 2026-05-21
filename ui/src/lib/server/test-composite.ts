// The grid a test page shows is a COMPOSITE across all of the test's runs:
// for each LoRA, the most recent run that actually produced an image for it
// (a run may have only regenerated a few newer checkpoints). This module is
// the single source of truth for that selection — both the test-detail page
// loader and the run export consume it, so the rule lives in one place.
//
// Presentation-agnostic on purpose: cells carry `image_filename` + the source
// `run_id` lives on the entry, but NO `image_url` and NO test name. Callers add
// their own surface — the page builds `/tests/output/...` URLs, the exporter
// pulls the files out of the run folders.
import { db } from './db';

export interface CompositeRunRow {
    id: number;
    test_id: number;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    started_at: string;
    finished_at: string | null;
    config_json: string;
    base_loras_json: string | null;
    face_meta_json: string | null;
    error: string | null;
}
export interface PromptCol {
    idx: number;
    text: string;
    width: number;
    height: number;
}
export interface CompositeCell {
    prompt_idx: number;
    prompt_text: string;
    prompt_width: number;
    prompt_height: number;
    image_filename: string | null;
    face_score: number | null;
}
export interface LoraRunEntry {
    lora_display: string;
    run_id: number;
    lora_idx: number;
    metrics: Record<string, number> | null;
    cells: CompositeCell[];
}
export interface Composite {
    /** All runs of the test, newest first (id DESC). */
    allRuns: CompositeRunRow[];
    /** Canonical column set = the newest run that has prompts. */
    prompts: PromptCol[];
    /** Distinct LoRA display names, alphabetically (stable layout). */
    loraDisplays: string[];
    /** Every (lora_display, run) pairing with that run's cells. */
    loraRuns: LoraRunEntry[];
    /** The composite choice: lora_display → the run_id that wins for it. */
    defaultRunByLora: Record<string, number>;
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

const allRunsStmt = db.prepare<[number]>('SELECT * FROM test_runs WHERE test_id = ? ORDER BY id DESC');
const rowsForRunStmt = db.prepare<[number]>(
    'SELECT * FROM test_run_rows WHERE run_id = ? ORDER BY lora_idx ASC'
);
const cellsForRunStmt = db.prepare<[number]>(
    'SELECT * FROM test_run_cells WHERE run_id = ? ORDER BY lora_idx ASC, prompt_idx ASC'
);

export function buildComposite(testId: number): Composite {
    const allRuns = (allRunsStmt.all(testId) as CompositeRunRow[]) ?? [];

    // Index every (lora_display, run) pairing with its cells, and collect each
    // run's prompt set. lora_idx is per-run (a checkpoint can be idx 0 in one
    // run and idx 3 in another if min_step changed), so we key on display name.
    const loraRuns: LoraRunEntry[] = [];
    const promptsByRun = new Map<number, Map<number, PromptCol>>();

    for (const run of allRuns) {
        const rows = rowsForRunStmt.all(run.id) as DbRow[];
        const cells = cellsForRunStmt.all(run.id) as DbCell[];

        const promptMap = new Map<number, PromptCol>();
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
                    face_score: c.face_score
                }))
            });
        }
    }

    // Distinct LoRAs (newest-first discovery), then sorted for a stable layout.
    const seen = new Set<string>();
    const loraDisplays: string[] = [];
    for (const e of loraRuns) {
        if (!seen.has(e.lora_display)) {
            seen.add(e.lora_display);
            loraDisplays.push(e.lora_display);
        }
    }
    loraDisplays.sort();

    // Canonical prompts = the newest run that has any (the latest run may still
    // be queued with no cells yet).
    let prompts: PromptCol[] = [];
    for (const run of allRuns) {
        const m = promptsByRun.get(run.id);
        if (m && m.size > 0) {
            prompts = [...m.values()].sort((a, b) => a.idx - b.idx);
            break;
        }
    }

    // The composite choice: newest run that produced an IMAGE for the LoRA,
    // falling back to newest run containing it (so a never-completed LoRA still
    // shows coherent placeholders). loraRuns is in newest-first run order.
    const defaultRunByLora: Record<string, number> = {};
    for (const lora of loraDisplays) {
        let chosen: number | null = null;
        for (const e of loraRuns) {
            if (e.lora_display !== lora) continue;
            if (e.cells.some((c) => c.image_filename != null)) {
                chosen = e.run_id;
                break;
            }
        }
        if (chosen == null) {
            for (const e of loraRuns) {
                if (e.lora_display === lora) {
                    chosen = e.run_id;
                    break;
                }
            }
        }
        if (chosen != null) defaultRunByLora[lora] = chosen;
    }

    return { allRuns, prompts, loraDisplays, loraRuns, defaultRunByLora };
}
