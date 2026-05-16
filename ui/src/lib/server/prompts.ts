// Prompt sets: named bundles of N prompts, stored as a JSON array.
// Same flat-schema convention as dataset_groups — no separate prompt table
// since prompts are short, never queried individually, and edited as a block.
import { db } from './db';

export interface PromptSet {
    id: number;
    name: string;
    prompts: string[];
    created_at: string;
    updated_at: string;
}

export interface PromptSetRow extends PromptSet {
    prompt_count: number;
}

type PromptSetRecord = {
    id: number;
    name: string;
    prompts_json: string;
    created_at: string;
    updated_at: string;
};

const listStmt = db.prepare(
    'SELECT id, name, prompts_json, created_at, updated_at FROM prompt_sets ORDER BY name ASC'
);
const getStmt = db.prepare(
    'SELECT id, name, prompts_json, created_at, updated_at FROM prompt_sets WHERE id = ?'
);
const insertStmt = db.prepare(
    'INSERT INTO prompt_sets(name, prompts_json) VALUES(?, ?) RETURNING id'
);
const updateStmt = db.prepare(
    "UPDATE prompt_sets SET name = ?, prompts_json = ?, updated_at = datetime('now') WHERE id = ?"
);
const deleteStmt = db.prepare('DELETE FROM prompt_sets WHERE id = ?');

function parseRow(r: PromptSetRecord): PromptSet {
    let prompts: string[] = [];
    try {
        const parsed = JSON.parse(r.prompts_json);
        if (Array.isArray(parsed)) prompts = parsed.filter((p) => typeof p === 'string');
    } catch {
        // ignore — corrupt JSON degrades to []. The user can re-save.
    }
    return {
        id: r.id,
        name: r.name,
        prompts,
        created_at: r.created_at,
        updated_at: r.updated_at
    };
}

function decorate(p: PromptSet): PromptSetRow {
    return { ...p, prompt_count: p.prompts.length };
}

/** Normalize a textarea blob into prompts: split on newlines, trim, drop
 * empties and `#`-comment lines. Same convention as the Python CLI. */
export function parsePromptsBlob(blob: string): string[] {
    return blob
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0 && !s.startsWith('#'));
}

export function listPromptSets(): PromptSetRow[] {
    return (listStmt.all() as PromptSetRecord[]).map(parseRow).map(decorate);
}

export function getPromptSet(id: number): PromptSetRow | null {
    const row = getStmt.get(id) as PromptSetRecord | undefined;
    return row ? decorate(parseRow(row)) : null;
}

export function createPromptSet(name: string, prompts: string[]): PromptSetRow {
    const trimmed = name.trim();
    if (!trimmed) throw new Error('Prompt set name is required');
    const res = insertStmt.get(trimmed, JSON.stringify(prompts)) as { id: number };
    return getPromptSet(res.id)!;
}

export function updatePromptSet(id: number, name: string, prompts: string[]): PromptSetRow {
    const trimmed = name.trim();
    if (!trimmed) throw new Error('Prompt set name is required');
    updateStmt.run(trimmed, JSON.stringify(prompts), id);
    const out = getPromptSet(id);
    if (!out) throw new Error(`Prompt set ${id} not found after update`);
    return out;
}

export function deletePromptSet(id: number): void {
    deleteStmt.run(id);
}
