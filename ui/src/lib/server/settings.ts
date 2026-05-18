// Typed settings layer — values land in the `settings` key/value table as
// strings; this module is the single place that knows the schema of each key.
import { resolve } from 'node:path';

import { db } from './db';

export type SettingKey =
    | 'dataset_root'
    | 'tests_root'
    | 'lora_root'
    | 'python_bin'
    /** GPU memory cap (in GiB) the Python worker passes to InsightFace's
     * ONNX CUDA sessions. Bounds VRAM so it doesn't grow unbounded under the
     * default cuDNN EXHAUSTIVE + max-workspace policy. Stored as a string in
     * the kv table; parsed as a number when forwarded to the worker. */
    | 'face_gpu_mem_limit_gb'
    /** Minimum image area (in megapixels) for a connector picture to be
     * eligible as a suggestion. Low-res candidates pollute LoRA training
     * datasets — at 1 MP we're around 1024×1024 which is the comfortable
     * floor for modern face / portrait models. */
    | 'suggestion_min_image_mp';

export type Settings = Record<SettingKey, string>;

export const DEFAULT_SETTINGS: Settings = {
    dataset_root: '',
    tests_root: '',
    lora_root: '',
    python_bin: '',
    // 0 = no cap. We kept the plumbing but a hard cap on ORT's BFC arena
    // causes mid-run crashes when the arena hits the limit (the allocator
    // can't release memory back to grow). Setting >0 is at your own risk.
    face_gpu_mem_limit_gb: '0',
    suggestion_min_image_mp: '1'
};

const SETTING_KEYS: SettingKey[] = [
    'dataset_root',
    'tests_root',
    'lora_root',
    'python_bin',
    'face_gpu_mem_limit_gb',
    'suggestion_min_image_mp'
];

/** Absolute path to the GridLoraTester repo root — where the `glt/` Python
 * package lives. Derived rather than user-configured because the
 * dashboard literally lives at `<repo>/ui/`, so the repo root is one
 * level above the dashboard's cwd. Override with the `GLT_ROOT` env var
 * for unusual deployments (e.g., dashboard packaged independently). */
export function gltRoot(): string {
    return process.env.GLT_ROOT
        ? resolve(process.env.GLT_ROOT)
        : resolve(process.cwd(), '..');
}

const getStmt = db.prepare<[string]>('SELECT value FROM settings WHERE key = ?');
const setStmt = db.prepare<[string, string]>(
    'INSERT INTO settings(key, value) VALUES(?, ?) ' +
        'ON CONFLICT(key) DO UPDATE SET value = excluded.value'
);

export function getSettings(): Settings {
    const out: Settings = { ...DEFAULT_SETTINGS };
    for (const key of SETTING_KEYS) {
        const row = getStmt.get(key) as { value: string } | undefined;
        if (row?.value != null) out[key] = row.value;
    }
    return out;
}

export function updateSettings(patch: Partial<Settings>): Settings {
    const writeMany = db.transaction((entries: [SettingKey, string][]) => {
        for (const [k, v] of entries) setStmt.run(k, v);
    });
    const entries: [SettingKey, string][] = [];
    for (const key of SETTING_KEYS) {
        if (key in patch) entries.push([key, (patch[key] ?? '').trim()]);
    }
    writeMany(entries);
    return getSettings();
}
