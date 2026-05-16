// Typed settings layer — values land in the `settings` key/value table as
// strings; this module is the single place that knows the schema of each key.
import { db } from './db';

export type SettingKey =
    | 'dataset_root'
    | 'tests_root'
    | 'lora_root'
    | 'python_bin'
    | 'glt_root';

export type Settings = Record<SettingKey, string>;

export const DEFAULT_SETTINGS: Settings = {
    dataset_root: '',
    tests_root: '',
    lora_root: '',
    python_bin: '',
    glt_root: ''
};

const SETTING_KEYS: SettingKey[] = [
    'dataset_root',
    'tests_root',
    'lora_root',
    'python_bin',
    'glt_root'
];

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
