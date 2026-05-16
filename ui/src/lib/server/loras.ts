// Lists subfolders of `lora_root` — each is a LoRA family / training run.
// Cheap count of `.safetensors` files inside, no recursion.
import { readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

export interface LoraFamily {
    name: string;
    path: string;
    safetensors_count: number;
    /** Highest training step parsed from `<base>_<digits>.safetensors`.
     * `null` when there are no step-suffixed files, or when at least one
     * file has no step suffix (treated as "final"). */
    max_step: number | null;
    /** True when at least one safetensors file has no `_<digits>` suffix
     * (e.g. `final.safetensors`). Mirrors the Python `math.inf` step. */
    has_final: boolean;
}

const STEP_RE = /^(.*?)_(\d+)$/;

/** Extract step from a safetensors basename (extension stripped):
 *   "my-lora_000001400.safetensors" → 1400
 *   "final.safetensors"             → null  (treated as "final")
 */
function parseStep(filename: string): number | null {
    // Strip the .safetensors extension before matching.
    const stem = filename.replace(/\.safetensors$/, '');
    const m = STEP_RE.exec(stem);
    return m ? parseInt(m[2], 10) : null;
}

export function listLoraFamilies(rootPath: string): LoraFamily[] {
    if (!rootPath) return [];
    const root = resolve(rootPath);
    let entries: string[];
    try {
        entries = readdirSync(root);
    } catch {
        return [];
    }
    const out: LoraFamily[] = [];
    for (const name of entries) {
        const full = join(root, name);
        let st: ReturnType<typeof statSync>;
        try {
            st = statSync(full);
        } catch {
            continue;
        }
        if (!st.isDirectory()) continue;
        let count = 0;
        let max_step: number | null = null;
        let has_final = false;
        try {
            for (const f of readdirSync(full)) {
                if (!f.endsWith('.safetensors')) continue;
                // Make sure it's actually a file (not a directory that
                // happens to end in `.safetensors`, and not a broken symlink).
                try {
                    if (!statSync(join(full, f)).isFile()) continue;
                } catch {
                    continue; // broken symlink → skip
                }
                count++;
                const step = parseStep(f);
                if (step == null) has_final = true;
                else if (max_step == null || step > max_step) max_step = step;
            }
        } catch {
            // unreadable — surface with 0
        }
        out.push({ name, path: full, safetensors_count: count, max_step, has_final });
    }
    out.sort((a, b) => a.name.localeCompare(b.name));
    return out;
}

/** Names of every `.safetensors` file directly under `loraPath` (no recursion).
 * Used to detect "new LoRAs since last test run" for the out-of-sync status. */
export function listLoraSafetensorNames(loraPath: string): string[] {
    if (!loraPath) return [];
    try {
        return readdirSync(loraPath).filter((f) => f.endsWith('.safetensors'));
    } catch {
        return [];
    }
}
