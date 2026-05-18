// Lightweight per-phase timing aggregator for long-running jobs. Keeps every
// sample (handfuls of KB at job scale we care about, ~5KB per 1k images per
// phase) and computes percentiles on demand by sort-and-index — exact and
// simple, no t-digest needed for the volumes we run.
export class PhaseStats {
    private samples = new Map<string, number[]>();

    /** Record one timing sample (ms) for a phase. Phase names are free-form
     * — the snapshot reports whatever phases have data. */
    record(phase: string, ms: number): void {
        if (!Number.isFinite(ms) || ms < 0) return;
        let arr = this.samples.get(phase);
        if (!arr) {
            arr = [];
            this.samples.set(phase, arr);
        }
        arr.push(ms);
    }

    /** Snapshot a JSON-serializable view of all phases with count/p50/p95.
     * Phases with zero samples are omitted. Sort cost is O(n log n) per
     * phase per snapshot — fine at our debounce cadence (~1s). */
    snapshot(): PhaseSnapshot {
        const phases: Record<string, PhaseStat> = {};
        for (const [name, arr] of this.samples) {
            if (arr.length === 0) continue;
            const sorted = [...arr].sort((a, b) => a - b);
            phases[name] = {
                count: sorted.length,
                p50: pct(sorted, 0.5),
                p95: pct(sorted, 0.95),
                mean: sorted.reduce((s, v) => s + v, 0) / sorted.length
            };
        }
        return { phases };
    }
}

export interface PhaseStat {
    count: number;
    p50: number;
    p95: number;
    mean: number;
}
export interface PhaseSnapshot {
    phases: Record<string, PhaseStat>;
}

function pct(sortedAsc: number[], q: number): number {
    if (sortedAsc.length === 0) return 0;
    // Nearest-rank — fine for monitoring, no need for linear interpolation.
    const idx = Math.min(sortedAsc.length - 1, Math.floor(q * sortedAsc.length));
    return sortedAsc[idx];
}
