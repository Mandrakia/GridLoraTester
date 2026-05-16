<script lang="ts">
    // 3 × 5 head-pose coverage grid for a single dataset. Cells show the
    // image count (winning face's bucket), with a subtle heatmap relative
    // to the max count so empty/low-coverage cells stand out.
    import { POSE_BUCKETS, type PoseCoverage } from '$lib/pose-grid';

    interface Props {
        coverage: PoseCoverage;
    }

    let { coverage }: Props = $props();

    const yawLabels = ['profile L', '¾ L', 'front', '¾ R', 'profile R'];
    const pitchLabels: { key: 'up' | 'level' | 'down'; label: string }[] = [
        { key: 'up', label: 'up' },
        { key: 'level', label: 'level' },
        { key: 'down', label: 'down' }
    ];

    let max = $derived(
        Math.max(0, ...POSE_BUCKETS.map((b) => coverage.counts[b.id] ?? 0))
    );

    function cellClass(count: number): string {
        if (count === 0) return 'text-fg-faint bg-bg-2/40';
        // Tailwind doesn't support runtime arbitrary alphas easily — bucket
        // into three tiers based on the ratio to max.
        const ratio = max > 0 ? count / max : 0;
        if (ratio >= 0.66) return 'bg-emerald-500/30 text-fg';
        if (ratio >= 0.33) return 'bg-emerald-500/15 text-fg';
        return 'bg-emerald-500/[0.06] text-fg-muted';
    }

    function cellByBand(pitchBand: string, yawIdx: number): number {
        // POSE_BUCKETS is row-major (pitch outer × yaw inner). Pick the right
        // entry from the flat list.
        const pitchIdx = pitchBand === 'up' ? 0 : pitchBand === 'level' ? 1 : 2;
        return coverage.counts[POSE_BUCKETS[pitchIdx * 5 + yawIdx].id] ?? 0;
    }
</script>

<div class="space-y-2">
    <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
        <table class="w-full text-sm">
            <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                <tr>
                    <th class="w-20 px-3 py-2 text-left font-medium"></th>
                    {#each yawLabels as l (l)}
                        <th class="px-3 py-2 text-center font-medium">{l}</th>
                    {/each}
                </tr>
            </thead>
            <tbody class="divide-y divide-border">
                {#each pitchLabels as p (p.key)}
                    <tr>
                        <th class="bg-bg-2/60 px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-fg-muted"
                            >{p.label}</th
                        >
                        {#each yawLabels as _yl, yawIdx (yawIdx)}
                            {@const n = cellByBand(p.key, yawIdx)}
                            <td
                                class="px-3 py-2 text-center tabular-nums {cellClass(n)}"
                                title={POSE_BUCKETS[
                                    (p.key === 'up' ? 0 : p.key === 'level' ? 1 : 2) * 5 + yawIdx
                                ].description}
                            >
                                {n || '—'}
                            </td>
                        {/each}
                    </tr>
                {/each}
            </tbody>
        </table>
    </div>
    <p class="text-xs text-fg-faint">
        {coverage.total - coverage.unknown}/{coverage.total} image{coverage.total === 1
            ? ''
            : 's'} classified
        {#if coverage.unknown > 0}
            · <span class="text-amber-400">{coverage.unknown} without pose</span>
            (no face detected, or pose missing — run "Calculate centroid")
        {/if}
    </p>
</div>
