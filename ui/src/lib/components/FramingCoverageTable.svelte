<script lang="ts">
    // Single-row "framing distance" coverage table. Buckets go from extreme
    // close-up on the left to extreme wide on the right — same column count
    // as the pose grid so the two tables line up visually when stacked.
    import { FRAMING_BUCKETS, type FramingCoverage } from '$lib/framing-grid';

    interface Props {
        coverage: FramingCoverage;
    }

    let { coverage }: Props = $props();

    let max = $derived(
        Math.max(0, ...FRAMING_BUCKETS.map((b) => coverage.counts[b.id] ?? 0))
    );

    function cellClass(count: number): string {
        if (count === 0) return 'text-fg-faint bg-bg-2/40';
        const ratio = max > 0 ? count / max : 0;
        if (ratio >= 0.66) return 'bg-sky-500/30 text-fg';
        if (ratio >= 0.33) return 'bg-sky-500/15 text-fg';
        return 'bg-sky-500/[0.06] text-fg-muted';
    }
</script>

<div class="space-y-2">
    <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
        <table class="w-full text-sm">
            <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                <tr>
                    <th class="w-20 px-3 py-2 text-left font-medium">framing</th>
                    {#each FRAMING_BUCKETS as b (b.id)}
                        <th class="px-3 py-2 text-center font-medium" title={b.description}
                            >{b.label}</th
                        >
                    {/each}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th
                        class="bg-bg-2/60 px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-fg-muted"
                        >count</th
                    >
                    {#each FRAMING_BUCKETS as b (b.id)}
                        {@const n = coverage.counts[b.id] ?? 0}
                        <td
                            class="px-3 py-2 text-center tabular-nums {cellClass(n)}"
                            title={b.description}
                        >
                            {n || '—'}
                        </td>
                    {/each}
                </tr>
            </tbody>
        </table>
    </div>
    <p class="text-xs text-fg-faint">
        {coverage.total - coverage.unknown}/{coverage.total} image{coverage.total === 1
            ? ''
            : 's'} classified
        {#if coverage.unknown > 0}
            · <span class="text-amber-400">{coverage.unknown} without framing</span>
            (no face detected, or image dims missing — run "Calculate centroid")
        {/if}
    </p>
</div>
