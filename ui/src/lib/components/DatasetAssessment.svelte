<script lang="ts">
    import {
        assessDataset,
        TARGET_SOURCES,
        type Severity
    } from '$lib/dataset-targets';
    import type { FramingCoverage } from '$lib/framing-grid';
    import type { PoseCoverage } from '$lib/pose-grid';

    interface Props {
        pose: PoseCoverage | null;
        framing: FramingCoverage | null;
        /** Compact mode hides the sources block (used inside dense layouts). */
        compact?: boolean;
    }

    let { pose, framing, compact = false }: Props = $props();
    let assessment = $derived(assessDataset(framing, pose));

    const severityClasses: Record<Severity, { icon: string; classes: string }> = {
        good: { icon: '✓', classes: 'text-emerald-300' },
        warn: { icon: '⚠', classes: 'text-amber-300' },
        bad: { icon: '✗', classes: 'text-red-300' },
        info: { icon: '•', classes: 'text-fg-muted' }
    };

    function healthClass(h: number): string {
        if (h >= 0.75) return 'bg-emerald-500/15 text-emerald-300';
        if (h >= 0.5) return 'bg-amber-500/15 text-amber-300';
        return 'bg-red-500/15 text-red-300';
    }
</script>

<div class="rounded-lg border border-border bg-bg-1 p-4">
    <header class="mb-3 flex items-baseline justify-between gap-3">
        <h3 class="text-sm font-medium">Dataset health</h3>
        <span
            class="rounded-full px-2 py-0.5 text-xs font-medium tabular-nums {healthClass(
                assessment.health
            )}"
            title="Share of checks passing as 'good' (with half credit for 'warn')"
        >
            {Math.round(assessment.health * 100)}%
        </span>
    </header>

    <ul class="space-y-1.5">
        {#each assessment.recommendations as rec, i (i)}
            {@const s = severityClasses[rec.severity]}
            <li class="flex items-baseline gap-2 text-xs">
                <span class="w-3 shrink-0 text-center font-bold {s.classes}">{s.icon}</span>
                <span class={s.classes}>{rec.text}</span>
            </li>
        {/each}
    </ul>

    {#if !compact}
        <details class="mt-3 text-xs text-fg-faint">
            <summary class="cursor-pointer select-none hover:text-fg-muted">
                Diversity reminders + sources
            </summary>
            <div class="mt-2 space-y-2">
                <p class="text-fg-muted">
                    Beyond the framing / pose split, aim for variety in:
                </p>
                <ul class="list-disc space-y-1 pl-5">
                    <li>3-4 distinct facial expressions</li>
                    <li>3-4 lighting conditions (soft, hard, indoor, outdoor…)</li>
                    <li>varied clothing, backgrounds, and accessories</li>
                    <li>one person per image, 768-1024 px, no Instagram-style filters</li>
                </ul>
                <p class="pt-1">Targets above are pulled from community guides:</p>
                <ul class="list-disc space-y-0.5 pl-5">
                    {#each TARGET_SOURCES as src (src.url)}
                        <li>
                            <a href={src.url} class="hover:text-accent-hover" target="_blank" rel="noopener"
                                >{src.title}</a
                            >
                        </li>
                    {/each}
                </ul>
            </div>
        </details>
    {/if}
</div>
