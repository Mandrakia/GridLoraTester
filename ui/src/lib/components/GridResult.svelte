<script lang="ts">
    // Native Svelte rendering of a grid test result. Reads the manifest.json
    // produced by the Python pipeline (one row per LoRA, one column per
    // prompt) and lays it out as a sticky-header table with thumbnails and
    // face-similarity score badges. Click an image to open a lightbox.
    //
    // Image URLs go through the asset proxy at
    //   /tests/output/<test_name>/<filename>
    // so we don't have to expose tests_root statically.

    interface ManifestMetrics {
        n_total?: number;
        n_faces?: number;
        mean?: number;
        median?: number;
        std?: number;
        p20?: number;
        p80?: number;
        min?: number;
        max?: number;
    }

    interface ManifestRow {
        lora: string;
        images: (string | null)[];
        scores?: (number | null)[];
        metrics?: ManifestMetrics;
    }

    interface ManifestMeta {
        model?: string;
        width?: number;
        height?: number;
        seed?: number;
        steps?: number;
        guidance?: number;
        lora_scale?: number;
        format?: string;
        fp8?: boolean;
        base_loras?: { name: string; weight: number }[];
        face_recognition?: {
            thresholds?: { good?: number; ok?: number };
            model_name?: string;
        };
    }

    interface Manifest {
        meta?: ManifestMeta;
        prompts?: string[];
        rows?: ManifestRow[];
    }

    interface Props {
        testName: string;
        manifest: Manifest;
    }

    let { testName, manifest }: Props = $props();

    // Color thresholds come from the manifest's face_recognition section
    // when present, falling back to the same defaults the Python HTML uses.
    let thresholds = $derived({
        good: manifest.meta?.face_recognition?.thresholds?.good ?? 0.5,
        ok: manifest.meta?.face_recognition?.thresholds?.ok ?? 0.35
    });

    let rows = $derived(manifest.rows ?? []);
    let prompts = $derived(manifest.prompts ?? []);
    let meta = $derived(manifest.meta ?? {});

    // Pulled out so we can show a face-recognition column only when the
    // pipeline actually scored anything for this run.
    let faceOn = $derived(
        rows.some((r) => (r.metrics?.n_faces ?? 0) > 0 || r.scores != null)
    );

    function imageUrl(filename: string): string {
        // Encode the path segments so spaces / unicode / special chars
        // survive the trip to disk.
        return `/tests/output/${encodeURIComponent(testName)}/${encodeURIComponent(filename)}`;
    }

    function scoreClass(s: number | null | undefined): string {
        if (s == null) return 'text-fg-faint';
        if (s >= thresholds.good) return 'text-emerald-300';
        if (s >= thresholds.ok) return 'text-amber-300';
        return 'text-red-300';
    }

    function scoreBg(s: number | null | undefined): string {
        if (s == null) return '';
        if (s >= thresholds.good) return 'ring-1 ring-emerald-500/40';
        if (s >= thresholds.ok) return 'ring-1 ring-amber-500/40';
        return 'ring-1 ring-red-500/40';
    }

    // Lightbox state.
    let zoom = $state<{ src: string; prompt: string; lora: string; score: number | null } | null>(
        null
    );

    function openZoom(row: ManifestRow, p_idx: number) {
        const img = row.images[p_idx];
        if (!img) return;
        zoom = {
            src: imageUrl(img),
            prompt: prompts[p_idx] ?? '',
            lora: row.lora,
            score: row.scores?.[p_idx] ?? null
        };
    }

    function closeZoom() {
        zoom = null;
    }

    function onZoomKey(e: KeyboardEvent) {
        if (e.key === 'Escape') closeZoom();
    }
</script>

<svelte:window onkeydown={onZoomKey} />

<div class="space-y-4">
    <!-- Meta header (model / size / seed / etc.) -->
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-fg-muted">
        <span><span class="text-fg-faint">Test:</span> <span class="font-mono">{testName}</span></span>
        {#if meta.model}
            <span><span class="text-fg-faint">Model:</span> <span class="font-mono">{meta.model}</span></span>
        {/if}
        {#if meta.width && meta.height}
            <span
                ><span class="text-fg-faint">Size:</span>
                <span class="font-mono">{meta.width}×{meta.height}</span></span
            >
        {/if}
        {#if meta.seed != null}
            <span><span class="text-fg-faint">Seed:</span> <span class="font-mono">{meta.seed}</span></span>
        {/if}
        {#if meta.steps != null}
            <span
                ><span class="text-fg-faint">Steps:</span>
                <span class="font-mono">{meta.steps}</span></span
            >
        {/if}
        {#if meta.guidance != null}
            <span
                ><span class="text-fg-faint">Guidance:</span>
                <span class="font-mono">{meta.guidance}</span></span
            >
        {/if}
        {#if meta.lora_scale != null}
            <span
                ><span class="text-fg-faint">LoRA scale:</span>
                <span class="font-mono">{meta.lora_scale}</span></span
            >
        {/if}
        {#if meta.fp8 != null}
            <span class="rounded-full bg-bg-3 px-2 py-0.5 text-[10px] uppercase tracking-wide">
                {meta.fp8 ? 'fp8' : 'bf16'}
            </span>
        {/if}
    </div>

    {#if rows.length === 0 || prompts.length === 0}
        <div class="card text-sm text-fg-muted">Manifest contains no rows yet.</div>
    {:else}
        <!-- Scrollable grid: sticky first column (LoRA name) + sticky header (prompts) -->
        <div class="overflow-auto rounded-lg border border-border bg-bg-1">
            <table class="border-separate border-spacing-0 text-xs">
                <thead>
                    <tr>
                        <th
                            class="sticky left-0 top-0 z-30 min-w-[14rem] border-b border-r border-border bg-bg-2 px-3 py-2 text-left font-medium text-fg-muted"
                        >
                            LoRA
                        </th>
                        {#each prompts as p, i (i)}
                            <th
                                class="sticky top-0 z-20 min-w-[10rem] max-w-[18rem] border-b border-border bg-bg-2 px-3 py-2 text-left font-normal text-fg-muted"
                            >
                                <div class="line-clamp-3" title={p}>
                                    <span class="mr-1 text-fg-faint">#{i + 1}</span>{p}
                                </div>
                            </th>
                        {/each}
                        {#if faceOn}
                            <th
                                class="sticky top-0 z-20 border-b border-l border-border bg-bg-2 px-3 py-2 text-right font-medium text-fg-muted"
                            >
                                Median
                            </th>
                        {/if}
                    </tr>
                </thead>
                <tbody>
                    {#each rows as row (row.lora)}
                        <tr class="group">
                            <th
                                class="sticky left-0 z-10 max-w-[14rem] truncate border-b border-r border-border bg-bg-1 px-3 py-2 text-left text-fg-muted group-hover:bg-bg-2/40"
                                title={row.lora}
                            >
                                <div class="truncate font-mono text-xs">{row.lora}</div>
                                {#if row.metrics?.n_faces != null && row.metrics?.n_total != null}
                                    <div class="mt-0.5 text-[10px] text-fg-faint">
                                        {row.metrics.n_faces}/{row.metrics.n_total} faces
                                    </div>
                                {/if}
                            </th>
                            {#each row.images as img, i (i)}
                                {@const score = row.scores?.[i] ?? null}
                                <td
                                    class="border-b border-border bg-bg-1 p-1.5 align-top group-hover:bg-bg-2/40"
                                >
                                    {#if img}
                                        <button
                                            type="button"
                                            class="block overflow-hidden rounded-md {scoreBg(
                                                score
                                            )} transition-transform hover:scale-[1.01]"
                                            onclick={() => openZoom(row, i)}
                                            title={prompts[i]}
                                        >
                                            <img
                                                src={imageUrl(img)}
                                                alt={prompts[i] ?? img}
                                                loading="lazy"
                                                class="block h-32 w-full object-cover"
                                            />
                                            {#if score != null}
                                                <span
                                                    class="absolute -mt-6 ml-1 rounded bg-black/60 px-1 text-[10px] font-medium tabular-nums {scoreClass(
                                                        score
                                                    )}"
                                                >
                                                    {score.toFixed(2)}
                                                </span>
                                            {/if}
                                        </button>
                                    {:else}
                                        <div
                                            class="flex h-32 items-center justify-center rounded-md border border-dashed border-border bg-bg-2 text-[10px] text-fg-faint"
                                        >
                                            —
                                        </div>
                                    {/if}
                                </td>
                            {/each}
                            {#if faceOn}
                                {@const m = row.metrics?.median ?? null}
                                <td
                                    class="border-b border-l border-border bg-bg-1 px-3 py-2 text-right align-middle tabular-nums {scoreClass(
                                        m
                                    )} group-hover:bg-bg-2/40"
                                >
                                    {m != null ? m.toFixed(3) : '—'}
                                </td>
                            {/if}
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {/if}
</div>

<!-- Lightbox -->
{#if zoom}
    <div
        class="fixed inset-0 z-50 flex items-stretch justify-center bg-black/80 p-6"
        onclick={closeZoom}
        onkeydown={(e) => {
            if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') closeZoom();
        }}
        role="button"
        tabindex="-1"
        aria-label="Close image"
    >
        <!-- See DatasetView.svelte for the same flex pattern: image area
             flex-grows with min-h-0, caption stays at its natural height. -->
        <div
            class="flex h-full w-full max-w-screen-2xl flex-col gap-2"
            onclick={(e) => e.stopPropagation()}
            role="presentation"
        >
            <div class="flex min-h-0 flex-1 items-center justify-center">
                <img
                    src={zoom.src}
                    alt={zoom.prompt}
                    class="max-h-full max-w-full rounded-md object-contain"
                />
            </div>
            <div class="shrink-0 rounded-md bg-bg-1 p-3 text-xs">
                <div class="mb-1 flex items-center justify-between gap-3">
                    <span class="font-mono text-fg-muted">{zoom.lora}</span>
                    {#if zoom.score != null}
                        <span class="tabular-nums {scoreClass(zoom.score)}"
                            >score: {zoom.score.toFixed(3)}</span
                        >
                    {/if}
                </div>
                <div class="text-fg">{zoom.prompt}</div>
            </div>
        </div>
    </div>
{/if}
