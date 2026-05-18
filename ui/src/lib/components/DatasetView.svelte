<script lang="ts">
    // Grid view for a single dataset (image + caption per cell), shared by
    // /datasets/folder/[name] (1 view) and /datasets/group/[id] (N views).
    // The slider that controls `cols` lives at the page level so a group page
    // can drive every grid with one shared knob.
    import {
        DELTA_AMBER,
        DELTA_RED,
        poseSimOffset,
        type PoseOffsets
    } from '$lib/centroid-thresholds';

    interface DatasetItem {
        filename: string;
        caption: string;
        /** Active vs excluded vs no DB row yet (manual file not analyzed).
         * Excluded items render greyscale + an "excluded" badge so the user
         * sees them in the grid without confusing them for active members. */
        status?: 'active' | 'excluded' | null;
        similarity: number | null;
        face_count: number | null;
        yaw: number | null;
        pitch: number | null;
    }

    interface Props {
        /** Display label shown above the grid (folder basename). */
        name: string;
        /** Full path to the dataset (for the muted subtitle). */
        path: string;
        items: DatasetItem[];
        /** Server endpoint that streams the raw image for a given filename. */
        imageUrlFor: (filename: string) => string;
        /** Current columns count (driven by the parent slider). */
        cols: number;
        /** Optional badge to show next to the header (e.g. "(in group)"). */
        badge?: string;
        /** Median similarity for this dataset (from the centroid run). Drives
         * the per-cell delta badges. `null` when no centroid yet. */
        medianSim?: number | null;
        /** Data-driven pose offsets (override the hardcoded defaults in
         * `centroid-thresholds.ts`). Pass null to use defaults across the
         * board. */
        poseOffsets?: PoseOffsets | null;
    }

    let {
        name,
        path,
        items,
        imageUrlFor,
        cols,
        badge,
        medianSim = null,
        poseOffsets = null
    }: Props = $props();

    function deltaClasses(d: number): string {
        if (d < DELTA_RED) return 'bg-red-500/85 text-white';
        if (d < DELTA_AMBER) return 'bg-amber-500/85 text-black';
        return 'bg-emerald-500/85 text-black';
    }

    // Pick a fixed thumbnail width once: the browser will downscale via CSS
    // when columns get tighter, but never has to re-decode the source. We
    // pick 512 because (a) it covers a wide grid (2 cols on a ~1100px panel
    // ≈ 550px each), and (b) the cache key includes width so this is the
    // single bucket the disk cache will grow into.
    const THUMB_W = 512;
    const thumbUrl = (filename: string) => `${imageUrlFor(filename)}?w=${THUMB_W}`;

    let zoom = $state<{ src: string; filename: string; caption: string } | null>(null);

    function onKey(e: KeyboardEvent) {
        if (e.key === 'Escape') zoom = null;
    }
</script>

<svelte:window onkeydown={onKey} />

<section>
    <header class="mb-3 flex items-baseline justify-between gap-4">
        <div class="min-w-0">
            <h2 class="truncate text-base font-medium" title={name}>
                {name}
                {#if badge}
                    <span class="ml-1 rounded-full bg-bg-3 px-2 py-0.5 text-[10px] uppercase tracking-wide text-fg-muted"
                        >{badge}</span
                    >
                {/if}
            </h2>
            <p class="truncate font-mono text-xs text-fg-faint" title={path}>{path}</p>
        </div>
        <span class="shrink-0 text-xs text-fg-muted">{items.length} image{items.length === 1 ? '' : 's'}</span>
    </header>

    {#if items.length === 0}
        <div class="card text-sm text-fg-muted">No images found in this folder.</div>
    {:else}
        <div
            class="grid gap-3"
            style="grid-template-columns: repeat({Math.max(1, cols)}, minmax(0, 1fr));"
        >
            {#each items as item (item.filename)}
                {@const isExcluded = item.status === 'excluded'}
                <figure
                    class="overflow-hidden rounded-md border bg-bg-1 transition-opacity {isExcluded
                        ? 'border-red-500/40 opacity-60 hover:opacity-100'
                        : 'border-border'}"
                    style="content-visibility: auto; contain-intrinsic-size: 1px 320px;"
                >
                    <button
                        type="button"
                        class="relative block aspect-square w-full overflow-hidden bg-bg-2 transition-transform hover:scale-[1.02]"
                        onclick={() =>
                            (zoom = {
                                src: imageUrlFor(item.filename),
                                filename: item.filename,
                                caption: item.caption
                            })}
                        title={item.filename}
                    >
                        <img
                            src={thumbUrl(item.filename)}
                            alt={item.filename}
                            loading="lazy"
                            decoding="async"
                            class="block h-full w-full object-cover {isExcluded ? 'grayscale' : ''}"
                        />

                        {#if isExcluded}
                            <span
                                class="pointer-events-none absolute left-1 top-1 rounded bg-red-500/85 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white shadow-sm"
                                title="Excluded from centroid + suggestions. Restore from the Stats tab."
                            >
                                excluded
                            </span>
                        {/if}

                        {#if item.face_count != null}
                            {#if item.similarity != null && medianSim != null}
                                {@const rawDelta = item.similarity - medianSim}
                                {@const poseOffset = poseSimOffset(
                                    item.yaw,
                                    item.pitch,
                                    poseOffsets
                                )}
                                {@const tempered = rawDelta + poseOffset}
                                <span
                                    class="absolute right-1 top-1 rounded px-1.5 py-0.5 text-[10px] font-semibold tabular-nums shadow-sm {deltaClasses(
                                        tempered
                                    )}"
                                    title={`sim=${item.similarity.toFixed(3)} · median=${medianSim.toFixed(3)}
raw delta = ${rawDelta >= 0 ? '+' : ''}${rawDelta.toFixed(3)}${poseOffset > 0 ? `\npose offset = +${poseOffset.toFixed(3)} (off-axis shot)\ntempered delta = ${tempered >= 0 ? '+' : ''}${tempered.toFixed(3)}` : ''}`}
                                >
                                    {tempered >= 0 ? '+' : ''}{tempered.toFixed(2)}
                                </span>
                            {:else if item.face_count === 0}
                                <span
                                    class="absolute right-1 top-1 rounded bg-bg-0/80 px-1.5 py-0.5 text-[10px] font-medium text-fg-faint shadow-sm"
                                    title="No face detected during last centroid run"
                                >
                                    no face
                                </span>
                            {:else if item.face_count > 1}
                                <span
                                    class="absolute right-1 top-1 rounded bg-bg-0/80 px-1.5 py-0.5 text-[10px] font-medium text-amber-300 shadow-sm"
                                    title={`${item.face_count} faces detected — target picked via two-pass`}
                                >
                                    {item.face_count}f
                                </span>
                            {/if}
                        {/if}
                    </button>
                    <figcaption class="space-y-1 p-2">
                        <div class="truncate font-mono text-[10px] text-fg-faint" title={item.filename}>
                            {item.filename}
                        </div>
                        <div
                            class="line-clamp-3 text-xs leading-snug text-fg-muted"
                            title={item.caption}
                        >
                            {item.caption || '—'}
                        </div>
                    </figcaption>
                </figure>
            {/each}
        </div>
    {/if}
</section>

<!-- Lightbox -->
{#if zoom}
    <div
        class="fixed inset-0 z-50 flex items-stretch justify-center bg-black/80 p-6"
        onclick={() => (zoom = null)}
        onkeydown={(e) => {
            if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') zoom = null;
        }}
        role="button"
        tabindex="-1"
        aria-label="Close image"
    >
        <!-- inner column: takes all available vertical space, lets the image area
             flex-grow while the caption sits below with its natural height.
             min-h-0 on the image area is critical: without it, flex children
             default to min-content and the image can push the column past 100%. -->
        <div
            class="flex h-full w-full max-w-screen-2xl flex-col gap-2"
            onclick={(e) => e.stopPropagation()}
            role="presentation"
        >
            <div class="flex min-h-0 flex-1 items-center justify-center">
                <img
                    src={zoom.src}
                    alt={zoom.filename}
                    class="max-h-full max-w-full rounded-md object-contain"
                />
            </div>
            <div class="shrink-0 rounded-md bg-bg-1 p-3 text-xs">
                <div class="mb-1 font-mono text-fg-faint">{zoom.filename}</div>
                <div class="whitespace-pre-wrap text-fg">{zoom.caption || '—'}</div>
            </div>
        </div>
    </div>
{/if}
