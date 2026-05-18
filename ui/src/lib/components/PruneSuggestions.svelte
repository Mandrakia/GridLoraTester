<script lang="ts">
    // Prune section for the Stats tab: surfaces dataset images flagged as
    // redundant within over-represented buckets, plus a collapsible
    // "Excluded" list so the user can restore a mis-prune. Mirrors the
    // visual rhythm of ConnectorSuggestions so the user can scan both
    // panels with one mental model.
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';

    interface PruneNeighbor {
        image_path: string;
        folder_path: string;
        filename: string;
        thumbnail_url: string;
        full_url: string;
        /** Cosine vs the candidate (not vs centroid). */
        similarity: number;
    }

    interface PruneCandidate {
        image_path: string;
        folder_path: string;
        filename: string;
        thumbnail_url: string;
        full_url: string;
        pose_bucket: string | null;
        framing_bucket: string | null;
        /** Cosine vs centroid. */
        similarity: number | null;
        yaw: number | null;
        pitch: number | null;
        redundancy_score: number;
        over_rep_buckets: string[];
        /** Closest in-bucket peers, sorted by similarity desc. Rendered in
         * the lightbox so the user can pick which of the cluster to drop. */
        neighbors: PruneNeighbor[];
    }

    interface PruneBucket {
        dimension: 'pose' | 'framing';
        group_key: string;
        label: string;
        target: number;
        actual: number;
        over: number;
        candidates: PruneCandidate[];
    }

    interface ExcludedItem {
        image_path: string;
        filename: string;
        thumbnail_url: string;
        excluded_at: string | null;
        excluded_reason: string | null;
    }

    interface Props {
        prune: {
            n_active: number;
            max_size: number | null;
            buckets: PruneBucket[];
        };
        excluded: ExcludedItem[];
        /** Form action endpoint for the exclude post. Folder page: '?/exclude'. */
        excludeAction: string;
        /** Form action endpoint for the restore post. */
        restoreAction: string;
    }

    let { prune, excluded, excludeAction, restoreAction }: Props = $props();

    let showExcluded = $state(false);
    /** image_path currently being acted on — used to disable its tile + spin. */
    let pending = $state<string | null>(null);

    function fmtCount(n: number): string {
        return n === Math.floor(n) ? `${n}` : n.toFixed(1);
    }

    function fmtPct(score: number): string {
        return (score * 100).toFixed(1) + '%';
    }

    function gapBadge(key: string): { cls: string; label: string } {
        if (key.startsWith('framing:')) {
            return { cls: 'bg-sky-500/20 text-sky-200', label: key.slice('framing:'.length) };
        }
        const k = key.slice('pose:'.length);
        if (k === 'tilted') {
            return { cls: 'bg-violet-500/20 text-violet-200', label: 'tilted' };
        }
        return { cls: 'bg-emerald-500/20 text-emerald-200', label: k };
    }

    /** Main image displayed in the lightbox. Can be a candidate, an
     * excluded item, or a peer the user clicked from the rail. */
    let zoom = $state<PruneCandidate | ExcludedItem | PruneNeighbor | null>(null);
    /** The originating PruneCandidate that drives the lightbox rail. Stays
     * fixed when the user clicks a neighbor to swap the main image — so
     * the rail keeps showing the same cluster. Null when the lightbox was
     * opened from an excluded item. */
    let cluster = $state<PruneCandidate | null>(null);

    function isPruneCandidate(
        v: PruneCandidate | ExcludedItem | PruneNeighbor
    ): v is PruneCandidate {
        return 'redundancy_score' in v;
    }
    function isExcluded(
        v: PruneCandidate | ExcludedItem | PruneNeighbor
    ): v is ExcludedItem {
        return 'excluded_at' in v;
    }

    function openCandidate(c: PruneCandidate) {
        zoom = c;
        cluster = c;
    }
    function openExcluded(e: ExcludedItem) {
        zoom = e;
        cluster = null;
    }
    function openNeighbor(n: PruneNeighbor) {
        // Keep the cluster — the rail stays the originating candidate's peers.
        zoom = n;
    }
    function closeLightbox() {
        zoom = null;
        cluster = null;
    }
</script>

<section class="space-y-3 rounded-lg border border-border bg-bg-1 p-4">
    <header class="flex items-start justify-between gap-3">
        <div>
            <h3 class="text-sm font-medium">Prune candidates</h3>
            <p class="text-xs text-fg-faint">
                Most-redundant images within over-represented buckets. Excluding one
                lowers a saturated bucket without hurting diversity (mean cosine to
                its 3 nearest in-bucket neighbors).
            </p>
        </div>
        <div class="text-right text-xs text-fg-muted">
            <div class="tabular-nums">
                {prune.n_active}
                {#if prune.max_size != null}
                    / {prune.max_size}
                    {#if prune.n_active >= prune.max_size}
                        <span class="ml-1 rounded bg-amber-500/20 px-1 text-amber-200">at cap</span>
                    {/if}
                {/if}
            </div>
            <div class="text-fg-faint">active images</div>
        </div>
    </header>

    {#if prune.buckets.length === 0}
        <p class="text-xs text-fg-muted">
            No over-represented bucket{prune.max_size == null
                ? ' — set a max dataset size above to compare actuals against an explicit target'
                : ''}.
        </p>
    {:else}
        <div class="space-y-4">
            {#each prune.buckets as g (g.dimension + ':' + g.group_key)}
                <div class="space-y-2">
                    <div class="flex items-baseline justify-between gap-3">
                        <h4 class="text-sm font-medium">
                            <span
                                class="rounded-full px-1.5 py-0.5 text-[10px] uppercase tracking-wide {g.dimension ===
                                'pose'
                                    ? 'bg-emerald-500/15 text-emerald-300'
                                    : 'bg-sky-500/15 text-sky-300'}"
                            >
                                {g.dimension}
                            </span>
                            <span class="ml-1">{g.label}</span>
                        </h4>
                        <span class="text-xs text-fg-muted tabular-nums">
                            +{fmtCount(g.over)} over · have {fmtCount(g.actual)}/{fmtCount(g.target)}
                        </span>
                    </div>
                    <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
                        {#each g.candidates as c (c.image_path)}
                            {@render pruneTile(c)}
                        {/each}
                    </div>
                </div>
            {/each}
        </div>
    {/if}

    {#if excluded.length > 0}
        <div class="mt-4 rounded-md border border-fg-faint/20 bg-bg-2/40 p-3">
            <button
                type="button"
                class="flex w-full items-baseline justify-between text-left"
                onclick={() => (showExcluded = !showExcluded)}
            >
                <div class="flex items-baseline gap-2">
                    <span class="text-fg-muted">{showExcluded ? '▾' : '▸'}</span>
                    <h4 class="text-sm font-medium">
                        Excluded
                        <span class="ml-1 text-xs text-fg-faint tabular-nums">({excluded.length})</span>
                    </h4>
                </div>
                <span class="text-[10px] text-fg-faint">click to {showExcluded ? 'hide' : 'review'}</span>
            </button>
            {#if showExcluded}
                <p class="mt-2 text-[11px] text-fg-faint">
                    Hidden from centroid + suggestions but still on disk. Restore to bring
                    one back into the active set.
                </p>
                <div
                    class="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8"
                >
                    {#each excluded as e (e.image_path)}
                        {@render excludedTile(e)}
                    {/each}
                </div>
            {/if}
        </div>
    {/if}
</section>

{#snippet pruneTile(c: PruneCandidate)}
    {@const isPending = pending === c.image_path}
    <div
        class="group relative overflow-hidden rounded-md border border-border bg-bg-2"
        title={`redundancy ${fmtPct(c.redundancy_score)} (mean cos to top-3 in-bucket)\n` +
            `pose=${c.pose_bucket ?? '?'} · framing=${c.framing_bucket ?? '?'}` +
            (c.similarity != null ? `\nsim to centroid = ${c.similarity.toFixed(3)}` : '') +
            `\n${c.filename}`}
    >
        <button
            type="button"
            class="block aspect-square w-full cursor-zoom-in"
            onclick={() => openCandidate(c)}
            aria-label={`Preview ${c.filename}`}
        >
            <img
                src={c.thumbnail_url}
                alt={c.filename}
                loading="lazy"
                decoding="async"
                class="h-full w-full object-cover transition-transform group-hover:scale-[1.02]"
            />
        </button>
        <span
            class="pointer-events-none absolute right-1 top-1 rounded bg-black/70 px-1 text-[10px] font-medium tabular-nums text-red-300"
        >
            {fmtPct(c.redundancy_score)}
        </span>
        {#if c.over_rep_buckets.length > 0}
            <div class="pointer-events-none absolute left-1 top-1 flex flex-wrap gap-0.5">
                {#each c.over_rep_buckets as g (g)}
                    {@const b = gapBadge(g)}
                    <span
                        class="rounded px-1 text-[9px] font-medium uppercase tracking-tight {b.cls}"
                        >{b.label}</span
                    >
                {/each}
            </div>
        {/if}
        <form
            method="POST"
            action={excludeAction}
            use:enhance={() => {
                pending = c.image_path;
                return async ({ update }) => {
                    await update({ reset: false });
                    pending = null;
                    await invalidateAll();
                };
            }}
            class="absolute inset-x-1 bottom-1"
        >
            <input type="hidden" name="image_path" value={c.image_path} />
            <button
                type="submit"
                disabled={isPending}
                class="block w-full rounded bg-red-500/85 px-1.5 py-1 text-[10px] font-medium text-white backdrop-blur transition-colors hover:bg-red-500 disabled:opacity-50"
            >
                {isPending ? 'Excluding…' : '− Exclude'}
            </button>
        </form>
    </div>
{/snippet}

{#snippet excludedTile(e: ExcludedItem)}
    {@const isPending = pending === e.image_path}
    <div
        class="group relative overflow-hidden rounded-md border border-border bg-bg-2 opacity-60 transition-opacity hover:opacity-100"
        title={`Excluded${e.excluded_reason ? ': ' + e.excluded_reason : ''}\n${e.filename}`}
    >
        <button
            type="button"
            class="block aspect-square w-full cursor-zoom-in"
            onclick={() => openExcluded(e)}
            aria-label={`Preview ${e.filename}`}
        >
            <img
                src={e.thumbnail_url}
                alt={e.filename}
                loading="lazy"
                decoding="async"
                class="h-full w-full object-cover grayscale"
            />
        </button>
        <form
            method="POST"
            action={restoreAction}
            use:enhance={() => {
                pending = e.image_path;
                return async ({ update }) => {
                    await update({ reset: false });
                    pending = null;
                    await invalidateAll();
                };
            }}
            class="absolute inset-x-1 bottom-1"
        >
            <input type="hidden" name="image_path" value={e.image_path} />
            <button
                type="submit"
                disabled={isPending}
                class="block w-full rounded bg-emerald-500/85 px-1.5 py-1 text-[10px] font-medium text-white backdrop-blur transition-colors hover:bg-emerald-500 disabled:opacity-50"
            >
                {isPending ? 'Restoring…' : '↺ Restore'}
            </button>
        </form>
    </div>
{/snippet}

{#if zoom}
    <!-- Outer overlay closes on click; the inner container stops propagation
         so users can interact with the neighbor rail + exclude forms
         without the lightbox dismissing under them. -->
    <div
        class="fixed inset-0 z-50 flex items-stretch justify-center bg-black/80 p-6"
        onclick={closeLightbox}
        onkeydown={(e) => {
            if (e.key === 'Escape') closeLightbox();
        }}
        role="button"
        tabindex="-1"
        aria-label="Close image"
    >
        <div
            class="flex h-full w-full max-w-screen-2xl gap-3"
            onclick={(e) => e.stopPropagation()}
            role="presentation"
        >
            <!-- Main image column. The image being displayed is `zoom`,
                 which can swap to a neighbor without breaking the rail
                 below (rail follows `cluster`). -->
            <div class="flex min-w-0 flex-1 flex-col gap-2">
                <div class="flex min-h-0 flex-1 items-center justify-center">
                    <img
                        src={'full_url' in zoom ? zoom.full_url : zoom.thumbnail_url}
                        alt={zoom.filename}
                        class="max-h-full max-w-full object-contain"
                    />
                </div>
                <div class="shrink-0 space-y-2 rounded-md bg-bg-1 p-3 text-xs">
                    <div class="flex flex-wrap items-baseline gap-2">
                        <span class="font-mono text-fg">{zoom.filename}</span>
                        {#if isPruneCandidate(zoom)}
                            <span class="text-fg-faint">
                                redundancy {fmtPct(zoom.redundancy_score)}
                            </span>
                            {#if zoom.similarity != null}
                                <span class="text-fg-faint">
                                    · sim to centroid {zoom.similarity.toFixed(3)}
                                </span>
                            {/if}
                            {#if zoom.pose_bucket || zoom.framing_bucket}
                                <span class="text-fg-faint">
                                    · pose={zoom.pose_bucket ?? '?'} · framing={zoom.framing_bucket ??
                                        '?'}
                                </span>
                            {/if}
                        {:else if isExcluded(zoom) && zoom.excluded_reason}
                            <span class="text-fg-faint">· {zoom.excluded_reason}</span>
                        {:else if cluster && !isPruneCandidate(zoom)}
                            <!-- zoom is a neighbor: show its sim TO the cluster
                                 candidate (not to centroid) since that's
                                 the relevant comparison in this view. -->
                            {@const zoomedPath = zoom.image_path}
                            {@const peer = cluster.neighbors.find(
                                (p) => p.image_path === zoomedPath
                            )}
                            {#if peer}
                                <span class="text-fg-faint">
                                    sim to candidate {peer.similarity.toFixed(3)}
                                </span>
                            {/if}
                            <span class="text-fg-faint">
                                · peer of <span class="font-mono">{cluster.filename}</span>
                            </span>
                        {/if}
                    </div>
                    {#if !isExcluded(zoom)}
                        {@const targetPath = zoom.image_path}
                        {@const isPending = pending === targetPath}
                        <form
                            method="POST"
                            action={excludeAction}
                            use:enhance={() => {
                                pending = targetPath;
                                return async ({ update }) => {
                                    await update({ reset: false });
                                    pending = null;
                                    closeLightbox();
                                    await invalidateAll();
                                };
                            }}
                        >
                            <input type="hidden" name="image_path" value={targetPath} />
                            <button
                                type="submit"
                                disabled={isPending}
                                class="rounded bg-red-500/85 px-3 py-1 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-50"
                            >
                                {isPending ? 'Excluding…' : '− Exclude this one'}
                            </button>
                        </form>
                    {/if}
                </div>
            </div>

            <!-- Neighbor rail. Driven by `cluster` (the originating
                 candidate), so clicking a peer to swap the main image
                 doesn't change the rail. Skipped entirely on the excluded
                 path — no cluster context to compare against. -->
            {#if cluster && cluster.neighbors.length > 0}
                <aside
                    class="flex w-56 shrink-0 flex-col gap-2 overflow-y-auto rounded-md bg-bg-1/80 p-2"
                >
                    <p class="px-1 text-[11px] text-fg-faint">
                        Closest in-bucket peers — click to compare, then exclude whichever
                        member of the cluster to drop.
                    </p>
                    <!-- Candidate itself at the top of the rail so the user
                         can swap back after exploring peers. -->
                    {@render railTile(cluster, null, zoom.image_path === cluster.image_path)}
                    {#each cluster.neighbors as n (n.image_path)}
                        {@render railTile(n, n.similarity, zoom.image_path === n.image_path)}
                    {/each}
                </aside>
            {/if}
        </div>
    </div>
{/if}

{#snippet railTile(
    item: PruneCandidate | PruneNeighbor,
    sim: number | null,
    isActive: boolean
)}
    {@const isPending = pending === item.image_path}
    <div
        class="rounded-md border bg-bg-2 p-1.5 {isActive
            ? 'border-accent ring-1 ring-accent'
            : 'border-border'}"
    >
        <button
            type="button"
            class="relative block aspect-square w-full overflow-hidden rounded"
            onclick={() => {
                if (isPruneCandidate(item)) {
                    // Swapping back to the originating candidate.
                    zoom = item;
                } else {
                    openNeighbor(item);
                }
            }}
            title={(sim != null ? `sim ${sim.toFixed(3)} · ` : '') + item.filename}
            aria-label={`Show ${item.filename}`}
        >
            <img
                src={item.thumbnail_url}
                alt={item.filename}
                loading="lazy"
                decoding="async"
                class="h-full w-full object-cover"
            />
            {#if sim != null}
                <span
                    class="pointer-events-none absolute right-1 top-1 rounded bg-black/70 px-1 text-[10px] font-medium tabular-nums text-amber-200"
                >
                    {sim.toFixed(2)}
                </span>
            {:else}
                <span
                    class="pointer-events-none absolute left-1 top-1 rounded bg-black/70 px-1 text-[10px] font-medium uppercase tracking-tight text-red-300"
                >
                    cand
                </span>
            {/if}
        </button>
        <div class="mt-1 truncate font-mono text-[9px] text-fg-faint" title={item.filename}>
            {item.filename}
        </div>
        <form
            method="POST"
            action={excludeAction}
            use:enhance={() => {
                pending = item.image_path;
                return async ({ update }) => {
                    await update({ reset: false });
                    pending = null;
                    closeLightbox();
                    await invalidateAll();
                };
            }}
            class="mt-1"
        >
            <input type="hidden" name="image_path" value={item.image_path} />
            <button
                type="submit"
                disabled={isPending}
                class="block w-full rounded bg-red-500/75 px-1.5 py-1 text-[10px] font-medium text-white hover:bg-red-500 disabled:opacity-50"
            >
                {isPending ? '…' : '− Exclude'}
            </button>
        </form>
    </div>
{/snippet}
