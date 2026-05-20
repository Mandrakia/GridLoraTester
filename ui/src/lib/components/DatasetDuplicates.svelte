<script lang="ts">
    // Duplicates section for the Stats tab: clusters of near-identical
    // dataset images (burst frames, re-edits, re-imports) detected by
    // perceptual-hash Hamming distance. Each cluster suggests one image to
    // keep (highest resolution) and lets the user exclude the rest — either
    // one at a time or the whole tail in one shot. Mirrors the visual rhythm
    // of PruneSuggestions so both panels scan with one mental model.
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';

    interface DuplicateMember {
        image_path: string;
        folder_path: string;
        filename: string;
        image_width: number | null;
        image_height: number | null;
        megapixels: number | null;
        hamming_to_keep: number;
        suggested_keep: boolean;
        thumbnail_url: string;
        full_url: string;
    }

    interface DuplicateCluster {
        id: string;
        members: DuplicateMember[];
        max_hamming: number;
    }

    interface Props {
        duplicates: {
            threshold: number;
            clusters: DuplicateCluster[];
            hashed_images: number;
            total_active: number;
        };
        /** Single-image exclude form action (folder + group both: '?/exclude'). */
        excludeAction: string;
        /** Batch exclude form action ('?/excludeMany') — collapses a cluster
         * with one centroid resync. */
        excludeManyAction: string;
    }

    let { duplicates, excludeAction, excludeManyAction }: Props = $props();

    /** image_path currently being excluded on its own — disables + spins. */
    let pending = $state<string | null>(null);
    /** cluster id currently collapsing via the batch action. */
    let pendingCluster = $state<string | null>(null);
    /** Full-res preview. */
    let zoom = $state<DuplicateMember | null>(null);

    function fmtMp(mp: number | null): string {
        return mp == null ? '?' : mp >= 10 ? `${Math.round(mp)} MP` : `${mp.toFixed(1)} MP`;
    }

    function extras(c: DuplicateCluster): DuplicateMember[] {
        return c.members.filter((m) => !m.suggested_keep);
    }
    function extraPathsJson(c: DuplicateCluster): string {
        return JSON.stringify(extras(c).map((m) => m.image_path));
    }
</script>

<section class="space-y-3 rounded-lg border border-border bg-bg-1 p-4">
    <header class="flex items-start justify-between gap-3">
        <div>
            <h3 class="text-sm font-medium">Duplicates</h3>
            <p class="text-xs text-fg-faint">
                Clusters of near-identical images (perceptual hash within
                <span class="font-mono">{duplicates.threshold}</span>/256 bits — tune in Settings).
                Keep the best take, exclude the redundant ones.
            </p>
        </div>
        <div class="text-right text-xs text-fg-muted">
            <div class="tabular-nums">{duplicates.clusters.length}</div>
            <div class="text-fg-faint">cluster{duplicates.clusters.length === 1 ? '' : 's'}</div>
        </div>
    </header>

    {#if duplicates.hashed_images < duplicates.total_active}
        <p class="rounded bg-amber-500/10 px-2 py-1 text-[11px] text-amber-300/90">
            {duplicates.hashed_images} of {duplicates.total_active} active images are hashed —
            run <span class="font-medium">Analyze</span> so dedup sees the whole set.
        </p>
    {/if}

    {#if duplicates.clusters.length === 0}
        <p class="text-xs text-fg-muted">
            No duplicate clusters at the current sensitivity. Raise the threshold in Settings
            to catch looser same-scene shots.
        </p>
    {:else}
        <div class="space-y-4">
            {#each duplicates.clusters as c (c.id)}
                {@const extraCount = extras(c).length}
                {@const collapsing = pendingCluster === c.id}
                <div class="space-y-2 rounded-md border border-border/70 bg-bg-2/40 p-3">
                    <div class="flex items-baseline justify-between gap-3">
                        <h4 class="text-xs font-medium text-fg-muted">
                            {c.members.length} near-identical
                            <span class="ml-1 text-fg-faint tabular-nums">· Δ≤{c.max_hamming} bits</span>
                        </h4>
                        {#if extraCount > 0}
                            <form
                                method="POST"
                                action={excludeManyAction}
                                use:enhance={() => {
                                    pendingCluster = c.id;
                                    return async ({ update }) => {
                                        await update({ reset: false });
                                        pendingCluster = null;
                                        await invalidateAll();
                                    };
                                }}
                            >
                                <input type="hidden" name="image_paths" value={extraPathsJson(c)} />
                                <input type="hidden" name="reason" value="duplicate" />
                                <button
                                    type="submit"
                                    disabled={collapsing}
                                    class="rounded bg-red-500/85 px-2 py-1 text-[10px] font-medium text-white hover:bg-red-500 disabled:opacity-50"
                                >
                                    {collapsing
                                        ? 'Excluding…'
                                        : `− Exclude the other ${extraCount}`}
                                </button>
                            </form>
                        {/if}
                    </div>
                    <div class="flex flex-wrap gap-2">
                        {#each c.members as m (m.image_path)}
                            {@render memberTile(m, collapsing)}
                        {/each}
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</section>

{#snippet memberTile(m: DuplicateMember, collapsing: boolean)}
    {@const isPending = pending === m.image_path || (collapsing && !m.suggested_keep)}
    <div
        class="group relative w-28 overflow-hidden rounded-md border bg-bg-2 {m.suggested_keep
            ? 'border-emerald-500/70 ring-1 ring-emerald-500/50'
            : 'border-border'}"
        title={`${m.filename}\n${fmtMp(m.megapixels)}` +
            (m.suggested_keep ? '\nsuggested keep (highest resolution)' : `\nΔ${m.hamming_to_keep} bits from keep`)}
    >
        <button
            type="button"
            class="block aspect-square w-full cursor-zoom-in"
            onclick={() => (zoom = m)}
            aria-label={`Preview ${m.filename}`}
        >
            <img
                src={m.thumbnail_url}
                alt={m.filename}
                loading="lazy"
                decoding="async"
                class="h-full w-full object-cover transition-transform group-hover:scale-[1.02]"
            />
        </button>
        {#if m.suggested_keep}
            <span
                class="pointer-events-none absolute left-1 top-1 rounded bg-emerald-500/90 px-1 text-[9px] font-semibold uppercase tracking-tight text-white"
            >
                keep
            </span>
        {:else}
            <span
                class="pointer-events-none absolute left-1 top-1 rounded bg-black/70 px-1 text-[10px] font-medium tabular-nums text-amber-200"
            >
                Δ{m.hamming_to_keep}
            </span>
        {/if}
        <span
            class="pointer-events-none absolute right-1 top-1 rounded bg-black/70 px-1 text-[9px] tabular-nums text-fg-faint"
        >
            {fmtMp(m.megapixels)}
        </span>
        {#if !m.suggested_keep}
            <form
                method="POST"
                action={excludeAction}
                use:enhance={() => {
                    pending = m.image_path;
                    return async ({ update }) => {
                        await update({ reset: false });
                        pending = null;
                        await invalidateAll();
                    };
                }}
                class="absolute inset-x-1 bottom-1"
            >
                <input type="hidden" name="image_path" value={m.image_path} />
                <input type="hidden" name="reason" value="duplicate" />
                <button
                    type="submit"
                    disabled={isPending}
                    class="block w-full rounded bg-red-500/85 px-1.5 py-1 text-[10px] font-medium text-white backdrop-blur transition-colors hover:bg-red-500 disabled:opacity-50"
                >
                    {isPending ? '…' : '− Exclude'}
                </button>
            </form>
        {/if}
    </div>
{/snippet}

{#if zoom}
    <div
        class="fixed inset-0 z-50 flex items-stretch justify-center bg-black/80 p-6"
        onclick={() => (zoom = null)}
        onkeydown={(e) => {
            if (e.key === 'Escape') zoom = null;
        }}
        role="button"
        tabindex="-1"
        aria-label="Close image"
    >
        <div
            class="flex h-full w-full max-w-screen-xl flex-col gap-2"
            onclick={(e) => e.stopPropagation()}
            role="presentation"
        >
            <div class="flex min-h-0 flex-1 items-center justify-center">
                <img src={zoom.full_url} alt={zoom.filename} class="max-h-full max-w-full object-contain" />
            </div>
            <div class="flex shrink-0 flex-wrap items-center gap-3 rounded-md bg-bg-1 p-3 text-xs">
                <span class="font-mono text-fg">{zoom.filename}</span>
                <span class="text-fg-faint">{fmtMp(zoom.megapixels)}</span>
                {#if zoom.suggested_keep}
                    <span class="rounded bg-emerald-500/20 px-1.5 py-0.5 text-emerald-200">suggested keep</span>
                {:else}
                    <span class="text-fg-faint">Δ{zoom.hamming_to_keep} bits from keep</span>
                    {@const z = zoom}
                    {@const isPending = pending === z.image_path}
                    <form
                        method="POST"
                        action={excludeAction}
                        use:enhance={() => {
                            pending = z.image_path;
                            return async ({ update }) => {
                                await update({ reset: false });
                                pending = null;
                                zoom = null;
                                await invalidateAll();
                            };
                        }}
                    >
                        <input type="hidden" name="image_path" value={z.image_path} />
                        <input type="hidden" name="reason" value="duplicate" />
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
    </div>
{/if}
