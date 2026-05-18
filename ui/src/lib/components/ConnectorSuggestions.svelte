<script lang="ts">
    // Render the external-picture suggestions surfaced by
    // suggestExternalPictures(). One section per under-represented pose or
    // framing group, with a row of thumbnail tiles ordered by best identity
    // match (after pose-tempering). Hover a tile to see sim + delta + bucket.
    // Each tile has an "Add to dataset" button that pulls the file in and
    // recomputes the centroid in-place.
    import { invalidateAll } from '$app/navigation';

    interface SuggestionCandidate {
        connector_id: string;
        picture_id: string;
        filename: string | null;
        thumbnail_url: string;
        full_url: string;
        image_width: number | null;
        image_height: number | null;
        similarity: number;
        tempered_delta: number;
        pose_offset_applied: number;
        pose_bucket: string | null;
        framing_bucket: string | null;
        yaw: number | null;
        pitch: number | null;
        ambiguous_identity: boolean;
        runner_up_area_ratio: number | null;
        dedup_match: {
            dataset_path: string;
            dataset_basename: string;
            hamming: number;
        } | null;
        gaps_filled: string[];
        gap_score: number;
    }

    interface SuggestionGroup {
        dimension: 'pose' | 'framing';
        group_key: string;
        label: string;
        target: number;
        actual: number;
        gap: number;
        candidates: SuggestionCandidate[];
    }

    interface Props {
        suggestions: {
            has_linked_connectors: boolean;
            no_data: boolean;
            multi_gap: SuggestionCandidate[];
            groups: SuggestionGroup[];
            ambiguous: SuggestionCandidate[];
            candidates_pool: number;
            candidates_qualifying: number;
            min_image_mp: number;
            rejected_low_res: number;
            rejected_duplicates: number;
            rejected_duplicate_samples: SuggestionCandidate[];
            dataset_hashes_indexed: boolean;
        };
        /** Scope identity for the import call. */
        scope_kind: 'folder' | 'group';
        scope_key: string;
        /** Where the imported file lands. For folder scope = [folder_path];
         * for group scope = the member folders, user picks one in the UI. */
        target_folders: string[];
    }

    let showAmbiguous = $state(false);
    let showDedup = $state(false);

    let { suggestions, scope_kind, scope_key, target_folders }: Props = $props();

    let targetFolder = $state('');
    let adding = $state<string | null>(null);
    let addError = $state<string | null>(null);
    /** Candidate being shown in the fullscreen lightbox (null = closed).
     * Loaded image source is `c.full_url` — the original asset, not the
     * thumbnail — so the user can audit at real quality before importing. */
    let zoom = $state<SuggestionCandidate | null>(null);

    // Re-sync the target when the prop set changes — but only when the
    // current selection is no longer valid (so we don't trample on the user's
    // explicit pick within the same set). Also handles initial mount, which
    // is why $state above starts empty: the effect picks the first folder.
    $effect(() => {
        if (target_folders.length > 0 && !target_folders.includes(targetFolder)) {
            targetFolder = target_folders[0];
        }
    });

    function basename(p: string): string {
        return p.split('/').filter(Boolean).pop() ?? p;
    }

    async function addToDataset(c: SuggestionCandidate) {
        if (!targetFolder) {
            addError = 'No target folder available.';
            return;
        }
        const key = `${c.connector_id}::${c.picture_id}`;
        adding = key;
        addError = null;
        try {
            const res = await fetch('/api/import-picture', {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify({
                    scope_kind,
                    scope_key,
                    target_folder: targetFolder,
                    connector_id: c.connector_id,
                    picture_id: c.picture_id
                })
            });
            const body = await res.json().catch(() => null);
            if (!res.ok) {
                addError = body?.error ?? `HTTP ${res.status}`;
                return;
            }
            // Successful — refetch the page so the suggestion drops out and
            // the centroid pills downstream pick up the new face.
            await invalidateAll();
        } catch (e) {
            addError = (e as Error).message;
        } finally {
            adding = null;
        }
    }

    function fmtCount(n: number) {
        return n === Math.floor(n) ? `${n}` : n.toFixed(1);
    }

    function fmtDelta(d: number) {
        return (d >= 0 ? '+' : '') + d.toFixed(3);
    }

    /** "1024×768 (0.79 MP)" — null when dims unknown. */
    function fmtRes(w: number | null, h: number | null): string | null {
        if (!w || !h) return null;
        const mp = (w * h) / 1_000_000;
        return `${w}×${h} (${mp.toFixed(2)} MP)`;
    }

    /** Per-axis color + short label for a gap key like "framing:close"
     * or "pose:tilted". Pitch axis (tilted) gets its own color to stand
     * out from yaw groups. */
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
</script>

<section class="space-y-3 rounded-lg border border-border bg-bg-1 p-4">
    <header class="flex items-start justify-between gap-3">
        <div>
            <h3 class="text-sm font-medium">Suggested external photos</h3>
            <p class="text-xs text-fg-faint">
                Best matches from connector libraries to fill under-represented
                pose / framing buckets. Identity check uses pose-tempered
                similarity (profile shots aren't penalized for being off-axis).
            </p>
        </div>
        {#if scope_kind === 'group' && target_folders.length > 1}
            <label class="flex flex-col gap-1 text-xs text-fg-muted">
                <span>Add to</span>
                <select bind:value={targetFolder} class="input font-mono text-xs">
                    {#each target_folders as f (f)}
                        <option value={f}>{basename(f)}</option>
                    {/each}
                </select>
            </label>
        {/if}
    </header>
    {#if addError}
        <p class="text-xs text-red-300">{addError}</p>
    {/if}

    {#if !suggestions.has_linked_connectors}
        <p class="text-xs text-fg-muted">
            No connector linked to this dataset yet — open the chain icon next to its name on
            <a href="/datasets" class="text-accent hover:text-accent-hover">/datasets</a> to pick a person.
        </p>
    {:else if suggestions.no_data}
        <p class="text-xs text-fg-muted">
            Nothing to suggest yet. Make sure the connector face-detect job has finished, then come
            back. (Compute the centroid first if you haven't.)
        </p>
    {:else if suggestions.multi_gap.length === 0 && suggestions.groups.length === 0}
        <p class="text-xs text-emerald-300">
            All target buckets are covered ✓ —
            <span class="text-fg-muted"
                >{suggestions.candidates_qualifying}/{suggestions.candidates_pool} connector pics match
                identity, but nothing's under-represented.</span
            >
        </p>
    {:else}
        <p class="text-xs text-fg-muted">
            {suggestions.candidates_qualifying}/{suggestions.candidates_pool} connector pic{suggestions.candidates_pool ===
            1
                ? ''
                : 's'} match identity, surfaced where they fill a gap.
        </p>

        {#if suggestions.multi_gap.length > 0}
            <div class="space-y-2 rounded-md border border-amber-400/30 bg-amber-400/5 p-3">
                <div class="flex items-baseline justify-between gap-3">
                    <h4 class="text-sm font-medium text-amber-200">
                        🏆 Multi-gap candidates
                        <span class="ml-1 text-[10px] uppercase tracking-wide text-amber-300/70"
                            >fills ≥ 2 gaps each</span
                        >
                    </h4>
                    <span class="text-xs text-amber-300/80 tabular-nums">
                        {suggestions.multi_gap.length} top
                    </span>
                </div>
                <p class="text-[11px] text-amber-200/80">
                    These pictures fix more than one under-represented bucket at once — import
                    them first to maximize coverage per addition. Badges below each tile show
                    which buckets they fill.
                </p>
                <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
                    {#each suggestions.multi_gap as c (c.connector_id + c.picture_id)}
                        {@render tile(c)}
                    {/each}
                </div>
            </div>
        {/if}

        <div class="space-y-4">
            {#each suggestions.groups as g (g.dimension + ':' + g.group_key)}
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
                            need {fmtCount(g.gap)} more · have {fmtCount(g.actual)}/{fmtCount(g.target)}
                        </span>
                    </div>
                    <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
                        {#each g.candidates as c (c.connector_id + c.picture_id)}
                            {@render tile(c)}
                        {/each}
                    </div>
                </div>
            {/each}
        </div>

        <!-- Ambiguous identity bucket: pictures with another face whose
             area is comparable to the target's. Collapsed by default so
             they don't visually compete with clean suggestions — the user
             opens this only when they want to audit. -->
        {#if suggestions.ambiguous.length > 0}
            <div class="mt-4 rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
                <button
                    type="button"
                    class="flex w-full items-baseline justify-between text-left"
                    onclick={() => (showAmbiguous = !showAmbiguous)}
                >
                    <div class="flex items-baseline gap-2">
                        <span class="text-amber-300">{showAmbiguous ? '▾' : '▸'}</span>
                        <h4 class="text-sm font-medium text-amber-200">
                            Group photos — uncertain identity
                            <span class="ml-1 text-xs text-amber-300/70 tabular-nums"
                                >({suggestions.ambiguous.length})</span
                            >
                        </h4>
                    </div>
                    <span class="text-[10px] text-amber-300/70">click to {showAmbiguous ? 'hide' : 'review'}</span>
                </button>
                {#if showAmbiguous}
                    <p class="mt-2 text-xs text-amber-200/80">
                        Each of these contains at least one OTHER face roughly the same size as
                        the matched one. Eyeball the click-through full-res before importing —
                        the subject might not be who you think.
                    </p>
                    <div
                        class="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8"
                    >
                        {#each suggestions.ambiguous as c (c.connector_id + c.picture_id)}
                            {@render tile(c)}
                        {/each}
                    </div>
                {/if}
            </div>
        {/if}

        <!-- Footer: filter receipts. Each line surfaces a class of
             candidates dropped before scoring, so the user knows their
             knobs are eating something and where to tune. -->
        {#if suggestions.rejected_low_res > 0}
            <p class="mt-3 text-[11px] text-fg-faint">
                {suggestions.rejected_low_res} picture{suggestions.rejected_low_res === 1
                    ? ''
                    : 's'} hidden — below the
                <span class="font-mono">{suggestions.min_image_mp} MP</span> floor
                (change in <a href="/settings" class="text-accent hover:text-accent-hover">Settings</a>).
            </p>
        {/if}
        {#if suggestions.rejected_duplicates > 0}
            <div class="mt-3 rounded-md border border-sky-500/20 bg-sky-500/5 p-3">
                <button
                    type="button"
                    class="flex w-full items-baseline justify-between text-left"
                    onclick={() => (showDedup = !showDedup)}
                >
                    <div class="flex items-baseline gap-2">
                        <span class="text-sky-300">{showDedup ? '▾' : '▸'}</span>
                        <h4 class="text-xs font-medium text-sky-200">
                            {suggestions.rejected_duplicates} near-duplicate{suggestions.rejected_duplicates ===
                            1
                                ? ''
                                : 's'} hidden
                            {#if suggestions.rejected_duplicate_samples.length < suggestions.rejected_duplicates}
                                <span class="text-sky-300/60 tabular-nums"
                                    >· showing first {suggestions.rejected_duplicate_samples.length}</span
                                >
                            {/if}
                        </h4>
                    </div>
                    <span class="text-[10px] text-sky-300/70">
                        click to {showDedup ? 'hide' : 'inspect'} matches
                    </span>
                </button>
                {#if showDedup}
                    <p class="mt-2 text-[11px] text-sky-200/80">
                        Each tile here matched a dataset image via BlockHash (Hamming distance
                        shown). Click a tile to open it full-res and visually verify the match
                        — false positives suggest tightening the threshold; real dupes
                        confirm the filter is doing its job.
                    </p>
                    <div
                        class="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8"
                    >
                        {#each suggestions.rejected_duplicate_samples as c (c.connector_id + c.picture_id)}
                            {@render tile(c)}
                        {/each}
                    </div>
                {/if}
            </div>
        {/if}
        {#if !suggestions.dataset_hashes_indexed}
            <p class="text-[11px] text-amber-300/70">
                Duplicate detection inactive — dataset images haven't been hashed yet. Click
                <span class="font-mono">Analyze dataset</span> at the top of this page; the
                hashes are computed in the background once the centroid is done.
            </p>
        {/if}
    {/if}
</section>

<!-- Single source of truth for the candidate tile markup — reused by the
     main groups, the ambiguous bucket, AND the dedup-rejected panel
     (where the dedup_match badge replaces the Add button). -->
{#snippet tile(c: SuggestionCandidate)}
    {@const key = c.connector_id + '::' + c.picture_id}
    {@const res = fmtRes(c.image_width, c.image_height)}
    <div
        class="group relative overflow-hidden rounded-md border border-border bg-bg-2"
        title={`sim=${c.similarity.toFixed(3)} · tempered=${fmtDelta(c.tempered_delta)}${
            c.pose_offset_applied > 0
                ? ` (pose offset +${c.pose_offset_applied.toFixed(3)})`
                : ''
        }
pose=${c.pose_bucket ?? '?'} · framing=${c.framing_bucket ?? '?'}${
            res ? '\n' + res : ''
        }${
            c.ambiguous_identity && c.runner_up_area_ratio
                ? `\nambiguous: runner-up face = ${c.runner_up_area_ratio.toFixed(2)}× target area`
                : ''
        }${
            c.dedup_match
                ? `\nduplicate of ${c.dedup_match.dataset_basename} (hamming ${c.dedup_match.hamming}/256)`
                : ''
        }
${c.filename ?? c.picture_id}`}
    >
        <button
            type="button"
            class="block aspect-square w-full cursor-zoom-in"
            onclick={() => (zoom = c)}
            aria-label={`Preview ${c.filename ?? c.picture_id}`}
        >
            <img
                src={c.thumbnail_url}
                alt={c.filename ?? c.picture_id}
                loading="lazy"
                decoding="async"
                class="h-full w-full object-cover transition-transform group-hover:scale-[1.02]"
            />
        </button>
        <span
            class="pointer-events-none absolute right-1 top-1 rounded bg-black/70 px-1 text-[10px] font-medium tabular-nums {c.tempered_delta >=
            0
                ? 'text-emerald-300'
                : c.tempered_delta >= -0.03
                  ? 'text-amber-300'
                  : 'text-red-300'}"
        >
            {fmtDelta(c.tempered_delta)}
        </span>
        {#if c.gaps_filled.length > 0}
            <!-- Reserve space on the right for the similarity delta pill
                 (right-1, ~36px wide). Without `right-10` here, three
                 gap-fill pills push under the delta badge and overlap. -->
            <div
                class="pointer-events-none absolute left-1 right-10 top-1 flex flex-wrap gap-0.5"
            >
                {#each c.gaps_filled as g (g)}
                    {@const b = gapBadge(g)}
                    <span
                        class="rounded px-1 text-[9px] font-medium uppercase tracking-tight {b.cls}"
                        >{b.label}</span
                    >
                {/each}
            </div>
        {/if}
        {#if c.dedup_match}
            <!-- Dedup-rejected: non-interactive badge with the matched
                 dataset basename + Hamming distance. The user can still
                 click the image to open the lightbox and visually compare
                 against the dataset file they have on disk. -->
            <div
                class="pointer-events-none absolute inset-x-1 bottom-1 truncate rounded bg-sky-500/85 px-1.5 py-1 text-center text-[10px] font-medium text-white backdrop-blur"
                title={`Matched ${c.dedup_match.dataset_path}`}
            >
                ≈ {c.dedup_match.dataset_basename} · h{c.dedup_match.hamming}
            </div>
        {:else}
            <button
                type="button"
                class="absolute inset-x-1 bottom-1 rounded bg-bg-0/80 px-1.5 py-1 text-[10px] font-medium text-fg backdrop-blur transition-colors hover:bg-accent hover:text-white disabled:opacity-50"
                disabled={adding === key || !targetFolder}
                onclick={() => addToDataset(c)}
            >
                {adding === key ? 'Adding…' : '+ Add to dataset'}
            </button>
        {/if}
    </div>
{/snippet}

<!-- Fullscreen lightbox. Loads `full_url` (the original, not the thumbnail)
     so the user can audit at real quality before importing. Click overlay
     or press Escape/Enter/Space to close — same shortcuts as
     DatasetView / GridResult lightboxes for muscle-memory consistency. -->
{#if zoom}
    {@const res = fmtRes(zoom.image_width, zoom.image_height)}
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
        <div class="flex h-full w-full max-w-screen-2xl flex-col gap-2">
            <div class="flex min-h-0 flex-1 items-center justify-center">
                <img
                    src={zoom.full_url}
                    alt={zoom.filename ?? zoom.picture_id}
                    class="max-h-full max-w-full object-contain"
                />
            </div>
            <div class="shrink-0 text-center text-xs text-fg-muted">
                <span class="font-mono">{zoom.filename ?? zoom.picture_id}</span>
                {#if res}
                    <span class="text-fg-faint"> · {res}</span>
                {/if}
                <span class="text-fg-faint">
                    · sim={zoom.similarity.toFixed(3)} · delta={fmtDelta(zoom.tempered_delta)}
                </span>
                {#if zoom.ambiguous_identity}
                    <span class="ml-2 rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-200">
                        ambiguous identity
                    </span>
                {/if}
                {#if zoom.dedup_match}
                    <div class="mt-1 text-[11px] text-sky-200">
                        ≈ duplicate of
                        <span class="font-mono">{zoom.dedup_match.dataset_basename}</span>
                        <span class="text-sky-300/70">
                            · hamming {zoom.dedup_match.hamming}/256
                        </span>
                        <div class="mt-0.5 truncate font-mono text-[10px] text-fg-faint">
                            {zoom.dedup_match.dataset_path}
                        </div>
                    </div>
                {/if}
            </div>
        </div>
    </div>
{/if}
