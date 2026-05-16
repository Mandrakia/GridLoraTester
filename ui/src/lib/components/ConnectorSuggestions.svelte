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
        similarity: number;
        tempered_delta: number;
        pose_offset_applied: number;
        pose_bucket: string | null;
        framing_bucket: string | null;
        yaw: number | null;
        pitch: number | null;
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
            groups: SuggestionGroup[];
            candidates_pool: number;
            candidates_qualifying: number;
        };
        /** Scope identity for the import call. */
        scope_kind: 'folder' | 'group';
        scope_key: string;
        /** Where the imported file lands. For folder scope = [folder_path];
         * for group scope = the member folders, user picks one in the UI. */
        target_folders: string[];
    }

    let { suggestions, scope_kind, scope_key, target_folders }: Props = $props();

    let targetFolder = $state('');
    let adding = $state<string | null>(null);
    let addError = $state<string | null>(null);

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
    {:else if suggestions.groups.length === 0}
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
                            {@const key = c.connector_id + '::' + c.picture_id}
                            <div
                                class="group relative overflow-hidden rounded-md border border-border bg-bg-2"
                                title={`sim=${c.similarity.toFixed(3)} · tempered=${fmtDelta(
                                    c.tempered_delta
                                )}${
                                    c.pose_offset_applied > 0
                                        ? ` (pose offset +${c.pose_offset_applied.toFixed(3)})`
                                        : ''
                                }
pose=${c.pose_bucket ?? '?'} · framing=${c.framing_bucket ?? '?'}
${c.filename ?? c.picture_id}`}
                            >
                                <a
                                    href={c.thumbnail_url}
                                    target="_blank"
                                    rel="noopener"
                                    class="block aspect-square"
                                >
                                    <img
                                        src={c.thumbnail_url}
                                        alt={c.filename ?? c.picture_id}
                                        loading="lazy"
                                        decoding="async"
                                        class="h-full w-full object-cover transition-transform group-hover:scale-[1.02]"
                                    />
                                </a>
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
                                <button
                                    type="button"
                                    class="absolute inset-x-1 bottom-1 rounded bg-bg-0/80 px-1.5 py-1 text-[10px] font-medium text-fg backdrop-blur transition-colors hover:bg-accent hover:text-white disabled:opacity-50"
                                    disabled={adding === key || !targetFolder}
                                    onclick={() => addToDataset(c)}
                                >
                                    {adding === key ? 'Adding…' : '+ Add to dataset'}
                                </button>
                            </div>
                        {/each}
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</section>
