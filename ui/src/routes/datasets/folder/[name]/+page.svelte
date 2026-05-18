<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import CentroidAction from '$lib/components/CentroidAction.svelte';
    import ConnectorSuggestions from '$lib/components/ConnectorSuggestions.svelte';
    import DatasetAssessment from '$lib/components/DatasetAssessment.svelte';
    import DatasetView from '$lib/components/DatasetView.svelte';
    import FramingCoverageTable from '$lib/components/FramingCoverageTable.svelte';
    import MainPanel from '$lib/components/MainPanel.svelte';
    import PoseCoverageTable from '$lib/components/PoseCoverageTable.svelte';
    import PruneSuggestions from '$lib/components/PruneSuggestions.svelte';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    // Local mirror of max_size for the input. Initialized + re-synced via
    // the $effect so we never capture a stale initial value of `data`.
    let maxSizeInput = $state('');
    let savingMaxSize = $state(false);
    $effect(() => {
        maxSizeInput = data.max_size != null ? String(data.max_size) : '';
    });

    // Column count is local to this page; persisted in localStorage so it
    // sticks between navigations.
    let cols = $state(4);
    let tab = $state<'images' | 'stats'>('images');

    $effect(() => {
        try {
            const v = Number(localStorage.getItem('glt:dataset-cols'));
            if (Number.isFinite(v) && v >= 2 && v <= 10) cols = v;
            const t = localStorage.getItem('glt:dataset-tab');
            if (t === 'images' || t === 'stats') tab = t;
        } catch {
            // localStorage unavailable (SSR or private mode) — keep default.
        }
    });

    $effect(() => {
        try {
            localStorage.setItem('glt:dataset-cols', String(cols));
            localStorage.setItem('glt:dataset-tab', tab);
        } catch {
            // ignore
        }
    });

    const imageUrl = (filename: string) =>
        `/datasets/folder/${encodeURIComponent(data.dataset.name)}/raw/${encodeURIComponent(filename)}`;
</script>

<svelte:head>
    <title>{data.dataset.name} — Dataset — GridLoraTester</title>
</svelte:head>

<MainPanel>
    {#snippet header()}
        <div class="space-y-4 px-6 pt-6">
            <header>
                <a href="/datasets" class="text-xs text-fg-muted hover:text-fg">← Datasets</a>
                <h1 class="mt-1 text-2xl font-semibold tracking-tight">{data.dataset.name}</h1>
            </header>
            <div class="flex flex-wrap items-start gap-6">
                <CentroidAction centroid={data.centroid} error={form?.error ?? null} />
                <form
                    method="POST"
                    action="?/setMaxSize"
                    use:enhance={() => {
                        savingMaxSize = true;
                        return async ({ update }) => {
                            await update({ reset: false });
                            savingMaxSize = false;
                            await invalidateAll();
                        };
                    }}
                    class="flex items-center gap-2 text-xs text-fg-muted"
                    title="Caps the number of active images in this dataset. When set, prune suggestions appear once the buckets fill up so you can keep the best ones."
                >
                    <label for="max-size-input" class="whitespace-nowrap">Max dataset size</label>
                    <input
                        id="max-size-input"
                        name="max_size"
                        type="number"
                        min="0"
                        step="1"
                        placeholder="—"
                        bind:value={maxSizeInput}
                        class="input w-24 px-2 py-1 text-xs"
                    />
                    <span class="tabular-nums text-fg-faint">
                        {data.prune.n_active}
                        {#if data.max_size != null}/ {data.max_size}{/if}
                    </span>
                    <button
                        type="submit"
                        class="btn btn-secondary px-2 py-1 text-xs"
                        disabled={savingMaxSize}
                    >
                        {savingMaxSize ? '…' : 'Save'}
                    </button>
                </form>
            </div>
        </div>

        <!-- Real tab strip with underline indicator, in its own row so the
             tabs read as the primary navigation device of this panel. -->
        <div
            class="mt-3 flex items-center justify-between border-b border-border px-6"
            role="tablist"
            aria-label="View"
        >
            <div class="flex">
                <button
                    type="button"
                    role="tab"
                    aria-selected={tab === 'images'}
                    onclick={() => (tab = 'images')}
                    class="-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors {tab ===
                    'images'
                        ? 'border-accent text-fg'
                        : 'border-transparent text-fg-muted hover:text-fg'}"
                >
                    Images
                    <span class="ml-1 text-xs text-fg-faint tabular-nums"
                        >{data.dataset.items.length}</span
                    >
                </button>
                <button
                    type="button"
                    role="tab"
                    aria-selected={tab === 'stats'}
                    onclick={() => (tab = 'stats')}
                    class="-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors {tab ===
                    'stats'
                        ? 'border-accent text-fg'
                        : 'border-transparent text-fg-muted hover:text-fg'}"
                >
                    Stats
                </button>
            </div>

            <!-- Tab-contextual controls: only the Images view needs the
                 column slider, so it only renders there. Keeps the tab row
                 visually anchored to whichever tab is active. -->
            {#if tab === 'images'}
                <label class="flex items-center gap-3 text-xs text-fg-muted">
                    Columns
                    <input
                        type="range"
                        min="2"
                        max="10"
                        step="1"
                        bind:value={cols}
                        class="h-1 w-40 cursor-pointer appearance-none rounded-full bg-bg-3 accent-accent"
                    />
                    <span class="w-6 text-right tabular-nums text-fg">{cols}</span>
                </label>
            {/if}
        </div>
    {/snippet}

    {#if tab === 'images'}
        <DatasetView
            name={data.dataset.name}
            path={data.dataset.path}
            items={data.dataset.items}
            imageUrlFor={imageUrl}
            medianSim={data.centroid?.median_sim ?? null}
            poseOffsets={{
                threequarter: data.pose_calibration.offset_threequarter,
                profile: data.pose_calibration.offset_profile,
                tilted: data.pose_calibration.offset_tilted
            }}
            {cols}
        />
    {:else}
        <div class="space-y-6">
            <DatasetAssessment pose={data.pose_coverage} framing={data.framing_coverage} />
            <section class="space-y-2">
                <h2 class="text-base font-medium">Pose coverage</h2>
                <PoseCoverageTable coverage={data.pose_coverage} />
            </section>
            <section class="space-y-2">
                <h2 class="text-base font-medium">Framing distance</h2>
                <FramingCoverageTable coverage={data.framing_coverage} />
            </section>
            <PruneSuggestions
                prune={data.prune}
                excluded={data.excluded}
                excludeAction="?/exclude"
                restoreAction="?/restore"
            />
            <ConnectorSuggestions
                suggestions={data.suggestions}
                scope_kind="folder"
                scope_key={data.dataset.path}
                target_folders={[data.dataset.path]}
            />
        </div>
    {/if}
</MainPanel>
