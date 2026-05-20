<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import CentroidAction from '$lib/components/CentroidAction.svelte';
    import ConnectorSuggestions from '$lib/components/ConnectorSuggestions.svelte';
    import DatasetAssessment from '$lib/components/DatasetAssessment.svelte';
    import DatasetDuplicates from '$lib/components/DatasetDuplicates.svelte';
    import DatasetView from '$lib/components/DatasetView.svelte';
    import FramingCoverageTable from '$lib/components/FramingCoverageTable.svelte';
    import MainPanel from '$lib/components/MainPanel.svelte';
    import PoseCoverageTable from '$lib/components/PoseCoverageTable.svelte';
    import PruneSuggestions from '$lib/components/PruneSuggestions.svelte';
    import {
        FRAMING_BUCKETS,
        type FramingBand,
        type FramingCoverage
    } from '$lib/framing-grid';
    import { POSE_BUCKETS, type PoseBucketId, type PoseCoverage } from '$lib/pose-grid';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    // Mirror max_size for the input. Initialized + re-synced via the
    // $effect so we never capture a stale initial `data` value.
    let maxSizeInput = $state('');
    let savingMaxSize = $state(false);
    $effect(() => {
        maxSizeInput = data.max_size != null ? String(data.max_size) : '';
    });

    /** Sum every dataset's per-bucket counts into a single group-level
     * coverage. Done client-side since we already have the per-dataset
     * tallies in `data` — saves a server round-trip. */
    let globalCoverage = $derived.by<PoseCoverage>(() => {
        const counts = {} as Record<PoseBucketId, number>;
        for (const b of POSE_BUCKETS) counts[b.id] = 0;
        let unknown = 0;
        let total = 0;
        for (const ds of data.datasets) {
            const c = ds.pose_coverage;
            for (const b of POSE_BUCKETS) counts[b.id] += c.counts[b.id] ?? 0;
            unknown += c.unknown;
            total += c.total;
        }
        return { counts, unknown, total };
    });

    /** Same aggregation for the framing-distance buckets. */
    let globalFraming = $derived.by<FramingCoverage>(() => {
        const counts = {} as Record<FramingBand, number>;
        for (const b of FRAMING_BUCKETS) counts[b.id] = 0;
        let unknown = 0;
        let total = 0;
        for (const ds of data.datasets) {
            const c = ds.framing_coverage;
            for (const b of FRAMING_BUCKETS) counts[b.id] += c.counts[b.id] ?? 0;
            unknown += c.unknown;
            total += c.total;
        }
        return { counts, unknown, total };
    });

    let cols = $state(4);
    /** Which centroid drives the per-cell delta badges and the median we
     * show as the reference: each dataset's own centroid, or the global
     * group centroid. */
    let refMode = $state<'dataset' | 'group'>('dataset');
    let tab = $state<'images' | 'stats'>('images');

    $effect(() => {
        try {
            const v = Number(localStorage.getItem('glt:dataset-cols'));
            if (Number.isFinite(v) && v >= 2 && v <= 10) cols = v;
            const rm = localStorage.getItem('glt:dataset-ref-mode');
            if (rm === 'group' || rm === 'dataset') refMode = rm;
            const t = localStorage.getItem('glt:dataset-tab');
            if (t === 'images' || t === 'stats') tab = t;
        } catch {
            // localStorage unavailable
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

    $effect(() => {
        try {
            localStorage.setItem('glt:dataset-ref-mode', refMode);
        } catch {
            // ignore
        }
    });

    /** Disabled when the group centroid hasn't been computed yet. */
    let canCompareToGroup = $derived(data.global_centroid != null);

    function imageUrl(slug: string, filename: string): string {
        return `/datasets/group/${data.group.id}/raw/${encodeURIComponent(slug)}/${encodeURIComponent(filename)}`;
    }

    const totalImages = $derived(
        data.datasets.reduce((sum, d) => sum + d.items.length, 0)
    );
</script>

<svelte:head>
    <title>{data.group.name} — Group — GridLoraTester</title>
</svelte:head>

<MainPanel>
    {#snippet header()}
        <div class="space-y-4 px-6 pt-6">
            <header>
                <a href="/datasets" class="text-xs text-fg-muted hover:text-fg">← Datasets</a>
                <h1 class="mt-1 text-2xl font-semibold tracking-tight">{data.group.name}</h1>
                <p class="mt-1 text-sm text-fg-muted">
                    {data.datasets.length} dataset{data.datasets.length === 1 ? '' : 's'} · {totalImages} image{totalImages ===
                    1
                        ? ''
                        : 's'}
                    {#if data.group.missing_paths.length > 0}
                        <span class="text-amber-400"
                            >· {data.group.missing_paths.length} missing path{data.group
                                .missing_paths.length === 1
                                ? ''
                                : 's'}</span
                        >
                    {/if}
                </p>
            </header>

            <div class="flex flex-wrap items-start gap-6">
                <CentroidAction
                    label="Analyze group"
                    centroid={data.global_centroid}
                    error={form?.error ?? null}
                />
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
                    title="Caps the number of active images in this group. When set, prune suggestions appear once the buckets fill up so you can keep the best ones."
                >
                    <label for="max-size-input" class="whitespace-nowrap">Max group size</label>
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
                        {data.n_active}
                        {#if data.max_size != null}/ {data.max_size}{/if}
                    </span>
                    <button
                        type="submit"
                        class="btn-secondary px-2.5 py-1 text-xs"
                        disabled={savingMaxSize}
                    >
                        {savingMaxSize ? '…' : 'Save'}
                    </button>
                </form>
                <a
                    href="/datasets/group/{data.group.id}/export"
                    class="btn-secondary px-3 py-1 text-xs"
                    title="Download a zip of every member dataset, one subfolder per member, with caption sidecars"
                >
                    Export zip
                </a>
            </div>
        </div>

        <!-- Real tab strip with underline indicator, on its own row. The
             Images-only secondary controls (vs dataset / vs group, columns
             slider) move to the right of the strip so they're visually
             attached to the active tab. -->
        <div
            class="mt-3 flex flex-wrap items-center justify-between gap-3 border-b border-border px-6"
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
                    <span class="ml-1 text-xs text-fg-faint tabular-nums">{totalImages}</span>
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

            {#if tab === 'images'}
                <div class="flex items-center gap-4 pb-2">
                    <div
                        class="inline-flex rounded-md border border-border bg-bg-2 p-0.5 text-xs"
                        role="tablist"
                        aria-label="Centroid reference"
                    >
                        <button
                            type="button"
                            role="tab"
                            aria-selected={refMode === 'dataset'}
                            class="rounded px-2.5 py-1 transition-colors {refMode === 'dataset'
                                ? 'bg-bg-0 text-fg'
                                : 'text-fg-muted hover:text-fg'}"
                            onclick={() => (refMode = 'dataset')}
                            title="Each cell's delta is computed vs that dataset's own centroid"
                        >
                            vs dataset
                        </button>
                        <button
                            type="button"
                            role="tab"
                            aria-selected={refMode === 'group'}
                            disabled={!canCompareToGroup}
                            class="rounded px-2.5 py-1 transition-colors {refMode === 'group'
                                ? 'bg-bg-0 text-fg'
                                : 'text-fg-muted hover:text-fg'} disabled:cursor-not-allowed disabled:opacity-50"
                            onclick={() => (refMode = 'group')}
                            title={canCompareToGroup
                                ? "Every cell's delta is computed vs the global group centroid"
                                : 'Compute the group centroid first (Calculate centroids button below)'}
                        >
                            vs group
                        </button>
                    </div>

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
                </div>
            {/if}
        </div>
    {/snippet}

    <div class="space-y-6">
    {#if tab === 'images'}
        {#each data.datasets as ds (ds.slug)}
            {@const effectiveMedian =
                refMode === 'group'
                    ? data.global_centroid?.median_sim ?? null
                    : ds.centroid?.median_sim ?? null}
            {@const effectiveItems =
                refMode === 'group'
                    ? ds.items.map((it) => ({ ...it, similarity: it.similarity_group }))
                    : ds.items}
            {@const effectiveCalib =
                refMode === 'group' ? data.group_pose_calibration : ds.pose_calibration}
            <div class="space-y-2">
                {#if ds.centroid}
                    <div class="text-xs text-fg-muted">
                        <span class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-300"
                            >centroid</span
                        >
                        <span class="ml-2 tabular-nums"
                            >{ds.centroid.n_single_face} single · {ds.centroid
                                .n_multi_face} multi{ds.centroid.n_no_face > 0
                                ? ` · ${ds.centroid.n_no_face} no-face`
                                : ''}</span
                        >
                        {#if effectiveMedian != null}
                            <span class="ml-2 tabular-nums">
                                ref median
                                <span class="text-fg">{effectiveMedian.toFixed(3)}</span>
                                <span class="text-fg-faint"
                                    >({refMode === 'group' ? 'group' : 'dataset'})</span
                                >
                            </span>
                        {/if}
                    </div>
                {/if}
                <DatasetView
                    name={ds.name}
                    path={ds.path}
                    items={effectiveItems}
                    imageUrlFor={(fn) => imageUrl(ds.slug, fn)}
                    medianSim={effectiveMedian}
                    poseOffsets={{
                        threequarter: effectiveCalib.offset_threequarter,
                        profile: effectiveCalib.offset_profile,
                        tilted: effectiveCalib.offset_tilted
                    }}
                    {cols}
                    badge="in group"
                />
            </div>
        {/each}
    {:else}
        <section class="space-y-3">
            <header class="flex items-baseline justify-between gap-3">
                <h2 class="text-base font-medium">
                    Global
                    <span
                        class="ml-2 rounded-full bg-bg-3 px-2 py-0.5 align-middle text-[10px] uppercase tracking-wide text-fg-muted"
                        >group</span
                    >
                </h2>
                <span class="text-xs text-fg-faint"
                    >{data.datasets.length} dataset{data.datasets.length === 1 ? '' : 's'}
                    combined</span
                >
            </header>
            <DatasetAssessment pose={globalCoverage} framing={globalFraming} />
            <div class="space-y-1">
                <h3 class="text-xs uppercase tracking-wide text-fg-muted">Pose</h3>
                <PoseCoverageTable coverage={globalCoverage} />
            </div>
            <div class="space-y-1">
                <h3 class="text-xs uppercase tracking-wide text-fg-muted">Framing</h3>
                <FramingCoverageTable coverage={globalFraming} />
            </div>
            {#await data.duplicates}
                <p class="text-xs text-fg-faint">Scanning for duplicates…</p>
            {:then duplicates}
                <DatasetDuplicates
                    {duplicates}
                    excludeAction="?/exclude"
                    excludeManyAction="?/excludeMany"
                />
            {/await}
            {#await data.prune}
                <p class="text-xs text-fg-faint">Computing prune candidates…</p>
            {:then prune}
                <PruneSuggestions
                    {prune}
                    excluded={data.excluded}
                    excludeAction="?/exclude"
                    restoreAction="?/restore"
                />
            {/await}
            {#await data.group_suggestions}
                <p class="text-xs text-fg-faint">Loading group-wide suggestions…</p>
            {:then group_suggestions}
                <ConnectorSuggestions
                    suggestions={group_suggestions}
                    scope_kind="group"
                    scope_key={String(data.group.id)}
                    target_folders={data.datasets.map((d) => d.path)}
                />
            {/await}
        </section>

        <hr class="border-border/70" />

        {#each data.datasets as ds (ds.slug)}
            <section class="space-y-3">
                <header class="flex items-baseline justify-between gap-3">
                    <h2 class="truncate text-base font-medium" title={ds.name}>{ds.name}</h2>
                    <span class="truncate font-mono text-xs text-fg-faint" title={ds.path}
                        >{ds.path}</span
                    >
                </header>
                <DatasetAssessment
                    pose={ds.pose_coverage}
                    framing={ds.framing_coverage}
                    compact
                />
                <div class="space-y-1">
                    <h3 class="text-xs uppercase tracking-wide text-fg-muted">Pose</h3>
                    <PoseCoverageTable coverage={ds.pose_coverage} />
                </div>
                <div class="space-y-1">
                    <h3 class="text-xs uppercase tracking-wide text-fg-muted">Framing</h3>
                    <FramingCoverageTable coverage={ds.framing_coverage} />
                </div>
                {#await ds.suggestions}
                    <p class="text-xs text-fg-faint">Loading suggestions…</p>
                {:then suggestions}
                    <ConnectorSuggestions
                        {suggestions}
                        scope_kind="folder"
                        scope_key={ds.path}
                        target_folders={[ds.path]}
                    />
                {/await}
            </section>
        {/each}
    {/if}

    {#if data.group.missing_paths.length > 0}
        <section class="card text-sm text-amber-300">
            <h3 class="mb-2 font-medium">Missing paths</h3>
            <ul class="space-y-1">
                {#each data.group.missing_paths as mp (mp)}
                    <li class="font-mono text-xs text-fg-muted">{mp}</li>
                {/each}
            </ul>
            <p class="mt-2 text-xs text-fg-faint">
                These folders no longer exist on disk. Edit the group on /datasets to remove them.
            </p>
        </section>
    {/if}
    </div>
</MainPanel>
