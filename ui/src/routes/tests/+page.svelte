<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import GridResult from '$lib/components/GridResult.svelte';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    // Editor state: null = closed; -1 = create new; positive = edit that id.
    let editingId = $state<number | null>(null);
    let saving = $state(false);
    let deletingId = $state<number | null>(null);
    let showAdvanced = $state(false);
    // Active tab in the details panel. New tests skip straight to settings
    // (no result to show yet); existing tests with output default to the grid.
    let activeTab = $state<'grid' | 'settings'>('settings');

    /** The "open in details panel" row from `data.tests` (or null when creating). */
    let openedTest = $derived(
        editingId != null && editingId > 0 ? data.tests.find((t) => t.id === editingId) ?? null : null
    );
    let hasResult = $derived(openedTest != null && openedTest.status !== 'not_started');

    // Manifest fetched lazily when a test with output is opened. Kept in
    // component state so flipping back to the Settings tab and back again
    // doesn't re-fetch. The $effect tears it down whenever the opened test
    // changes (or no test is selected).
    let manifest = $state<unknown | null>(null);
    let manifestLoading = $state(false);
    let manifestError = $state<string | null>(null);

    $effect(() => {
        const name = openedTest?.name;
        if (!name || !hasResult) {
            manifest = null;
            manifestError = null;
            return;
        }
        manifestLoading = true;
        manifestError = null;
        const ac = new AbortController();
        fetch(`/tests/output/${encodeURIComponent(name)}/manifest.json`, { signal: ac.signal })
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then((j) => {
                manifest = j;
            })
            .catch((e) => {
                if (e?.name === 'AbortError') return;
                manifestError = (e as Error).message;
                manifest = null;
            })
            .finally(() => {
                manifestLoading = false;
            });
        return () => ac.abort();
    });

    // Editable fields (kept in sync with editor state).
    let f = $state({
        name: '',
        lora_path: '',
        dataset_value: '', // encoded "dataset:<path>" or "group:<id>"
        prompt_set_id: '', // stringified id or '' (matches <select> value)
        prompts_path: '', // legacy: free-text path, passed through hidden field
        width: 1024,
        height: 1024,
        batch_size: 0,
        quant: 'fp8_weight_only',
        offload: 'text-encoder',
        // advanced
        seed: 42,
        steps: 4,
        guidance: 1.0,
        lora_scale: 1.0,
        format: 'png',
        sage_attention: false,
        compile_transformer: false,
        preload_loras: false,
        comfyui_noise: false,
        shift: '',
        min_step: 0,
        skip_face: false,
        force: false
    });

    function reset() {
        f = {
            name: '',
            lora_path: data.loras[0]?.path ?? '',
            dataset_value: '',
            prompt_set_id: '',
            prompts_path: '',
            width: 1024,
            height: 1024,
            batch_size: 0,
            quant: 'fp8_weight_only',
            offload: 'text-encoder',
            seed: 42,
            steps: 4,
            guidance: 1.0,
            lora_scale: 1.0,
            format: 'png',
            sage_attention: false,
            compile_transformer: false,
            preload_loras: false,
            comfyui_noise: false,
            shift: '',
            min_step: 0,
            skip_face: false,
            force: false
        };
    }

    function openNew() {
        reset();
        editingId = -1;
        showAdvanced = false;
        activeTab = 'settings'; // nothing to show on the grid tab for a brand-new test
    }

    function openEdit(t: PageData['tests'][number]) {
        const adv = t.advanced as Record<string, unknown>;
        f = {
            name: t.name,
            lora_path: t.lora_path,
            dataset_value:
                t.dataset?.kind === 'group'
                    ? `group:${t.dataset.id}`
                    : t.dataset?.kind === 'path'
                      ? `dataset:${t.dataset.path}`
                      : '',
            prompt_set_id: t.prompt_set_id != null ? String(t.prompt_set_id) : '',
            prompts_path: t.prompts_path ?? '',
            width: t.width,
            height: t.height,
            batch_size: t.batch_size,
            quant: t.quant,
            offload: t.offload,
            seed: Number(adv.seed ?? 42),
            steps: Number(adv.steps ?? 4),
            guidance: Number(adv.guidance ?? 1.0),
            lora_scale: Number(adv.lora_scale ?? 1.0),
            format: String(adv.format ?? 'png'),
            sage_attention: Boolean(adv.sage_attention),
            compile_transformer: Boolean(adv.compile_transformer),
            preload_loras: Boolean(adv.preload_loras),
            comfyui_noise: Boolean(adv.comfyui_noise),
            shift: adv.shift != null ? String(adv.shift) : '',
            min_step: Number(adv.min_step ?? 0),
            skip_face: Boolean(adv.skip_face),
            force: Boolean(adv.force)
        };
        editingId = t.id;
        showAdvanced = false;
        // Default to the grid tab if there's actually something to look at.
        activeTab = t.status === 'not_started' ? 'settings' : 'grid';
    }

    function closeEditor() {
        editingId = null;
    }

    const STATUS_LABELS: Record<string, { label: string; classes: string }> = {
        not_started: {
            label: 'Not started',
            classes: 'bg-bg-3 text-fg-muted'
        },
        in_progress: {
            label: 'Running',
            classes: 'bg-amber-500/15 text-amber-300'
        },
        completed: {
            label: 'Completed',
            classes: 'bg-emerald-500/15 text-emerald-300'
        },
        out_of_sync: {
            label: 'Out of sync',
            classes: 'bg-orange-500/15 text-orange-300'
        }
    };

    function scoreClass(s: number | null): string {
        if (s == null) return 'text-fg-faint';
        if (s >= 0.5) return 'text-emerald-300';
        if (s >= 0.35) return 'text-amber-300';
        return 'text-red-300';
    }
</script>

<svelte:head>
    <title>Tests — GridLoraTester</title>
</svelte:head>

<div class="mx-auto max-w-7xl space-y-6 p-8">
    <header class="flex items-baseline justify-between gap-4">
        <div>
            <h1 class="text-2xl font-semibold tracking-tight">Tests</h1>
            <p class="mt-1 text-sm text-fg-muted">
                Saved grid recipes. Status is read live from <span class="font-mono text-fg"
                    >{data.tests_root || 'tests_root'}</span
                >.
            </p>
        </div>
        <div class="flex items-center gap-2">
            <a href="/settings" class="btn-ghost text-xs">Configure roots…</a>
            <button
                type="button"
                class="btn-primary"
                disabled={editingId !== null}
                onclick={openNew}
            >
                + Add test
            </button>
        </div>
    </header>

    <!-- =================== TABLE =================== -->
    <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
        <table class="w-full text-sm">
            <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                <tr>
                    <th class="px-4 py-2.5 text-left font-medium">Name</th>
                    <th class="px-4 py-2.5 text-left font-medium">LoRA</th>
                    <th class="px-4 py-2.5 text-left font-medium">Dataset</th>
                    <th class="px-4 py-2.5 text-right font-medium">Prompts</th>
                    <th class="px-4 py-2.5 text-right font-medium">Images</th>
                    <th class="px-4 py-2.5 text-left font-medium">Status</th>
                    <th class="px-4 py-2.5 text-right font-medium">Best&nbsp;med</th>
                    <th class="w-28 px-4 py-2.5 text-right font-medium">Actions</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-border">
                {#if data.tests.length === 0}
                    <tr>
                        <td colspan="8" class="px-4 py-6 text-center text-sm text-fg-faint">
                            No tests defined. Click "+ Add test" to create one.
                        </td>
                    </tr>
                {:else}
                    {#each data.tests as t (t.id)}
                        {@const s = STATUS_LABELS[t.status]}
                        <tr class="transition-colors hover:bg-bg-2/40">
                            <td class="px-4 py-2.5 align-top">
                                <div class="font-medium">{t.name}</div>
                                <div class="text-xs text-fg-faint">
                                    {t.width}×{t.height} · batch {t.batch_size || 'auto'}
                                </div>
                            </td>
                            <td
                                class="px-4 py-2.5 align-top font-mono text-xs text-fg-muted"
                                title={t.lora_path}
                            >
                                {t.lora_path.split('/').pop()}
                            </td>
                            <td
                                class="px-4 py-2.5 align-top text-xs text-fg-muted"
                                title={t.dataset?.kind === 'path' ? t.dataset.path : ''}
                            >
                                {t.dataset_label}
                            </td>
                            <td
                                class="px-4 py-2.5 text-right align-top tabular-nums"
                                title={t.prompts_label}
                                >{t.prompt_count}</td
                            >
                            <td class="px-4 py-2.5 text-right align-top tabular-nums">
                                {t.images_generated}{t.images_target
                                    ? `/${t.images_target}`
                                    : ''}
                            </td>
                            <td class="px-4 py-2.5 align-top">
                                <span
                                    class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {s.classes}"
                                    title={t.status === 'out_of_sync'
                                        ? `${t.new_loras_count} new LoRA file(s) since last run`
                                        : ''}
                                >
                                    {s.label}
                                    {#if t.status === 'out_of_sync'}
                                        <span class="ml-1">·+{t.new_loras_count}</span>
                                    {/if}
                                </span>
                            </td>
                            <td
                                class="px-4 py-2.5 text-right align-top tabular-nums {scoreClass(
                                    t.best_median_score
                                )}"
                            >
                                {t.best_median_score != null
                                    ? t.best_median_score.toFixed(3)
                                    : '—'}
                            </td>
                            <td class="px-4 py-2.5 align-top">
                                <div class="flex justify-end gap-1">
                                    <button
                                        type="button"
                                        class="btn-ghost px-2 py-1 text-xs"
                                        onclick={() => openEdit(t)}
                                        disabled={editingId !== null}
                                    >
                                        Edit
                                    </button>
                                    <form
                                        method="POST"
                                        action="?/delete"
                                        use:enhance={() => {
                                            deletingId = t.id;
                                            return ({ update }) => {
                                                update({ reset: false }).finally(
                                                    () => (deletingId = null)
                                                );
                                            };
                                        }}
                                    >
                                        <input type="hidden" name="id" value={t.id} />
                                        <button
                                            type="submit"
                                            class="btn-ghost px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                            onclick={(e) => {
                                                if (!confirm(`Delete test "${t.name}"?`))
                                                    e.preventDefault();
                                            }}
                                            disabled={deletingId === t.id}
                                        >
                                            {deletingId === t.id ? '…' : 'Delete'}
                                        </button>
                                    </form>
                                </div>
                            </td>
                        </tr>
                    {/each}
                {/if}
            </tbody>
        </table>
    </div>

    <!-- =================== DETAILS (tabs) =================== -->
    {#if editingId !== null}
        <section class="card">
            <div class="mb-4 flex items-center justify-between gap-4">
                <h3 class="text-sm font-medium">
                    {editingId === -1
                        ? 'New test'
                        : openedTest
                          ? openedTest.name
                          : 'Edit test'}
                </h3>
                {#if editingId !== -1}
                    <div
                        class="inline-flex rounded-md border border-border bg-bg-2 p-0.5 text-xs"
                        role="tablist"
                    >
                        <button
                            type="button"
                            role="tab"
                            aria-selected={activeTab === 'grid'}
                            disabled={!hasResult}
                            class="rounded px-2.5 py-1 transition-colors {activeTab === 'grid'
                                ? 'bg-bg-0 text-fg'
                                : 'text-fg-muted hover:text-fg'} disabled:cursor-not-allowed disabled:opacity-50"
                            onclick={() => (activeTab = 'grid')}
                            title={hasResult ? '' : 'Test has not been run yet'}
                        >
                            Result / Grid
                        </button>
                        <button
                            type="button"
                            role="tab"
                            aria-selected={activeTab === 'settings'}
                            class="rounded px-2.5 py-1 transition-colors {activeTab === 'settings'
                                ? 'bg-bg-0 text-fg'
                                : 'text-fg-muted hover:text-fg'}"
                            onclick={() => (activeTab = 'settings')}
                        >
                            Settings
                        </button>
                    </div>
                {/if}
            </div>

            {#if editingId !== -1 && activeTab === 'grid'}
                <!-- ------------ RESULT / GRID ------------ -->
                {#if !hasResult}
                    <div class="card text-sm text-fg-muted">
                        Test hasn't been run yet — nothing to show. Edit settings and run it
                        from the CLI for now (`python -m glt ...`); the dashboard runner is
                        coming in the next pass.
                    </div>
                {:else if manifestLoading}
                    <div class="text-sm text-fg-muted">Loading manifest…</div>
                {:else if manifestError}
                    <div class="card text-sm text-red-300">
                        Failed to load manifest: <code class="font-mono">{manifestError}</code>
                    </div>
                {:else if manifest && openedTest}
                    <GridResult testName={openedTest.name} manifest={manifest as never} />
                {:else}
                    <div class="text-sm text-fg-muted">No manifest data.</div>
                {/if}
            {:else}
                <!-- ------------ SETTINGS (form) ------------ -->
                <form
                method="POST"
                action={editingId === -1 ? '?/create' : '?/update'}
                use:enhance={() => {
                    saving = true;
                    return async ({ result, update }) => {
                        await update({ reset: false });
                        saving = false;
                        if (result.type === 'success') {
                            editingId = null;
                            await invalidateAll();
                        }
                    };
                }}
                class="space-y-5"
            >
                {#if editingId > 0}
                    <input type="hidden" name="id" value={editingId} />
                {/if}

                <!-- Row 1: name + lora -->
                <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div class="space-y-1.5">
                        <label for="t-name" class="text-sm font-medium">Name</label>
                        <input
                            id="t-name"
                            name="name"
                            type="text"
                            class="input"
                            bind:value={f.name}
                            placeholder="my-test-1"
                            autocomplete="off"
                            required
                        />
                        <p class="text-xs text-fg-faint">
                            Output folder name inside <span class="font-mono"
                                >{data.tests_root || 'tests_root'}</span
                            >.
                        </p>
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-lora" class="text-sm font-medium">LoRA folder</label>
                        <select
                            id="t-lora"
                            name="lora_path"
                            class="input"
                            bind:value={f.lora_path}
                            required
                        >
                            <option value="" disabled>Pick a LoRA folder…</option>
                            {#each data.loras as l (l.path)}
                                <option value={l.path}>
                                    {l.name} · {l.safetensors_count} LoRA{l.safetensors_count === 1
                                        ? ''
                                        : 's'} · {l.has_final
                                        ? 'final'
                                        : l.max_step != null
                                          ? `max step ${l.max_step}`
                                          : 'no steps'}
                                </option>
                            {/each}
                        </select>
                        {#if data.loras.length === 0}
                            <p class="text-xs text-amber-400">
                                No LoRA families found under
                                <span class="font-mono"
                                    >{data.lora_root || 'lora_root'}</span
                                >. Set
                                <a href="/settings" class="underline">lora_root</a>.
                            </p>
                        {/if}
                    </div>
                </div>

                <!-- Row 2: dataset + prompts -->
                <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div class="space-y-1.5">
                        <label for="t-dataset" class="text-sm font-medium">Dataset</label>
                        <select
                            id="t-dataset"
                            name="dataset"
                            class="input"
                            bind:value={f.dataset_value}
                        >
                            <option value="">— (none)</option>
                            {#if data.dataset_groups.length > 0}
                                <optgroup label="Groups">
                                    {#each data.dataset_groups as g (g.id)}
                                        <option value="group:{g.id}"
                                            >{g.name} · {g.total_images} img</option
                                        >
                                    {/each}
                                </optgroup>
                            {/if}
                            {#if data.datasets.length > 0}
                                <optgroup label="Datasets">
                                    {#each data.datasets as d (d.path)}
                                        <option value="dataset:{d.path}"
                                            >{d.name} · {d.image_count} img</option
                                        >
                                    {/each}
                                </optgroup>
                            {/if}
                        </select>
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-prompts" class="text-sm font-medium">Prompts</label>
                        <select
                            id="t-prompts"
                            name="prompt_set_id"
                            class="input"
                            bind:value={f.prompt_set_id}
                        >
                            <option value="">— (none)</option>
                            {#each data.prompt_sets as p (p.id)}
                                <option value={p.id}>{p.name} · {p.prompt_count} prompts</option>
                            {/each}
                        </select>
                        {#if data.prompt_sets.length === 0}
                            <p class="text-xs text-amber-400">
                                No prompt sets yet.
                                <a href="/prompts" class="underline">Create one →</a>
                            </p>
                        {/if}
                        <!-- Preserve any legacy free-text path that's already on the row.
                             The server clears it whenever prompt_set_id is set. -->
                        <input type="hidden" name="prompts_path" value={f.prompts_path} />
                    </div>
                </div>

                <!-- Row 3: resolution / batch -->
                <div class="grid grid-cols-2 gap-4 md:grid-cols-4">
                    <div class="space-y-1.5">
                        <label for="t-w" class="text-sm font-medium">Width</label>
                        <input
                            id="t-w"
                            name="width"
                            type="number"
                            min="64"
                            step="8"
                            class="input"
                            bind:value={f.width}
                        />
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-h" class="text-sm font-medium">Height</label>
                        <input
                            id="t-h"
                            name="height"
                            type="number"
                            min="64"
                            step="8"
                            class="input"
                            bind:value={f.height}
                        />
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-batch" class="text-sm font-medium">Batch size</label>
                        <input
                            id="t-batch"
                            name="batch_size"
                            type="number"
                            min="0"
                            step="1"
                            class="input"
                            bind:value={f.batch_size}
                        />
                        <p class="text-xs text-fg-faint">0 = all missing in one go.</p>
                    </div>
                    <div class="space-y-1.5">
                        <span class="text-sm font-medium">&nbsp;</span>
                        <!-- spacer to keep grid aligned -->
                    </div>
                </div>

                <!-- Row 4: quant / offload -->
                <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div class="space-y-1.5">
                        <label for="t-quant" class="text-sm font-medium">Quantization</label>
                        <select
                            id="t-quant"
                            name="quant"
                            class="input"
                            bind:value={f.quant}
                        >
                            <option value="fp8_weight_only"
                                >FP8 weight-only (recommended, ~9 GB)</option
                            >
                            <option value="fp8_dynamic">FP8 dynamic (Hopper/Ada FP8 cores)</option
                            >
                            <option value="bf16">bf16 (no quant, ~22 GB)</option>
                        </select>
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-offload" class="text-sm font-medium">Offload</label>
                        <select
                            id="t-offload"
                            name="offload"
                            class="input"
                            bind:value={f.offload}
                        >
                            <option value="text-encoder">Text encoder only (24 GB)</option>
                            <option value="full">Full (small VRAM, slower)</option>
                            <option value="none">None (kept on GPU, fastest)</option>
                        </select>
                    </div>
                </div>

                <!-- Advanced -->
                <details class="rounded-md border border-border bg-bg-2/40" bind:open={showAdvanced}>
                    <summary class="cursor-pointer select-none px-3 py-2 text-sm text-fg-muted">
                        Advanced
                    </summary>
                    <div class="grid grid-cols-2 gap-4 p-4 md:grid-cols-4">
                        <div class="space-y-1.5">
                            <label for="a-seed" class="text-xs font-medium">Seed</label>
                            <input
                                id="a-seed"
                                name="advanced.seed"
                                type="number"
                                class="input"
                                bind:value={f.seed}
                            />
                        </div>
                        <div class="space-y-1.5">
                            <label for="a-steps" class="text-xs font-medium">Steps</label>
                            <input
                                id="a-steps"
                                name="advanced.steps"
                                type="number"
                                min="1"
                                class="input"
                                bind:value={f.steps}
                            />
                        </div>
                        <div class="space-y-1.5">
                            <label for="a-guidance" class="text-xs font-medium">Guidance</label>
                            <input
                                id="a-guidance"
                                name="advanced.guidance"
                                type="number"
                                step="0.1"
                                class="input"
                                bind:value={f.guidance}
                            />
                        </div>
                        <div class="space-y-1.5">
                            <label for="a-loras" class="text-xs font-medium">LoRA scale</label>
                            <input
                                id="a-loras"
                                name="advanced.lora_scale"
                                type="number"
                                step="0.05"
                                class="input"
                                bind:value={f.lora_scale}
                            />
                        </div>
                        <div class="space-y-1.5">
                            <label for="a-shift" class="text-xs font-medium">Shift (μ)</label>
                            <input
                                id="a-shift"
                                name="advanced.shift"
                                type="text"
                                placeholder="auto"
                                class="input"
                                bind:value={f.shift}
                            />
                        </div>
                        <div class="space-y-1.5">
                            <label for="a-minstep" class="text-xs font-medium">Min step</label>
                            <input
                                id="a-minstep"
                                name="advanced.min_step"
                                type="number"
                                min="0"
                                class="input"
                                bind:value={f.min_step}
                            />
                        </div>
                        <div class="space-y-1.5">
                            <label for="a-format" class="text-xs font-medium">Format</label>
                            <select
                                id="a-format"
                                name="advanced.format"
                                class="input"
                                bind:value={f.format}
                            >
                                <option value="png">png</option>
                                <option value="jpg">jpg</option>
                            </select>
                        </div>
                        <div class="space-y-1.5">
                            <span class="text-xs font-medium">&nbsp;</span>
                        </div>

                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.sage_attention"
                                type="checkbox"
                                bind:checked={f.sage_attention}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>SageAttention</span>
                        </label>
                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.compile_transformer"
                                type="checkbox"
                                bind:checked={f.compile_transformer}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>torch.compile transformer</span>
                        </label>
                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.preload_loras"
                                type="checkbox"
                                bind:checked={f.preload_loras}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>Preload all LoRAs as named adapters</span>
                        </label>
                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.comfyui_noise"
                                type="checkbox"
                                bind:checked={f.comfyui_noise}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>ComfyUI noise (bit-exact parity)</span>
                        </label>
                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.skip_face"
                                type="checkbox"
                                bind:checked={f.skip_face}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>Skip face scoring</span>
                        </label>
                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.force"
                                type="checkbox"
                                bind:checked={f.force}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>Force regenerate (ignore existing images)</span>
                        </label>
                    </div>
                </details>

                {#if form?.error}
                    <p class="text-sm text-red-400">{form.error}</p>
                {/if}

                <div class="flex items-center gap-3 pt-1">
                    <button type="submit" class="btn-primary" disabled={saving}>
                        {saving ? 'Saving…' : editingId === -1 ? 'Create test' : 'Save changes'}
                    </button>
                    <button
                        type="button"
                        class="btn-ghost"
                        onclick={closeEditor}
                        disabled={saving}
                    >
                        Cancel
                    </button>
                </div>
            </form>
            {/if}
        </section>
    {/if}
</div>
