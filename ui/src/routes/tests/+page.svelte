<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import MainPanel from '$lib/components/MainPanel.svelte';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    // Editor state: null = closed; -1 = create new; positive = edit that id.
    let editingId = $state<number | null>(null);
    let saving = $state(false);
    let deletingId = $state<number | null>(null);
    let runningId = $state<number | null>(null);

    // Live refresh while any test has an active grid-test-run job. Polls
    // every 2s — invalidateAll re-runs the page load, which re-reads the
    // job's progress + the test_runs cells count. Stops as soon as no
    // test is active anymore (server returns active_job=null).
    let anyActive = $derived(data.tests.some((t) => t.active_job != null));
    $effect(() => {
        if (!anyActive) return;
        const id = setInterval(() => invalidateAll(), 2000);
        return () => clearInterval(id);
    });
    let showAdvanced = $state(false);

    /** The "open in details panel" row from `data.tests` (or null when creating). */
    let openedTest = $derived(
        editingId != null && editingId > 0 ? data.tests.find((t) => t.id === editingId) ?? null : null
    );

    // Editable fields (kept in sync with editor state).
    let f = $state({
        name: '',
        lora_path: '',
        dataset_value: '', // encoded "dataset:<path>" or "group:<id>"
        prompt_set_id: '', // stringified id or '' (matches <select> value)
        prompts_path: '', // legacy: free-text path, passed through hidden field
        trigger: '',
        resolution: '1MP',
        batch_size: 0,
        quant: 'auto',
        compile_mode: 'on' as 'on' | 'auto' | 'off',
        // advanced
        seed: 42,
        steps: 4,
        guidance: 1.0,
        lora_scale: 1.0,
        format: 'png',
        sage_attention: false,
        preload_loras: false,
        shift: '',
        min_step: 0,
        qwen_dtype: 'bf16',
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
            trigger: '',
            resolution: '1MP',
            batch_size: 0,
            quant: 'auto',
        compile_mode: 'on' as 'on' | 'auto' | 'off',
            seed: 42,
            steps: 4,
            guidance: 1.0,
            lora_scale: 1.0,
            format: 'png',
            sage_attention: false,
            preload_loras: false,
            shift: '',
            min_step: 0,
            qwen_dtype: 'bf16',
            skip_face: false,
            force: false
        };
    }

    function openNew() {
        reset();
        editingId = -1;
        showAdvanced = false;
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
            trigger: t.trigger ?? '',
            resolution: t.resolution ?? '1024x1024',
            batch_size: t.batch_size,
            quant: t.quant,
            compile_mode: t.compile_mode,
            seed: Number(adv.seed ?? 42),
            steps: Number(adv.steps ?? 4),
            guidance: Number(adv.guidance ?? 1.0),
            lora_scale: Number(adv.lora_scale ?? 1.0),
            format: String(adv.format ?? 'png'),
            sage_attention: Boolean(adv.sage_attention),
            preload_loras: Boolean(adv.preload_loras),
            shift: adv.shift != null ? String(adv.shift) : '',
            min_step: Number(adv.min_step ?? 0),
            qwen_dtype: String(adv.qwen_dtype ?? 'bf16'),
            skip_face: Boolean(adv.skip_face),
            force: Boolean(adv.force)
        };
        editingId = t.id;
        showAdvanced = false;
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
        },
        failed: {
            label: 'Failed',
            classes: 'bg-red-500/15 text-red-300'
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

<MainPanel>
    <div class="space-y-6">
    <header class="flex items-baseline justify-between gap-4">
        <div>
            <h1 class="text-2xl font-semibold tracking-tight">Tests</h1>
            <p class="mt-1 text-sm text-fg-muted">
                Saved grid recipes — pick a LoRA family, a dataset, and a prompt set, then run.
            </p>
        </div>
        <div class="flex items-center gap-2">
            <a href="/settings" class="btn-ghost text-xs">Settings…</a>
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
                                <a
                                    href={`/tests/${t.id}`}
                                    class="font-medium text-fg hover:text-accent"
                                    title="Open live grid view">{t.name}</a
                                >
                                <div class="text-xs text-fg-faint">
                                    {t.resolution || '—'} · batch {t.batch_size || 'auto'}
                                    {#if t.trigger}
                                        · trigger <span class="font-mono">{t.trigger}</span>
                                    {/if}
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
                                <div class="flex flex-col items-start gap-1">
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
                                        {#if t.active_job && t.active_job.status === 'queued'}
                                            <span class="ml-1 text-fg-faint">· queued</span>
                                        {/if}
                                    </span>
                                    {#if t.active_job}
                                        {@const aj = t.active_job}
                                        {@const pct =
                                            aj.progress_total && aj.progress_total > 0
                                                ? Math.round(
                                                      (aj.progress_current / aj.progress_total) *
                                                          100
                                                  )
                                                : null}
                                        <a
                                            href={`/jobs?focus=${aj.id}`}
                                            class="text-[10px] text-accent hover:text-accent-hover"
                                            title="Open job logs"
                                        >
                                            job #{aj.id}
                                            {#if pct != null}
                                                · {aj.progress_current}/{aj.progress_total} ({pct}%)
                                            {:else if aj.current_label}
                                                · {aj.current_label}
                                            {:else}
                                                · starting…
                                            {/if}
                                        </a>
                                        {#if pct != null}
                                            <div class="h-1 w-32 overflow-hidden rounded bg-bg-3">
                                                <div
                                                    class="h-full bg-accent transition-all"
                                                    style="width: {pct}%"
                                                ></div>
                                            </div>
                                        {/if}
                                    {/if}
                                </div>
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
                                    <form
                                        method="POST"
                                        action="?/run"
                                        use:enhance={() => {
                                            runningId = t.id;
                                            return ({ update }) => {
                                                update({ reset: false }).finally(
                                                    () => (runningId = null)
                                                );
                                            };
                                        }}
                                    >
                                        <input type="hidden" name="id" value={t.id} />
                                        <button
                                            type="submit"
                                            class="btn-ghost px-2 py-1 text-xs text-emerald-300 hover:bg-emerald-500/10 hover:text-emerald-200 disabled:opacity-50"
                                            disabled={runningId === t.id || t.status === 'in_progress'}
                                            title={t.status === 'in_progress'
                                                ? 'A run is already in progress for this test'
                                                : 'Queue a grid run as a background job'}
                                        >
                                            {runningId === t.id ? 'Queuing…' : '▶ Run'}
                                        </button>
                                    </form>
                                    {#if t.status === 'in_progress' && t.active_job == null}
                                        <!-- Stuck row: status says running but no
                                             queued/active job backs it. The PID reaper
                                             handles this automatically (10s cadence on
                                             dead pids, restart-time sweep), but expose
                                             a manual button for impatient users. -->
                                        <form
                                            method="POST"
                                            action="?/reconcile-runs"
                                            use:enhance={() => {
                                                return ({ update }) => update();
                                            }}
                                        >
                                            <input type="hidden" name="id" value={t.id} />
                                            <button
                                                type="submit"
                                                class="btn-ghost px-2 py-1 text-xs text-amber-300 hover:bg-amber-500/10 hover:text-amber-200"
                                                title="Mark the stuck 'running' test_run as failed (PID is gone)"
                                            >
                                                ⚠ Reconcile
                                            </button>
                                        </form>
                                    {/if}
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

    <!-- =================== DETAILS =================== -->
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
            </div>

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
                                <!-- value must be a string because f.prompt_set_id is
                                     a string ('' or stringified id) — Svelte 5 uses
                                     strict equality to match select↔option. Passing
                                     the bare numeric id leaves the select stuck on
                                     "— (none)" when editing an existing test. -->
                                <option value={String(p.id)}>
                                    {p.name} · {p.prompt_count} prompts
                                </option>
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

                <!-- Row 3: trigger / resolution / batch -->
                <div class="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <div class="space-y-1.5">
                        <label for="t-trigger" class="text-sm font-medium">Trigger word</label>
                        <input
                            id="t-trigger"
                            name="trigger"
                            type="text"
                            placeholder="e.g. m4nd1234"
                            class="input"
                            bind:value={f.trigger}
                        />
                        <p class="text-xs text-fg-faint">
                            Substituted into <span class="font-mono">[trigger]</span> by the
                            grid script. Leave empty to keep placeholders.
                        </p>
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-resolution" class="text-sm font-medium">Resolution</label>
                        <select
                            id="t-resolution"
                            name="resolution"
                            class="input"
                            bind:value={f.resolution}
                        >
                            <option value="0.5MP">0.5 MP (~724²)</option>
                            <option value="1MP">1 MP (1024²)</option>
                            <option value="1.5MP">1.5 MP (~1254²)</option>
                            <option value="2MP">2 MP (~1448²)</option>
                            <option value="3MP">3 MP (~1773²)</option>
                            <option value="4MP">4 MP (~2048²)</option>
                        </select>
                        <p class="text-xs text-fg-faint">
                            Target image area. The grid script derives W×H per cell
                            from this + the per-prompt AR tag
                            (<span class="font-mono">[3:4]</span>, <span class="font-mono">[16:9]</span>…).
                        </p>
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
                </div>

                <!-- Row 4: quant + compile -->
                <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div class="space-y-1.5">
                        <label for="t-quant" class="text-sm font-medium">Quantization</label>
                        <select
                            id="t-quant"
                            name="quant"
                            class="input"
                            bind:value={f.quant}
                        >
                            <option value="auto">Auto (pick by GPU — recommended)</option>
                            <option value="int8_convrot"
                                >INT8 ConvRot (Ampere best, ~2× vs FP8 weight)</option
                            >
                            <option value="fp8_weight"
                                >FP8 weight-only (Ada/Hopper native, ~9 GB)</option
                            >
                            <option value="fp8_dynamic">FP8 dynamic (Ada/Hopper FP8 cores)</option>
                            <option value="fp8_quanto">FP8 quanto (legacy, needs CUDA toolkit)</option>
                            <option value="none">bf16 (no quant, ~22 GB)</option>
                        </select>
                    </div>
                    <div class="space-y-1.5">
                        <label for="t-compile" class="text-sm font-medium">torch.compile</label>
                        <select
                            id="t-compile"
                            name="compile_mode"
                            class="input"
                            bind:value={f.compile_mode}
                        >
                            <option value="on">On (default — ~2× faster, persistent disk cache)</option>
                            <option value="auto"
                                >Auto (enable when n_loras × n_prompts ≥ n_shapes × 8)</option
                            >
                            <option value="off">Off (skip compile — useful for one-off tests)</option>
                        </select>
                    </div>
                </div>
                <p class="text-xs text-fg-faint">
                    Qwen3 and the transformer are never co-resident — Qwen is loaded for the
                    encode pass only, then unloaded before the transformer loads. Compile
                    artifacts persist in <code>~/.cache/glt/torchinductor/</code> across runs.
                </p>

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
                            <label for="a-qwen-dtype" class="text-xs font-medium">
                                Qwen dtype
                            </label>
                            <select
                                id="a-qwen-dtype"
                                name="advanced.qwen_dtype"
                                class="input"
                                bind:value={f.qwen_dtype}
                            >
                                <option value="bf16">bf16 (~16 GB, default)</option>
                                <option value="fp16">fp16 (~16 GB)</option>
                                <option value="fp8_e4m3fn">fp8_e4m3fn (~8 GB, for ≤16 GB cards)</option>
                                <option value="fp8_e5m2">fp8_e5m2 (~8 GB)</option>
                            </select>
                            <p class="text-[10px] text-fg-faint">
                                Qwen is unloaded after encode — dtype only matters during the
                                brief encode pass. Drop to fp8 if you only have 16 GB of VRAM.
                            </p>
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
                        <!-- torch.compile moved to the main form (next to
                             Quantization). It's a top-level performance knob,
                             not an advanced experiment, and it's on by default
                             with the disk cache. -->
                        <label class="col-span-2 flex items-center gap-2 text-xs">
                            <input
                                name="advanced.preload_loras"
                                type="checkbox"
                                bind:checked={f.preload_loras}
                                class="h-4 w-4 rounded border-border bg-bg-1 text-accent"
                            />
                            <span>Preload all LoRAs as named adapters</span>
                        </label>
                        <!-- ComfyUI noise: forced-on for bit-exact parity, no toggle.
                             Hidden field ensures every create/update writes
                             advanced.comfyui_noise = true. -->
                        <input
                            type="hidden"
                            name="advanced.comfyui_noise"
                            value="on"
                        />
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
        </section>
    {/if}
    </div>
</MainPanel>
