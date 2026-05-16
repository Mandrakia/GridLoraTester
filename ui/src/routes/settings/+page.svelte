<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import type { ConnectorId, ConnectorTypeInfo } from '$lib/connectors/types';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    interface Field {
        key:
            | 'dataset_root'
            | 'tests_root'
            | 'lora_root'
            | 'python_bin'
            | 'glt_root';
        label: string;
        hint: string;
        placeholder: string;
    }

    const fields: Field[] = [
        {
            key: 'dataset_root',
            label: 'Dataset paths',
            hint: 'Root folder for datasets. Each immediate subfolder is treated as one dataset.',
            placeholder: '/home/you/datasets'
        },
        {
            key: 'tests_root',
            label: 'Tests paths',
            hint: 'Root folder where grid test results are written (one subfolder per run).',
            placeholder: '/home/you/glt-tests'
        },
        {
            key: 'lora_root',
            label: 'Lora paths',
            hint: 'Root folder for LoRA outputs. Each immediate subfolder is one family / training run.',
            placeholder: '/home/you/loras'
        },
        {
            key: 'python_bin',
            label: 'Python interpreter',
            hint: 'Path to the Python binary of the venv that has glt installed (insightface, onnxruntime, sharp deps). Used to compute dataset centroids.',
            placeholder: '/home/you/glt/.venv/bin/python'
        },
        {
            key: 'glt_root',
            label: 'GridLoraTester repo root',
            hint: 'Absolute path to the GridLoraTester checkout (the folder that contains the `glt/` package). The dashboard cd-s here before spawning `python -m glt`.',
            placeholder: '/home/you/GridLoraTester'
        }
    ];

    let saving = $state(false);

    // ---- Connectors ----
    // Modal state: null = closed; ConnectorTypeInfo = "Add this connector".
    let addingType = $state<ConnectorTypeInfo | null>(null);
    let connectorSaving = $state(false);
    let removingId = $state<ConnectorId | null>(null);
    let testingId = $state<ConnectorId | null>(null);

    /** Types not yet configured — what the + Add menu offers. */
    let availableToAdd = $derived(
        data.connector_types.filter(
            (t) => !data.connectors.find((c) => c.id === t.id && c.configured)
        )
    );

    function findType(id: ConnectorId): ConnectorTypeInfo | undefined {
        return data.connector_types.find((t) => t.id === id);
    }

    function fmtDate(iso: string | null) {
        if (!iso) return null;
        try {
            return new Date(iso.replace(' ', 'T') + (iso.endsWith('Z') ? '' : 'Z')).toLocaleString();
        } catch {
            return iso;
        }
    }
</script>

<svelte:head>
    <title>Settings — GridLoraTester</title>
</svelte:head>

<div class="space-y-10 p-6">
    <header>
        <h1 class="text-2xl font-semibold tracking-tight">Settings</h1>
        <p class="mt-1 text-sm text-fg-muted">Configure where GridLoraTester looks for your data.</p>
    </header>

    <!-- ============================================================ -->
    <!-- Paths                                                        -->
    <!-- ============================================================ -->
    <section>
        <h2 class="mb-3 text-base font-medium">Paths</h2>
        <form
            method="POST"
            action="?/save-paths"
            use:enhance={() => {
                saving = true;
                return ({ update }) => {
                    update({ reset: false }).finally(() => (saving = false));
                };
            }}
            class="max-w-3xl space-y-5"
        >
            {#each fields as f (f.key)}
                <div class="space-y-1.5">
                    <label for={f.key} class="text-sm font-medium">{f.label}</label>
                    <input
                        id={f.key}
                        name={f.key}
                        type="text"
                        value={(form?.settings ?? data.settings)[f.key]}
                        placeholder={f.placeholder}
                        autocomplete="off"
                        spellcheck="false"
                        class="input font-mono"
                    />
                    <p class="text-xs text-fg-faint">{f.hint}</p>
                    {#if form?.warnings?.[f.key]}
                        <p class="text-xs text-amber-400">{form.warnings[f.key]}</p>
                    {/if}
                </div>
            {/each}

            <div class="flex items-center gap-3 pt-2">
                <button type="submit" class="btn-primary" disabled={saving}>
                    {saving ? 'Saving…' : 'Save'}
                </button>
                {#if form?.saved}
                    <span class="text-xs text-fg-muted">Saved.</span>
                {/if}
            </div>
        </form>
    </section>

    <!-- ============================================================ -->
    <!-- Connectors                                                   -->
    <!-- ============================================================ -->
    <section>
        <div class="mb-3 flex items-baseline justify-between gap-3">
            <div>
                <h2 class="text-base font-medium">Connectors</h2>
                <p class="text-xs text-fg-faint">
                    External photo databases. Used as sources to build datasets.
                </p>
            </div>
            {#if availableToAdd.length > 0}
                <button
                    type="button"
                    class="btn-primary"
                    onclick={() => (addingType = availableToAdd[0])}
                    title={availableToAdd.length > 1
                        ? `Pick one of: ${availableToAdd.map((t) => t.label).join(', ')}`
                        : `Add ${availableToAdd[0].label}`}
                >
                    + Add connector
                </button>
            {/if}
        </div>

        <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
            <table class="w-full text-sm">
                <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                    <tr>
                        <th class="px-4 py-2.5 text-left font-medium">Connector</th>
                        <th class="px-4 py-2.5 text-left font-medium">Status</th>
                        <th class="px-4 py-2.5 text-left font-medium">Last check</th>
                        <th class="w-40 px-4 py-2.5 text-right font-medium">Actions</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-border">
                    {#if data.connectors.filter((c) => c.configured).length === 0}
                        <tr>
                            <td colspan="4" class="px-4 py-6 text-center text-sm text-fg-faint">
                                No connectors yet — click "+ Add connector" to wire one up.
                            </td>
                        </tr>
                    {:else}
                        {#each data.connectors.filter((c) => c.configured) as c (c.id)}
                            <tr class="transition-colors hover:bg-bg-2/40">
                                <td class="px-4 py-2.5">
                                    <div class="font-medium">{c.label}</div>
                                    <div class="font-mono text-[10px] text-fg-faint">{c.id}</div>
                                </td>
                                <td class="px-4 py-2.5">
                                    {#if c.signed_in}
                                        <span
                                            class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs text-emerald-300"
                                            >signed in</span
                                        >
                                    {:else}
                                        <span
                                            class="rounded-full bg-red-500/15 px-2 py-0.5 text-xs text-red-300"
                                            >error</span
                                        >
                                        {#if c.last_error}
                                            <div class="mt-1 text-[10px] text-red-300/80">
                                                {c.last_error}
                                            </div>
                                        {/if}
                                    {/if}
                                </td>
                                <td class="px-4 py-2.5 text-xs text-fg-muted">
                                    {fmtDate(c.last_check_at) ?? '—'}
                                </td>
                                <td class="px-4 py-2.5">
                                    <div class="flex justify-end gap-1">
                                        <form
                                            method="POST"
                                            action="?/connector-test"
                                            use:enhance={() => {
                                                testingId = c.id;
                                                return async ({ update }) => {
                                                    // Re-fetch page data so the status pill
                                                    // + last_error reflect the fresh DB row.
                                                    await update({ reset: false });
                                                    await invalidateAll();
                                                    testingId = null;
                                                };
                                            }}
                                        >
                                            <input type="hidden" name="connector_id" value={c.id} />
                                            <button
                                                type="submit"
                                                class="btn-ghost px-2 py-1 text-xs"
                                                disabled={testingId === c.id}
                                                >{testingId === c.id ? 'Testing…' : 'Test'}</button
                                            >
                                        </form>
                                        <button
                                            type="button"
                                            class="btn-ghost px-2 py-1 text-xs"
                                            onclick={() => {
                                                const t = findType(c.id);
                                                if (t) addingType = t;
                                            }}>Edit</button
                                        >
                                        <form
                                            method="POST"
                                            action="?/connector-remove"
                                            use:enhance={() => {
                                                removingId = c.id;
                                                return ({ update }) => {
                                                    update({ reset: false }).finally(
                                                        () => (removingId = null)
                                                    );
                                                };
                                            }}
                                        >
                                            <input type="hidden" name="connector_id" value={c.id} />
                                            <button
                                                type="submit"
                                                class="btn-ghost px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                                disabled={removingId === c.id}
                                                onclick={(e) => {
                                                    if (!confirm(`Remove ${c.label} connector?`))
                                                        e.preventDefault();
                                                }}
                                                >{removingId === c.id
                                                    ? '…'
                                                    : 'Remove'}</button
                                            >
                                        </form>
                                    </div>
                                </td>
                            </tr>
                        {/each}
                    {/if}
                </tbody>
            </table>
        </div>

        {#if form?.connector_saved}
            <p class="mt-2 text-xs text-emerald-300">
                {findType(form.connector_saved)?.label} saved.
            </p>
        {/if}
        {#if form?.connector_tested && form?.connector_test_ok}
            <p class="mt-2 text-xs text-emerald-300">
                Test OK on {findType(form.connector_tested)?.label}.
            </p>
        {/if}
        {#if form?.connector_removed}
            <p class="mt-2 text-xs text-fg-muted">
                {findType(form.connector_removed)?.label} removed.
            </p>
        {/if}
    </section>
</div>

<!-- ============================================================ -->
<!-- Add / Edit connector modal                                    -->
<!-- ============================================================ -->
{#if addingType}
    {@const t = addingType}
    {@const existing = data.connectors.find((c) => c.id === t.id)}
    <div
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
        role="button"
        tabindex="-1"
        aria-label="Close"
        onclick={() => (addingType = null)}
        onkeydown={(e) => {
            if (e.key === 'Escape') addingType = null;
        }}
    >
        <div
            class="w-full max-w-md rounded-lg border border-border bg-bg-1 p-5"
            role="presentation"
            onclick={(e) => e.stopPropagation()}
        >
            <h3 class="mb-1 text-base font-medium">
                {existing?.configured ? `Edit ${t.label}` : `Add ${t.label}`}
            </h3>
            <p class="mb-4 text-xs text-fg-faint">Saving validates the credentials against the server.</p>

            <form
                method="POST"
                action="?/connector-save"
                use:enhance={() => {
                    connectorSaving = true;
                    return async ({ result, update }) => {
                        await update({ reset: false });
                        connectorSaving = false;
                        if (result.type === 'success') {
                            addingType = null;
                            await invalidateAll();
                        }
                    };
                }}
                class="space-y-4"
            >
                <input type="hidden" name="connector_id" value={t.id} />

                {#each t.credentials_fields as f (f.key)}
                    <div class="space-y-1.5">
                        <label for={`cred-${t.id}-${f.key}`} class="text-sm font-medium"
                            >{f.label}{f.required ? ' *' : ''}</label
                        >
                        <input
                            id={`cred-${t.id}-${f.key}`}
                            name={f.key}
                            type={f.type}
                            class="input font-mono"
                            placeholder={f.placeholder ?? ''}
                            autocomplete="off"
                            spellcheck="false"
                            required={f.required}
                        />
                        {#if f.help}
                            <p class="text-xs text-fg-faint">{f.help}</p>
                        {/if}
                    </div>
                {/each}

                {#if form?.connector_error && form?.connector_id === t.id}
                    <p class="text-sm text-red-300">{form.connector_error}</p>
                {/if}

                <div class="flex items-center justify-end gap-2 pt-1">
                    <button
                        type="button"
                        class="btn-ghost"
                        onclick={() => (addingType = null)}
                        disabled={connectorSaving}>Cancel</button
                    >
                    <button type="submit" class="btn-primary" disabled={connectorSaving}>
                        {connectorSaving ? 'Validating…' : existing?.configured ? 'Update' : 'Add'}
                    </button>
                </div>
            </form>
        </div>
    </div>
{/if}
