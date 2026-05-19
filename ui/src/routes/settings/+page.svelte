<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import { page } from '$app/state';
    import MainPanel from '$lib/components/MainPanel.svelte';
    import type { ConnectorId, ConnectorTypeInfo } from '$lib/connectors/types';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    interface Field {
        key: 'dataset_root' | 'tests_root' | 'lora_root' | 'python_bin';
        label: string;
        hint: string;
        placeholder: string;
    }

    const fields: Field[] = [
        {
            key: 'dataset_root',
            label: 'Dataset folder',
            hint: 'Where your training datasets live. Each subfolder shows up as one dataset.',
            placeholder: '/path/to/datasets or D:\\datasets'
        },
        {
            key: 'tests_root',
            label: 'Tests folder',
            hint: 'Where grid test results are written. Each run gets its own subfolder.',
            placeholder: '/path/to/glt-tests or D:\\glt-tests'
        },
        {
            key: 'lora_root',
            label: 'LoRA folder',
            hint: 'Where your LoRA training output lives. Each subfolder is one family.',
            placeholder: '/path/to/loras or D:\\loras'
        },
        {
            key: 'python_bin',
            label: 'Python interpreter',
            hint: 'Path to the Python that runs inference. Usually the .venv at the repo root.',
            placeholder: '/path/to/.venv/bin/python or C:\\path\\to\\.venv\\Scripts\\python.exe'
        }
    ];

    let saving = $state(false);

    // ---- Connectors ----
    // Modal state: null = closed; ConnectorTypeInfo = "Add this connector".
    let addingType = $state<ConnectorTypeInfo | null>(null);
    let connectorSaving = $state(false);
    let removingId = $state<ConnectorId | null>(null);
    let testingId = $state<ConnectorId | null>(null);
    /** + Add dropdown state (lists every connector type that isn't already
     * configured). Wrapper bound so the click-outside listener can detect
     * dismissal events. */
    let addDropdownOpen = $state(false);
    let addDropdownWrapper = $state<HTMLDivElement | null>(null);

    /** Types not yet configured — what the + Add menu offers. */
    let availableToAdd = $derived(
        data.connector_types.filter(
            (t) => !data.connectors.find((c) => c.id === t.id && c.configured)
        )
    );

    function findType(id: ConnectorId): ConnectorTypeInfo | undefined {
        return data.connector_types.find((t) => t.id === id);
    }

    /** Click an item in the + Add dropdown. OAuth-flavored connectors
     * navigate straight to their start route; others open the credentials
     * modal. */
    function pickAdd(t: ConnectorTypeInfo) {
        addDropdownOpen = false;
        if (t.oauth_start_url) {
            window.location.href = t.oauth_start_url;
            return;
        }
        addingType = t;
    }

    /** Same as pickAdd but for an already-configured connector's "Edit"
     * action. OAuth connectors restart their flow (re-consent); others
     * reopen the credentials modal pre-filled. */
    function editConnector(id: ConnectorId) {
        const t = findType(id);
        if (!t) return;
        if (t.oauth_start_url) {
            window.location.href = t.oauth_start_url;
            return;
        }
        addingType = t;
    }

    $effect(() => {
        if (!addDropdownOpen) return;
        function onGlobalClick(e: MouseEvent) {
            const target = e.target as Node | null;
            if (addDropdownWrapper && target && addDropdownWrapper.contains(target)) {
                return;
            }
            addDropdownOpen = false;
        }
        document.addEventListener('click', onGlobalClick, true);
        return () => document.removeEventListener('click', onGlobalClick, true);
    });

    /** ?google_oauth=... flash params set by the OAuth callback. Mapped to
     * user-facing copy + tone for the connectors section banner. */
    let oauthFlash = $derived.by(() => {
        const v = page.url.searchParams.get('google_oauth');
        if (!v) return null;
        const map: Record<string, { tone: 'ok' | 'err'; msg: string }> = {
            ok: { tone: 'ok', msg: 'Signed in to Google Photos.' },
            state_mismatch: {
                tone: 'err',
                msg: 'OAuth callback state mismatch — try again.'
            },
            exchange_failed: {
                tone: 'err',
                msg: 'Could not exchange the authorization code with Google.'
            },
            signin_failed: {
                tone: 'err',
                msg: 'Google returned tokens but signing in failed.'
            }
        };
        return map[v] ?? { tone: 'err' as const, msg: `OAuth error: ${v}` };
    });

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

<MainPanel size="narrow">
    <div class="space-y-10">
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

            <!-- face_gpu_mem_limit_gb is intentionally NOT exposed in the
                 UI — capping ORT's BFC arena causes mid-run OOM crashes
                 (the allocator can't release memory to satisfy peak
                 requests once at the cap). The setting + plumbing remain
                 in case we ever need to re-surface it for a specific
                 contention scenario; default 0 = no cap. -->

            <!-- Suggestions tuning section -->
            <div class="space-y-1.5 border-t border-border pt-5">
                <label for="suggestion_min_image_mp" class="text-sm font-medium"
                    >Suggestion min. image resolution (megapixels)</label
                >
                <input
                    id="suggestion_min_image_mp"
                    name="suggestion_min_image_mp"
                    type="number"
                    step="0.1"
                    min="0"
                    max="200"
                    value={(form?.settings ?? data.settings).suggestion_min_image_mp}
                    placeholder="1"
                    class="input max-w-[12rem] font-mono"
                />
                <p class="text-xs text-fg-faint">
                    Photos smaller than this won't be suggested for import. Default
                    <span class="font-mono">1 MP</span> ≈ 1000×1000 — a sensible floor for
                    portrait training. Set <span class="font-mono">0</span> to disable.
                </p>
                {#if form?.warnings?.suggestion_min_image_mp}
                    <p class="text-xs text-amber-400">
                        {form.warnings.suggestion_min_image_mp}
                    </p>
                {/if}
            </div>

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
                    External photo sources used to build your datasets.
                </p>
            </div>
            {#if availableToAdd.length > 0}
                <div class="relative inline-block" bind:this={addDropdownWrapper}>
                    <button
                        type="button"
                        class="btn-primary"
                        aria-haspopup="menu"
                        aria-expanded={addDropdownOpen}
                        onclick={() => (addDropdownOpen = !addDropdownOpen)}
                    >
                        + Add connector
                    </button>
                    {#if addDropdownOpen}
                        <div
                            class="absolute right-0 top-full z-30 mt-1 w-56 rounded-md border border-border bg-bg-1 p-1 shadow-lg"
                            role="menu"
                        >
                            {#each availableToAdd as t (t.id)}
                                <button
                                    type="button"
                                    class="flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left text-xs transition-colors hover:bg-bg-2"
                                    onclick={() => pickAdd(t)}
                                >
                                    <span>{t.label}</span>
                                    {#if t.oauth_start_url}
                                        <span class="text-[10px] text-fg-faint">OAuth</span>
                                    {/if}
                                </button>
                            {/each}
                        </div>
                    {/if}
                </div>
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
                                            onclick={() => editConnector(c.id)}
                                            title={findType(c.id)?.oauth_start_url
                                                ? 'Restart OAuth consent'
                                                : 'Edit credentials'}
                                            >{findType(c.id)?.oauth_start_url
                                                ? 'Re-sign-in'
                                                : 'Edit'}</button
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

        {#if oauthFlash}
            <p
                class="mt-2 text-xs {oauthFlash.tone === 'ok'
                    ? 'text-emerald-300'
                    : 'text-red-300'}"
            >
                {oauthFlash.msg}
            </p>
        {/if}
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
</MainPanel>

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
                    {@const current = data.credentials?.[t.id]?.[f.key]}
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
                            value={typeof current === 'string' ? current : ''}
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
