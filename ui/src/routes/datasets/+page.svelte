<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import ConnectorLinks from '$lib/components/ConnectorLinks.svelte';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    // Editor state: when `editingId === null` the editor is closed; -1 means
    // "creating new", any positive id means "editing that group".
    let editingId = $state<number | null>(null);
    let editorName = $state('');
    let editorPaths = $state<Set<string>>(new Set());
    let saving = $state(false);
    let deletingId = $state<number | null>(null);

    function openNew() {
        editingId = -1;
        editorName = '';
        editorPaths = new Set();
    }

    function openEdit(g: PageData['groups'][number]) {
        editingId = g.id;
        editorName = g.name;
        editorPaths = new Set(g.paths);
    }

    function closeEditor() {
        editingId = null;
    }

    function togglePath(p: string) {
        const next = new Set(editorPaths);
        if (next.has(p)) next.delete(p);
        else next.add(p);
        editorPaths = next;
    }
</script>

<svelte:head>
    <title>Datasets — GridLoraTester</title>
</svelte:head>

<div class="mx-auto max-w-6xl space-y-8 p-8">
    <header class="flex items-baseline justify-between gap-4">
        <div>
            <h1 class="text-2xl font-semibold tracking-tight">Datasets</h1>
            <p class="mt-1 text-sm text-fg-muted">
                Browse the dataset root and bundle folders into reusable groups.
            </p>
        </div>
        <a href="/settings" class="btn-ghost">Configure roots…</a>
    </header>

    <!-- ============================================================ -->
    <!-- Dataset groups (DB-backed, user-curated)                     -->
    <!-- ============================================================ -->
    <section>
        <div class="mb-3 flex items-center justify-between">
            <div>
                <h2 class="text-base font-medium">Dataset groups</h2>
                <p class="text-xs text-fg-faint">
                    Named bundles of dataset folders, stored in <code class="font-mono">glt.db</code>.
                </p>
            </div>
            <button
                type="button"
                class="btn-primary"
                disabled={editingId !== null || data.datasets.length === 0}
                onclick={openNew}
            >
                + New group
            </button>
        </div>

        <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
            <table class="w-full text-sm">
                <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                    <tr>
                        <th class="px-4 py-2.5 text-left font-medium">Name</th>
                        <th class="px-4 py-2.5 text-left font-medium">Datasets</th>
                        <th class="px-4 py-2.5 text-right font-medium">Images</th>
                        <th class="w-32 px-4 py-2.5 text-right font-medium">Actions</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-border">
                    {#if data.groups.length === 0}
                        <tr>
                            <td colspan="4" class="px-4 py-6 text-center text-sm text-fg-faint">
                                No groups yet. {data.datasets.length === 0
                                    ? 'Add a dataset_root first.'
                                    : 'Click "New group" above to create one.'}
                            </td>
                        </tr>
                    {:else}
                        {#each data.groups as g (g.id)}
                            <tr class="transition-colors hover:bg-bg-2/40">
                                <td class="px-4 py-2.5 align-top">
                                    <div class="flex items-center gap-2">
                                        <a
                                            href={`/datasets/group/${g.id}`}
                                            class="font-medium hover:text-accent-hover"
                                        >
                                            {g.name}
                                        </a>
                                        <ConnectorLinks
                                            scope_kind="group"
                                            scope_key={String(g.id)}
                                            connectors={data.connectors}
                                            connector_types={data.all_connector_types}
                                            links={data.links_by_group[String(g.id)] ?? []}
                                        />
                                    </div>
                                    <div class="text-xs text-fg-faint">
                                        {g.paths.length} path{g.paths.length === 1 ? '' : 's'}
                                        {#if g.missing_paths.length > 0}
                                            <span class="ml-1 text-amber-400"
                                                >· {g.missing_paths.length} missing</span
                                            >
                                        {/if}
                                    </div>
                                </td>
                                <td class="px-4 py-2.5 align-top">
                                    <div
                                        class="truncate text-fg-muted"
                                        title={g.paths.join('\n')}
                                    >
                                        {g.dataset_names || '—'}
                                    </div>
                                </td>
                                <td class="px-4 py-2.5 text-right align-top tabular-nums"
                                    >{g.total_images}</td
                                >
                                <td class="px-4 py-2.5 align-top">
                                    <div class="flex justify-end gap-1">
                                        <button
                                            type="button"
                                            class="btn-ghost px-2 py-1 text-xs"
                                            onclick={() => openEdit(g)}
                                            disabled={editingId !== null}
                                        >
                                            Edit
                                        </button>
                                        <form
                                            method="POST"
                                            action="?/delete"
                                            use:enhance={() => {
                                                deletingId = g.id;
                                                return ({ update }) => {
                                                    update({ reset: false }).finally(() => {
                                                        deletingId = null;
                                                    });
                                                };
                                            }}
                                        >
                                            <input type="hidden" name="id" value={g.id} />
                                            <button
                                                type="submit"
                                                class="btn-ghost px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                                onclick={(e) => {
                                                    if (!confirm(`Delete group "${g.name}"?`))
                                                        e.preventDefault();
                                                }}
                                                disabled={deletingId === g.id}
                                            >
                                                {deletingId === g.id ? '…' : 'Delete'}
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
    </section>

    <!-- ============================================================ -->
    <!-- Editor (create / edit a group)                               -->
    <!-- ============================================================ -->
    {#if editingId !== null}
        <section class="card">
            <h3 class="mb-3 text-sm font-medium">
                {editingId === -1 ? 'New dataset group' : 'Edit dataset group'}
            </h3>
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
                class="space-y-4"
            >
                {#if editingId > 0}
                    <input type="hidden" name="id" value={editingId} />
                {/if}

                <div class="space-y-1.5">
                    <label for="grp-name" class="text-sm font-medium">Name</label>
                    <input
                        id="grp-name"
                        name="name"
                        type="text"
                        bind:value={editorName}
                        placeholder="e.g. portraits-v2"
                        autocomplete="off"
                        class="input"
                        required
                    />
                </div>

                <div class="space-y-1.5">
                    <span class="text-sm font-medium">Datasets</span>
                    <p class="text-xs text-fg-faint">
                        Pick one or more subfolders of
                        <span class="font-mono text-fg-muted">{data.dataset_root || 'dataset_root'}</span
                        >.
                    </p>
                    {#if data.datasets.length === 0}
                        <p class="text-xs text-amber-400">
                            No datasets found — configure <a href="/settings" class="underline"
                                >dataset_root</a
                            > first.
                        </p>
                    {:else}
                        <div
                            class="max-h-64 space-y-1 overflow-y-auto rounded-md border border-border bg-bg-2 p-2"
                        >
                            {#each data.datasets as ds (ds.path)}
                                <label
                                    class="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-bg-3"
                                >
                                    <input
                                        type="checkbox"
                                        name="path"
                                        value={ds.path}
                                        checked={editorPaths.has(ds.path)}
                                        onchange={() => togglePath(ds.path)}
                                        class="h-4 w-4 rounded border-border bg-bg-1 text-accent focus:ring-1 focus:ring-accent"
                                    />
                                    <span class="flex-1 truncate font-mono text-xs">{ds.name}</span>
                                    <span class="text-xs text-fg-faint">{ds.image_count} img</span>
                                </label>
                            {/each}
                        </div>
                    {/if}
                    <p class="text-xs text-fg-faint">
                        Selected: {editorPaths.size}
                    </p>
                </div>

                {#if form?.error}
                    <p class="text-sm text-red-400">{form.error}</p>
                {/if}

                <div class="flex items-center gap-3">
                    <button type="submit" class="btn-primary" disabled={saving}>
                        {saving ? 'Saving…' : editingId === -1 ? 'Create group' : 'Save changes'}
                    </button>
                    <button type="button" class="btn-ghost" onclick={closeEditor} disabled={saving}>
                        Cancel
                    </button>
                </div>
            </form>
        </section>
    {/if}

    <!-- ============================================================ -->
    <!-- Available datasets (FS-iter on dataset_root)                 -->
    <!-- ============================================================ -->
    <section>
        <div class="mb-3">
            <h2 class="text-base font-medium">Available datasets</h2>
            <p class="text-xs text-fg-faint">
                Live listing of <span class="font-mono text-fg-muted"
                    >{data.dataset_root || 'dataset_root'}</span
                > — not stored in the DB.
            </p>
        </div>

        {#if !data.dataset_root}
            <div class="card text-sm text-fg-muted">
                No <span class="font-mono text-fg">dataset_root</span> configured.
                <a href="/settings" class="text-accent hover:text-accent-hover"
                    >Set it in Settings →</a
                >
            </div>
        {:else if data.datasets.length === 0}
            <div class="card text-sm text-fg-muted">
                No subfolders found under
                <span class="font-mono text-fg">{data.dataset_root}</span>.
            </div>
        {:else}
            <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {#each data.datasets as ds (ds.path)}
                    <a
                        href={`/datasets/folder/${encodeURIComponent(ds.name)}`}
                        class="card transition-colors hover:border-border-strong hover:bg-bg-2/40"
                    >
                        <div class="flex items-start justify-between gap-2">
                            <h3 class="truncate text-sm font-medium" title={ds.path}>{ds.name}</h3>
                            <span
                                class="shrink-0 rounded-full bg-bg-3 px-2 py-0.5 text-xs text-fg-muted"
                            >
                                {ds.image_count} img
                            </span>
                        </div>
                        <!-- Connector links replace the redundant full path
                             (the basename is already shown above; the path
                             title= attribute keeps the full one on hover). -->
                        <div class="mt-2">
                            <ConnectorLinks
                                scope_kind="folder"
                                scope_key={ds.path}
                                connectors={data.connectors}
                                connector_types={data.all_connector_types}
                                links={data.links_by_folder[ds.path] ?? []}
                            />
                        </div>
                    </a>
                {/each}
            </div>
        {/if}
    </section>
</div>
