<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import MainPanel from '$lib/components/MainPanel.svelte';
    import type { ActionData, PageData } from './$types';

    let { data, form }: { data: PageData; form: ActionData } = $props();

    let editingId = $state<number | null>(null);
    let editorName = $state('');
    let editorBlob = $state('');
    let saving = $state(false);
    let deletingId = $state<number | null>(null);

    function openNew() {
        editingId = -1;
        editorName = '';
        editorBlob = '';
    }

    function openEdit(p: PageData['prompt_sets'][number]) {
        editingId = p.id;
        editorName = p.name;
        editorBlob = p.prompts.join('\n');
    }

    function closeEditor() {
        editingId = null;
    }

    function liveCount(blob: string): number {
        return blob
            .split(/\r?\n/)
            .map((s) => s.trim())
            .filter((s) => s && !s.startsWith('#')).length;
    }
</script>

<svelte:head>
    <title>Prompts — GridLoraTester</title>
</svelte:head>

<MainPanel size="narrow">
    <div class="space-y-6">
    <header class="flex items-baseline justify-between gap-4">
        <div>
            <h1 class="text-2xl font-semibold tracking-tight">Prompts</h1>
            <p class="mt-1 text-sm text-fg-muted">
                Named lists of prompts, reusable across tests.
            </p>
        </div>
        <button
            type="button"
            class="btn-primary"
            disabled={editingId !== null}
            onclick={openNew}
        >
            + New prompt set
        </button>
    </header>

    <!-- ============================================================ -->
    <!-- Table                                                        -->
    <!-- ============================================================ -->
    <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
        <table class="w-full text-sm">
            <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                <tr>
                    <th class="px-4 py-2.5 text-left font-medium">Name</th>
                    <th class="px-4 py-2.5 text-right font-medium">Prompts</th>
                    <th class="w-32 px-4 py-2.5 text-right font-medium">Actions</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-border">
                {#if data.prompt_sets.length === 0}
                    <tr>
                        <td colspan="3" class="px-4 py-6 text-center text-sm text-fg-faint">
                            No prompt sets yet. Click "+ New prompt set" to create one.
                        </td>
                    </tr>
                {:else}
                    {#each data.prompt_sets as p (p.id)}
                        <tr class="transition-colors hover:bg-bg-2/40">
                            <td class="px-4 py-2.5 align-top">
                                <div class="font-medium">{p.name}</div>
                                <div class="text-xs text-fg-faint" title={p.prompts[0] ?? ''}>
                                    {p.prompts[0]
                                        ? p.prompts[0].length > 80
                                            ? p.prompts[0].slice(0, 80) + '…'
                                            : p.prompts[0]
                                        : '(empty)'}
                                </div>
                            </td>
                            <td class="px-4 py-2.5 text-right align-top tabular-nums"
                                >{p.prompt_count}</td
                            >
                            <td class="px-4 py-2.5 align-top">
                                <div class="flex justify-end gap-1">
                                    <button
                                        type="button"
                                        class="btn-ghost px-2 py-1 text-xs"
                                        onclick={() => openEdit(p)}
                                        disabled={editingId !== null}
                                    >
                                        Edit
                                    </button>
                                    <form
                                        method="POST"
                                        action="?/delete"
                                        use:enhance={() => {
                                            deletingId = p.id;
                                            return ({ update }) => {
                                                update({ reset: false }).finally(
                                                    () => (deletingId = null)
                                                );
                                            };
                                        }}
                                    >
                                        <input type="hidden" name="id" value={p.id} />
                                        <button
                                            type="submit"
                                            class="btn-ghost px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                            onclick={(e) => {
                                                if (!confirm(`Delete prompt set "${p.name}"?`))
                                                    e.preventDefault();
                                            }}
                                            disabled={deletingId === p.id}
                                        >
                                            {deletingId === p.id ? '…' : 'Delete'}
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

    <!-- ============================================================ -->
    <!-- Editor                                                       -->
    <!-- ============================================================ -->
    {#if editingId !== null}
        <section class="card">
            <h3 class="mb-4 text-sm font-medium">
                {editingId === -1 ? 'New prompt set' : 'Edit prompt set'}
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
                    <label for="p-name" class="text-sm font-medium">Name</label>
                    <input
                        id="p-name"
                        name="name"
                        type="text"
                        class="input"
                        bind:value={editorName}
                        placeholder="e.g. portraits-default"
                        autocomplete="off"
                        required
                    />
                </div>

                <div class="space-y-1.5">
                    <div class="flex items-baseline justify-between">
                        <label for="p-prompts" class="text-sm font-medium">Prompts</label>
                        <span class="text-xs text-fg-faint">
                            {liveCount(editorBlob)} prompt{liveCount(editorBlob) === 1 ? '' : 's'}
                        </span>
                    </div>
                    <textarea
                        id="p-prompts"
                        name="prompts"
                        bind:value={editorBlob}
                        rows="16"
                        spellcheck="false"
                        placeholder={`One prompt per line. Lines starting with # are ignored.\n\n[trigger] sitting on a couch. Candid shot, soft lighting\n[trigger] as a jedi with a green lightsaber, sci-fi cinematic still`}
                        class="input min-h-[16rem] resize-y font-mono text-xs leading-relaxed"
                    ></textarea>
                    <p class="text-xs text-fg-faint">
                        One prompt per line. Blank lines and <code class="font-mono">#</code>-comments
                        are skipped on save.
                    </p>
                </div>

                {#if form?.error}
                    <p class="text-sm text-red-400">{form.error}</p>
                {/if}

                <div class="flex items-center gap-3 pt-1">
                    <button type="submit" class="btn-primary" disabled={saving}>
                        {saving ? 'Saving…' : editingId === -1 ? 'Create' : 'Save changes'}
                    </button>
                    <button type="button" class="btn-ghost" onclick={closeEditor} disabled={saving}>
                        Cancel
                    </button>
                </div>
            </form>
        </section>
    {/if}
    </div>
</MainPanel>
