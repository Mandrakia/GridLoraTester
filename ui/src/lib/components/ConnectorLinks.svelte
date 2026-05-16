<script lang="ts">
    // Row of chain-link icons (one per configured connector) shown next to a
    // dataset or dataset-group name. Gray = unlinked, accent-colored = linked.
    // Click any (linked or not) opens a person-picker modal — upsert flow.
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import type {
        ConnectorPerson,
        ConnectorStatus,
        ConnectorTypeInfo,
        LinkerKind
    } from '$lib/connectors/types';

    interface LinkRow {
        scope_kind: 'folder' | 'group';
        scope_key: string;
        connector_id: string;
        person_id: string;
        person_name: string | null;
        person_thumb_url: string | null;
    }

    interface Props {
        scope_kind: 'folder' | 'group';
        scope_key: string;
        /** Connectors usable for linking — credential-backed (signed-in)
         * + always-on (hard-drive). */
        connectors: ConnectorStatus[];
        /** Per-id metadata describing the linker shape. Falls back to
         * 'persons' when missing. */
        connector_types?: ConnectorTypeInfo[];
        /** Existing links for this scope_key (any subset of `connectors`). */
        links: LinkRow[];
    }

    let {
        scope_kind,
        scope_key,
        connectors,
        connector_types = [],
        links
    }: Props = $props();

    function linkerKindFor(id: string): LinkerKind {
        return connector_types.find((c) => c.id === id)?.linker_kind ?? 'persons';
    }

    /** Currently-open modal connector, or null. */
    let openFor = $state<ConnectorStatus | null>(null);
    let persons = $state<ConnectorPerson[] | null>(null);
    let loading = $state(false);
    let loadError = $state<string | null>(null);
    let filter = $state('');
    let saving = $state<string | null>(null); // person_id we're submitting
    /** Folder path typed by the user for folder-kind connectors. */
    let folderPath = $state('');

    let linkByConnector = $derived(
        new Map(links.map((l) => [l.connector_id, l]))
    );

    /** Current link for the open connector (for the "currently linked to X"
     * banner inside the modal). */
    let openLink = $derived(openFor ? linkByConnector.get(openFor.id) ?? null : null);

    let filteredPersons = $derived.by(() => {
        if (!persons) return [];
        const q = filter.trim().toLowerCase();
        if (!q) return persons;
        return persons.filter((p) => p.name.toLowerCase().includes(q));
    });

    async function openModal(c: ConnectorStatus) {
        openFor = c;
        persons = null;
        loadError = null;
        filter = '';
        const kind = linkerKindFor(c.id);

        // Folder-kind connectors: pre-fill with any currently-linked path
        // so editing is a one-character tweak instead of retyping.
        if (kind === 'folder') {
            const existing = linkByConnector.get(c.id);
            folderPath = existing?.person_id ?? '';
            return;
        }

        if (!c.signed_in) {
            loadError =
                'This connector is not signed in. Open Settings and run "Test" to see why.';
            return;
        }
        loading = true;
        try {
            const res = await fetch(`/connectors/${c.id}/persons`);
            const body = await res.json().catch(() => null);
            if (!res.ok) {
                loadError =
                    (body && typeof body.error === 'string' && body.error) ||
                    `HTTP ${res.status}`;
            } else {
                persons = (body?.persons ?? []) as ConnectorPerson[];
            }
        } catch (e) {
            loadError = (e as Error).message;
        } finally {
            loading = false;
        }
    }

    function closeModal(e?: Event) {
        // Stop the click from bubbling up to the dataset card's <a> wrapper,
        // which would otherwise navigate away as soon as we dismiss.
        e?.preventDefault();
        e?.stopPropagation();
        openFor = null;
        persons = null;
        loadError = null;
    }

    function onKey(e: KeyboardEvent) {
        // Escape isn't a click event — no navigation risk — but we still
        // route through closeModal for consistency.
        if (e.key === 'Escape' && openFor) closeModal();
    }
</script>

<svelte:window onkeydown={onKey} />

{#if connectors.length > 0}
    <span class="inline-flex items-center gap-1 align-middle">
        {#each connectors as c (c.id)}
            {@const link = linkByConnector.get(c.id)}
            <button
                type="button"
                class="inline-flex h-6 w-6 items-center justify-center rounded-md transition-colors {link
                    ? 'bg-accent/15 text-accent hover:bg-accent/25'
                    : 'text-fg-faint hover:bg-bg-2 hover:text-fg-muted'}"
                onclick={(e) => {
                    // Stop click from triggering the parent <a> that wraps
                    // dataset cards (would navigate to the detail page).
                    e.preventDefault();
                    e.stopPropagation();
                    openModal(c);
                }}
                title={link
                    ? `${c.label} → ${link.person_name ?? link.person_id}`
                    : c.signed_in
                      ? `Link to a ${c.label} person`
                      : `${c.label} (not signed in)`}
            >
                <!-- chain-link icon -->
                <svg
                    width="13"
                    height="13"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    aria-hidden="true"
                >
                    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                </svg>
            </button>
        {/each}
    </span>
{/if}

{#if openFor}
    {@const c = openFor}
    <div
        class="fixed inset-0 z-50 flex items-stretch justify-center bg-black/70 p-6"
        role="button"
        tabindex="-1"
        aria-label="Close"
        onclick={closeModal}
        onkeydown={(e) => {
            if (e.key === 'Escape') closeModal();
        }}
    >
        <div
            class="flex h-full w-full max-w-4xl flex-col gap-3 rounded-lg border border-border bg-bg-1 p-5"
            role="presentation"
            onclick={(e) => e.stopPropagation()}
        >
            <header class="flex items-baseline justify-between gap-3">
                <div>
                    <h3 class="text-base font-medium">
                        Link {c.label}
                        <span class="ml-1 font-mono text-xs text-fg-faint">{c.id}</span>
                    </h3>
                    {#if openLink}
                        <p class="mt-1 text-xs text-fg-muted">
                            currently linked to <span class="text-fg"
                                >{openLink.person_name ?? openLink.person_id}</span
                            >
                        </p>
                    {/if}
                </div>
                <div class="flex items-center gap-2">
                    {#if openLink}
                        <form
                            method="POST"
                            action="?/link-remove"
                            use:enhance={() => {
                                saving = '__unlink';
                                return async ({ update }) => {
                                    await update({ reset: false });
                                    saving = null;
                                    await invalidateAll();
                                    closeModal();
                                };
                            }}
                        >
                            <input type="hidden" name="scope_kind" value={scope_kind} />
                            <input type="hidden" name="scope_key" value={scope_key} />
                            <input type="hidden" name="connector_id" value={c.id} />
                            <button
                                type="submit"
                                class="btn-ghost px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                disabled={saving !== null}>Unlink</button
                            >
                        </form>
                    {/if}
                    <button type="button" class="btn-ghost text-xs" onclick={closeModal}
                        >Close</button
                    >
                </div>
            </header>

            {#if linkerKindFor(c.id) === 'folder'}
                <form
                    method="POST"
                    action="?/link-set"
                    use:enhance={() => {
                        saving = '__folder';
                        return async ({ result, update }) => {
                            await update({ reset: false });
                            saving = null;
                            if (result.type === 'success') {
                                await invalidateAll();
                                closeModal();
                            }
                        };
                    }}
                    class="space-y-3"
                >
                    <input type="hidden" name="scope_kind" value={scope_kind} />
                    <input type="hidden" name="scope_key" value={scope_key} />
                    <input type="hidden" name="connector_id" value={c.id} />
                    <!-- person_id is the folder path for hard-drive. Same
                         link table row shape; downstream is identity. -->
                    <input type="hidden" name="person_id" value={folderPath} />
                    <input
                        type="hidden"
                        name="person_name"
                        value={folderPath ? folderPath.split('/').pop() ?? folderPath : ''}
                    />
                    <label for="hd-folder-{c.id}" class="block text-sm font-medium"
                        >Folder path</label
                    >
                    <input
                        id="hd-folder-{c.id}"
                        type="text"
                        bind:value={folderPath}
                        placeholder="/home/you/photos/some-person"
                        class="input font-mono"
                        autocomplete="off"
                        spellcheck="false"
                        required
                    />
                    <p class="text-xs text-fg-faint">
                        Every image file in the folder (no recursion) becomes a candidate.
                        Subsequent runs only process newly-added pictures (idempotent by file
                        path).
                    </p>
                    <div class="flex justify-end gap-2">
                        <button
                            type="submit"
                            class="btn-primary"
                            disabled={saving !== null || !folderPath.trim()}
                        >
                            {saving === '__folder' ? 'Linking…' : 'Save folder link'}
                        </button>
                    </div>
                </form>
            {:else}
            <input
                type="text"
                placeholder="Filter persons…"
                bind:value={filter}
                class="input"
                autocomplete="off"
            />

            <div class="flex-1 overflow-y-auto rounded-md border border-border bg-bg-0/40 p-2">
                {#if loading}
                    <p class="p-6 text-center text-sm text-fg-muted">Loading persons…</p>
                {:else if loadError}
                    <div class="p-4 text-sm">
                        <p class="text-red-300">Could not load persons:</p>
                        <pre class="mt-1 overflow-auto text-xs text-red-300/80">{loadError}</pre>
                    </div>
                {:else if !persons || persons.length === 0}
                    <p class="p-6 text-center text-sm text-fg-muted">
                        {persons === null ? '—' : 'No persons returned by the connector.'}
                    </p>
                {:else if filteredPersons.length === 0}
                    <p class="p-6 text-center text-sm text-fg-muted">No match.</p>
                {:else}
                    <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
                        {#each filteredPersons as p (p.id)}
                            {@const isCurrent = openLink?.person_id === p.id}
                            <form
                                method="POST"
                                action="?/link-set"
                                use:enhance={() => {
                                    saving = p.id;
                                    return async ({ update }) => {
                                        await update({ reset: false });
                                        saving = null;
                                        await invalidateAll();
                                        closeModal();
                                    };
                                }}
                            >
                                <input type="hidden" name="scope_kind" value={scope_kind} />
                                <input type="hidden" name="scope_key" value={scope_key} />
                                <input type="hidden" name="connector_id" value={c.id} />
                                <input type="hidden" name="person_id" value={p.id} />
                                <input type="hidden" name="person_name" value={p.name} />
                                <input
                                    type="hidden"
                                    name="person_thumb_url"
                                    value={p.thumbnail_url}
                                />
                                <button
                                    type="submit"
                                    class="group flex w-full flex-col items-center gap-1 rounded-md border p-2 transition-colors {isCurrent
                                        ? 'border-accent bg-accent/10'
                                        : 'border-border bg-bg-1 hover:border-border-strong hover:bg-bg-2'}"
                                    disabled={saving !== null}
                                    title={p.name}
                                >
                                    <img
                                        src={p.thumbnail_url}
                                        alt={p.name}
                                        loading="lazy"
                                        decoding="async"
                                        class="h-16 w-16 rounded-full object-cover"
                                    />
                                    <span class="w-full truncate text-center text-xs">
                                        {p.name}
                                    </span>
                                    {#if isCurrent}
                                        <span class="text-[10px] text-accent">linked</span>
                                    {/if}
                                </button>
                            </form>
                        {/each}
                    </div>
                {/if}
            </div>

            {#if persons && persons.length > 0}
                <p class="text-xs text-fg-faint">
                    {filteredPersons.length}/{persons.length} person{persons.length === 1 ? '' : 's'}
                </p>
            {/if}
            {/if}
        </div>
    </div>
{/if}
