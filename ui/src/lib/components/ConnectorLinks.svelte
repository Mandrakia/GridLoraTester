<script lang="ts">
    // "Connectors:" badge row next to a dataset / dataset-group name.
    // Each LINKED connector renders as a clickable pill (icon + label of the
    // linked person / folder). A trailing "+" button drops down the list of
    // configured-but-unlinked connectors. Clicking either path opens the
    // person-picker / folder modal that owns the upsert flow.
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import FolderPicker from '$lib/components/FolderPicker.svelte';
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
        last_sync_at?: string | null;
        last_sync_count?: number | null;
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

    let linkByConnector = $derived(
        new Map(links.map((l) => [l.connector_id, l]))
    );
    /** Connectors that already have a link on this scope → render as pills. */
    let linkedConnectors = $derived(
        connectors.filter((c) => linkByConnector.has(c.id))
    );
    /** Connectors available to ADD (no existing link) → shown in the + dropdown. */
    let availableConnectors = $derived(
        connectors.filter((c) => !linkByConnector.has(c.id))
    );

    /** Currently-open modal connector, or null. */
    let openFor = $state<ConnectorStatus | null>(null);
    let persons = $state<ConnectorPerson[] | null>(null);
    let loading = $state(false);
    let loadError = $state<string | null>(null);
    let filter = $state('');
    let saving = $state<string | null>(null);
    let folderPath = $state('');
    /** Add-connector popover state. */
    let dropdownOpen = $state(false);
    /** Wrapper for the + button + its dropdown panel. Used by the global
     * click listener to detect "click outside" → dismiss-and-swallow. */
    let dropdownWrapper = $state<HTMLDivElement | null>(null);

    /** Google Photos picker session lifecycle inside the modal. The modal
     * opens directly in `creating` (no intermediate button to click) and
     * advances on its own once Google's response lands. */
    type PickerPhase = 'creating' | 'awaiting_pick' | 'finalizing' | 'done' | 'error';
    let pickerPhase = $state<PickerPhase>('creating');
    let pickerSessionId = $state<string | null>(null);
    let pickerErrorMsg = $state<string | null>(null);
    let pickerImportedCount = $state<number | null>(null);
    let pickerJobId = $state<number | null>(null);
    /** Live progress from the finalize NDJSON stream. */
    let pickerProgressDone = $state(0);
    let pickerProgressTotal = $state(0);
    /** When non-null, the in-flight poll timer for the active picker
     * session. Cleared on completion / modal close / unmount. */
    let pickerPollTimer = $state<ReturnType<typeof setInterval> | null>(null);

    function clearPicker() {
        if (pickerPollTimer) clearInterval(pickerPollTimer);
        pickerPollTimer = null;
        pickerPhase = 'creating';
        pickerSessionId = null;
        pickerErrorMsg = null;
        pickerImportedCount = null;
        pickerJobId = null;
        pickerProgressDone = 0;
        pickerProgressTotal = 0;
    }

    /** Parse a Google Duration like "5s" / "5.123s" into ms, fallback 3s. */
    function pollMsFromDuration(d: string | undefined): number {
        if (!d) return 3000;
        const m = /^([\d.]+)s$/.exec(d);
        if (!m) return 3000;
        return Math.max(1000, Math.round(parseFloat(m[1]) * 1000));
    }

    async function startPickerSession() {
        pickerErrorMsg = null;
        pickerPhase = 'creating';
        try {
            const r = await fetch('/api/connectors/google-photos/sessions', {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: '{}'
            });
            const body = (await r.json().catch(() => null)) as {
                session?: {
                    id: string;
                    pickerUri: string;
                    pollingConfig?: { pollInterval?: string };
                };
                error?: string;
            } | null;
            if (!r.ok || !body?.session) {
                pickerErrorMsg = body?.error ?? `HTTP ${r.status}`;
                pickerPhase = 'error';
                return;
            }
            pickerSessionId = body.session.id;
            pickerPhase = 'awaiting_pick';
            // Open Google's UI in a new tab. We don't track the window —
            // the user may close it after picking; only the server-side
            // session state matters.
            window.open(body.session.pickerUri, '_blank', 'noopener');

            const intervalMs = pollMsFromDuration(body.session.pollingConfig?.pollInterval);
            pickerPollTimer = setInterval(() => pollPickerSession(), intervalMs);
        } catch (e) {
            pickerErrorMsg = (e as Error).message;
            pickerPhase = 'error';
        }
    }

    async function pollPickerSession() {
        if (!pickerSessionId) return;
        try {
            const r = await fetch(
                `/api/connectors/google-photos/sessions/${encodeURIComponent(pickerSessionId)}`
            );
            if (!r.ok) return; // transient; keep polling
            const body = (await r.json()) as {
                session?: { mediaItemsSet?: boolean };
            };
            if (body.session?.mediaItemsSet) {
                if (pickerPollTimer) clearInterval(pickerPollTimer);
                pickerPollTimer = null;
                await finalizePickerSession();
            }
        } catch {
            // transient — leave timer running
        }
    }

    /** NDJSON event shapes from /finalize. */
    type FinalizeEvent =
        | { phase: 'start'; total: number }
        | { phase: 'progress'; done: number; total: number }
        | { phase: 'done'; count: number; total: number; job_id: number | null }
        | { phase: 'error'; message: string };

    function handleFinalizeEvent(evt: FinalizeEvent) {
        if (evt.phase === 'start') {
            pickerProgressTotal = evt.total;
            pickerProgressDone = 0;
        } else if (evt.phase === 'progress') {
            pickerProgressDone = evt.done;
            pickerProgressTotal = evt.total;
        } else if (evt.phase === 'done') {
            pickerImportedCount = evt.count;
            pickerJobId = evt.job_id;
            pickerProgressDone = evt.count;
            pickerProgressTotal = evt.total;
            pickerPhase = 'done';
            void invalidateAll();
        } else if (evt.phase === 'error') {
            pickerErrorMsg = evt.message;
            pickerPhase = 'error';
        }
    }

    async function finalizePickerSession() {
        if (!pickerSessionId) return;
        pickerPhase = 'finalizing';
        pickerProgressDone = 0;
        pickerProgressTotal = 0;
        try {
            const r = await fetch(
                `/api/connectors/google-photos/sessions/${encodeURIComponent(pickerSessionId)}/finalize`,
                {
                    method: 'POST',
                    headers: { 'content-type': 'application/json' },
                    body: JSON.stringify({ scope_kind, scope_key })
                }
            );
            if (!r.ok || !r.body) {
                const errText = await r.text().catch(() => '');
                let msg: string;
                try {
                    msg = (JSON.parse(errText) as { error?: string }).error ?? errText;
                } catch {
                    msg = errText || `HTTP ${r.status}`;
                }
                pickerErrorMsg = msg;
                pickerPhase = 'error';
                return;
            }
            const reader = r.body.getReader();
            const dec = new TextDecoder();
            let buf = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += dec.decode(value, { stream: true });
                let nl: number;
                while ((nl = buf.indexOf('\n')) >= 0) {
                    const line = buf.slice(0, nl).trim();
                    buf = buf.slice(nl + 1);
                    if (!line) continue;
                    try {
                        handleFinalizeEvent(JSON.parse(line) as FinalizeEvent);
                    } catch {
                        // ignore malformed line — the stream may still emit
                        // the next valid event
                    }
                }
            }
        } catch (e) {
            pickerErrorMsg = (e as Error).message;
            pickerPhase = 'error';
        }
    }
    /** Folder picker state (hard-drive connector). Allowed roots are
     * fetched from /api/fs/roots on first open and cached. */
    let pickerOpen = $state(false);
    let pickerRoots = $state<{ label: string; path: string }[] | null>(null);
    let pickerError = $state<string | null>(null);

    async function openFolderPicker() {
        pickerError = null;
        if (!pickerRoots) {
            try {
                const r = await fetch('/api/fs/roots');
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const body = (await r.json()) as {
                    roots: { label: string; path: string }[];
                };
                if (!body.roots?.length) throw new Error('no roots configured');
                pickerRoots = body.roots;
            } catch (e) {
                pickerError = (e as Error).message;
                return;
            }
        }
        pickerOpen = true;
    }

    function onFolderPicked(p: string) {
        folderPath = p;
        pickerOpen = false;
    }

    /** While the dropdown is open, every click anywhere else dismisses it
     * AND is swallowed so it doesn't trigger the underlying UI (e.g.
     * clicking on a dataset card while the dropdown is open just closes
     * the dropdown — no navigation). Capture phase so we run BEFORE
     * Svelte / SvelteKit's own click handlers. */
    $effect(() => {
        if (!dropdownOpen) return;
        function onGlobalClick(e: MouseEvent) {
            const target = e.target as Node | null;
            if (dropdownWrapper && target && dropdownWrapper.contains(target)) {
                return;
            }
            e.preventDefault();
            e.stopImmediatePropagation();
            dropdownOpen = false;
        }
        document.addEventListener('click', onGlobalClick, true);
        return () => document.removeEventListener('click', onGlobalClick, true);
    });

    let openLink = $derived(openFor ? linkByConnector.get(openFor.id) ?? null : null);

    let filteredPersons = $derived.by(() => {
        if (!persons) return [];
        const q = filter.trim().toLowerCase();
        if (!q) return persons;
        return persons.filter((p) => p.name.toLowerCase().includes(q));
    });

    async function openModal(c: ConnectorStatus, e?: Event) {
        e?.preventDefault();
        e?.stopPropagation();
        dropdownOpen = false;
        openFor = c;
        persons = null;
        loadError = null;
        filter = '';
        clearPicker();
        const kind = linkerKindFor(c.id);

        if (kind === 'folder') {
            const existing = linkByConnector.get(c.id);
            folderPath = existing?.person_id ?? '';
            return;
        }

        if (kind === 'picker') {
            // Open Google's picker tab immediately on click. The modal
            // shows a polling-waiting state until the user finishes
            // selecting. If they're not signed in, the template surfaces
            // a "Sign in in Settings" hint instead.
            if (c.signed_in) {
                await startPickerSession();
            }
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
        e?.preventDefault();
        e?.stopPropagation();
        openFor = null;
        persons = null;
        loadError = null;
        clearPicker();
    }

    function onKey(e: KeyboardEvent) {
        if (e.key === 'Escape') {
            if (openFor) closeModal();
            else if (dropdownOpen) dropdownOpen = false;
        }
    }

    function labelFor(c: ConnectorStatus): string {
        const link = linkByConnector.get(c.id);
        if (!link) return c.label;
        const kind = linkerKindFor(c.id);
        if (kind === 'folder') {
            const path = link.person_id;
            return path.split('/').filter(Boolean).pop() ?? path;
        }
        if (kind === 'picker') {
            const n = link.last_sync_count ?? 0;
            return `${n} photo${n === 1 ? '' : 's'}`;
        }
        return link.person_name ?? link.person_id;
    }

    function titleFor(c: ConnectorStatus): string {
        const link = linkByConnector.get(c.id);
        const kind = linkerKindFor(c.id);
        if (link) {
            if (kind === 'folder') return `${c.label} → ${link.person_id}`;
            if (kind === 'picker') {
                const n = link.last_sync_count ?? 0;
                const when = link.last_sync_at ? ` · synced ${formatSyncDate(link.last_sync_at)}` : '';
                return `${c.label} · ${n} photo${n === 1 ? '' : 's'}${when}`;
            }
            return `${c.label} → ${link.person_name ?? link.person_id}`;
        }
        if (!c.signed_in) return `${c.label} (not signed in)`;
        if (kind === 'picker') return `Pick photos from ${c.label}`;
        if (kind === 'folder') return `Link a folder via ${c.label}`;
        return `Link to a ${c.label} person`;
    }

    /** SQLite stores 'YYYY-MM-DD HH:MM:SS' in UTC without a Z. Parse
     * explicitly so toLocaleString reflects the user's tz. */
    function formatSyncDate(iso: string): string {
        try {
            const normalized = iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z';
            return new Date(normalized).toLocaleString();
        } catch {
            return iso;
        }
    }
</script>

<svelte:window onkeydown={onKey} />

{#if connectors.length > 0}
    <span class="inline-flex flex-wrap items-center gap-1.5 align-middle">
        <!-- Section label with chain icon -->
        <span class="inline-flex items-center gap-1 text-[11px] uppercase tracking-wide text-fg-faint">
            {@render chainIcon()}
            connectors
        </span>

        <!-- Linked connector pills -->
        {#each linkedConnectors as c (c.id)}
            <button
                type="button"
                class="inline-flex items-center gap-1 rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-xs text-accent transition-colors hover:bg-accent/20"
                onclick={(e) => openModal(c, e)}
                title={titleFor(c)}
            >
                {@render connectorIcon(c.id)}
                <span class="max-w-[14ch] truncate">{labelFor(c)}</span>
            </button>
        {/each}

        <!-- + dropdown for adding another connector -->
        {#if availableConnectors.length > 0}
            <div class="relative inline-block" bind:this={dropdownWrapper}>
                <button
                    type="button"
                    class="inline-flex h-6 w-6 items-center justify-center rounded-full border border-border bg-bg-2 text-fg-muted transition-colors hover:border-accent/60 hover:text-fg"
                    onclick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        dropdownOpen = !dropdownOpen;
                    }}
                    title="Add a connector link"
                    aria-label="Add a connector link"
                    aria-expanded={dropdownOpen}
                >
                    +
                </button>
                {#if dropdownOpen}
                    <div
                        class="absolute left-0 top-full z-30 mt-1 w-56 rounded-md border border-border bg-bg-1 p-1 shadow-lg"
                        role="menu"
                    >
                        {#each availableConnectors as c (c.id)}
                            {@const folderKind = linkerKindFor(c.id) === 'folder'}
                            {@const disabled = !folderKind && !c.signed_in}
                            <button
                                type="button"
                                class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs transition-colors hover:bg-bg-2 disabled:cursor-not-allowed disabled:opacity-50"
                                {disabled}
                                onclick={(e) => openModal(c, e)}
                            >
                                {@render connectorIcon(c.id)}
                                <span class="flex-1">{c.label}</span>
                                {#if disabled}
                                    <span class="text-[10px] text-amber-300">sign in</span>
                                {/if}
                            </button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/if}
    </span>
{/if}

<!-- ============== Icons ============== -->

{#snippet chainIcon()}
    <svg
        width="11"
        height="11"
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
{/snippet}

{#snippet connectorIcon(id: string)}
    {#if id === 'immich'}
        <!-- Immich brand mark (5-petal flower). Brand colors are hardcoded
             inline so they survive whatever color the surrounding badge
             uses for its text. -->
        <svg
            width="12"
            height="12"
            viewBox="0 0 792 792"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
        >
            <path
                fill="#FA2921"
                d="M375.48,267.63c38.64,34.21,69.78,70.87,89.82,105.42c34.42-61.56,57.42-134.71,57.71-181.3 c0-0.33,0-0.63,0-0.91c0-68.94-68.77-95.77-128.01-95.77s-128.01,26.83-128.01,95.77c0,0.94,0,2.2,0,3.72 C300.01,209.24,339.15,235.47,375.48,267.63z"
            />
            <path
                fill="#ED79B5"
                d="M164.7,455.63c24.15-26.87,61.2-55.99,103.01-80.61c44.48-26.18,88.97-44.47,128.02-52.84 c-47.91-51.76-110.37-96.24-154.6-110.91c-0.31-0.1-0.6-0.19-0.86-0.28c-65.57-21.3-112.34,35.81-130.64,92.15 c-18.3,56.34-14.04,130.04,51.53,151.34C162.05,454.77,163.25,455.16,164.7,455.63z"
            />
            <path
                fill="#FFB400"
                d="M681.07,302.19c-18.3-56.34-65.07-113.45-130.64-92.15c-0.9,0.29-2.1,0.68-3.54,1.15 c-3.75,35.93-16.6,81.27-35.96,125.76c-20.59,47.32-45.84,88.27-72.51,118c69.18,13.72,145.86,12.98,190.26-1.14 c0.31-0.1,0.6-0.2,0.86-0.28C695.11,432.22,699.37,358.52,681.07,302.19z"
            />
            <path
                fill="#1E83F7"
                d="M336.54,510.71c-11.15-50.39-14.8-98.36-10.7-138.08c-64.03,29.57-125.63,75.23-153.26,112.76 c-0.19,0.26-0.37,0.51-0.53,0.73c-40.52,55.78-0.66,117.91,47.27,152.72c47.92,34.82,119.33,53.54,159.86-2.24 c0.56-0.76,1.3-1.78,2.19-3.01C363.28,602.32,347.02,558.08,336.54,510.71z"
            />
            <path
                fill="#18C249"
                d="M617.57,482.52c-35.33,7.54-82.42,9.33-130.72,4.66c-51.37-4.96-98.11-16.32-134.63-32.5 c8.33,70.03,32.73,142.73,59.88,180.6c0.19,0.26,0.37,0.51,0.53,0.73c40.52,55.78,111.93,37.06,159.86,2.24 c47.92-34.82,87.79-96.95,47.27-152.72C619.2,484.77,618.46,483.75,617.57,482.52z"
            />
        </svg>
    {:else if id === 'hard-drive'}
        <!-- folder glyph for the local-folder connector -->
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
        </svg>
    {:else}
        <!-- generic plug glyph for unknown future connectors -->
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M9 7V3M15 7V3M7 7h10v5a5 5 0 0 1-10 0V7zM12 17v4" />
        </svg>
    {/if}
{/snippet}

<!-- ============== Modal (unchanged from before) ============== -->

{#if openFor}
    {@const c = openFor}
    {@const _kind = linkerKindFor(c.id)}
    <div
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
        role="button"
        tabindex="-1"
        aria-label="Close"
        onclick={closeModal}
        onkeydown={(e) => {
            if (e.key === 'Escape') closeModal();
        }}
    >
        <div
            class="flex flex-col gap-3 rounded-lg border border-border bg-bg-1 p-5 {_kind ===
            'picker'
                ? 'w-full max-w-md'
                : 'h-full w-full max-w-4xl'}"
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
                        {@const kind = linkerKindFor(c.id)}
                        <p class="mt-1 text-xs text-fg-muted">
                            {#if kind === 'picker'}
                                {openLink.last_sync_count ?? 0} photo{openLink.last_sync_count ===
                                1
                                    ? ''
                                    : 's'} cached{#if openLink.last_sync_at}, last synced
                                    {formatSyncDate(openLink.last_sync_at)}{/if}
                            {:else if kind === 'folder'}
                                currently linked to
                                <span class="font-mono text-fg">{openLink.person_id}</span>
                            {:else}
                                currently linked to
                                <span class="text-fg"
                                    >{openLink.person_name ?? openLink.person_id}</span
                                >
                            {/if}
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
                    <input type="hidden" name="person_id" value={folderPath} />
                    <input
                        type="hidden"
                        name="person_name"
                        value={folderPath ? folderPath.split('/').pop() ?? folderPath : ''}
                    />
                    <label for="hd-folder-{c.id}" class="block text-sm font-medium"
                        >Folder path</label
                    >
                    <div class="flex gap-2">
                        <input
                            id="hd-folder-{c.id}"
                            type="text"
                            bind:value={folderPath}
                            placeholder="/home/you/photos/some-person"
                            class="input flex-1 font-mono"
                            autocomplete="off"
                            spellcheck="false"
                            required
                        />
                        <button
                            type="button"
                            class="btn-ghost shrink-0 px-3 text-xs"
                            onclick={openFolderPicker}
                            title="Browse for a folder"
                        >
                            Browse…
                        </button>
                    </div>
                    {#if pickerError}
                        <p class="text-xs text-amber-300">
                            Folder picker unavailable: <code class="font-mono">{pickerError}</code>
                        </p>
                    {/if}
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
            {:else if linkerKindFor(c.id) === 'picker'}
                <!-- Google-Photos-style picker flow: the user picks photos
                     in Google's UI (separate tab), we poll the session
                     until they're done, then finalize (download into the
                     local cache + queue face-detect). -->
                {#if !c.signed_in}
                    <div class="rounded-md border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
                        <p class="text-amber-200">
                            Not signed in to {c.label}. Open
                            <a href="/settings" class="underline">Settings</a> and sign in
                            first.
                        </p>
                    </div>
                {:else}
                    <div class="flex flex-col gap-3">
                        {#if openLink}
                            <p class="text-xs text-fg-faint">
                                Re-picking adds new photos to the existing set — items already
                                imported are skipped automatically.
                            </p>
                        {/if}

                        {#if pickerPhase === 'creating'}
                            <div class="flex items-center gap-3 text-sm text-fg">
                                <svg
                                    class="h-5 w-5 animate-spin text-accent"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="2.5"
                                    aria-hidden="true"
                                >
                                    <path d="M12 3a9 9 0 1 0 9 9" stroke-linecap="round" />
                                </svg>
                                Opening Google Photos…
                            </div>
                        {:else if pickerPhase === 'awaiting_pick'}
                            <div class="flex items-center gap-3">
                                <svg
                                    class="h-5 w-5 animate-spin text-accent"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="2.5"
                                    aria-hidden="true"
                                >
                                    <path d="M12 3a9 9 0 1 0 9 9" stroke-linecap="round" />
                                </svg>
                                <div class="text-sm">
                                    <p class="text-fg">
                                        Waiting for your selection in Google Photos…
                                    </p>
                                    <p class="mt-0.5 text-xs text-fg-muted">
                                        Pick the photos you want to add, then come back — this
                                        dialog continues automatically.
                                    </p>
                                </div>
                            </div>
                            <p class="text-[11px] text-fg-faint">
                                Bytes must be downloaded within ~60 minutes of finishing your
                                selection (Google's limit). Very large picks may fail at the
                                tail — re-pick the leftovers to finish.
                            </p>
                            <button
                                type="button"
                                class="btn-ghost self-start text-xs"
                                onclick={closeModal}
                            >
                                Cancel
                            </button>
                        {:else if pickerPhase === 'finalizing'}
                            {@const pct =
                                pickerProgressTotal > 0
                                    ? Math.round((pickerProgressDone / pickerProgressTotal) * 100)
                                    : 0}
                            <div class="space-y-2">
                                <div class="flex items-baseline justify-between gap-3 text-sm">
                                    <span class="text-fg">Downloading picked photos…</span>
                                    <span class="tabular-nums text-fg-muted">
                                        {pickerProgressDone}
                                        {#if pickerProgressTotal > 0}/ {pickerProgressTotal}{/if}
                                    </span>
                                </div>
                                <div class="h-1.5 w-full overflow-hidden rounded bg-bg-3">
                                    <div
                                        class="h-full bg-accent transition-all duration-200"
                                        style="width: {pct}%"
                                    ></div>
                                </div>
                            </div>
                        {:else if pickerPhase === 'done'}
                            <div class="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-200">
                                Imported {pickerImportedCount} photo{pickerImportedCount === 1
                                    ? ''
                                    : 's'}.
                                {#if pickerJobId != null}
                                    Face detection is running in the background —
                                    <a
                                        href={`/jobs?focus=${pickerJobId}`}
                                        class="underline">job #{pickerJobId}</a
                                    >.
                                {/if}
                            </div>
                            <button
                                type="button"
                                class="btn-primary self-start"
                                onclick={closeModal}
                            >
                                Done
                            </button>
                        {:else if pickerPhase === 'error'}
                            <div class="rounded-md border border-red-500/30 bg-red-500/5 p-3 text-sm">
                                <p class="text-red-300">Picker flow failed:</p>
                                <pre class="mt-1 overflow-auto text-xs text-red-300/80">{pickerErrorMsg}</pre>
                            </div>
                            <div class="flex gap-2">
                                <button
                                    type="button"
                                    class="btn-primary text-xs"
                                    onclick={startPickerSession}
                                >
                                    Try again
                                </button>
                                <button
                                    type="button"
                                    class="btn-ghost text-xs"
                                    onclick={closeModal}
                                >
                                    Close
                                </button>
                            </div>
                        {/if}
                    </div>
                {/if}
            {:else}
                <input
                    type="text"
                    placeholder="Filter persons…"
                    bind:value={filter}
                    class="input"
                    autocomplete="off"
                />

                <div
                    class="flex-1 overflow-y-auto rounded-md border border-border bg-bg-0/40 p-2"
                >
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

{#if pickerOpen && pickerRoots}
    <FolderPicker
        roots={pickerRoots}
        initialPath={folderPath || undefined}
        onSelect={onFolderPicked}
        onCancel={() => (pickerOpen = false)}
    />
{/if}
