<script lang="ts">
    // Modal folder picker. Server-side cage: every list request names one
    // of the server's allowed roots (see /api/fs/roots), and the server
    // rejects any path that would escape it. The caller passes the full
    // root list — typically fetched via /api/fs/roots — and the picker
    // shows a tab strip when there's more than one (Home, /mnt, /media,
    // /Volumes …). Hidden folders (.dotfiles) render at half opacity to
    // match OS file-manager convention.
    interface Root {
        label: string;
        path: string;
    }
    interface Entry {
        name: string;
        isHidden: boolean;
    }
    interface ListResponse {
        root: string;
        path: string;
        parent: string | null;
        entries: Entry[];
    }

    interface Props {
        /** Server-advertised browsable roots. Must be non-empty. */
        roots: Root[];
        /** Starting folder. Must live inside one of the roots; we pick the
         *  containing root automatically. Defaults to roots[0].path. */
        initialPath?: string;
        onSelect: (path: string) => void;
        onCancel: () => void;
    }

    let { roots, initialPath, onSelect, onCancel }: Props = $props();

    /** Currently active root (the one whose subtree we're navigating). */
    let activeRoot = $state('');
    let currentPath = $state('');
    let parent = $state<string | null>(null);
    let entries = $state<Entry[]>([]);
    let loading = $state(false);
    let loadError = $state<string | null>(null);

    function rootContaining(p: string): Root | null {
        for (const r of roots) {
            if (p === r.path || p.startsWith(r.path + '/')) return r;
        }
        return null;
    }

    async function load(rootPath: string, target: string) {
        loading = true;
        loadError = null;
        try {
            const r = await fetch(
                `/api/fs/list?root=${encodeURIComponent(rootPath)}&path=${encodeURIComponent(target)}`
            );
            const body = (await r.json().catch(() => null)) as
                | (ListResponse & { message?: string })
                | null;
            if (!r.ok) {
                loadError = body?.message ?? `HTTP ${r.status}`;
                return;
            }
            if (!body) {
                loadError = 'empty response';
                return;
            }
            activeRoot = body.root;
            currentPath = body.path;
            parent = body.parent;
            entries = body.entries;
        } catch (e) {
            loadError = (e as Error).message;
        } finally {
            loading = false;
        }
    }

    // Initial load. Pick the root that contains initialPath, or the first
    // root otherwise. Re-fires if the caller swaps the roots list or
    // initialPath.
    $effect.pre(() => {
        if (roots.length === 0) {
            loadError = 'No roots available.';
            return;
        }
        const target = initialPath ?? roots[0].path;
        const containing = rootContaining(target) ?? roots[0];
        load(containing.path, target);
    });

    /** Breadcrumb segments from activeRoot down to currentPath. */
    let crumbs = $derived.by(() => {
        const list: { label: string; path: string }[] = [];
        const activeLabel = roots.find((r) => r.path === activeRoot)?.label ?? activeRoot;
        list.push({ label: activeLabel, path: activeRoot });
        if (currentPath === activeRoot) return list;
        if (!currentPath.startsWith(activeRoot)) return list;
        const rel = currentPath.slice(activeRoot.length).replace(/^\/+/, '');
        let acc = activeRoot.endsWith('/') ? activeRoot.slice(0, -1) : activeRoot;
        for (const seg of rel.split('/')) {
            if (!seg) continue;
            acc = `${acc}/${seg}`;
            list.push({ label: seg, path: acc });
        }
        return list;
    });

    function enter(name: string) {
        const sep = currentPath.endsWith('/') ? '' : '/';
        load(activeRoot, `${currentPath}${sep}${name}`);
    }

    function up() {
        if (parent) load(activeRoot, parent);
    }

    function switchRoot(r: Root) {
        if (r.path === activeRoot) return;
        load(r.path, r.path);
    }

    function navigateTo(p: string) {
        load(activeRoot, p);
    }

    function pick() {
        onSelect(currentPath);
    }

    function onKey(e: KeyboardEvent) {
        if (e.key === 'Escape') onCancel();
        else if (e.key === 'Enter') pick();
    }
</script>

<svelte:window onkeydown={onKey} />

<div
    class="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-6"
    role="button"
    tabindex="-1"
    aria-label="Close folder picker"
    onclick={onCancel}
    onkeydown={(e) => {
        if (e.key === 'Escape') onCancel();
    }}
>
    <div
        class="flex h-[70vh] w-full max-w-2xl flex-col gap-3 rounded-lg border border-border bg-bg-1 p-5"
        role="presentation"
        onclick={(e) => e.stopPropagation()}
    >
        <header class="flex items-baseline justify-between gap-3">
            <h3 class="text-base font-medium">Select folder</h3>
            <button type="button" class="btn-ghost text-xs" onclick={onCancel}>Close</button>
        </header>

        <!-- Roots tab strip (only when there's more than one). -->
        {#if roots.length > 1}
            <div
                class="flex flex-wrap gap-1 border-b border-border"
                role="tablist"
                aria-label="Locations"
            >
                {#each roots as r (r.path)}
                    <button
                        type="button"
                        role="tab"
                        aria-selected={activeRoot === r.path}
                        class="-mb-px border-b-2 px-3 py-1.5 text-xs font-medium transition-colors {activeRoot ===
                        r.path
                            ? 'border-accent text-fg'
                            : 'border-transparent text-fg-muted hover:text-fg'}"
                        onclick={() => switchRoot(r)}
                    >
                        {r.label}
                    </button>
                {/each}
            </div>
        {/if}

        <!-- Breadcrumbs anchored at the active root. -->
        <nav class="flex flex-wrap items-center gap-0.5 font-mono text-xs">
            {#each crumbs as c, i (c.path)}
                {#if i > 0}
                    <span class="px-0.5 text-fg-faint">/</span>
                {/if}
                <button
                    type="button"
                    class="rounded px-1.5 py-0.5 text-fg-muted hover:bg-bg-2 hover:text-fg disabled:cursor-default disabled:text-fg disabled:hover:bg-transparent"
                    disabled={c.path === currentPath}
                    onclick={() => navigateTo(c.path)}
                >
                    {c.label}
                </button>
            {/each}
        </nav>

        <!-- Folder list -->
        <div class="flex-1 overflow-y-auto rounded-md border border-border bg-bg-0/40">
            {#if loading}
                <p class="p-6 text-center text-sm text-fg-muted">Loading…</p>
            {:else if loadError}
                <div class="p-4 text-sm">
                    <p class="text-red-300">Could not list folder:</p>
                    <pre class="mt-1 overflow-auto text-xs text-red-300/80">{loadError}</pre>
                </div>
            {:else}
                <ul class="divide-y divide-border">
                    {#if parent}
                        <li>
                            <button
                                type="button"
                                class="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-bg-2"
                                onclick={up}
                            >
                                <svg
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="2"
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    class="text-fg-muted"
                                    aria-hidden="true"
                                >
                                    <path d="M5 12h14M5 12l4-4M5 12l4 4" />
                                </svg>
                                <span class="font-mono text-fg-muted">..</span>
                            </button>
                        </li>
                    {/if}
                    {#if entries.length === 0 && !parent}
                        <li class="p-6 text-center text-sm text-fg-muted">No subfolders.</li>
                    {/if}
                    {#each entries as e (e.name)}
                        <li>
                            <button
                                type="button"
                                class="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-bg-2 {e.isHidden
                                    ? 'opacity-50'
                                    : ''}"
                                onclick={() => enter(e.name)}
                                title={e.isHidden ? 'Hidden folder' : ''}
                            >
                                <svg
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="2"
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    class="text-fg-muted"
                                    aria-hidden="true"
                                >
                                    <path
                                        d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"
                                    />
                                </svg>
                                <span class="font-mono">{e.name}</span>
                            </button>
                        </li>
                    {/each}
                </ul>
            {/if}
        </div>

        <!-- Bottom bar: current path + buttons. "Pick" takes the displayed
             folder; to pick a child, enter it first. -->
        <div class="flex items-center justify-between gap-3">
            <p class="min-w-0 flex-1 truncate font-mono text-xs text-fg-muted" title={currentPath}>
                {currentPath}
            </p>
            <div class="flex shrink-0 gap-2">
                <button type="button" class="btn-ghost" onclick={onCancel}>Cancel</button>
                <button type="button" class="btn-primary" onclick={pick} disabled={loading}>
                    Pick this folder
                </button>
            </div>
        </div>
    </div>
</div>
