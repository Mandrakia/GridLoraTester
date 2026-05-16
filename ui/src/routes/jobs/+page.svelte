<script lang="ts">
    import { onMount } from 'svelte';
    import { invalidateAll } from '$app/navigation';
    import type { PageData } from './$types';

    interface JobRow {
        id: number;
        type: string;
        params_json: string;
        status: 'queued' | 'running' | 'cancelled' | 'completed' | 'failed';
        progress_current: number;
        progress_total: number | null;
        current_label: string | null;
        error: string | null;
        created_at: string;
        started_at: string | null;
        finished_at: string | null;
    }
    interface JobLog {
        id: number;
        job_id: number;
        level: 'info' | 'warn' | 'error';
        message: string;
        created_at: string;
    }

    let { data }: { data: PageData } = $props();

    // Local mirror of the server-loaded list. The $effect below copies SSR
    // refreshes (invalidateAll() etc.) into it; polling (`refresh()`) writes
    // straight to it without needing a server round-trip on each tick.
    let jobs = $state<JobRow[]>([]);
    let expandedId = $state<number | null>(null);
    let detailLogs = $state<JobLog[]>([]);
    let cancelling = $state<Set<number>>(new Set());

    $effect(() => {
        jobs = data.jobs as JobRow[];
    });

    // Poll while the page is visible. Faster cadence when there's an active
    // job, slow when everything is settled.
    let pollTimer: ReturnType<typeof setTimeout> | undefined;
    function schedule() {
        const hasActive = jobs.some(
            (j) => j.status === 'queued' || j.status === 'running'
        );
        pollTimer = setTimeout(refresh, hasActive ? 1500 : 5000);
    }
    async function refresh() {
        try {
            const res = await fetch('/api/jobs');
            const body = await res.json();
            jobs = body.jobs as JobRow[];
            if (expandedId != null) {
                const r = await fetch(`/api/jobs/${expandedId}`);
                if (r.ok) {
                    const b = await r.json();
                    detailLogs = b.logs as JobLog[];
                }
            }
        } catch {
            // ignore — next tick will retry
        } finally {
            schedule();
        }
    }
    onMount(() => {
        schedule();
        return () => {
            if (pollTimer) clearTimeout(pollTimer);
        };
    });

    async function toggleExpand(id: number) {
        if (expandedId === id) {
            expandedId = null;
            detailLogs = [];
            return;
        }
        expandedId = id;
        detailLogs = [];
        try {
            const r = await fetch(`/api/jobs/${id}`);
            if (r.ok) {
                const b = await r.json();
                detailLogs = b.logs as JobLog[];
            }
        } catch {
            // ignore
        }
    }

    async function cancelJob(id: number) {
        cancelling.add(id);
        cancelling = cancelling;
        try {
            await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
            await refresh();
            await invalidateAll();
        } finally {
            cancelling.delete(id);
            cancelling = cancelling;
        }
    }

    function statusClass(s: JobRow['status']): string {
        switch (s) {
            case 'queued':
                return 'bg-bg-3 text-fg-muted';
            case 'running':
                return 'bg-amber-500/15 text-amber-300';
            case 'completed':
                return 'bg-emerald-500/15 text-emerald-300';
            case 'failed':
                return 'bg-red-500/15 text-red-300';
            case 'cancelled':
                return 'bg-fg-faint/20 text-fg-faint';
        }
    }

    function fmtDate(iso: string | null): string {
        if (!iso) return '—';
        try {
            return new Date(iso.replace(' ', 'T') + (iso.endsWith('Z') ? '' : 'Z')).toLocaleString();
        } catch {
            return iso;
        }
    }

    function parseParams(json: string): Record<string, unknown> {
        try {
            const o = JSON.parse(json);
            return o && typeof o === 'object' ? o : {};
        } catch {
            return {};
        }
    }

    function progressPct(j: JobRow): number | null {
        if (!j.progress_total || j.progress_total <= 0) return null;
        return Math.min(100, Math.round((j.progress_current / j.progress_total) * 100));
    }
</script>

<svelte:head>
    <title>Jobs — GridLoraTester</title>
</svelte:head>

<div class="space-y-4 p-6">
    <header class="flex items-baseline justify-between gap-4">
        <div>
            <h1 class="text-2xl font-semibold tracking-tight">Jobs</h1>
            <p class="mt-1 text-sm text-fg-muted">
                Background tasks (face detection, future imports, …). Polls live.
            </p>
        </div>
    </header>

    <div class="overflow-hidden rounded-lg border border-border bg-bg-1">
        <table class="w-full text-sm">
            <thead class="bg-bg-2 text-xs uppercase tracking-wide text-fg-muted">
                <tr>
                    <th class="px-3 py-2 text-right font-medium">#</th>
                    <th class="px-4 py-2 text-left font-medium">Type</th>
                    <th class="px-4 py-2 text-left font-medium">Status</th>
                    <th class="px-4 py-2 text-left font-medium">Progress</th>
                    <th class="px-4 py-2 text-left font-medium">Started</th>
                    <th class="px-4 py-2 text-left font-medium">Finished</th>
                    <th class="w-24 px-4 py-2 text-right font-medium">Actions</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-border">
                {#if jobs.length === 0}
                    <tr>
                        <td colspan="7" class="px-4 py-6 text-center text-sm text-fg-faint">
                            No jobs yet.
                        </td>
                    </tr>
                {:else}
                    {#each jobs as j (j.id)}
                        {@const pct = progressPct(j)}
                        {@const params = parseParams(j.params_json)}
                        <tr class="transition-colors hover:bg-bg-2/40">
                            <td class="px-3 py-2 text-right align-top tabular-nums text-fg-faint"
                                >{j.id}</td
                            >
                            <td class="px-4 py-2 align-top">
                                <button
                                    type="button"
                                    class="font-mono text-xs hover:text-accent-hover"
                                    onclick={() => toggleExpand(j.id)}
                                    title="Click to show logs"
                                >
                                    {j.type}
                                </button>
                                {#if params.person_name || params.person_id}
                                    <div class="text-xs text-fg-muted">
                                        {(params.person_name as string) ??
                                            (params.person_id as string)}
                                    </div>
                                {/if}
                            </td>
                            <td class="px-4 py-2 align-top">
                                <span
                                    class="rounded-full px-2 py-0.5 text-xs font-medium {statusClass(
                                        j.status
                                    )}"
                                >
                                    {j.status}
                                </span>
                                {#if j.error}
                                    <div class="mt-1 text-[10px] text-red-300/80">{j.error}</div>
                                {/if}
                            </td>
                            <td class="px-4 py-2 align-top">
                                {#if pct != null}
                                    <div class="flex items-center gap-2">
                                        <div class="h-1.5 w-32 rounded-full bg-bg-3">
                                            <div
                                                class="h-1.5 rounded-full bg-accent"
                                                style="width: {pct}%"
                                            ></div>
                                        </div>
                                        <span class="text-xs tabular-nums text-fg-muted">
                                            {j.progress_current}/{j.progress_total} · {pct}%
                                        </span>
                                    </div>
                                {:else if j.progress_current > 0}
                                    <span class="text-xs tabular-nums text-fg-muted"
                                        >{j.progress_current} done</span
                                    >
                                {:else}
                                    <span class="text-xs text-fg-faint">—</span>
                                {/if}
                                {#if j.current_label && (j.status === 'running' || j.status === 'queued')}
                                    <div class="mt-1 truncate text-[10px] text-fg-faint"
                                        title={j.current_label}
                                        >{j.current_label}</div
                                    >
                                {/if}
                            </td>
                            <td class="px-4 py-2 align-top text-xs text-fg-muted"
                                >{fmtDate(j.started_at)}</td
                            >
                            <td class="px-4 py-2 align-top text-xs text-fg-muted"
                                >{fmtDate(j.finished_at)}</td
                            >
                            <td class="px-4 py-2 align-top">
                                <div class="flex justify-end gap-1">
                                    {#if j.status === 'running' || j.status === 'queued'}
                                        <button
                                            type="button"
                                            class="btn-ghost px-2 py-1 text-xs text-red-400 hover:bg-red-500/10 hover:text-red-300"
                                            onclick={() => cancelJob(j.id)}
                                            disabled={cancelling.has(j.id)}
                                            >{cancelling.has(j.id) ? '…' : 'Cancel'}</button
                                        >
                                    {/if}
                                </div>
                            </td>
                        </tr>
                        {#if expandedId === j.id}
                            <tr>
                                <td></td>
                                <td colspan="6" class="px-4 py-2 align-top">
                                    <div
                                        class="max-h-64 overflow-y-auto rounded-md border border-border bg-bg-0/40 p-3 font-mono text-[11px]"
                                    >
                                        {#if detailLogs.length === 0}
                                            <p class="text-fg-faint">No log lines yet.</p>
                                        {:else}
                                            {#each detailLogs as l (l.id)}
                                                <div
                                                    class="whitespace-pre-wrap {l.level === 'error'
                                                        ? 'text-red-300'
                                                        : l.level === 'warn'
                                                          ? 'text-amber-300'
                                                          : 'text-fg-muted'}"
                                                >
                                                    <span class="text-fg-faint">{l.created_at}</span>
                                                    {l.message}
                                                </div>
                                            {/each}
                                        {/if}
                                    </div>
                                </td>
                            </tr>
                        {/if}
                    {/each}
                {/if}
            </tbody>
        </table>
    </div>
</div>
