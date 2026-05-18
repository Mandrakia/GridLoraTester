<script lang="ts">
    import { onMount } from 'svelte';
    import { invalidateAll } from '$app/navigation';
    import MainPanel from '$lib/components/MainPanel.svelte';
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
        metrics_json: string | null;
    }

    interface PhaseStat {
        count: number;
        p50: number;
        p95: number;
        mean: number;
    }
    interface PipelineMetrics {
        active_producers: number;
        queue_depth: number;
        queue_capacity: number;
        consumer_wait_p50_ms: number;
        scale_ups: number;
    }
    interface Metrics {
        phases: Record<string, PhaseStat>;
        throughput_per_s?: number;
        with_faces?: number;
        no_face?: number;
        failed?: number;
        pipeline?: PipelineMetrics | null;
    }

    /** Human-readable label per phase key the handler records. */
    const PHASE_LABEL: Record<string, string> = {
        download: 'Download',
        detect_http: 'HTTP→worker',
        python_decode: 'Python decode',
        python_detect: 'Python detect',
        http_overhead: 'HTTP overhead',
        db_write: 'DB write'
    };
    /** Display order — bottlenecks readers care about first. */
    const PHASE_ORDER = [
        'download',
        'python_detect',
        'python_decode',
        'db_write',
        'detect_http',
        'http_overhead'
    ];
    interface JobLog {
        id: number;
        job_id: number;
        level: 'info' | 'warn' | 'error';
        message: string;
        created_at: string;
    }

    let { data }: { data: PageData } = $props();

    type Tab = 'latest' | 'archive';
    let tab = $state<Tab>('latest');

    // Local mirrors of the server-loaded lists. The $effect below copies SSR
    // refreshes (invalidateAll() etc.) into them; polling (`refresh()`)
    // writes straight to the active tab without needing a server round-trip.
    let latest = $state<JobRow[]>([]);
    let archive = $state<JobRow[]>([]);
    let expandedId = $state<number | null>(null);
    let detailLogs = $state<JobLog[]>([]);
    let cancelling = $state<Set<number>>(new Set());
    let retrying = $state<Set<number>>(new Set());
    /** ID of the most recent retry'd job — highlighted briefly so the user
     * sees where its rerun landed in the list. */
    let flashId = $state<number | null>(null);

    $effect(() => {
        latest = data.latest as JobRow[];
        archive = data.archive as JobRow[];
    });

    /** The currently visible list — what the table renders. */
    const jobs = $derived<JobRow[]>(tab === 'latest' ? latest : archive);

    // Poll while the page is visible. Faster cadence when there's an
    // active job, slow when everything is settled. Polling is paused
    // automatically when the tab is hidden (the browser would throttle us
    // anyway) and forced-refreshed the moment the tab regains focus.
    let pollTimer: ReturnType<typeof setTimeout> | undefined;
    /** Tracks whether a refresh is in flight, so a visibilitychange-driven
     * forced refresh doesn't double-trigger. */
    let refreshing = false;
    /** Single-source-of-truth for the timeout on every fetch — keeps the
     * chain self-healing if SQLite write contention from a noisy job
     * delays the read side. */
    const FETCH_TIMEOUT_MS = 8000;

    async function fetchWithTimeout(url: string, init?: RequestInit) {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
        try {
            return await fetch(url, { ...init, signal: ctrl.signal });
        } finally {
            clearTimeout(t);
        }
    }

    function schedule() {
        if (pollTimer) clearTimeout(pollTimer);
        // Don't queue polls while the tab is hidden — the browser throttles
        // setTimeout heavily in background and we'd just accumulate stale
        // requests. `visibilitychange` re-fires on return.
        if (typeof document !== 'undefined' && document.hidden) return;
        // Archives is historical-only — no active jobs by definition. Poll
        // it slowly so a freshly-archived row eventually shows up without
        // hammering the DB.
        if (tab === 'archive') {
            pollTimer = setTimeout(refresh, 10_000);
            return;
        }
        const hasActive = jobs.some(
            (j) => j.status === 'queued' || j.status === 'running'
        );
        pollTimer = setTimeout(refresh, hasActive ? 1500 : 5000);
    }

    async function refresh() {
        if (refreshing) return;
        refreshing = true;
        try {
            const url = `/api/jobs?view=${tab}`;
            const res = await fetchWithTimeout(url);
            const body = await res.json();
            if (tab === 'latest') latest = body.jobs as JobRow[];
            else archive = body.jobs as JobRow[];
            if (expandedId != null) {
                const r = await fetchWithTimeout(`/api/jobs/${expandedId}`);
                if (r.ok) {
                    const b = await r.json();
                    detailLogs = b.logs as JobLog[];
                }
            }
        } catch (e) {
            // Don't let a transient error halt the chain — but DO surface
            // it in the console so a recurring problem is visible. Most
            // common: AbortError from the 8s timeout when SQLite write
            // contention from a busy job blocks the read for too long.
            console.warn('[jobs] refresh failed:', (e as Error).message || e);
        } finally {
            refreshing = false;
            schedule();
        }
    }

    function switchTab(t: Tab) {
        if (tab === t) return;
        tab = t;
        // Collapse any open detail row — its target may not even be in the
        // new tab's list, and the user clearly switched contexts.
        expandedId = null;
        detailLogs = [];
        // Refresh immediately so the new tab feels live (SSR data may be
        // stale by now if the user has been on the page for a while).
        refresh();
    }

    function onVisibility() {
        if (!document.hidden) {
            // Coming back to the tab — refresh immediately and resume polling.
            refresh();
        } else if (pollTimer) {
            // Tab going hidden — pause the chain. refresh() will reschedule
            // on the next visibility return.
            clearTimeout(pollTimer);
            pollTimer = undefined;
        }
    }

    onMount(() => {
        schedule();
        document.addEventListener('visibilitychange', onVisibility);
        return () => {
            if (pollTimer) clearTimeout(pollTimer);
            document.removeEventListener('visibilitychange', onVisibility);
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

    async function retryJob(id: number) {
        retrying.add(id);
        retrying = retrying;
        try {
            const res = await fetch(`/api/jobs/${id}/retry`, { method: 'POST' });
            const body = await res.json().catch(() => null);
            if (res.ok && body?.new_job_id) {
                flashId = body.new_job_id as number;
                // Drop the highlight after a few seconds — long enough to see
                // it scroll into view at the top, short enough that it
                // doesn't shout at the user while watching progress.
                setTimeout(() => {
                    if (flashId === body.new_job_id) flashId = null;
                }, 4000);
            }
            await refresh();
            await invalidateAll();
        } finally {
            retrying.delete(id);
            retrying = retrying;
        }
    }

    /** Label for the retry button based on the source job's terminal status.
     * Same backend action — just a label nuance so the affordance reads
     * naturally for each case. */
    function retryLabel(s: JobRow['status']): string {
        if (s === 'failed') return 'Retry';
        if (s === 'cancelled') return 'Resume';
        return 'Re-run'; // completed — pick up new items added since
    }

    /** Returns the pill's tailwind classes for a job.
     *
     * The DB `status` reflects whether the JOB itself ran to completion;
     * but a job can complete cleanly while many of its items failed
     * (e.g. compute-image-hashes downloading from a flaky connector). A
     * green "completed" pill is misleading in that case — we render an
     * amber variant when the live metrics report any in-flight failures.
     */
    function statusClass(j: JobRow, m: Metrics | null): string {
        switch (j.status) {
            case 'queued':
                return 'bg-bg-3 text-fg-muted';
            case 'running':
                return 'bg-amber-500/15 text-amber-300';
            case 'completed':
                if (m && (m.failed ?? 0) > 0) {
                    return 'bg-amber-500/15 text-amber-300';
                }
                return 'bg-emerald-500/15 text-emerald-300';
            case 'failed':
                return 'bg-red-500/15 text-red-300';
            case 'cancelled':
                return 'bg-fg-faint/20 text-fg-faint';
        }
    }

    /** Human label for the pill — adds "· N failed" suffix when a
     * completed job had in-flight errors, so the user knows what the
     * amber pill is signaling without expanding the row. */
    function statusLabel(j: JobRow, m: Metrics | null): string {
        if (j.status === 'completed' && m && (m.failed ?? 0) > 0) {
            return `completed · ${m.failed} failed`;
        }
        return j.status;
    }

    /** Parse a SQLite datetime('now') string ("YYYY-MM-DD HH:MM:SS",
     * implicit UTC) into a Date in the local timezone. We add the missing
     * 'Z' so `new Date` doesn't drift into ambiguous "space-separated =
     * local" parsing across browsers. */
    function parseSqliteUtc(iso: string): Date {
        return new Date(iso.replace(' ', 'T') + (iso.endsWith('Z') ? '' : 'Z'));
    }

    function fmtDate(iso: string | null): string {
        if (!iso) return '—';
        try {
            return parseSqliteUtc(iso).toLocaleString();
        } catch {
            return iso;
        }
    }

    /** Compact local-time formatter for log lines — HH:MM:SS today, with
     * the date prefixed only when the line is from another day. Avoids
     * bloating the log column with redundant full timestamps. */
    function fmtLogTime(iso: string): string {
        try {
            const d = parseSqliteUtc(iso);
            const now = new Date();
            const sameDay =
                d.getFullYear() === now.getFullYear() &&
                d.getMonth() === now.getMonth() &&
                d.getDate() === now.getDate();
            const time = d.toLocaleTimeString();
            return sameDay ? time : `${d.toLocaleDateString()} ${time}`;
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

    function parseMetrics(json: string | null): Metrics | null {
        if (!json) return null;
        try {
            const o = JSON.parse(json);
            if (o && typeof o === 'object' && o.phases) return o as Metrics;
        } catch {
            // ignore
        }
        return null;
    }

    /** Phases in display order, only those with data. */
    function orderedPhases(m: Metrics): [string, PhaseStat][] {
        const out: [string, PhaseStat][] = [];
        for (const k of PHASE_ORDER) {
            const s = m.phases[k];
            if (s && s.count > 0) out.push([k, s]);
        }
        // Any other phases we haven't seen the name of, appended at the end.
        for (const [k, s] of Object.entries(m.phases)) {
            if (!PHASE_ORDER.includes(k) && s.count > 0) out.push([k, s]);
        }
        return out;
    }

    /** Bottleneck = phase with the highest p50. Returned as the phase key,
     * or null when there's nothing to compare. */
    function dominantPhase(m: Metrics): string | null {
        let best: [string, number] | null = null;
        for (const [k, s] of Object.entries(m.phases)) {
            if (s.count === 0) continue;
            if (!best || s.p50 > best[1]) best = [k, s.p50];
        }
        return best?.[0] ?? null;
    }

    function fmtMs(ms: number): string {
        if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
        if (ms >= 10) return Math.round(ms) + ' ms';
        return ms.toFixed(1) + ' ms';
    }
</script>

<svelte:head>
    <title>Jobs — GridLoraTester</title>
</svelte:head>

<MainPanel>
    <div class="space-y-4">
    <header class="flex items-baseline justify-between gap-4">
        <div>
            <h1 class="text-2xl font-semibold tracking-tight">Jobs</h1>
            <p class="mt-1 text-sm text-fg-muted">
                Background tasks (face detection, imports, grid runs). Updates live.
            </p>
        </div>
    </header>

    <!-- Latest = most recent run per (type, key_arg1, key_arg2). Archives =
         older runs of the same logical job. A retry pushes the previous
         row into archives automatically. -->
    <div class="flex items-center gap-1 border-b border-border">
        {#each [{ id: 'latest' as Tab, label: 'Latest' }, { id: 'archive' as Tab, label: 'Archives' }] as t (t.id)}
            <button
                type="button"
                class="-mb-px border-b-2 px-3 py-1.5 text-sm transition-colors {tab === t.id
                    ? 'border-accent text-fg'
                    : 'border-transparent text-fg-muted hover:text-fg'}"
                onclick={() => switchTab(t.id)}
            >
                {t.label}
                <span class="ml-1 text-[10px] tabular-nums text-fg-faint">
                    {t.id === 'latest' ? latest.length : archive.length}
                </span>
            </button>
        {/each}
    </div>

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
                        {@const metrics = parseMetrics(j.metrics_json)}
                        {@const dominant = metrics ? dominantPhase(metrics) : null}
                        <tr
                            class="transition-colors hover:bg-bg-2/40 {flashId === j.id
                                ? 'bg-accent/10'
                                : ''}"
                        >
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
                                        j,
                                        metrics
                                    )}"
                                >
                                    {statusLabel(j, metrics)}
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
                                {#if metrics?.throughput_per_s && metrics.throughput_per_s > 0}
                                    {@const tp = metrics.throughput_per_s}
                                    {@const eta =
                                        j.progress_total && j.progress_total > j.progress_current
                                            ? (j.progress_total - j.progress_current) / tp
                                            : null}
                                    <div
                                        class="mt-1 flex items-center gap-1.5 text-[10px] tabular-nums text-fg-faint"
                                    >
                                        <span class="text-accent/80"
                                            >{tp.toFixed(2)} img/s</span
                                        >
                                        {#if eta != null && eta > 0}
                                            <span>· ETA {eta < 60 ? Math.round(eta) + 's' : Math.round(eta / 60) + 'm'}</span>
                                        {/if}
                                        {#if dominant}
                                            <span title="Dominant phase (highest p50)"
                                                >· bottleneck: <span class="text-fg-muted"
                                                    >{PHASE_LABEL[dominant] ?? dominant}</span
                                                ></span
                                            >
                                        {/if}
                                    </div>
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
                                    {:else}
                                        <!-- Retry / Resume / Re-run — same backend (creates a
                                             new job with the same params; the handler's per-item
                                             idempotency skips the already-processed work). -->
                                        <button
                                            type="button"
                                            class="btn-ghost px-2 py-1 text-xs text-accent hover:bg-accent/10 hover:text-accent-hover"
                                            onclick={() => retryJob(j.id)}
                                            disabled={retrying.has(j.id)}
                                            title="Re-enqueue with the same params — already-processed items are skipped."
                                            >{retrying.has(j.id) ? '…' : retryLabel(j.status)}</button
                                        >
                                    {/if}
                                </div>
                            </td>
                        </tr>
                        {#if expandedId === j.id}
                            <tr>
                                <td></td>
                                <td colspan="6" class="px-4 py-2 align-top">
                                    <div class="grid gap-3 lg:grid-cols-[1fr_22rem]">
                                        <!-- Logs (existing) -->
                                        <div
                                            class="max-h-64 overflow-y-auto rounded-md border border-border bg-bg-0/40 p-3 font-mono text-[11px]"
                                        >
                                            {#if detailLogs.length === 0}
                                                <p class="text-fg-faint">No log lines yet.</p>
                                            {:else}
                                                {#each detailLogs as l (l.id)}
                                                    <div
                                                        class="whitespace-pre-wrap {l.level ===
                                                        'error'
                                                            ? 'text-red-300'
                                                            : l.level === 'warn'
                                                              ? 'text-amber-300'
                                                              : 'text-fg-muted'}"
                                                    >
                                                        <span
                                                            class="text-fg-faint"
                                                            title={l.created_at + ' UTC'}
                                                            >{fmtLogTime(l.created_at)}</span
                                                        >
                                                        {l.message}
                                                    </div>
                                                {/each}
                                            {/if}
                                        </div>

                                        <!-- Live timings panel — p50/p95 per
                                             phase. Bar widths normalize to the
                                             max p50 across phases so the
                                             bottleneck is visually obvious. -->
                                        {#if metrics}
                                            {@const rows = orderedPhases(metrics)}
                                            {@const maxP50 = rows.reduce(
                                                (m, [, s]) => Math.max(m, s.p50),
                                                0
                                            )}
                                            <div
                                                class="rounded-md border border-border bg-bg-0/40 p-3"
                                            >
                                                <div
                                                    class="mb-2 flex items-baseline justify-between gap-2"
                                                >
                                                    <h4 class="text-xs font-medium text-fg">
                                                        Timings
                                                    </h4>
                                                    {#if metrics.throughput_per_s}
                                                        <span
                                                            class="text-[10px] tabular-nums text-fg-faint"
                                                        >
                                                            {metrics.throughput_per_s.toFixed(2)}
                                                            img/s
                                                        </span>
                                                    {/if}
                                                </div>
                                                {#if rows.length === 0}
                                                    <p class="text-[11px] text-fg-faint">
                                                        Waiting for first sample…
                                                    </p>
                                                {:else}
                                                    <table class="w-full text-[11px]">
                                                        <thead
                                                            class="text-[10px] uppercase tracking-wide text-fg-faint"
                                                        >
                                                            <tr>
                                                                <th
                                                                    class="text-left font-medium"
                                                                    >Phase</th
                                                                >
                                                                <th
                                                                    class="text-right font-medium"
                                                                    >p50</th
                                                                >
                                                                <th
                                                                    class="text-right font-medium"
                                                                    >p95</th
                                                                >
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {#each rows as [k, s] (k)}
                                                                {@const w = maxP50 > 0
                                                                    ? (s.p50 / maxP50) * 100
                                                                    : 0}
                                                                <tr
                                                                    class={dominant === k
                                                                        ? 'text-fg'
                                                                        : 'text-fg-muted'}
                                                                >
                                                                    <td
                                                                        class="py-1 pr-2 align-middle"
                                                                    >
                                                                        <div
                                                                            class="font-mono text-[10px]"
                                                                        >
                                                                            {PHASE_LABEL[k] ?? k}
                                                                        </div>
                                                                        <div
                                                                            class="h-1 w-full rounded-full bg-bg-3"
                                                                        >
                                                                            <div
                                                                                class="h-1 rounded-full {dominant ===
                                                                                k
                                                                                    ? 'bg-accent'
                                                                                    : 'bg-accent/40'}"
                                                                                style="width: {w}%"
                                                                            ></div>
                                                                        </div>
                                                                    </td>
                                                                    <td
                                                                        class="text-right tabular-nums"
                                                                        >{fmtMs(s.p50)}</td
                                                                    >
                                                                    <td
                                                                        class="text-right tabular-nums"
                                                                        >{fmtMs(s.p95)}</td
                                                                    >
                                                                </tr>
                                                            {/each}
                                                        </tbody>
                                                    </table>
                                                    {#if metrics.failed != null && metrics.failed > 0}
                                                        <p
                                                            class="mt-2 text-[10px] text-red-300/80"
                                                        >
                                                            {metrics.failed} failed
                                                        </p>
                                                    {/if}
                                                {/if}

                                                <!-- Pipeline panel —
                                                     downloader pool state,
                                                     queue depth, consumer
                                                     starvation signal. -->
                                                {#if metrics.pipeline}
                                                    {@const p = metrics.pipeline}
                                                    {@const qFillPct =
                                                        p.queue_capacity > 0
                                                            ? Math.min(
                                                                  100,
                                                                  (p.queue_depth /
                                                                      p.queue_capacity) *
                                                                      100
                                                              )
                                                            : 0}
                                                    <div
                                                        class="mt-3 border-t border-border pt-2"
                                                    >
                                                        <h5
                                                            class="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-fg-faint"
                                                        >
                                                            Pipeline
                                                        </h5>
                                                        <div
                                                            class="space-y-1 text-[11px] tabular-nums"
                                                        >
                                                            <div
                                                                class="flex items-center justify-between gap-2"
                                                            >
                                                                <span
                                                                    class="text-fg-muted"
                                                                    >Downloaders</span
                                                                >
                                                                <span class="text-fg"
                                                                    >{p.active_producers}
                                                                    <span
                                                                        class="text-fg-faint"
                                                                        >· {p.scale_ups}
                                                                        scale-up{p.scale_ups ===
                                                                        1
                                                                            ? ''
                                                                            : 's'}</span
                                                                    ></span
                                                                >
                                                            </div>
                                                            <div
                                                                class="flex items-center justify-between gap-2"
                                                            >
                                                                <span
                                                                    class="text-fg-muted"
                                                                    >Queue</span
                                                                >
                                                                <span class="text-fg"
                                                                    >{p.queue_depth}/{p.queue_capacity}</span
                                                                >
                                                            </div>
                                                            <div
                                                                class="h-1 w-full rounded-full bg-bg-3"
                                                            >
                                                                <div
                                                                    class="h-1 rounded-full bg-emerald-400/60"
                                                                    style="width: {qFillPct}%"
                                                                ></div>
                                                            </div>
                                                            <div
                                                                class="flex items-center justify-between gap-2 pt-0.5"
                                                            >
                                                                <span
                                                                    class="text-fg-muted"
                                                                    title="Consumer wait on queue.get() — high = starved, scale signal"
                                                                    >Starvation p50</span
                                                                >
                                                                <span
                                                                    class={p.consumer_wait_p50_ms >
                                                                    50
                                                                        ? 'text-amber-300'
                                                                        : 'text-fg'}
                                                                    >{fmtMs(
                                                                        p.consumer_wait_p50_ms
                                                                    )}</span
                                                                >
                                                            </div>
                                                        </div>
                                                    </div>
                                                {/if}
                                            </div>
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
</MainPanel>
