<script lang="ts">
    // Floating bottom-right indicator showing active jobs (queued + running).
    // Polls /api/jobs?active=1 on a slow tick; hides entirely when nothing
    // is active. Clicks navigate to /jobs for the full view.
    import { onMount } from 'svelte';
    import { page } from '$app/state';

    interface ActiveJob {
        id: number;
        type: string;
        status: 'queued' | 'running';
        progress_current: number;
        progress_total: number | null;
        current_label: string | null;
    }

    let jobs = $state<ActiveJob[]>([]);
    let pollTimer: ReturnType<typeof setTimeout> | undefined;

    async function refresh() {
        try {
            const res = await fetch('/api/jobs?active=1');
            const body = await res.json();
            jobs = (body.jobs ?? []) as ActiveJob[];
        } catch {
            // ignore — the page itself stays usable even when polling fails
        } finally {
            pollTimer = setTimeout(refresh, jobs.length > 0 ? 1500 : 5000);
        }
    }

    onMount(() => {
        refresh();
        return () => {
            if (pollTimer) clearTimeout(pollTimer);
        };
    });

    let onJobsPage = $derived(page.url.pathname.startsWith('/jobs'));

    function pct(j: ActiveJob): number | null {
        if (!j.progress_total) return null;
        return Math.min(100, Math.round((j.progress_current / j.progress_total) * 100));
    }
</script>

{#if jobs.length > 0 && !onJobsPage}
    <a
        href="/jobs"
        class="fixed bottom-4 right-4 z-40 flex max-w-xs flex-col gap-1.5 rounded-lg border border-border bg-bg-1/95 px-3 py-2 text-xs shadow-lg backdrop-blur transition-colors hover:border-border-strong"
    >
        <div class="flex items-center gap-2">
            <span class="relative flex h-2 w-2">
                <span
                    class="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-75"
                ></span>
                <span class="relative inline-flex h-2 w-2 rounded-full bg-accent"></span>
            </span>
            <span class="font-medium text-fg">
                {jobs.length} active job{jobs.length === 1 ? '' : 's'}
            </span>
        </div>
        {#each jobs.slice(0, 3) as j (j.id)}
            {@const p = pct(j)}
            <div class="flex items-center gap-2 text-fg-muted">
                <span class="truncate font-mono text-[10px]">{j.type} #{j.id}</span>
                {#if p != null}
                    <span class="tabular-nums text-[10px] text-fg-faint">{p}%</span>
                {/if}
            </div>
        {/each}
        {#if jobs.length > 3}
            <span class="text-[10px] text-fg-faint">+{jobs.length - 3} more…</span>
        {/if}
    </a>
{/if}
