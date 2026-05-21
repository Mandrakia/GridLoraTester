<script lang="ts">
    // Button + status pill that triggers the `?/analyze` form action on
    // whichever route includes this component. The action only *enqueues* a
    // compute-centroid job and returns its id; we then poll /api/jobs/<id>
    // and keep the button in its spinner until the job reaches a terminal
    // state. Reads persisted state from props — the caller already loaded it,
    // and update() refreshes it once the job finishes.
    import { onDestroy } from 'svelte';
    import { enhance } from '$app/forms';

    interface CentroidSummary {
        centroid_b64: string;
        n_single_face: number;
        n_multi_face: number;
        n_no_face: number;
        median_sim?: number | null;
        mean_sim?: number | null;
        computed_at: string;
    }

    interface Props {
        label?: string;
        /** Existing persisted state, or null when never computed. */
        centroid: CentroidSummary | null;
        /** The form action endpoint suffix, e.g. '?/analyze'. */
        action?: string;
        /** Last error from a previous attempt, surfaced under the button. */
        error?: string | null;
    }

    let { label = 'Analyze dataset', centroid, action = '?/analyze', error = null }: Props =
        $props();

    let running = $state(false);
    // Failure raised by the background job itself. The POST only enqueues, so
    // it succeeds even when the analysis later fails — `error` (from the form
    // action's fail()) won't cover that case, this does.
    let jobError = $state<string | null>(null);

    let destroyed = false;
    onDestroy(() => {
        destroyed = true;
    });

    interface JobView {
        status: string;
        error: string | null;
    }

    /** Poll a job until it reaches a terminal state, returning the final
     * status + error. Transient fetch failures (HMR reload, brief blip) are
     * swallowed so a momentary hiccup doesn't drop the spinner mid-run. */
    async function waitForJob(jobId: number): Promise<JobView> {
        while (!destroyed) {
            try {
                const res = await fetch(`/api/jobs/${jobId}`);
                if (res.ok) {
                    const { job } = (await res.json()) as { job: JobView | null };
                    // 404 / vanished row → treat as done; nothing left to wait on.
                    if (!job) return { status: 'completed', error: null };
                    if (
                        job.status === 'completed' ||
                        job.status === 'failed' ||
                        job.status === 'cancelled'
                    ) {
                        return { status: job.status, error: job.error ?? null };
                    }
                }
            } catch {
                // transient — keep polling
            }
            await new Promise((r) => setTimeout(r, 1200));
        }
        return { status: 'cancelled', error: null };
    }

    function fmtDate(iso: string): string {
        try {
            return new Date(iso.replace(' ', 'T') + 'Z').toLocaleString();
        } catch {
            return iso;
        }
    }
</script>

<div class="flex flex-wrap items-center gap-3">
    <form
        method="POST"
        {action}
        use:enhance={() => {
            running = true;
            jobError = null;
            return async ({ result, update }) => {
                // Keep spinning until the enqueued job finishes. Defer
                // update() (which applies the result + refreshes the loaded
                // centroid) until then, so the pill doesn't flip to "ready"
                // before the analysis has actually run.
                const jobId =
                    result.type === 'success'
                        ? Number((result.data as { job_id?: unknown } | undefined)?.job_id)
                        : NaN;
                if (Number.isFinite(jobId) && jobId > 0) {
                    const final = await waitForJob(jobId);
                    if (final.status === 'failed') {
                        jobError = final.error ?? 'Analysis failed.';
                    }
                }
                running = false;
                if (!destroyed) await update({ reset: false });
            };
        }}
    >
        <button type="submit" class="btn-primary" disabled={running}>
            {#if running}
                <svg
                    class="h-3.5 w-3.5 animate-spin"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2.5"
                >
                    <path d="M12 3a9 9 0 1 0 9 9" stroke-linecap="round" />
                </svg>
                Analyzing…
            {:else}
                {label}
            {/if}
        </button>
    </form>

    {#if centroid}
        <div class="flex flex-wrap items-center gap-3 text-xs text-fg-muted">
            <span class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-300">centroid ready</span>
            <span class="tabular-nums">
                {centroid.n_single_face} single · {centroid.n_multi_face} multi
                {#if centroid.n_no_face > 0}
                    · <span class="text-fg-faint">{centroid.n_no_face} no-face</span>
                {/if}
            </span>
            {#if centroid.median_sim != null || centroid.mean_sim != null}
                <span class="tabular-nums">
                    {#if centroid.median_sim != null}
                        median <span class="text-fg">{centroid.median_sim.toFixed(3)}</span>
                    {/if}
                    {#if centroid.mean_sim != null}
                        · mean <span class="text-fg">{centroid.mean_sim.toFixed(3)}</span>
                    {/if}
                </span>
            {/if}
            <span class="text-fg-faint">computed {fmtDate(centroid.computed_at)}</span>
        </div>
    {:else}
        <span class="text-xs text-fg-faint">no centroid yet</span>
    {/if}
</div>

{#if jobError ?? error}
    <pre class="mt-2 max-h-40 overflow-auto rounded-md border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-300">{jobError ?? error}</pre>
{/if}
