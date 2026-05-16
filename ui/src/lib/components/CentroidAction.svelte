<script lang="ts">
    // Button + status pill that triggers the `?/compute-centroid` form
    // action on whichever route includes this component. Reads from props,
    // doesn't touch the DB — the caller already loaded the persisted state.
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';

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
        /** The form action endpoint suffix, e.g. '?/compute-centroid'. */
        action?: string;
        /** Last error from a previous attempt, surfaced under the button. */
        error?: string | null;
    }

    let { label = 'Calculate centroid', centroid, action = '?/compute-centroid', error = null }: Props =
        $props();

    let running = $state(false);

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
            return async ({ update }) => {
                await update({ reset: false });
                running = false;
                await invalidateAll();
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
                Detecting…
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

{#if error}
    <pre class="mt-2 max-h-40 overflow-auto rounded-md border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-300">{error}</pre>
{/if}
