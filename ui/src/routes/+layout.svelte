<script lang="ts">
    import '../app.css';
    import JobsBadge from '$lib/components/JobsBadge.svelte';
    import Sidebar from '$lib/components/Sidebar.svelte';
    import { page } from '$app/state';

    let { children, data } = $props();

    // The /login page renders standalone — no sidebar / jobs chrome around it.
    const showChrome = $derived(page.url.pathname !== '/login');
</script>

{#if showChrome}
    <div class="flex h-full">
        <Sidebar authEnabled={data.authEnabled} />
        <main class="flex-1 overflow-y-auto">
            {@render children()}
        </main>
    </div>

    <JobsBadge />
{:else}
    {@render children()}
{/if}
