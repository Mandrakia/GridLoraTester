<script lang="ts">
    import '../app.css';
    import JobsBadge from '$lib/components/JobsBadge.svelte';
    import Sidebar from '$lib/components/Sidebar.svelte';

    let { children } = $props();
</script>

<div class="flex h-full">
    <Sidebar />
    <main class="flex-1 overflow-y-auto">
        {@render children()}
    </main>
</div>

<JobsBadge />
