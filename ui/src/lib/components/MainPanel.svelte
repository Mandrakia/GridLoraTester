<script lang="ts">
    // Unified page container. Every route should wrap its content in
    // <MainPanel> so padding + width stay consistent across the app.
    //
    //   size="wide"   — full viewport width (minus sidebar), padded.
    //                   Use for tables, grids, dashboards.
    //   size="narrow" — max-w-3xl centered, padded. Use for forms-heavy
    //                   pages where wide content reads badly (Settings,
    //                   Prompts, …).
    //
    // Two-zone pages (pinned header + scrolling body) pass a `header`
    // snippet. The header is rendered at full width (so tab strips with
    // a `border-b` can still span edge-to-edge), and the body becomes
    // its own overflow-auto scroll container that fills the remaining
    // height. The body's padding + max-width come from this component.
    import type { Snippet } from 'svelte';

    let {
        size = 'wide',
        header,
        children
    }: {
        size?: 'narrow' | 'wide';
        header?: Snippet;
        children: Snippet;
    } = $props();

    const maxW = $derived(size === 'narrow' ? 'max-w-3xl' : '');
</script>

{#if header}
    <div class="flex h-full flex-col">
        <div class="shrink-0 bg-bg-0">
            {@render header()}
        </div>
        <div class="flex-1 overflow-auto">
            <div class="mx-auto {maxW} flex min-h-full flex-col px-6 py-6">
                {@render children()}
            </div>
        </div>
    </div>
{:else}
    <div class="mx-auto {maxW} px-6 py-6">
        {@render children()}
    </div>
{/if}
