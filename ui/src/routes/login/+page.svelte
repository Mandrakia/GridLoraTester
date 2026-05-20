<script lang="ts">
    import { enhance } from '$app/forms';

    let { form } = $props();
    let submitting = $state(false);
</script>

<svelte:head>
    <title>Sign in · GridLoraTester</title>
</svelte:head>

<div class="flex h-full min-h-screen items-center justify-center bg-bg-0 p-4">
    <form
        method="POST"
        use:enhance={() => {
            submitting = true;
            return async ({ update }) => {
                await update();
                submitting = false;
            };
        }}
        class="card w-full max-w-sm space-y-5"
    >
        <div class="flex items-center gap-2">
            <div
                class="flex h-8 w-8 items-center justify-center rounded-md bg-accent text-xs font-bold text-white"
            >
                glt
            </div>
            <h1 class="text-base font-semibold tracking-tight">GridLoraTester</h1>
        </div>

        <p class="text-sm text-fg-muted">This instance is password-protected.</p>

        <label class="block space-y-1.5">
            <span class="text-xs font-medium text-fg-faint">Password</span>
            <!-- svelte-ignore a11y_autofocus -->
            <input
                name="password"
                type="password"
                autocomplete="current-password"
                autofocus
                required
                class="input"
            />
        </label>

        {#if form?.error}
            <p class="text-sm text-red-400">{form.error}</p>
        {/if}

        <button type="submit" class="btn-primary w-full" disabled={submitting}>
            {submitting ? 'Signing in…' : 'Sign in'}
        </button>
    </form>
</div>
