import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    // Tailwind v4 plugin must come BEFORE sveltekit() per Tailwind's docs —
    // otherwise SvelteKit's CSS pre-processing fights the on-the-fly
    // utility generation.
    plugins: [tailwindcss(), sveltekit()],
    server: {
        port: 5273,
        strictPort: false
    }
});
