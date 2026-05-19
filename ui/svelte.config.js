import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),
    kit: {
        adapter: adapter(),
        // Self-hosted single-user tool: must work whatever hostname the user
        // reaches it under (localhost, 127.0.0.1, <lan-ip>, *.local, tailscale,
        // a domain pointed at the container, …). The Origin-based CSRF check
        // enforces strict same-origin POSTs and breaks every form submit as
        // soon as access URL diverges from ORIGIN. Cross-origin form attacks
        // aren't part of this app's threat model — it's not a public service.
        csrf: { checkOrigin: false }
    }
};

export default config;
