<script lang="ts">
    import { page } from '$app/state';

    interface NavItem {
        href: string;
        label: string;
        icon: string;
    }

    const items: NavItem[] = [
        {
            href: '/datasets',
            label: 'Datasets',
            // square-stack icon
            icon: 'M4 7h12M4 11h12M4 15h12M4 19h12'
        },
        {
            href: '/prompts',
            label: 'Prompts',
            // text-lines icon
            icon: 'M4 6h16M4 10h10M4 14h16M4 18h8'
        },
        {
            href: '/tests',
            label: 'Tests',
            // flask / beaker icon
            icon: 'M9 3h6 M10 3v6L5 19a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2L14 9V3'
        },
        {
            href: '/jobs',
            label: 'Jobs',
            // gear / cog icon
            icon: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h.1a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v.1a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z'
        },
        {
            href: '/settings',
            label: 'Settings',
            // sliders icon
            icon: 'M3 7h10M17 7h4M3 12h4M11 12h10M3 17h14M19 17h2 M13 5v4 M9 10v4 M17 15v4'
        }
    ];

    function isActive(href: string): boolean {
        return page.url.pathname === href || page.url.pathname.startsWith(href + '/');
    }
</script>

<aside class="flex h-full w-56 flex-col border-r border-border bg-bg-1">
    <div class="flex h-14 items-center gap-2 border-b border-border px-4">
        <div
            class="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-xs font-bold text-white"
        >
            glt
        </div>
        <span class="text-sm font-semibold tracking-tight">GridLoraTester</span>
    </div>

    <nav class="flex-1 space-y-0.5 p-2">
        {#each items as item (item.href)}
            <a
                href={item.href}
                class="nav-link"
                class:nav-link-active={isActive(item.href)}
            >
                <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.75"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    aria-hidden="true"
                >
                    <path d={item.icon} />
                </svg>
                <span>{item.label}</span>
            </a>
        {/each}
    </nav>

    <div class="border-t border-border p-3 text-xs text-fg-faint">v0.1.0</div>
</aside>
