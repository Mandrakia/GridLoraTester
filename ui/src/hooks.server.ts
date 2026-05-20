import type { Handle } from '@sveltejs/kit';
import { AUTH_ENABLED, SESSION_COOKIE, isValidSession } from '$lib/server/auth';

// Site-wide password gate (active only when GLT_PASSWORD is set — see
// $lib/server/auth). Runs on every dynamic request:
//   - authenticated, or /login // /logout      → pass through
//   - unauthenticated page navigation (GET doc) → 303 redirect to /login
//   - any other unauthenticated request         → 401
//
// Static client assets (/_app/**, favicon) are served by adapter-node BEFORE
// hooks run, so they stay public — which is fine: a browser only requests them
// after loading a page, and every page is gated, so an unauthenticated visitor
// never gets that far. A direct asset hit just serves public, source-available
// bundle code; all real data (pages, /api/**, the /connectors/** and
// /tests/output/** image routes) is dynamic and gated here.

const PUBLIC_PATHS = new Set(['/login', '/logout']);

export const handle: Handle = async ({ event, resolve }) => {
    if (!AUTH_ENABLED) {
        event.locals.authed = true;
        return resolve(event);
    }

    const authed = isValidSession(event.cookies.get(SESSION_COOKIE));
    event.locals.authed = authed;

    if (authed || PUBLIC_PATHS.has(event.url.pathname)) {
        return resolve(event);
    }

    const accept = event.request.headers.get('accept') ?? '';
    const isDocument =
        event.request.method === 'GET' &&
        (event.request.headers.get('sec-fetch-dest') === 'document' ||
            accept.includes('text/html'));

    if (isDocument) {
        const redirectTo = event.url.pathname + event.url.search;
        return new Response(null, {
            status: 303,
            headers: { location: `/login?redirectTo=${encodeURIComponent(redirectTo)}` }
        });
    }

    return new Response('Authentication required.\n', {
        status: 401,
        headers: { 'content-type': 'text/plain; charset=utf-8' }
    });
};
