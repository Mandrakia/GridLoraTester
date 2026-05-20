import { fail, redirect } from '@sveltejs/kit';
import {
    AUTH_ENABLED,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    passwordMatches,
    safeRedirect,
    sessionToken
} from '$lib/server/auth';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = ({ locals, url }) => {
    // Nothing to log into when no gate is configured.
    if (!AUTH_ENABLED) throw redirect(303, '/');
    // Already signed in → bounce to wherever they were headed.
    if (locals.authed) throw redirect(303, safeRedirect(url.searchParams.get('redirectTo')));
    return {};
};

export const actions: Actions = {
    default: async ({ request, cookies, url }) => {
        if (!AUTH_ENABLED) throw redirect(303, '/');

        const data = await request.formData();
        const password = String(data.get('password') ?? '');

        if (!passwordMatches(password)) {
            return fail(401, { error: 'Incorrect password.' });
        }

        cookies.set(SESSION_COOKIE, sessionToken(), {
            path: '/',
            httpOnly: true,
            sameSite: 'lax',
            maxAge: SESSION_MAX_AGE,
            // Explicit false: SvelteKit defaults `secure` to true for non-
            // localhost hosts, but the dashboard is routinely reached over
            // plain http on a LAN IP, where a Secure cookie would be dropped
            // and login would silently fail. (Over the RunPod https proxy a
            // non-Secure cookie is still sent fine.)
            secure: false
        });

        throw redirect(303, safeRedirect(url.searchParams.get('redirectTo')));
    }
};
