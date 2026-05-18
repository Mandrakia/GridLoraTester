// OAuth callback: Google redirects here with ?code=&state=. We validate
// the state cookie set by /start, exchange the code for tokens, persist
// them via the GooglePhotosConnector.signIn path, then bounce the user
// back to /settings.
import { redirect } from '@sveltejs/kit';
import { exchangeCode } from '$lib/server/connectors/google-oauth';
import { getConnector } from '$lib/server/connectors/registry';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url, cookies }) => {
    const code = url.searchParams.get('code');
    const state = url.searchParams.get('state');
    const stateCookie = cookies.get('glt_google_oauth_state');
    cookies.delete('glt_google_oauth_state', { path: '/' });

    if (!code || !state || !stateCookie || state !== stateCookie) {
        throw redirect(302, '/settings?google_oauth=state_mismatch');
    }
    const redirectUri = `${url.origin}/connectors/google-photos/oauth/callback`;
    const tokens = await exchangeCode(code, redirectUri);
    if (!tokens) throw redirect(302, '/settings?google_oauth=exchange_failed');

    const res = await getConnector('google-photos').signIn(
        tokens as unknown as Record<string, unknown>
    );
    if (!res.ok) throw redirect(302, '/settings?google_oauth=signin_failed');

    throw redirect(302, '/settings?google_oauth=ok');
};
