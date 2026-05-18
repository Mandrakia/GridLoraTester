// Kicks off the Google OAuth consent flow. Builds the auth URL with a
// CSRF state (random + 5-min cookie), redirects the user's browser to
// Google. The callback below validates state and exchanges the code.
import { redirect } from '@sveltejs/kit';
import { randomBytes } from 'node:crypto';
import { buildAuthUrl, loadClient } from '$lib/server/connectors/google-oauth';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url, cookies }) => {
    if (!loadClient()) {
        return new Response(
            'Google Photos connector is not configured. ' +
                'Set GLT_GOOGLE_CLIENT_SECRET to a Google Cloud OAuth client JSON file, ' +
                'or place it at <glt_root>/config/google-client-secret.json.',
            { status: 503, headers: { 'content-type': 'text/plain; charset=utf-8' } }
        );
    }
    const state = randomBytes(16).toString('hex');
    cookies.set('glt_google_oauth_state', state, {
        path: '/',
        httpOnly: true,
        sameSite: 'lax',
        maxAge: 300
    });
    // Mirror the callback URL the route below registers. We need the
    // current origin (could be http://localhost:5173, :5273, or whatever
    // SvelteKit landed on); url.origin reflects the actual port.
    const redirectUri = `${url.origin}/connectors/google-photos/oauth/callback`;
    const auth = buildAuthUrl(redirectUri, state);
    if (!auth) {
        return new Response('Could not build auth URL.', { status: 500 });
    }
    throw redirect(302, auth);
};
