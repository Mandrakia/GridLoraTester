import { redirect } from '@sveltejs/kit';
import { SESSION_COOKIE } from '$lib/server/auth';
import type { RequestHandler } from './$types';

// Clear the session and return to /login. POST-only so a stray <img>/link
// can't log the user out. The form lives in the sidebar.
export const POST: RequestHandler = ({ cookies }) => {
    // Match the attributes used when setting it (notably secure:false) so the
    // browser reliably clears the cookie.
    cookies.delete(SESSION_COOKIE, { path: '/', secure: false });
    throw redirect(303, '/login');
};
