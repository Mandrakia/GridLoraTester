import { AUTH_ENABLED } from '$lib/server/auth';
import type { LayoutServerLoad } from './$types';

// Expose whether the password gate is on so the sidebar can show a Log out
// control only when there's a session to end.
export const load: LayoutServerLoad = () => {
    return { authEnabled: AUTH_ENABLED };
};
