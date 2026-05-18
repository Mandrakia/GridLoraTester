// GET: thin proxy over Google's sessions.get so the front-end can poll
// without holding Google credentials. We don't cache — the
// pollInterval/timeoutIn that come back drive the front-end's cadence.
import { json } from '@sveltejs/kit';
import { getPickerSession } from '$lib/server/connectors/google-photos';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params }) => {
    const session = await getPickerSession(params.id);
    if (!session) return json({ error: 'session not found or expired' }, { status: 404 });
    return json({ session });
};
