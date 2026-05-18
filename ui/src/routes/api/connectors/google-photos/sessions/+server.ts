// POST: create a new Google Photos picker session. Returns the pickerUri
// the front-end should open in a new tab + the polling cadence Google
// suggests so we don't hammer their API.
import { json } from '@sveltejs/kit';
import { createPickerSession } from '$lib/server/connectors/google-photos';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
    let body: { maxItemCount?: number } = {};
    try {
        body = await request.json();
    } catch {
        // empty body OK
    }
    const session = await createPickerSession(body.maxItemCount);
    if (!session) {
        return json(
            {
                error:
                    'Could not create picker session — Google Photos connector not signed in or token rejected.'
            },
            { status: 400 }
        );
    }
    return json({ session });
};
