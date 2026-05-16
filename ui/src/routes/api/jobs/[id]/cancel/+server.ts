import type { RequestHandler } from './$types';
import { error, json } from '@sveltejs/kit';
import { cancel } from '$lib/server/jobs';

export const POST: RequestHandler = ({ params }) => {
    const id = Number(params.id);
    if (!Number.isFinite(id) || id <= 0) throw error(400, 'Bad id');
    const ok = cancel(id);
    return json({ ok });
};
