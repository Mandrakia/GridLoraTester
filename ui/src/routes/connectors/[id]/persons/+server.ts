// JSON listing of every person inside a connector. Used by the link modal
// to render the person picker — we don't pre-load this on every /datasets
// hit because it can be slow (hundreds of persons over the network).
import type { RequestHandler } from './$types';
import { error, json } from '@sveltejs/kit';
import { ALL_CONNECTORS, getConnector } from '$lib/server/connectors/registry';
import type { ConnectorId } from '$lib/connectors/types';

export const GET: RequestHandler = async ({ params }) => {
    const id = params.id;
    if (!ALL_CONNECTORS.some((c) => c.id === id)) throw error(404, 'Unknown connector');
    const connector = getConnector(id as ConnectorId);
    try {
        const persons = await connector.listPersons();
        return json({ persons });
    } catch (e) {
        // 502 — upstream connector failure (not configured, network, scope, …)
        return json({ error: (e as Error).message }, { status: 502 });
    }
};
