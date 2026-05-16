// Immich connector. Credentials: { base_url, api_key }. All requests carry
// the `x-api-key` header. Thumbnails are auth-gated upstream so we expose
// them through our own proxy route (see proxyFetch + the +server.ts under
// /connectors/immich/thumb).
import type {
    ConnectorPerson,
    ConnectorPicture,
    ConnectorSignInResult,
    ConnectorTypeInfo,
    ListPicturesOpts,
    ListPicturesPage,
    PhotoConnector
} from '$lib/connectors/types';
import { getCredentials, saveCredentials, setStatus } from './credentials';

export const IMMICH_TYPE_INFO: ConnectorTypeInfo = {
    id: 'immich',
    label: 'Immich',
    linker_kind: 'persons',
    needs_credentials: true,
    credentials_fields: [
        {
            key: 'base_url',
            label: 'Server URL',
            type: 'url',
            placeholder: 'https://immich.example.com',
            help: 'Base URL of the Immich server, no trailing slash.',
            required: true
        },
        {
            key: 'api_key',
            label: 'API key',
            type: 'password',
            placeholder: 'sk_…',
            help: 'Create one in Immich → Account Settings → API Keys.',
            required: true
        }
    ]
};

interface ImmichCreds {
    base_url: string;
    api_key: string;
    // Index signature lets the struct flow through saveCredentials() without
    // a cast at every call site. Concrete fields above keep the type-safety
    // we care about; the bag is what's persisted.
    [k: string]: unknown;
}

function normalizeBaseUrl(raw: unknown): string {
    if (typeof raw !== 'string') return '';
    return raw.trim().replace(/\/+$/, '');
}

function readCreds(): ImmichCreds | null {
    const row = getCredentials('immich');
    if (!row) return null;
    const base_url = normalizeBaseUrl(row.credentials.base_url);
    const api_key = typeof row.credentials.api_key === 'string' ? row.credentials.api_key : '';
    if (!base_url || !api_key) return null;
    return { base_url, api_key };
}

async function immichFetch(
    creds: ImmichCreds,
    path: string,
    init: RequestInit = {}
): Promise<Response> {
    const headers = new Headers(init.headers ?? {});
    headers.set('x-api-key', creds.api_key);
    headers.set('Accept', 'application/json');
    const url = path.startsWith('http') ? path : `${creds.base_url}${path}`;
    return fetch(url, { ...init, headers });
}

export class ImmichConnector implements PhotoConnector {
    readonly id = 'immich' as const;
    readonly label = 'Immich';

    async isSignedIn(): Promise<boolean> {
        // Cheap status read — does NOT hit the network. The persisted
        // `status='signed_in'` is set by `signIn()` (the only method that
        // actually validates the API key against Immich). Use signIn again
        // when you want to re-check.
        const row = getCredentials('immich');
        return row?.status === 'signed_in';
    }

    async signIn(input: Record<string, unknown>): Promise<ConnectorSignInResult> {
        const base_url = normalizeBaseUrl(input.base_url);
        const api_key = typeof input.api_key === 'string' ? input.api_key.trim() : '';
        if (!base_url || !api_key) {
            return { ok: false, error: 'Both server URL and API key are required.' };
        }
        const creds: ImmichCreds = { base_url, api_key };

        // Validate against the endpoint we'll actually use (listPersons →
        // /api/people). That way:
        //   - the key is exercised on the same permission we need at runtime
        //     (`person.read`), so we surface a "scope missing" error here
        //     instead of crashing later in listPersons
        //   - we don't require unrelated scopes like `user.read` just to
        //     light up the green pill
        try {
            const res = await immichFetch(creds, '/api/people?size=1');
            if (res.ok) {
                saveCredentials('immich', creds, 'signed_in', null);
                return { ok: true };
            }
            const body = (await res.text().catch(() => '')).slice(0, 200);
            let msg: string;
            if (res.status === 401) {
                msg = `Immich rejected the API key (HTTP 401).${body ? ` ${body}` : ''}`;
            } else if (res.status === 403) {
                // 403 = key is valid but lacks the required scope. Immich
                // expects `person.read` on this endpoint.
                msg = `API key lacks the 'person.read' permission (HTTP 403). Add it on the Immich API key page.`;
            } else if (res.status === 404) {
                msg = `Endpoint /api/people not found (HTTP 404). Check the server URL or upgrade Immich.`;
            } else {
                msg = `Validation failed: HTTP ${res.status}${body ? ` — ${body}` : ''}`;
            }
            saveCredentials('immich', creds, 'error', msg);
            return { ok: false, error: msg };
        } catch (e) {
            const msg = `Could not reach ${base_url}: ${(e as Error).message}`;
            saveCredentials('immich', creds, 'error', msg);
            return { ok: false, error: msg };
        }
    }

    async signOut(): Promise<void> {
        // The original API key keeps working on the Immich side until the
        // user revokes it there. We just forget it locally.
        const row = getCredentials('immich');
        if (row) {
            // Clear creds + status. Using save with empty rather than delete
            // so the UI can still show "configured but signed out" — but for
            // now, delete is simpler.
            const { deleteCredentials } = await import('./credentials');
            deleteCredentials('immich');
        }
    }

    async listPersons(): Promise<ConnectorPerson[]> {
        const creds = readCreds();
        if (!creds) throw new Error('Immich is not configured.');
        // Immich pages people by `page=1..N`. We fetch until empty — most
        // libraries fit in a single page (200 items).
        const out: ConnectorPerson[] = [];
        let page = 1;
        const pageSize = 200;
        while (true) {
            const res = await immichFetch(
                creds,
                `/api/people?withHidden=false&page=${page}&size=${pageSize}`
            );
            if (!res.ok) throw new Error(`Immich /people failed: HTTP ${res.status}`);
            const body = (await res.json()) as {
                people?: Array<{ id: string; name: string; thumbnailPath?: string }>;
                hasNextPage?: boolean;
            };
            const people = body.people ?? [];
            for (const p of people) {
                out.push({
                    id: p.id,
                    name: p.name?.trim() || '(unnamed)',
                    // Always go through our proxy so the browser doesn't
                    // need the x-api-key header. The trailing `/thumbnail`
                    // matters — without it the proxy hits the Person JSON
                    // endpoint instead of the image.
                    thumbnail_url: `/connectors/immich/thumb/people/${p.id}/thumbnail`
                });
            }
            if (!body.hasNextPage || people.length < pageSize) break;
            page++;
        }
        // Stable order, named first.
        out.sort((a, b) => {
            const an = a.name === '(unnamed)' ? 1 : 0;
            const bn = b.name === '(unnamed)' ? 1 : 0;
            if (an !== bn) return an - bn;
            return a.name.localeCompare(b.name);
        });
        setStatus('immich', 'signed_in', null);
        return out;
    }

    async listPictures(personId: string, opts: ListPicturesOpts = {}): Promise<ListPicturesPage> {
        const creds = readCreds();
        if (!creds) throw new Error('Immich is not configured.');
        const limit = Math.min(Math.max(opts.limit ?? 250, 1), 1000);
        // Immich's search/metadata endpoint accepts a personIds filter and
        // returns paginated metadata-rich results.
        const body = {
            personIds: [personId],
            withExif: true,
            page: opts.cursor ? Number(opts.cursor) : 1,
            size: limit
        };
        const res = await immichFetch(creds, '/api/search/metadata', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(`Immich /search/metadata failed: HTTP ${res.status}`);
        const json = (await res.json()) as {
            assets?: {
                items?: Array<{
                    id: string;
                    originalFileName?: string;
                    fileCreatedAt?: string;
                    originalMimeType?: string;
                    exifInfo?: { exifImageWidth?: number; exifImageHeight?: number };
                }>;
                total?: number;
                count?: number;
                nextPage?: number | null;
            };
        };
        const items = json.assets?.items ?? [];
        const pictures: ConnectorPicture[] = items.map((a) => ({
            id: a.id,
            filename: a.originalFileName ?? `${a.id}.bin`,
            // Both download_url and thumbnail_url go through our proxy so
            // the browser never has to know about x-api-key.
            download_url: `/connectors/immich/thumb/assets/${a.id}/original`,
            thumbnail_url: `/connectors/immich/thumb/assets/${a.id}/thumbnail`,
            created_date: a.fileCreatedAt ?? '',
            width: a.exifInfo?.exifImageWidth ?? 0,
            height: a.exifInfo?.exifImageHeight ?? 0,
            mime_type: a.originalMimeType
        }));
        const nextCursor = json.assets?.nextPage ? String(json.assets.nextPage) : null;
        return { pictures, nextCursor };
    }

    async downloadPicture(picture: ConnectorPicture): Promise<Buffer> {
        const res = await this.proxyFetch(picture.download_url);
        if (!res.ok) throw new Error(`Download failed: HTTP ${res.status}`);
        const ab = await res.arrayBuffer();
        return Buffer.from(ab);
    }

    /** Proxy a connector-managed URL (relative `/connectors/immich/...` or
     * an absolute Immich endpoint) with the auth header injected. Used by
     * the /thumb proxy and by downloadPicture. */
    async proxyFetch(upstreamUrl: string): Promise<Response> {
        const creds = readCreds();
        if (!creds) throw new Error('Immich is not configured.');

        // Translate our proxy paths back into real Immich endpoints.
        // Everything under /connectors/immich/thumb/<rest> maps to
        // <base_url>/api/<rest>.
        let target = upstreamUrl;
        const prefix = '/connectors/immich/thumb/';
        if (target.startsWith(prefix)) {
            target = `/api/${target.slice(prefix.length)}`;
        }
        return immichFetch(creds, target);
    }
}
