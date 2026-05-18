// Google Photos connector — backed by the Picker API.
//
// The OSS Photos APIs no longer expose the user's full library to apps;
// access happens through PICK sessions: we POST a session, the user
// picks photos in Google's own UI (separate tab), we poll until they're
// done, then we list and download the selected items.
//
// Once the bytes land in our local cache, every downstream concern (face
// detect, dedup hash, thumbnail proxy) reuses the local-folder pipeline
// — see ./local-folder.ts. The cache dir is the link's `person_id`.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

import type {
    ConnectorPicture,
    ConnectorSignInResult,
    ConnectorTypeInfo,
    ListPicturesOpts,
    ListPicturesPage,
    PhotoConnector
} from '$lib/connectors/types';
import { getCredentials, saveCredentials } from './credentials';
import { getAccessToken, type OAuthTokens } from './google-oauth';
import {
    downloadFromDir,
    isInsideAnyRoot,
    listLinkedFolders,
    listPicturesInDir,
    proxyServeFromDir
} from './local-folder';

export const GOOGLE_PHOTOS_TYPE_INFO: ConnectorTypeInfo = {
    id: 'google-photos',
    label: 'Google Photos',
    linker_kind: 'picker',
    needs_credentials: true,
    // No credentials form: Settings redirects the user to the OAuth flow,
    // and the callback persists the refresh_token into connector_credentials.
    credentials_fields: [],
    oauth_start_url: '/connectors/google-photos/oauth/start'
};

const THUMB_PREFIX = '/connectors/google-photos/thumb/';
const PICKER_API = 'https://photospicker.googleapis.com/v1';

function cacheRoot(): string {
    return process.env.GLT_CACHE_DIR
        ? resolve(process.env.GLT_CACHE_DIR)
        : join(homedir(), '.cache', 'glt');
}

/** Stable cache subdir per (scope_kind, scope_key). Opaque sha256 prefix
 * so folder renames don't strand the cache. Created on demand. */
export function cacheDirFor(scope_kind: string, scope_key: string): string {
    const h = createHash('sha256').update(`${scope_kind}:${scope_key}`).digest('hex').slice(0, 12);
    return join(cacheRoot(), 'google-photos', `${scope_kind}-${h}`);
}

export function ensureCacheDir(dir: string): void {
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function allowed(abs: string): boolean {
    return isInsideAnyRoot(abs, listLinkedFolders(['google-photos']));
}

/** ---- Picker API session helpers ---- */

export interface PickerSession {
    id: string;
    pickerUri: string;
    pollingConfig?: { pollInterval?: string; timeoutIn?: string };
    expireTime?: string;
    mediaItemsSet?: boolean;
}

export interface PickedMediaItem {
    id: string;
    createTime?: string;
    type?: string;
    mediaFile?: {
        baseUrl: string;
        mimeType: string;
        filename?: string;
        mediaFileMetadata?: {
            width?: string;
            height?: string;
            cameraMake?: string;
            cameraModel?: string;
            photoMetadata?: unknown;
        };
    };
}

async function authedFetch(token: string, url: string, init?: RequestInit): Promise<Response> {
    return fetch(url, {
        ...init,
        headers: {
            ...(init?.headers ?? {}),
            Authorization: `Bearer ${token}`
        }
    });
}

/** Persist any token refresh side-effects (access_token + expires_at
 * mutated by getAccessToken) back to the credentials row. */
function persistTokens(creds: Partial<OAuthTokens>): void {
    saveCredentials('google-photos', creds, 'signed_in', null);
}

async function token(): Promise<string | null> {
    const row = getCredentials('google-photos');
    if (!row) return null;
    const creds = row.credentials as Partial<OAuthTokens>;
    const t = await getAccessToken(creds);
    if (!t) return null;
    persistTokens(creds);
    return t;
}

export async function createPickerSession(maxItemCount?: number): Promise<PickerSession | null> {
    const t = await token();
    if (!t) return null;
    const body: Record<string, unknown> = {};
    if (maxItemCount && maxItemCount > 0) {
        body.pickingConfig = { maxItemCount: String(maxItemCount) };
    }
    const r = await authedFetch(t, `${PICKER_API}/sessions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!r.ok) return null;
    return (await r.json()) as PickerSession;
}

export async function getPickerSession(id: string): Promise<PickerSession | null> {
    const t = await token();
    if (!t) return null;
    const r = await authedFetch(t, `${PICKER_API}/sessions/${encodeURIComponent(id)}`);
    if (!r.ok) return null;
    return (await r.json()) as PickerSession;
}

export async function listPickedMediaItems(sessionId: string): Promise<PickedMediaItem[]> {
    const t = await token();
    if (!t) return [];
    const out: PickedMediaItem[] = [];
    let pageToken: string | undefined;
    do {
        const url = new URL(`${PICKER_API}/mediaItems`);
        url.searchParams.set('sessionId', sessionId);
        url.searchParams.set('pageSize', '100');
        if (pageToken) url.searchParams.set('pageToken', pageToken);
        const r = await authedFetch(t, url.toString());
        if (!r.ok) break;
        const j = (await r.json()) as {
            mediaItems?: PickedMediaItem[];
            nextPageToken?: string;
        };
        if (j.mediaItems) out.push(...j.mediaItems);
        pageToken = j.nextPageToken;
    } while (pageToken);
    return out;
}

export async function deletePickerSession(id: string): Promise<void> {
    const t = await token();
    if (!t) return;
    await authedFetch(t, `${PICKER_API}/sessions/${encodeURIComponent(id)}`, {
        method: 'DELETE'
    });
}

/** Fetches the original-quality bytes for one picked item. Caller must
 * call within 60 minutes of the picker session ending (Google's TTL on
 * baseUrls). Returns null on failure (TTL expired, network, etc.). */
export async function downloadPickedBytes(item: PickedMediaItem): Promise<Buffer | null> {
    const t = await token();
    if (!t || !item.mediaFile?.baseUrl) return null;
    const url = `${item.mediaFile.baseUrl}=d`;
    const r = await authedFetch(t, url);
    if (!r.ok) return null;
    return Buffer.from(await r.arrayBuffer());
}

/** Picks a file extension from the mimeType, fallback to '.jpg'. */
export function extFor(mimeType: string | undefined): string {
    if (!mimeType) return '.jpg';
    if (mimeType.includes('jpeg')) return '.jpg';
    if (mimeType.includes('png')) return '.png';
    if (mimeType.includes('webp')) return '.webp';
    if (mimeType.includes('heic')) return '.heic';
    if (mimeType.includes('gif')) return '.gif';
    if (mimeType.includes('bmp')) return '.bmp';
    return '.jpg';
}

/** Writes a picked item's bytes to its <id>.<ext> file in `dir`. Skips
 * silently if the file already exists (idempotent re-pick). Returns
 * the absolute path written (or pre-existing), or null on failure. */
export async function saveItemToCache(
    item: PickedMediaItem,
    dir: string
): Promise<string | null> {
    const ext = extFor(item.mediaFile?.mimeType);
    const dest = join(dir, `${item.id}${ext}`);
    if (existsSync(dest)) return dest;
    const bytes = await downloadPickedBytes(item);
    if (!bytes) return null;
    writeFileSync(dest, bytes);
    return dest;
}

/** ---- PhotoConnector contract ---- */

export class GooglePhotosConnector implements PhotoConnector {
    readonly id = 'google-photos' as const;
    readonly label = 'Google Photos';

    async isSignedIn(): Promise<boolean> {
        const row = getCredentials('google-photos');
        if (!row) return false;
        const creds = row.credentials as Partial<OAuthTokens>;
        return Boolean(creds.refresh_token);
    }

    async signIn(credentials: Record<string, unknown>): Promise<ConnectorSignInResult> {
        // signIn here is only ever called by the OAuth callback after a
        // successful code exchange. The route writes the full OAuthTokens
        // bag as `credentials` — we just confirm shape + persist via the
        // normal saveCredentials path.
        const refresh_token = credentials.refresh_token;
        if (typeof refresh_token !== 'string' || !refresh_token) {
            return { ok: false, error: 'missing refresh_token (OAuth flow incomplete)' };
        }
        // Validate by minting an access token.
        const t = await getAccessToken({ ...(credentials as Partial<OAuthTokens>) });
        if (!t) return { ok: false, error: 'refresh_token rejected by Google' };
        saveCredentials('google-photos', credentials, 'signed_in', null);
        return { ok: true };
    }

    async signOut(): Promise<void> {
        // Credentials row stays in place; callers (the link-remove flow)
        // delete it explicitly if they want a full purge. We could revoke
        // the refresh_token here via Google's revoke endpoint — left for
        // a follow-up.
    }

    async listPersons() {
        // 'picker' kind connectors don't surface persons.
        return [];
    }

    async listPictures(cacheDir: string, _opts: ListPicturesOpts = {}): Promise<ListPicturesPage> {
        return listPicturesInDir(cacheDir, {
            allowed,
            thumbUrlFor: (enc) => `${THUMB_PREFIX}${enc}`
        });
    }

    async downloadPicture(picture: ConnectorPicture): Promise<Buffer> {
        return downloadFromDir(picture, allowed);
    }

    async proxyFetch(upstreamUrl: string): Promise<Response> {
        return proxyServeFromDir(upstreamUrl, THUMB_PREFIX, allowed);
    }
}
