// OAuth 2.0 helpers for the Google Photos Picker. The client_secret JSON
// (the "installed" Desktop client downloaded from Google Cloud Console)
// is loaded from either:
//   - the GLT_GOOGLE_CLIENT_SECRET env var (a file path), or
//   - <glt_root>/config/google-client-secret.json (default location)
//
// We never ship a client_secret with the OSS distribution — every user
// creates their own OAuth client. The redirect URI we register at runtime
// is `http://localhost:<dev-port>/connectors/google-photos/oauth/callback`;
// Google's "installed" client type allows any loopback port, so this
// works for whatever port the SvelteKit dev server happens to bind.
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { gltRoot } from '../settings';

const TOKEN_URI = 'https://oauth2.googleapis.com/token';
const AUTH_URI = 'https://accounts.google.com/o/oauth2/auth';
export const PICKER_SCOPE = 'https://www.googleapis.com/auth/photospicker.mediaitems.readonly';

interface InstalledClient {
    client_id: string;
    client_secret: string;
    auth_uri: string;
    token_uri: string;
    redirect_uris: string[];
}

interface ClientSecretFile {
    installed?: InstalledClient;
    web?: InstalledClient;
}

let cached: InstalledClient | null | undefined;

/** Loads the OAuth client_secret JSON, supporting both "installed" and
 * "web" client types (we treat them the same — what matters is the
 * id+secret pair). Returns null when neither env var nor default file
 * is set; callers should surface that as "configure GLT_GOOGLE_CLIENT_SECRET
 * or place the JSON at <glt_root>/config/google-client-secret.json". */
export function loadClient(): InstalledClient | null {
    if (cached !== undefined) return cached;
    const candidates: string[] = [];
    if (process.env.GLT_GOOGLE_CLIENT_SECRET) {
        candidates.push(resolve(process.env.GLT_GOOGLE_CLIENT_SECRET));
    }
    candidates.push(join(gltRoot(), 'config', 'google-client-secret.json'));

    for (const path of candidates) {
        if (!existsSync(path)) continue;
        try {
            const raw = JSON.parse(readFileSync(path, 'utf-8')) as ClientSecretFile;
            const c = raw.installed ?? raw.web;
            if (c?.client_id && c?.client_secret) {
                cached = c;
                return c;
            }
        } catch {
            // ignore — fall through to next candidate
        }
    }
    cached = null;
    return null;
}

/** Drops the cached client_secret so a freshly-edited config file is
 * picked up without restarting the dashboard. */
export function reloadClient(): void {
    cached = undefined;
}

export interface OAuthTokens {
    /** Long-lived refresh token. Persisted in connector_credentials. */
    refresh_token: string;
    /** Short-lived access token (~1h). Refreshed on demand. */
    access_token: string;
    /** Epoch ms when the access_token expires. */
    expires_at: number;
    /** Scope(s) granted. Should always include PICKER_SCOPE. */
    scope: string;
}

/** Build the URL we redirect the user to for consent. `redirectUri` must
 * match one of the OAuth client's registered URIs — for "installed"
 * clients with the default `http://localhost` entry, any loopback URI
 * (any port, any path) is accepted by Google. */
export function buildAuthUrl(redirectUri: string, state: string): string | null {
    const c = loadClient();
    if (!c) return null;
    const params = new URLSearchParams({
        client_id: c.client_id,
        redirect_uri: redirectUri,
        response_type: 'code',
        scope: PICKER_SCOPE,
        access_type: 'offline',
        // 'consent' forces the refresh_token to be re-issued on every
        // sign-in; without it, repeat consents return no refresh_token
        // and we'd lose the durable credential.
        prompt: 'consent',
        state
    });
    return `${AUTH_URI}?${params.toString()}`;
}

/** Exchange a one-time auth code for the initial refresh_token +
 * access_token pair. Returns null on any failure (caller surfaces). */
export async function exchangeCode(
    code: string,
    redirectUri: string
): Promise<OAuthTokens | null> {
    const c = loadClient();
    if (!c) return null;
    const body = new URLSearchParams({
        code,
        client_id: c.client_id,
        client_secret: c.client_secret,
        redirect_uri: redirectUri,
        grant_type: 'authorization_code'
    });
    const r = await fetch(TOKEN_URI, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body
    });
    if (!r.ok) return null;
    const j = (await r.json()) as {
        access_token?: string;
        refresh_token?: string;
        expires_in?: number;
        scope?: string;
    };
    if (!j.access_token || !j.refresh_token) return null;
    return {
        refresh_token: j.refresh_token,
        access_token: j.access_token,
        expires_at: Date.now() + (j.expires_in ?? 3600) * 1000,
        scope: j.scope ?? PICKER_SCOPE
    };
}

/** Mint a fresh access_token from the saved refresh_token. */
export async function refreshAccessToken(
    refreshToken: string
): Promise<{ access_token: string; expires_at: number } | null> {
    const c = loadClient();
    if (!c) return null;
    const body = new URLSearchParams({
        client_id: c.client_id,
        client_secret: c.client_secret,
        refresh_token: refreshToken,
        grant_type: 'refresh_token'
    });
    const r = await fetch(TOKEN_URI, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body
    });
    if (!r.ok) return null;
    const j = (await r.json()) as { access_token?: string; expires_in?: number };
    if (!j.access_token) return null;
    return {
        access_token: j.access_token,
        expires_at: Date.now() + (j.expires_in ?? 3600) * 1000
    };
}

/** Returns a valid access_token, refreshing it transparently when the
 * cached one has < 60s left. Stored credentials are mutated in place so
 * the caller can persist the updated `expires_at` / `access_token`. */
export async function getAccessToken(creds: Partial<OAuthTokens>): Promise<string | null> {
    if (
        creds.access_token &&
        typeof creds.expires_at === 'number' &&
        creds.expires_at - Date.now() > 60_000
    ) {
        return creds.access_token;
    }
    if (!creds.refresh_token) return null;
    const fresh = await refreshAccessToken(creds.refresh_token);
    if (!fresh) return null;
    creds.access_token = fresh.access_token;
    creds.expires_at = fresh.expires_at;
    return fresh.access_token;
}
