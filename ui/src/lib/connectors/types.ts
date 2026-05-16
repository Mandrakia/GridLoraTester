// Photo-DB connectors (Immich, Google Photos, …) — pure types, no runtime
// behavior. Server-side implementations live under $lib/server/connectors;
// the client only sees DTOs that mirror these.

/** Stable identifier per connector type. Add new ones as we wire them. */
export type ConnectorId = 'immich' | 'google-photos' | 'hard-drive';

/** How the link UI should look. 'persons' = pick from a list returned by
 * listPersons (Immich, Google Photos, …); 'folder' = ask the user for a
 * filesystem path that becomes the "person_id" (hard-drive). */
export type LinkerKind = 'persons' | 'folder';

/** Per-connector metadata for the registry. */
export interface ConnectorTypeInfo {
    id: ConnectorId;
    label: string;
    /** Drives the link modal's shape. */
    linker_kind: LinkerKind;
    /** When false, the connector has no per-instance setup — Settings
     * never lists it, and it's always available in the link UI. */
    needs_credentials: boolean;
    /** Form fields the Settings UI should render for sign-in. Empty/absent
     * when needs_credentials is false. */
    credentials_fields?: CredentialField[];
}

export interface CredentialField {
    key: string;
    label: string;
    type: 'text' | 'password' | 'url';
    placeholder?: string;
    help?: string;
    required?: boolean;
}

export interface ConnectorPerson {
    id: string;
    name: string;
    /** Browser-loadable URL. Connectors that need auth headers expose their
     * thumbnails via our proxy at /connectors/<id>/thumb/... */
    thumbnail_url: string;
    picture_count?: number | null;
}

export interface ConnectorPicture {
    id: string;
    filename: string;
    /** URL the server uses to fetch the bytes. Not necessarily browser-
     * reachable — the connector handles auth on this URL. */
    download_url: string;
    /** Browser-loadable thumbnail (proxied when the upstream needs auth). */
    thumbnail_url?: string;
    /** ISO 8601 datetime. */
    created_date: string;
    width: number;
    height: number;
    mime_type?: string;
}

/** Result of any auth/credential-validating operation. */
export interface ConnectorSignInResult {
    ok: boolean;
    error?: string;
}

export interface ListPicturesOpts {
    /** Opaque pagination cursor — let the connector decide its shape. */
    cursor?: string | null;
    /** Max items per page; the connector may clamp lower. */
    limit?: number;
}

export interface ListPicturesPage {
    pictures: ConnectorPicture[];
    /** When set, pass back into `cursor` to fetch the next page. */
    nextCursor?: string | null;
}

/** Server-side connector contract. Every concrete connector implements this. */
export interface PhotoConnector {
    readonly id: ConnectorId;
    readonly label: string;

    isSignedIn(): Promise<boolean>;
    signIn(credentials: Record<string, unknown>): Promise<ConnectorSignInResult>;
    signOut(): Promise<void>;

    listPersons(): Promise<ConnectorPerson[]>;
    listPictures(personId: string, opts?: ListPicturesOpts): Promise<ListPicturesPage>;

    /** Fetch the raw bytes of a picture (already auth'd by the connector).
     * Returns a Buffer the caller can write to disk or hash. */
    downloadPicture(picture: ConnectorPicture): Promise<Buffer>;

    /** Proxy fetch for a connector-specific URL needing auth. Used by our
     * /connectors/<id>/thumb/... route to forward thumbnails to the browser
     * without exposing credentials. */
    proxyFetch(upstreamUrl: string): Promise<Response>;
}

/** Public-safe snapshot of one configured connector (no credentials). */
export interface ConnectorStatus {
    id: ConnectorId;
    label: string;
    configured: boolean;
    signed_in: boolean;
    last_check_at: string | null;
    last_error: string | null;
}
