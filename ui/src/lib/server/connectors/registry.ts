// Connector registry: maps an id to its factory and metadata. Adding a new
// connector = add an entry here. The settings UI reads
// ALL_CONNECTORS_WITH_CREDENTIALS to decide which forms to render; the
// link UI on /datasets reads listAvailableConnectors() so always-available
// connectors (no credentials) show up too.
import type {
    ConnectorId,
    ConnectorStatus,
    ConnectorTypeInfo,
    PhotoConnector
} from '$lib/connectors/types';
import { listCredentials } from './credentials';
import { HardDriveConnector, HARD_DRIVE_TYPE_INFO } from './hard-drive';
import { ImmichConnector, IMMICH_TYPE_INFO } from './immich';

export const ALL_CONNECTORS: ConnectorTypeInfo[] = [
    IMMICH_TYPE_INFO,
    HARD_DRIVE_TYPE_INFO
    // Google Photos to be added in a follow-up phase.
];

/** Types that need a credentials form in /settings. Hard-drive is excluded
 * because its "config" lives per link, not per connector instance. */
export const CREDENTIAL_BACKED_CONNECTORS = ALL_CONNECTORS.filter(
    (c) => c.needs_credentials
);

const factories: Record<ConnectorId, () => PhotoConnector> = {
    immich: () => new ImmichConnector(),
    'hard-drive': () => new HardDriveConnector(),
    'google-photos': () => {
        throw new Error('Google Photos connector is not implemented yet.');
    }
};

export function getConnector(id: ConnectorId): PhotoConnector {
    const f = factories[id];
    if (!f) throw new Error(`Unknown connector id: ${id}`);
    return f();
}

/** Snapshot of every connector with a credentials row (signed-in, error,
 * etc.). Used by /settings to show the configured table. */
export function listConnectorStatuses(): ConnectorStatus[] {
    const rows = listCredentials();
    const labelById = new Map(ALL_CONNECTORS.map((c) => [c.id, c.label]));
    return rows.map((r) => ({
        id: r.connector_id,
        label: labelById.get(r.connector_id) ?? r.connector_id,
        configured: true,
        signed_in: r.status === 'signed_in',
        last_check_at: r.last_check_at,
        last_error: r.last_error
    }));
}

/** Same as `listConnectorStatuses` but pads with a stub for every
 * credentials-backed connector that isn't configured yet — so the Settings
 * UI can show "Add" affordances for them. Always-available connectors are
 * excluded here (they don't belong in Settings). */
export function listAllConnectorStatuses(): ConnectorStatus[] {
    const configured = new Map(listConnectorStatuses().map((s) => [s.id, s]));
    return CREDENTIAL_BACKED_CONNECTORS.map(
        (c) =>
            configured.get(c.id) ?? {
                id: c.id,
                label: c.label,
                configured: false,
                signed_in: false,
                last_check_at: null,
                last_error: null
            }
    );
}

/** Every connector usable in the /datasets link UI: credentials-backed
 * (only when signed in) + always-available types like hard-drive. */
export function listAvailableConnectors(): ConnectorStatus[] {
    const fromCreds = listConnectorStatuses().filter((s) => s.signed_in);
    const alwaysOn = ALL_CONNECTORS.filter((c) => !c.needs_credentials).map(
        (c): ConnectorStatus => ({
            id: c.id,
            label: c.label,
            configured: true,
            signed_in: true,
            last_check_at: null,
            last_error: null
        })
    );
    return [...fromCreds, ...alwaysOn];
}
