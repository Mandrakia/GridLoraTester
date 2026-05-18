// "Hard drive" connector — link a dataset to a plain folder on disk. No
// per-instance setup: each link carries its own folder path (stored as the
// link's person_id). isSignedIn() is always true; signIn/signOut are no-ops.
//
// All the local-folder mechanics (dir scan, download, thumb proxy) live
// in ./local-folder.ts so the google-photos connector can reuse them
// against its own cache directory.
import type {
    ConnectorPicture,
    ConnectorSignInResult,
    ConnectorTypeInfo,
    ListPicturesOpts,
    ListPicturesPage,
    PhotoConnector
} from '$lib/connectors/types';
import {
    downloadFromDir,
    encodePath,
    isInsideAnyRoot,
    listLinkedFolders,
    listPicturesInDir,
    proxyServeFromDir
} from './local-folder';

export const HARD_DRIVE_TYPE_INFO: ConnectorTypeInfo = {
    id: 'hard-drive',
    label: 'Folder on disk',
    linker_kind: 'folder',
    needs_credentials: false
};

const THUMB_PREFIX = '/connectors/hard-drive/thumb/';

function allowed(abs: string): boolean {
    return isInsideAnyRoot(abs, listLinkedFolders(['hard-drive']));
}

export class HardDriveConnector implements PhotoConnector {
    readonly id = 'hard-drive' as const;
    readonly label = 'Folder on disk';

    async isSignedIn(): Promise<boolean> {
        return true;
    }

    async signIn(): Promise<ConnectorSignInResult> {
        return { ok: true };
    }

    async signOut(): Promise<void> {
        // No persisted credentials to clear.
    }

    async listPersons() {
        // The link UI doesn't call this for folder-kind connectors.
        return [];
    }

    async listPictures(folderPath: string, _opts: ListPicturesOpts = {}): Promise<ListPicturesPage> {
        return listPicturesInDir(folderPath, {
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

// Re-export so callers that imported encodePath from this module
// (none in tree right now, but keeps the API surface stable) still work.
export { encodePath };
