// Advertises the server-side allowlist of browsable roots to the UI.
// The FolderPicker uses these as switchable starting points.
import { json } from '@sveltejs/kit';
import { getAllowedRoots } from '$lib/server/fs-roots';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => json({ roots: getAllowedRoots() });
