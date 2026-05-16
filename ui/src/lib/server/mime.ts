// Tiny mime map shared by file-serving endpoints. Just enough extensions to
// cover what the dashboard streams (HTML/JSON for grid results, images for
// datasets); not a substitute for a real mime DB.
const MIME: Record<string, string> = {
    html: 'text/html; charset=utf-8',
    htm: 'text/html; charset=utf-8',
    json: 'application/json; charset=utf-8',
    txt: 'text/plain; charset=utf-8',
    css: 'text/css; charset=utf-8',
    js: 'application/javascript; charset=utf-8',
    svg: 'image/svg+xml',
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    webp: 'image/webp',
    bmp: 'image/bmp',
    gif: 'image/gif'
};

export function mimeFor(path: string): string {
    const dot = path.lastIndexOf('.');
    if (dot < 0) return 'application/octet-stream';
    const ext = path.slice(dot + 1).toLowerCase();
    return MIME[ext] ?? 'application/octet-stream';
}
