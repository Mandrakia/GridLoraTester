export function pathBasename(p: string): string {
    return p.split(/[\\/]+/).filter(Boolean).pop() ?? p;
}

export function isSameOrInsidePathString(root: string, target: string): boolean {
    const norm = (s: string) => {
        const out = s.replace(/\\/g, '/').replace(/\/+$/, '');
        return /^[A-Za-z]:/.test(out) ? out.toLowerCase() : out;
    };
    const r = norm(root);
    const t = norm(target);
    return t === r || t.startsWith(r + '/');
}
