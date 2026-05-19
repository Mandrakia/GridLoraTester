import { isAbsolute, relative, resolve } from 'node:path';

/** True when `target` is exactly `root` or is contained below it.
 * Uses path.relative() instead of string-prefix checks so it works with the
 * platform's native separator and does not confuse sibling paths. */
export function isPathInside(root: string, target: string): boolean {
    const rootAbs = resolve(root);
    const targetAbs = resolve(target);
    const rel = relative(rootAbs, targetAbs);
    return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}

export function isPathInsideAnyRoot(target: string, roots: string[]): boolean {
    return roots.some((root) => isPathInside(root, target));
}

export function pathBasename(p: string): string {
    return p.split(/[\\/]+/).filter(Boolean).pop() ?? p;
}
