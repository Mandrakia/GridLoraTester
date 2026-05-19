// Defer a sync compute to the next I/O turn so a SvelteKit `load` can flush
// the rest of its response before the heavy work runs. Returns a Promise
// that SvelteKit's streamed-loads pipeline treats as pending until then —
// the page renders with {#await} placeholders while the work happens, and
// the event loop is free to handle other requests (e.g. JobsBadge poll)
// between deferred slices.
export function defer<T>(fn: () => T): Promise<T> {
    return new Promise((resolve, reject) => {
        setImmediate(() => {
            try {
                resolve(fn());
            } catch (e) {
                reject(e);
            }
        });
    });
}
