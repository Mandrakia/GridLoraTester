// Generic producer/consumer pipeline with backpressure and adaptive
// concurrency on the producer side. Pattern:
//
//   [N producers] → [bounded queue] → [1 consumer]
//
// Use case: a long-running job processes a list of inputs in two phases.
// The first phase ("produce") is independent per-input and cheap to
// parallelize (HTTP downloads, disk reads). The second phase ("consume")
// is a single hot path that the upstream MUST feed at the right rate
// (a Python ONNX session, a serialized DB writer, etc).
//
// The pool watches the consumer's wait time on `queue.get()`. When the
// consumer keeps starving (p50 wait > `scaleUpThresholdMs` over a sliding
// window), the pool spawns another producer — up to `maxProducers`. The
// queue's hard capacity provides natural backpressure: producers block
// on `queue.put()` when the consumer is the actual bottleneck, so
// over-spawning past the optimum is a no-op (extra producers sit on
// `put` waits and don't run requests).
//
// No scale-down in v1: a producer blocked inside `await produce(...)` (an
// HTTP request) can't be cooperatively cancelled without an AbortSignal
// plumbed through every connector, and "extra" producers cost ~nothing
// while parked on `put`. They exit naturally when `inputs` is exhausted.
//
// Cancellation: pass `shouldCancel`. Producers check it at the start of
// each iteration; the consumer checks it before every `get`. In-flight
// produce/consume awaits run to completion (we don't interrupt them).
//
// Errors: `onProduceError` / `onConsumeError` are called per-failure and
// processing continues. Items that fail `produce` are dropped (never
// enqueued); items that fail `consume` are dropped (already dequeued).
// Failure counting is the caller's responsibility — pass it via the
// error callbacks.

/** Bounded async FIFO with backpressure. `put` resolves immediately when
 * there's capacity or a waiting consumer; otherwise it blocks until space
 * frees up. `get` resolves with the next item, or `null` when the queue
 * is closed AND empty (terminal state — the consumer should exit). */
class BoundedAsyncQueue<T> {
    private items: T[] = [];
    private waitingGets: Array<(v: T | null) => void> = [];
    private waitingPuts: Array<() => void> = [];
    private closed = false;

    constructor(private readonly capacity: number) {}

    get depth(): number {
        return this.items.length;
    }
    get capacityValue(): number {
        return this.capacity;
    }

    async put(item: T): Promise<void> {
        if (this.closed) return; // drop silently — we're shutting down
        // Fast path: hand directly to a waiting consumer.
        const waiter = this.waitingGets.shift();
        if (waiter) {
            waiter(item);
            return;
        }
        if (this.items.length < this.capacity) {
            this.items.push(item);
            return;
        }
        // Queue full — wait until a get() makes room.
        await new Promise<void>((resolve) => {
            this.waitingPuts.push(() => {
                this.items.push(item);
                resolve();
            });
        });
    }

    async get(): Promise<T | null> {
        if (this.items.length > 0) {
            const item = this.items.shift()!;
            const putter = this.waitingPuts.shift();
            if (putter) putter();
            return item;
        }
        if (this.closed) return null;
        return new Promise<T | null>((resolve) => {
            this.waitingGets.push(resolve);
        });
    }

    /** Mark the queue closed. Future `put`s become no-ops; pending `get`s
     * resolve with null. Already-buffered items are still drainable. */
    close(): void {
        if (this.closed) return;
        this.closed = true;
        for (const waiter of this.waitingGets.splice(0)) waiter(null);
    }
}

export interface AdaptivePoolConfig<TInput, TItem> {
    /** The work items to walk through. Producers claim them by index in
     * declaration order, but the consume order is then queue order (no
     * guarantee w.r.t. input order beyond rough FIFO). */
    inputs: TInput[];
    /** Producer phase: transform an input into an item to feed the
     * consumer. Runs in parallel across producers. */
    produce: (input: TInput) => Promise<TItem>;
    /** Consumer phase: process one item from the queue. Runs sequentially
     * (single consumer). */
    consume: (item: TItem) => Promise<void>;

    /** Max items buffered between producers and the consumer. Defaults
     * to 10 — large enough to absorb burstiness, small enough to bound
     * memory under worst-case item size. */
    queueCapacity?: number;
    /** Always keep at least this many producers running. Default 1. */
    minProducers?: number;
    /** Never spawn more than this. Default 4. */
    maxProducers?: number;
    /** Sliding window (in samples) over which the consumer's wait time
     * is measured to decide whether to scale up. Default 20. */
    scaleWindow?: number;
    /** If `p50(consumer_wait)` over the window exceeds this and we have
     * room, spawn another producer. Default 50 ms. */
    scaleUpThresholdMs?: number;

    /** Cooperative cancellation: producers check this between iterations,
     * the consumer checks before every queue get. In-flight produce/consume
     * runs to completion. */
    shouldCancel?: () => boolean;

    /** Reported after every consumed item — caller forwards to the job's
     * metrics channel. Cheap (no I/O). */
    onMetrics?: (m: AdaptivePoolMetrics) => void;

    onProduceError?: (input: TInput, error: Error) => void;
    onConsumeError?: (item: TItem, error: Error) => void;
}

export interface AdaptivePoolMetrics {
    processed: number;
    active_producers: number;
    queue_depth: number;
    queue_capacity: number;
    consumer_wait_p50_ms: number;
    consumer_wait_p95_ms: number;
    produce_p50_ms: number;
    consume_p50_ms: number;
    /** How many times the pool decided it was starved and added a
     * producer over the lifetime of this run. */
    scale_ups: number;
}

export async function runAdaptivePool<TInput, TItem>(
    cfg: AdaptivePoolConfig<TInput, TItem>
): Promise<void> {
    const queueCapacity = cfg.queueCapacity ?? 10;
    const minProducers = Math.max(1, cfg.minProducers ?? 1);
    const maxProducers = Math.max(minProducers, cfg.maxProducers ?? 4);
    const scaleWindow = Math.max(1, cfg.scaleWindow ?? 20);
    const scaleUpThresholdMs = cfg.scaleUpThresholdMs ?? 50;

    const queue = new BoundedAsyncQueue<TItem>(queueCapacity);
    let nextInputIdx = 0;
    let activeProducers = 0;
    let processed = 0;
    let scaleUps = 0;
    // After spawning a producer, wait this many samples before considering
    // another scale-up — the new producer needs a few iterations before
    // its effect on consumer wait time becomes visible.
    let samplesSinceLastScaleUp = scaleWindow;

    const consumerWaits: number[] = [];
    const produceTimes: number[] = [];
    const consumeTimes: number[] = [];
    const pushSliding = (arr: number[], v: number): void => {
        arr.push(v);
        if (arr.length > scaleWindow) arr.shift();
    };

    async function runProducer(): Promise<void> {
        activeProducers++;
        try {
            while (true) {
                if (cfg.shouldCancel?.()) return;
                const idx = nextInputIdx++;
                if (idx >= cfg.inputs.length) return;
                const input = cfg.inputs[idx];
                const t0 = performance.now();
                let item: TItem;
                try {
                    item = await cfg.produce(input);
                } catch (e) {
                    cfg.onProduceError?.(input, e as Error);
                    continue;
                }
                pushSliding(produceTimes, performance.now() - t0);
                await queue.put(item);
            }
        } finally {
            activeProducers--;
            // Last producer exiting closes the queue so the consumer's
            // next `get` returns null instead of hanging forever.
            if (activeProducers === 0) queue.close();
        }
    }

    function spawnProducer(): void {
        if (activeProducers >= maxProducers) return;
        runProducer().catch((e) => {
            // The inner loop already handles produce errors per-item — a
            // throw here means our wrapper has a bug. Log so it doesn't
            // disappear silently.
            console.error('[adaptive-pool] producer crashed:', e);
        });
    }

    for (let i = 0; i < minProducers; i++) spawnProducer();

    while (true) {
        if (cfg.shouldCancel?.()) {
            queue.close();
            break;
        }
        const tWait = performance.now();
        const item = await queue.get();
        if (item === null) break; // queue closed AND drained
        pushSliding(consumerWaits, performance.now() - tWait);

        const tConsume = performance.now();
        try {
            await cfg.consume(item);
        } catch (e) {
            cfg.onConsumeError?.(item, e as Error);
        }
        pushSliding(consumeTimes, performance.now() - tConsume);
        processed++;
        samplesSinceLastScaleUp++;

        // Scale-up decision. Need enough samples in the window AND enough
        // cooldown since last scale-up before we consider adding.
        if (
            samplesSinceLastScaleUp >= scaleWindow &&
            consumerWaits.length >= Math.min(scaleWindow, 5) &&
            activeProducers < maxProducers
        ) {
            if (p50(consumerWaits) > scaleUpThresholdMs) {
                spawnProducer();
                scaleUps++;
                samplesSinceLastScaleUp = 0;
            }
        }

        cfg.onMetrics?.({
            processed,
            active_producers: activeProducers,
            queue_depth: queue.depth,
            queue_capacity: queue.capacityValue,
            consumer_wait_p50_ms: p50(consumerWaits),
            consumer_wait_p95_ms: p95(consumerWaits),
            produce_p50_ms: p50(produceTimes),
            consume_p50_ms: p50(consumeTimes),
            scale_ups: scaleUps
        });
    }
}

function pct(arr: number[], q: number): number {
    if (arr.length === 0) return 0;
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}
function p50(arr: number[]): number {
    return pct(arr, 0.5);
}
function p95(arr: number[]): number {
    return pct(arr, 0.95);
}
