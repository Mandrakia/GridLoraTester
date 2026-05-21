<script lang="ts">
    import { enhance } from '$app/forms';
    import { invalidateAll } from '$app/navigation';
    import { pathBasename } from '$lib/path-display';
    import MainPanel from '$lib/components/MainPanel.svelte';
    import type { PageData } from './$types';

    let { data }: { data: PageData } = $props();

    let rescoring = $state(false);
    /** Rescore is meaningful only when a run exists and isn't currently
     * running (would conflict with the active job's writes). */
    let canRescore = $derived(
        data.run != null && data.run.status !== 'running' && data.active_job == null
    );

    // Live polling while a run is active.
    let isLive = $derived(
        data.active_job != null || (data.run != null && data.run.status === 'running')
    );
    $effect(() => {
        if (!isLive) return;
        const id = setInterval(() => invalidateAll(), 2000);
        return () => clearInterval(id);
    });

    const STATUS_LABELS: Record<string, { label: string; cls: string }> = {
        running: { label: 'Running', cls: 'bg-amber-500/20 text-amber-300' },
        completed: { label: 'Completed', cls: 'bg-emerald-500/20 text-emerald-300' },
        failed: { label: 'Failed', cls: 'bg-red-500/20 text-red-300' },
        cancelled: { label: 'Cancelled', cls: 'bg-fg-faint/20 text-fg-muted' },
        queued: { label: 'Queued', cls: 'bg-fg-faint/20 text-fg-muted' }
    };

    // ---- Face-score thresholds + helpers ----------------------------------
    const TH_GOOD = 0.5;
    const TH_OK = 0.35;
    type ScoreClass = 'good' | 'ok' | 'bad' | 'na';
    function scoreClass(s: number | null | undefined): ScoreClass {
        if (s == null || isNaN(s)) return 'na';
        if (s >= TH_GOOD) return 'good';
        if (s >= TH_OK) return 'ok';
        return 'bad';
    }
    function fmtScore(s: number | null | undefined): string {
        return s == null ? '—' : s.toFixed(2);
    }
    const BADGE_CLS: Record<ScoreClass, string> = {
        good: 'bg-emerald-500/85 text-white',
        ok: 'bg-amber-500/85 text-black',
        bad: 'bg-red-500/85 text-white',
        na: 'bg-bg-2 text-fg-faint'
    };
    const METRIC_CLS: Record<ScoreClass, string> = {
        good: 'border-emerald-500/45 bg-emerald-500/8 text-emerald-300',
        ok: 'border-amber-500/45 bg-amber-500/8 text-amber-300',
        bad: 'border-red-500/45 bg-red-500/8 text-red-300',
        na: 'border-border bg-bg-1 text-fg-faint'
    };

    // ---- Multi-run composite — per-row run selection ----------------------
    // The grid is now a "best-of-all-runs" composite. For each unique
    // lora_display, the user picks which run's cells back it. Default =
    // latest run containing that LoRA (see data.defaultRunByLora).
    //
    // The selection lives in localStorage scoped per-test so a user's
    // exploration of historical runs survives reloads. Polling keeps the
    // server data live; if a new run lands it shows up in the selectors
    // but the user's current pick is preserved.
    let storageDisabled = $derived(`flgrid_disabled_cols__${data.test.id}`);
    let storageSort = $derived(`flgrid_sort__${data.test.id}`);
    let storageRunSel = $derived(`flgrid_run_by_lora__${data.test.id}`);

    function loadRunSelFromStorage(): Record<string, number> {
        if (typeof localStorage === 'undefined') return {};
        try {
            const raw = localStorage.getItem(storageRunSel);
            if (!raw) return {};
            const parsed = JSON.parse(raw);
            if (parsed && typeof parsed === 'object') {
                const out: Record<string, number> = {};
                for (const [k, v] of Object.entries(parsed)) {
                    if (typeof v === 'number' && Number.isFinite(v)) out[k] = v;
                }
                return out;
            }
        } catch {
            // ignore corrupt entry
        }
        return {};
    }

    // Per-LoRA run pick. We seed from defaultRunByLora (latest run holding
    // each lora) and let stored choices override. The combined object is
    // re-derived on every server poll so newly-added LoRAs get their
    // default automatically without nuking the user's other picks.
    let runSelStored = $state<Record<string, number>>(loadRunSelFromStorage());
    let runByLora = $derived.by(() => {
        const out: Record<string, number> = { ...data.defaultRunByLora };
        for (const [lora, runId] of Object.entries(runSelStored)) {
            // Only honor a stored pick if the lora still exists somewhere
            // AND that specific run still has the lora (run may have been
            // deleted, lora may have been excluded by min_step). Otherwise
            // fall back to the default so the row keeps showing data.
            const hasLora = data.loraRuns.some(
                (e) => e.lora_display === lora && e.run_id === runId
            );
            if (hasLora) out[lora] = runId;
        }
        return out;
    });

    function selectRunForLora(lora: string, runId: number) {
        runSelStored = { ...runSelStored, [lora]: runId };
    }

    $effect(() => {
        if (typeof localStorage === 'undefined') return;
        try {
            localStorage.setItem(storageRunSel, JSON.stringify(runSelStored));
        } catch {
            // ignore
        }
    });

    // ---- Column-disable + sort state (persisted in localStorage) ----------
    type SortField = 'median' | `col-${number}`;
    type SortState = { field: SortField; dir: 'asc' | 'desc' };

    function loadDisabledFromStorage(): Set<number> {
        if (typeof localStorage === 'undefined') return new Set();
        try {
            const raw = localStorage.getItem(storageDisabled);
            if (!raw) return new Set();
            const arr = JSON.parse(raw);
            return Array.isArray(arr) ? new Set(arr.map((x) => +x)) : new Set();
        } catch {
            return new Set();
        }
    }
    function loadSortFromStorage(): SortState | null {
        if (typeof localStorage === 'undefined') return null;
        try {
            const raw = localStorage.getItem(storageSort);
            if (!raw) return null;
            const v = JSON.parse(raw) as SortState;
            if (!v || (v.field !== 'median' && !String(v.field).startsWith('col-'))) return null;
            if (v.dir !== 'asc' && v.dir !== 'desc') return null;
            return v;
        } catch {
            return null;
        }
    }

    let disabledCols = $state<Set<number>>(loadDisabledFromStorage());
    let currentSort = $state<SortState | null>(loadSortFromStorage());

    $effect(() => {
        if (typeof localStorage === 'undefined') return;
        try {
            localStorage.setItem(storageDisabled, JSON.stringify([...disabledCols]));
        } catch {
            // ignore
        }
    });
    $effect(() => {
        if (typeof localStorage === 'undefined') return;
        try {
            if (currentSort) localStorage.setItem(storageSort, JSON.stringify(currentSort));
            else localStorage.removeItem(storageSort);
        } catch {
            // ignore
        }
    });

    function toggleColumn(promptIdx: number) {
        const next = new Set(disabledCols);
        if (next.has(promptIdx)) next.delete(promptIdx);
        else next.add(promptIdx);
        disabledCols = next;
    }
    function resetColumns() {
        disabledCols = new Set();
    }
    function cycleSort(field: SortField) {
        if (!currentSort || currentSort.field !== field) currentSort = { field, dir: 'desc' };
        else if (currentSort.dir === 'desc') currentSort = { field, dir: 'asc' };
        else currentSort = null;
    }

    // ---- Lookup: (lora_display, prompt_idx) → cell ------------------------
    // Built by walking `data.loraRuns` for each lora's currently-selected
    // run. Re-derives on poll + on user-changes to runByLora.
    type Cell = (typeof data.loraRuns)[number]['cells'][number];
    let cellAt = $derived.by(() => {
        const m = new Map<string, Cell>();
        for (const lora of data.loraDisplays) {
            const runId = runByLora[lora];
            if (runId == null) continue;
            const entry = data.loraRuns.find(
                (e) => e.lora_display === lora && e.run_id === runId
            );
            if (!entry) continue;
            for (const c of entry.cells) m.set(`${lora}_${c.prompt_idx}`, c);
        }
        return m;
    });

    // Quick run-meta lookup keyed by id, for the per-row pill recap.
    let runById = $derived.by(() => {
        const m = new Map<number, (typeof data.runs)[number]>();
        for (const r of data.runs) m.set(r.id, r);
        return m;
    });

    // For each lora_display, the list of runs that contain it (newest
    // first) — feeds the per-row run picker dropdown.
    let runsForLora = $derived.by(() => {
        const m = new Map<string, typeof data.runs>();
        for (const lora of data.loraDisplays) {
            const ids = data.loraRuns
                .filter((e) => e.lora_display === lora)
                .map((e) => e.run_id);
            const runs = ids
                .map((id) => runById.get(id))
                .filter((r): r is (typeof data.runs)[number] => r != null);
            m.set(lora, runs);
        }
        return m;
    });

    // ---- Row metrics + sort -----------------------------------------------
    type RowMetrics = {
        n_faces: number;
        n_total: number;
        median: number | null;
        p20: number | null;
        max: number | null;
        std: number | null;
    };
    function quantile(sorted: number[], q: number): number | null {
        if (!sorted.length) return null;
        const pos = (sorted.length - 1) * q;
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        if (lo === hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }
    function recomputeRowMetrics(scores: (number | null)[]): RowMetrics {
        const total = scores.length;
        const valid = scores.filter((s): s is number => s != null).slice().sort((a, b) => a - b);
        if (!valid.length) {
            return { n_faces: 0, n_total: total, median: null, p20: null, max: null, std: null };
        }
        const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
        const varv = valid.reduce((s, v) => s + (v - mean) * (v - mean), 0) / valid.length;
        return {
            n_faces: valid.length,
            n_total: total,
            median: quantile(valid, 0.5),
            p20: quantile(valid, 0.2),
            max: valid[valid.length - 1],
            std: Math.sqrt(varv)
        };
    }

    /** Composite display rows: one per unique lora_display, with metrics
     * recomputed from the SELECTED run's cells minus disabled columns. */
    let displayRows = $derived.by(() => {
        const lookup = cellAt;
        const enabledPrompts = data.prompts.filter((p) => !disabledCols.has(p.idx));
        const enriched = data.loraDisplays.map((lora) => {
            const scores = enabledPrompts.map((p) => {
                const c = lookup.get(`${lora}_${p.idx}`);
                return c?.face_score ?? null;
            });
            return {
                lora_display: lora,
                run_id: runByLora[lora] ?? null,
                metrics: recomputeRowMetrics(scores)
            };
        });
        if (currentSort) {
            const dir = currentSort.dir === 'asc' ? 1 : -1;
            const valueFor = (row: (typeof enriched)[number]): number | null => {
                if (currentSort!.field === 'median') return row.metrics.median;
                const col = Number(currentSort!.field.slice(4));
                const c = lookup.get(`${row.lora_display}_${col}`);
                return c?.face_score ?? null;
            };
            enriched.sort((a, b) => {
                const av = valueFor(a);
                const bv = valueFor(b);
                if (av == null && bv == null) return 0;
                if (av == null) return 1;
                if (bv == null) return -1;
                return (av - bv) * dir;
            });
        }
        return enriched;
    });

    let faceOn = $derived(displayRows.some((r) => r.metrics.n_faces > 0));

    // ---- Lightbox ---------------------------------------------------------
    // zoomKey is keyed by lora_display now (not lora_idx) so it survives
    // when the user changes which run backs a row.
    let zoomKey = $state<{ lora_display: string; prompt_idx: number } | null>(null);
    let zoomCell = $derived(
        zoomKey ? cellAt.get(`${zoomKey.lora_display}_${zoomKey.prompt_idx}`) ?? null : null
    );

    let navMode = $state<'prompts' | 'loras'>('prompts');
    function toggleNavMode() {
        navMode = navMode === 'prompts' ? 'loras' : 'prompts';
    }

    function openZoom(lora_display: string, prompt_idx: number) {
        zoomKey = { lora_display, prompt_idx };
    }
    function closeZoom() {
        zoomKey = null;
    }
    function step(axis: 'prompt' | 'lora', dir: -1 | 1) {
        if (!zoomKey) return;
        if (axis === 'prompt') {
            const N = data.prompts.length;
            if (N === 0) return;
            const cur = data.prompts.findIndex((p) => p.idx === zoomKey!.prompt_idx);
            if (cur < 0) return;
            const next = (cur + dir + N) % N;
            zoomKey = { lora_display: zoomKey.lora_display, prompt_idx: data.prompts[next].idx };
        } else {
            const N = displayRows.length;
            if (N === 0) return;
            const cur = displayRows.findIndex((r) => r.lora_display === zoomKey!.lora_display);
            if (cur < 0) return;
            const next = (cur + dir + N) % N;
            zoomKey = {
                lora_display: displayRows[next].lora_display,
                prompt_idx: zoomKey.prompt_idx
            };
        }
    }

    function onKey(e: KeyboardEvent) {
        if (!zoomKey) return;
        if (e.key === 'Escape') closeZoom();
        else if (e.key === 'ArrowLeft') step(navMode === 'prompts' ? 'prompt' : 'lora', -1);
        else if (e.key === 'ArrowRight') step(navMode === 'prompts' ? 'prompt' : 'lora', +1);
        else if (e.key === 'ArrowUp') step(navMode === 'prompts' ? 'lora' : 'prompt', -1);
        else if (e.key === 'ArrowDown') step(navMode === 'prompts' ? 'lora' : 'prompt', +1);
        else if (e.key === 't' || e.key === 'T') toggleNavMode();
    }

    function sortIndicator(field: SortField): '↕' | '↑' | '↓' {
        if (!currentSort || currentSort.field !== field) return '↕';
        return currentSort.dir === 'asc' ? '↑' : '↓';
    }

    // ---- View tabs (Grid / Graph) -----------------------------------------
    type ViewKind = 'grid' | 'graph';
    let storageView = $derived(`flgrid_view__${data.test.id}`);
    function loadViewFromStorage(): ViewKind {
        if (typeof localStorage === 'undefined') return 'grid';
        const v = localStorage.getItem(storageView);
        return v === 'graph' ? 'graph' : 'grid';
    }
    let view = $state<ViewKind>(loadViewFromStorage());
    $effect(() => {
        if (typeof localStorage === 'undefined') return;
        try {
            localStorage.setItem(storageView, view);
        } catch {
            // ignore
        }
    });

    // ---- Graph state ------------------------------------------------------
    // Prompt selection: 'all' = aggregate (use displayRow.metrics), number =
    // single prompt index → series collapses to that prompt's raw score
    // across LoRAs (median/p20/max meaningless for a single value).
    let promptSel = $state<'all' | number>('all');
    let showMedian = $state(true);
    let showP20 = $state(true);
    let showMax = $state(true);
    // Trend smoothing: 0 = raw, 100 = widest window. Maps to a centered
    // moving-average radius below; symmetric so the peak stays on the
    // step where it actually occurred (no EMA-style rightward lag).
    let smoothPct = $state(0);
    // Y-axis scale. 'linear' = raw cosine [0,1] (default, unchanged). The two
    // "headroom" scales stretch the top so a +0.01 near the ceiling reads as
    // the bigger real gain it is (less room left to close):
    //   'fisher'  = Fisher z, arctanh(x) — the standard transform for cosine.
    //   'loghead' = -ln(1-x) — one-sided "fraction of remaining gap closed".
    let yScale = $state<'linear' | 'fisher' | 'loghead'>('linear');

    /** Extract training step from a LoRA filename. Default convention is
     * `<base>_<6-digit step>.safetensors` (ai-toolkit). Falls back to
     * NaN when the name doesn't carry a step. */
    function parseLoraStep(loraDisplay: string): number {
        const m = loraDisplay.match(/_(\d+)(?:\.safetensors)?$/);
        if (!m) return NaN;
        return parseInt(m[1], 10);
    }

    type ChartPoint = {
        x: number; // step or index
        xLabel: string; // human-readable
        median: number | null;
        p20: number | null;
        max: number | null;
        // For single-prompt mode, all three collapse to `single`. We
        // still populate median so the user sees a line.
        single: number | null;
        lora: string;
    };

    /** Chart data: one point per LoRA in displayRows order (already
     * sorted by user-chosen grid sort). X-axis prefers step parsed from
     * the lora_display; falls back to display index when no step found. */
    let chartPoints = $derived.by<ChartPoint[]>(() => {
        const lookup = cellAt;
        // First decide X: if every lora has a parsable step, use step;
        // otherwise fall back to 0..N-1 index for stability.
        const steps = displayRows.map((r) => parseLoraStep(r.lora_display));
        const useStep = steps.every((s) => Number.isFinite(s));

        return displayRows.map((r, idx) => {
            const step = useStep ? steps[idx] : idx;
            const xLabel = useStep
                ? String(steps[idx])
                : r.lora_display.length > 18
                  ? r.lora_display.slice(0, 16) + '…'
                  : r.lora_display;
            if (promptSel === 'all') {
                return {
                    x: step,
                    xLabel,
                    median: r.metrics.median,
                    p20: r.metrics.p20,
                    max: r.metrics.max,
                    single: null,
                    lora: r.lora_display
                };
            }
            const c = lookup.get(`${r.lora_display}_${promptSel}`);
            const s = c?.face_score ?? null;
            return {
                x: step,
                xLabel,
                median: null,
                p20: null,
                max: null,
                single: s,
                lora: r.lora_display
            };
        });
    });

    /** Selected-prompt display label for the toolbar dropdown. */
    function promptOptionLabel(idx: number, text: string): string {
        const trim = text.length > 32 ? text.slice(0, 30) + '…' : text;
        return `#${idx} · ${trim}`;
    }

    // ---- SVG chart geometry ----------------------------------------------
    // Viewbox is fixed; the parent <svg> stretches via width=100%. Tooltip
    // anchors to a hovered point's pixel coords (px relative to the SVG).
    const CHART_W = 1000;
    const CHART_H = 380;
    const CHART_PAD_L = 48;
    const CHART_PAD_R = 24;
    const CHART_PAD_T = 24;
    const CHART_PAD_B = 56;
    let chartPlotW = $derived(CHART_W - CHART_PAD_L - CHART_PAD_R);
    let chartPlotH = $derived(CHART_H - CHART_PAD_T - CHART_PAD_B);

    let chartXDomain = $derived.by<[number, number]>(() => {
        if (chartPoints.length === 0) return [0, 1];
        const xs = chartPoints.map((p) => p.x);
        const lo = Math.min(...xs);
        const hi = Math.max(...xs);
        return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
    });

    function xPos(x: number): number {
        const [lo, hi] = chartXDomain;
        const t = hi === lo ? 0.5 : (x - lo) / (hi - lo);
        return CHART_PAD_L + t * chartPlotW;
    }
    // Headroom scales: map raw similarity through arctanh / -ln(1-x) so the
    // top of the range is stretched. Both blow up at 1.0, so they share a
    // ceiling — scores at/above it pin to the top, scores below 0 pin to the
    // floor (early-training noise near 0 is meant to look flat; the real gain
    // lives up near the ceiling).
    const Y_CEIL = 0.85;
    const Y_ATANH_CEIL = Math.atanh(Y_CEIL); // ≈ 1.256  (Fisher z)
    const Y_LOG_CEIL = -Math.log(1 - Y_CEIL); // ≈ 1.897  (-ln(1-x))
    function yPos(score: number): number {
        // Invert because SVG y grows downward.
        if (yScale === 'fisher' || yScale === 'loghead') {
            const s = Math.min(Y_CEIL, Math.max(0, score));
            const norm =
                yScale === 'fisher'
                    ? Math.atanh(s) / Y_ATANH_CEIL
                    : -Math.log(1 - s) / Y_LOG_CEIL;
            return CHART_PAD_T + (1 - norm) * chartPlotH;
        }
        // Linear: fixed [0, 1] cosine range (unchanged default behavior).
        return CHART_PAD_T + (1 - score) * chartPlotH;
    }
    // Tick set depends on the scale: headroom packs more labels near the top
    // (capped at Y_CEIL) where the resolution now lives. TH_OK/TH_GOOD stay so
    // the threshold dashed lines render in all three.
    let yTicks = $derived(
        yScale === 'linear'
            ? [0, 0.25, TH_OK, TH_GOOD, 0.75, 1]
            : [0, TH_OK, TH_GOOD, 0.6, 0.7, 0.8, Y_CEIL]
    );

    /** Build an SVG path "M x,y L x,y …" from a series, skipping null
     * points by splitting the path into multiple subpaths (so a single
     * missing data point doesn't draw a misleading straight line across). */
    function pathFor(series: (number | null)[]): string {
        const segs: string[] = [];
        let cmd: 'M' | 'L' = 'M';
        for (let i = 0; i < chartPoints.length; i++) {
            const v = series[i];
            if (v == null) {
                cmd = 'M';
                continue;
            }
            segs.push(`${cmd}${xPos(chartPoints[i].x).toFixed(1)},${yPos(v).toFixed(1)}`);
            cmd = 'L';
        }
        return segs.join(' ');
    }

    /** Centered moving average over a series, ignoring nulls within the
     * window and preserving nulls at their own index (so gaps still split
     * the path in pathFor, never bridging missing LoRAs). The window
     * radius scales with the series length so the visual smoothness feels
     * consistent whether there are 8 or 60 LoRAs; smoothPct=0 is a no-op. */
    function smoothSeries(series: (number | null)[]): (number | null)[] {
        if (smoothPct <= 0) return series;
        const n = series.length;
        const maxR = Math.max(1, Math.round(n * 0.2));
        const r = Math.round((smoothPct / 100) * maxR);
        if (r <= 0) return series;
        const out: (number | null)[] = new Array(n).fill(null);
        for (let i = 0; i < n; i++) {
            if (series[i] == null) continue;
            let sum = 0;
            let cnt = 0;
            for (let j = Math.max(0, i - r); j <= Math.min(n - 1, i + r); j++) {
                const v = series[j];
                if (v != null) {
                    sum += v;
                    cnt++;
                }
            }
            out[i] = cnt > 0 ? sum / cnt : null;
        }
        return out;
    }

    // Smoothed series, derived once so the drawn line and the hover
    // tooltip read the exact same numbers (and we don't recompute the
    // moving average four times per render). Each tracks smoothPct +
    // chartPoints. In the mode where a series is all-null (e.g. median
    // in per-prompt mode) these collapse to all-null and go unused.
    let medianSmoothed = $derived(smoothSeries(chartPoints.map((p) => p.median)));
    let p20Smoothed = $derived(smoothSeries(chartPoints.map((p) => p.p20)));
    let maxSmoothed = $derived(smoothSeries(chartPoints.map((p) => p.max)));
    let singleSmoothed = $derived(smoothSeries(chartPoints.map((p) => p.single)));

    /** Pre-computed set of X-axis indices that should get a tick label.
     * Targets ~8 labels evenly spaced, but the "always show last" rule
     * is gated on a min-distance check so the last label doesn't collide
     * with the predecessor (e.g. 5900 / 6000 at step=8 with 58 points). */
    let xLabelIndices = $derived.by<Set<number>>(() => {
        const n = chartPoints.length;
        if (n === 0) return new Set();
        if (n <= 10) return new Set(chartPoints.map((_, i) => i));
        const target = 8;
        const step = Math.max(1, Math.ceil((n - 1) / target));
        const set = new Set<number>();
        for (let i = 0; i < n; i += step) set.add(i);
        const lastMultiple = Math.floor((n - 1) / step) * step;
        const gap = n - 1 - lastMultiple;
        if (gap === 0) {
            // Last tick already lands on n-1, nothing to add.
        } else if (gap >= Math.ceil(step / 2)) {
            set.add(n - 1);
        } else {
            // Last index is too close to the previous tick — drop the
            // predecessor and keep the actual end-of-axis label instead.
            set.delete(lastMultiple);
            set.add(n - 1);
        }
        return set;
    });

    let chartHover = $state<{ idx: number; x: number; y: number } | null>(null);
    function onPlotMove(e: MouseEvent) {
        if (chartPoints.length === 0) {
            chartHover = null;
            return;
        }
        const svg = e.currentTarget as SVGSVGElement;
        const rect = svg.getBoundingClientRect();
        // Map cursor px to viewBox px (SVG stretches with preserveAspectRatio
        // 'none'). x_view / W_view = (cx - rect.left) / rect.width.
        const cx = ((e.clientX - rect.left) / rect.width) * CHART_W;
        // Snap to nearest data point by X.
        let best = 0;
        let bestDist = Infinity;
        for (let i = 0; i < chartPoints.length; i++) {
            const d = Math.abs(xPos(chartPoints[i].x) - cx);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        }
        if (bestDist > 60) {
            chartHover = null;
            return;
        }
        const pt = chartPoints[best];
        const px = xPos(pt.x);
        // Anchor tooltip Y to the median (or single) value if present,
        // else the top of the chart.
        const yref =
            promptSel === 'all'
                ? pt.median ?? pt.max ?? pt.p20 ?? 0.5
                : pt.single ?? 0.5;
        chartHover = { idx: best, x: px, y: yPos(yref) };
    }

    /** Short, single-line summary of a run's config — used for the row
     * pill. Falls back to "run #N" when nothing characteristic is set. */
    function runShortLabel(r: (typeof data.runs)[number]): string {
        const parts: string[] = [`#${r.id}`];
        if (r.quant) parts.push(r.quant);
        return parts.join(' · ');
    }
    /** Multi-line full config for the (i) tooltip. */
    function runFullLabel(r: (typeof data.runs)[number]): string {
        const lines = [`Run #${r.id}  (${r.status})`];
        if (r.started_at) lines.push(`started: ${r.started_at}`);
        if (r.finished_at) lines.push(`finished: ${r.finished_at}`);
        const kv: [string, unknown][] = [
            ['quant', r.quant],
            ['compile', r.compile_mode],
            ['steps', r.steps],
            ['seed', r.seed],
            ['guidance', r.guidance],
            ['lora_scale', r.lora_scale],
            ['min_step', r.min_step],
            ['qwen_dtype', r.qwen_dtype],
            ['sage', r.sage_attention]
        ];
        for (const [k, v] of kv) {
            if (v != null && v !== '') lines.push(`${k}: ${v}`);
        }
        return lines.join('\n');
    }
</script>

<svelte:head>
    <title>{data.test.name} — GridLoraTester</title>
</svelte:head>

<MainPanel>
    {#snippet header()}
        <div class="border-b border-border px-6 py-4">
            <div class="flex items-center justify-between gap-4">
                <div>
                    <a href="/tests" class="text-xs text-fg-muted hover:text-fg">← Tests</a>
                    <h1 class="mt-1 text-xl font-semibold">{data.test.name}</h1>
                    <p class="mt-0.5 text-xs text-fg-faint">
                        {data.test.resolution || '—'}
                        {#if data.test.trigger}
                            · trigger <span class="font-mono">{data.test.trigger}</span>
                        {/if}
                        · lora <span class="font-mono">{pathBasename(data.test.lora_path)}</span>
                        {#if data.runs.length > 0}
                            · {data.runs.length} run{data.runs.length === 1 ? '' : 's'} in history
                        {/if}
                    </p>
                </div>
                <div class="flex items-center gap-3 text-right">
                    {#if canRescore}
                        <form
                            method="POST"
                            action="?/rescore"
                            use:enhance={() => {
                                rescoring = true;
                                return ({ update }) => {
                                    update({ reset: false }).finally(() => (rescoring = false));
                                };
                            }}
                        >
                            <button
                                type="submit"
                                class="btn-ghost px-2 py-1 text-xs text-sky-300 hover:bg-sky-500/10 hover:text-sky-200 disabled:opacity-50"
                                disabled={rescoring}
                                title="Re-score this run's images against the current centroid (no regeneration)"
                            >
                                {rescoring ? 'Queuing…' : '♺ Rescore'}
                            </button>
                        </form>
                    {/if}
                    {#if data.run}
                        <a
                            href="/tests/{data.test.id}/export"
                            class="btn-ghost px-2 py-1 text-xs text-emerald-300 hover:bg-emerald-500/10 hover:text-emerald-200"
                            title="Download the grid you see (composite of all runs, manifest + images) as a zip to import on another install"
                        >⬇ Export grid</a>
                    {/if}
                    {#if data.run}
                        {@const st = STATUS_LABELS[data.run.status] ?? STATUS_LABELS.queued}
                        <div class="flex flex-col items-end gap-1">
                            <span
                                class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium {st.cls}"
                            >
                                {#if isLive}
                                    <svg
                                        class="h-3 w-3 animate-spin"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        stroke-width="2.5"
                                    >
                                        <path d="M12 3a9 9 0 1 0 9 9" stroke-linecap="round" />
                                    </svg>
                                {/if}
                                {st.label}
                            </span>
                            <span class="text-[10px] text-fg-faint tabular-nums">
                                latest run #{data.run.id}
                                {#if data.active_job}
                                    · <a
                                        href={`/jobs?focus=${data.active_job.id}`}
                                        class="text-accent hover:text-accent-hover"
                                        >job #{data.active_job.id}</a
                                    >
                                    {#if data.active_job.progress_total && data.active_job.progress_total > 0}
                                        · {data.active_job.progress_current}/{data.active_job
                                            .progress_total}
                                    {/if}
                                {/if}
                            </span>
                        </div>
                    {:else}
                        <span
                            class="inline-flex items-center rounded-full bg-bg-3 px-2 py-0.5 text-xs text-fg-muted"
                        >
                            Never run
                        </span>
                    {/if}
                </div>
            </div>

            {#if data.active_job && data.active_job.progress_total}
                {@const aj = data.active_job}
                {@const pct =
                    aj.progress_total != null
                        ? Math.round((aj.progress_current / aj.progress_total) * 100)
                        : 0}
                <div class="mt-3 h-1.5 w-full overflow-hidden rounded bg-bg-3">
                    <div
                        class="h-full bg-accent transition-all duration-500"
                        style="width: {pct}%"
                    ></div>
                </div>
            {/if}

            {#if data.run?.status === 'failed' && data.run.error}
                <pre
                    class="mt-3 max-h-32 overflow-auto rounded-md border border-red-500/30 bg-red-500/5 p-2 text-xs text-red-300">{data.run.error}</pre>
            {/if}
        </div>
    {/snippet}

    {#if !data.run}
        <div class="flex flex-1 items-center justify-center text-sm text-fg-muted">
            No runs yet — click ▶ Run on <a
                href="/tests"
                class="text-accent hover:text-accent-hover">/tests</a
            > to start one.
        </div>
    {:else if displayRows.length === 0}
        <div class="flex flex-1 items-center justify-center text-sm text-fg-muted">
            Preparing run… (Python is booting the pipeline, rows will appear shortly.)
        </div>
    {:else}
        <!-- View tabs (Grid | Graph). Tabs sit above the per-view
             toolbar so the active view's controls can scroll/sticky
             independently underneath. -->
        <div class="flex items-center gap-1 border-b border-border bg-bg-1 px-6 pt-3">
            {#each [{ id: 'grid' as ViewKind, label: 'Grid' }, { id: 'graph' as ViewKind, label: 'Graph' }] as t (t.id)}
                <button
                    type="button"
                    class="-mb-px border-b-2 px-3 py-1.5 text-sm transition-colors {view ===
                    t.id
                        ? 'border-accent text-fg'
                        : 'border-transparent text-fg-muted hover:text-fg'}"
                    onclick={() => (view = t.id)}
                >
                    {t.label}
                </button>
            {/each}
        </div>

        {#if view === 'grid'}
            {#if faceOn}
                <div
                    class="sticky top-0 z-20 flex flex-wrap items-center gap-3 border-b border-border bg-bg-1/95 px-6 py-2 text-xs backdrop-blur"
                >
                    <span class="text-fg-muted">Face↔centroid similarity:</span>
                <span class="flex items-center gap-1">
                    <span class="inline-block h-2 w-2 rounded-full bg-emerald-500"></span>
                    ≥ {TH_GOOD.toFixed(2)}
                </span>
                <span class="flex items-center gap-1">
                    <span class="inline-block h-2 w-2 rounded-full bg-amber-500"></span>
                    ≥ {TH_OK.toFixed(2)}
                </span>
                <span class="flex items-center gap-1">
                    <span class="inline-block h-2 w-2 rounded-full bg-red-500"></span>
                    &lt; {TH_OK.toFixed(2)}
                </span>
                <span class="flex items-center gap-1">
                    <span class="inline-block h-2 w-2 rounded-full bg-bg-3"></span>
                    no face
                </span>
                <span class="mx-2 h-3 w-px bg-border"></span>
                <span class="text-fg-muted">Sort:</span>
                <button
                    type="button"
                    class="rounded-md border border-border px-2 py-0.5 text-xs hover:bg-bg-2 {currentSort?.field ===
                    'median'
                        ? 'border-accent text-accent'
                        : ''}"
                    title="Sort LoRAs by row median (cycles desc → asc → off)"
                    onclick={() => cycleSort('median')}
                >
                    median <span class="tabular-nums">{sortIndicator('median')}</span>
                </button>
                {#if disabledCols.size > 0}
                    <span class="mx-2 h-3 w-px bg-border"></span>
                    <span class="text-fg-faint">
                        {disabledCols.size} column{disabledCols.size === 1 ? '' : 's'} disabled
                    </span>
                    <button
                        type="button"
                        class="rounded-md border border-border px-2 py-0.5 text-xs hover:bg-bg-2"
                        title="Re-enable every column"
                        onclick={resetColumns}
                    >
                        reset cols
                    </button>
                {/if}
            </div>
        {/if}

        <div
            class="grid gap-2 p-6"
            style="grid-template-columns: minmax(240px, 320px) repeat({data.prompts
                .length}, minmax(180px, 280px));"
        >
            <!-- Header row: empty corner + one cell per prompt -->
            <div></div>
            {#each data.prompts as p (p.idx)}
                {@const isDisabled = disabledCols.has(p.idx)}
                {@const colSortField = `col-${p.idx}` as SortField}
                {@const isSortedByCol = !!currentSort && currentSort.field === colSortField}
                <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
                <div
                    class="sticky top-0 z-10 rounded-md border border-border bg-bg-2 p-2 text-xs transition-opacity {isDisabled
                        ? 'opacity-40'
                        : ''} {faceOn ? 'cursor-pointer hover:bg-bg-3' : ''}"
                    role={faceOn ? 'button' : undefined}
                    tabindex={faceOn ? 0 : undefined}
                    aria-pressed={faceOn ? isDisabled : undefined}
                    aria-label={faceOn
                        ? isDisabled
                            ? `Re-enable prompt #${p.idx}`
                            : `Disable prompt #${p.idx}`
                        : undefined}
                    title={(faceOn
                        ? isDisabled
                            ? 'Click to re-enable this column for row metrics'
                            : 'Click to disable this column for row metrics'
                        : '') +
                        '\n' +
                        p.text}
                    onclick={() => faceOn && toggleColumn(p.idx)}
                    onkeydown={(e) => {
                        if (!faceOn) return;
                        if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            toggleColumn(p.idx);
                        }
                    }}
                >
                    <div class="flex items-baseline justify-between gap-2">
                        <span
                            class="rounded-full bg-accent/20 px-1.5 py-0.5 text-[10px] font-medium text-accent"
                        >
                            #{p.idx}
                        </span>
                        <div class="flex items-center gap-1.5">
                            {#if faceOn}
                                <button
                                    type="button"
                                    class="flex h-6 w-6 items-center justify-center rounded border text-sm font-semibold leading-none transition-colors {isSortedByCol
                                        ? 'border-accent bg-accent/15 text-accent'
                                        : 'border-border text-fg-muted hover:border-accent/50 hover:bg-bg-2 hover:text-fg'}"
                                    title={isSortedByCol
                                        ? `Sorted by this column (${currentSort!.dir}). Click to cycle.`
                                        : "Sort rows by this column's score"}
                                    onclick={(e) => {
                                        e.stopPropagation();
                                        cycleSort(colSortField);
                                    }}
                                >
                                    {sortIndicator(colSortField)}
                                </button>
                            {/if}
                            <span class="font-mono text-[10px] text-fg-faint"
                                >{p.width}×{p.height}</span
                            >
                        </div>
                    </div>
                    <p class="mt-1 line-clamp-3 text-fg-muted">{p.text}</p>
                </div>
            {/each}

            <!-- Row per unique LoRA (across all runs) -->
            {#each displayRows as r (r.lora_display)}
                {@const selectedRun = r.run_id != null ? runById.get(r.run_id) ?? null : null}
                {@const availableRuns = runsForLora.get(r.lora_display) ?? []}
                <div
                    class="sticky left-0 z-10 flex flex-col gap-2 rounded-md border border-border bg-bg-2 p-2.5 text-xs"
                    title={r.lora_display}
                >
                    <div class="font-semibold leading-tight text-fg break-all">
                        {r.lora_display}
                    </div>

                    <!-- Per-row run selector. When the LoRA only exists in
                         one run there's nothing to pick — show it as a
                         static pill instead so the UI doesn't lie about
                         interactivity. (i) tooltip carries the full
                         config snapshot regardless. -->
                    {#if availableRuns.length > 1}
                        <div class="flex items-center gap-1.5">
                            <select
                                class="min-w-0 flex-1 rounded border border-border bg-bg-1 px-1.5 py-0.5 text-[10px] text-fg hover:border-accent/50"
                                value={r.run_id ?? ''}
                                title="Pick which run's cells back this LoRA row"
                                onchange={(e) => {
                                    const v = Number((e.target as HTMLSelectElement).value);
                                    if (Number.isFinite(v) && v > 0)
                                        selectRunForLora(r.lora_display, v);
                                }}
                            >
                                {#each availableRuns as run (run.id)}
                                    <option value={run.id}>{runShortLabel(run)}</option>
                                {/each}
                            </select>
                            {#if selectedRun}
                                <span
                                    class="inline-flex h-5 w-5 cursor-help items-center justify-center rounded-full border border-border text-[10px] text-fg-muted hover:border-accent/50 hover:text-fg"
                                    title={runFullLabel(selectedRun)}
                                    aria-label="Show full run config"
                                >
                                    i
                                </span>
                            {/if}
                        </div>
                    {:else if selectedRun}
                        <div class="flex items-center gap-1.5">
                            <span
                                class="inline-flex items-center rounded border border-border bg-bg-1 px-1.5 py-0.5 text-[10px] text-fg-muted"
                            >
                                {runShortLabel(selectedRun)}
                            </span>
                            <span
                                class="inline-flex h-5 w-5 cursor-help items-center justify-center rounded-full border border-border text-[10px] text-fg-muted hover:border-accent/50 hover:text-fg"
                                title={runFullLabel(selectedRun)}
                                aria-label="Show full run config"
                            >
                                i
                            </span>
                        </div>
                    {/if}

                    {#if r.metrics.n_faces > 0}
                        {@const medCls = scoreClass(r.metrics.median)}
                        {@const p20Cls = scoreClass(r.metrics.p20)}
                        {@const maxCls = scoreClass(r.metrics.max)}
                        <div class="grid grid-cols-2 gap-1">
                            <div
                                class="flex flex-col items-center justify-center rounded border-2 px-1 py-1 {METRIC_CLS[
                                    medCls
                                ]}"
                                title="median similarity"
                            >
                                <span class="text-sm font-bold tabular-nums leading-none"
                                    >{r.metrics.median!.toFixed(2)}</span
                                >
                                <span class="mt-0.5 text-[9px] uppercase tracking-wide text-fg-muted"
                                    >med</span
                                >
                            </div>
                            <div
                                class="flex flex-col items-center justify-center rounded border px-1 py-1 {METRIC_CLS[
                                    p20Cls
                                ]}"
                                title="20th percentile (worst-quartile floor)"
                            >
                                <span class="text-sm font-bold tabular-nums leading-none"
                                    >{r.metrics.p20!.toFixed(2)}</span
                                >
                                <span class="mt-0.5 text-[9px] uppercase tracking-wide text-fg-muted"
                                    >p20</span
                                >
                            </div>
                            <div
                                class="flex flex-col items-center justify-center rounded border px-1 py-1 {METRIC_CLS[
                                    maxCls
                                ]}"
                                title="max similarity in row"
                            >
                                <span class="text-sm font-bold tabular-nums leading-none"
                                    >{r.metrics.max!.toFixed(2)}</span
                                >
                                <span class="mt-0.5 text-[9px] uppercase tracking-wide text-fg-muted"
                                    >max</span
                                >
                            </div>
                            <div
                                class="flex flex-col items-center justify-center rounded border border-border bg-bg-1 px-1 py-1 text-fg-muted"
                                title="std dev / faces detected / total images"
                            >
                                <span class="text-xs font-semibold tabular-nums leading-none"
                                    >σ {r.metrics.std!.toFixed(2)}</span
                                >
                                <span class="mt-0.5 text-[9px] tabular-nums text-fg-faint"
                                    >{r.metrics.n_faces}/{r.metrics.n_total}</span
                                >
                            </div>
                        </div>
                    {:else if faceOn && r.metrics.n_total > 0}
                        <div class="text-[10px] italic text-fg-faint">
                            no face detected ({r.metrics.n_total} img)
                        </div>
                    {/if}
                </div>
                {#each data.prompts as p (p.idx)}
                    {@const c = cellAt.get(`${r.lora_display}_${p.idx}`)}
                    {@const colDisabled = disabledCols.has(p.idx)}
                    <div
                        class="relative overflow-hidden rounded-md border border-border bg-bg-2 transition-opacity {colDisabled
                            ? 'opacity-40'
                            : ''}"
                        style="aspect-ratio: {p.width} / {p.height};"
                    >
                        {#if c?.image_url}
                            <button
                                type="button"
                                class="block h-full w-full cursor-zoom-in"
                                onclick={() => openZoom(r.lora_display, p.idx)}
                                aria-label={`Zoom ${r.lora_display} × prompt ${p.idx}`}
                            >
                                <img
                                    src="{c.image_url}?w=512"
                                    alt={`${r.lora_display} :: ${p.text}`}
                                    class="block h-full w-full object-cover transition-transform hover:scale-[1.02]"
                                    loading="lazy"
                                />
                            </button>
                            {#if c.face_score != null}
                                <span
                                    class="pointer-events-none absolute right-1 top-1 rounded-full px-1.5 py-0.5 text-[10px] font-semibold tabular-nums shadow-sm {BADGE_CLS[
                                        scoreClass(c.face_score)
                                    ]}"
                                    title="face↔centroid similarity">{fmtScore(c.face_score)}</span
                                >
                            {/if}
                        {:else}
                            <div
                                class="flex h-full w-full items-center justify-center bg-bg-1 text-[10px] text-fg-faint"
                            >
                                {#if isLive}
                                    <svg
                                        class="h-4 w-4 animate-pulse opacity-50"
                                        viewBox="0 0 24 24"
                                        fill="currentColor"
                                    >
                                        <rect width="24" height="24" rx="4" />
                                    </svg>
                                {:else}
                                    —
                                {/if}
                            </div>
                        {/if}
                    </div>
                {/each}
            {/each}
        </div>
        {:else if view === 'graph'}
            <!-- Graph view. Same source data as the grid (displayRows +
                 disabledCols + cellAt), so disabling a column or
                 changing per-row run selection in the Grid tab is
                 reflected here on tab switch. -->
            <div class="flex flex-col gap-3 p-6">
                <div
                    class="flex flex-wrap items-center gap-3 rounded-md border border-border bg-bg-2 px-3 py-2 text-xs"
                >
                    <label class="flex items-center gap-1.5">
                        <span class="text-fg-muted">Prompt:</span>
                        <select
                            class="rounded border border-border bg-bg-1 px-1.5 py-0.5 text-xs text-fg hover:border-accent/50"
                            value={promptSel === 'all' ? 'all' : String(promptSel)}
                            onchange={(e) => {
                                const v = (e.target as HTMLSelectElement).value;
                                promptSel = v === 'all' ? 'all' : Number(v);
                            }}
                        >
                            <option value="all">All (aggregate)</option>
                            {#each data.prompts as p (p.idx)}
                                <option value={String(p.idx)}>{promptOptionLabel(p.idx, p.text)}</option>
                            {/each}
                        </select>
                    </label>
                    <span class="mx-1 h-3 w-px bg-border"></span>
                    <label class="flex items-center gap-1.5">
                        <span class="text-fg-muted">Smooth:</span>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            step="5"
                            bind:value={smoothPct}
                            class="h-1 w-28 cursor-pointer accent-accent"
                            aria-label="Trend smoothing amount"
                        />
                        <span class="w-8 tabular-nums text-fg-faint">{smoothPct}%</span>
                    </label>
                    <span class="mx-1 h-3 w-px bg-border"></span>
                    <label class="flex items-center gap-1.5">
                        <span class="text-fg-muted">Y-scale:</span>
                        <select
                            bind:value={yScale}
                            class="rounded border border-border bg-bg-1 px-1.5 py-0.5 text-xs text-fg"
                            title="Stretch the top of the range so a +0.01 near the ceiling — which closes far more of the remaining gap — reads as the bigger gain it is. Labels stay raw similarity; ceiling 0.85. Fisher z = arctanh(x) (standard for cosine); % left = -ln(1-x) (fraction of remaining gap)."
                        >
                            <option value="linear">Linear</option>
                            <option value="fisher">Headroom (z)</option>
                            <option value="loghead">Headroom (% left)</option>
                        </select>
                    </label>
                    {#if promptSel === 'all'}
                        <span class="mx-1 h-3 w-px bg-border"></span>
                        <span class="text-fg-muted">Series:</span>
                        <label class="flex cursor-pointer items-center gap-1 select-none">
                            <input
                                type="checkbox"
                                class="h-3.5 w-3.5 rounded border-border bg-bg-1 text-accent"
                                bind:checked={showMedian}
                            />
                            <span class="font-medium text-emerald-300">median</span>
                        </label>
                        <label class="flex cursor-pointer items-center gap-1 select-none">
                            <input
                                type="checkbox"
                                class="h-3.5 w-3.5 rounded border-border bg-bg-1 text-accent"
                                bind:checked={showP20}
                            />
                            <span class="font-medium text-amber-300">p20</span>
                        </label>
                        <label class="flex cursor-pointer items-center gap-1 select-none">
                            <input
                                type="checkbox"
                                class="h-3.5 w-3.5 rounded border-border bg-bg-1 text-accent"
                                bind:checked={showMax}
                            />
                            <span class="font-medium text-sky-300">max</span>
                        </label>
                    {:else}
                        <span class="text-fg-faint">
                            (median/p20/max collapse to a single score in per-prompt mode)
                        </span>
                    {/if}
                    {#if disabledCols.size > 0 && promptSel === 'all'}
                        <span class="mx-1 h-3 w-px bg-border"></span>
                        <span class="text-fg-faint">
                            metrics exclude {disabledCols.size} disabled column{disabledCols.size === 1
                                ? ''
                                : 's'}
                        </span>
                    {/if}
                </div>

                <div class="rounded-md border border-border bg-bg-2 p-3">
                    {#if chartPoints.length === 0}
                        <div class="flex h-64 items-center justify-center text-sm text-fg-muted">
                            No data points to plot.
                        </div>
                    {:else}
                        <!-- SVG chart: 0..1 Y axis, X axis = LoRA step (or
                             index). Threshold lines at 0.35 / 0.5 match
                             the cell badge color thresholds. -->
                        <!-- preserveAspectRatio defaults to xMidYMid meet —
                             the SVG keeps its viewBox aspect (1000×380) and
                             the browser computes the rendered height from
                             the parent's width. Text glyphs scale uniformly,
                             no horizontal squish + label overlap. -->
                        <svg
                            viewBox="0 0 {CHART_W} {CHART_H}"
                            class="block w-full h-auto"
                            onmousemove={onPlotMove}
                            onmouseleave={() => (chartHover = null)}
                            role="img"
                            aria-label="Score vs LoRA step"
                        >
                            <!-- Y grid + threshold bands -->
                            <rect
                                x={CHART_PAD_L}
                                y={yPos(1)}
                                width={chartPlotW}
                                height={yPos(TH_GOOD) - yPos(1)}
                                fill="rgba(16,185,129,0.06)"
                            />
                            <rect
                                x={CHART_PAD_L}
                                y={yPos(TH_GOOD)}
                                width={chartPlotW}
                                height={yPos(TH_OK) - yPos(TH_GOOD)}
                                fill="rgba(245,158,11,0.05)"
                            />
                            <rect
                                x={CHART_PAD_L}
                                y={yPos(TH_OK)}
                                width={chartPlotW}
                                height={yPos(0) - yPos(TH_OK)}
                                fill="rgba(239,68,68,0.05)"
                            />

                            <!-- Y axis ticks (raw similarity labels; positions
                                 follow the chosen scale). -->
                            {#each yTicks as ty (ty)}
                                <line
                                    x1={CHART_PAD_L}
                                    x2={CHART_PAD_L + chartPlotW}
                                    y1={yPos(ty)}
                                    y2={yPos(ty)}
                                    stroke={ty === TH_OK || ty === TH_GOOD
                                        ? 'rgba(255,255,255,0.18)'
                                        : 'rgba(255,255,255,0.06)'}
                                    stroke-dasharray={ty === TH_OK || ty === TH_GOOD ? '4 4' : '0'}
                                    stroke-width="1"
                                />
                                <text
                                    x={CHART_PAD_L - 6}
                                    y={yPos(ty) + 4}
                                    text-anchor="end"
                                    font-size="11"
                                    fill="#8a8a9a"
                                    font-family="ui-monospace, monospace"
                                >
                                    {ty.toFixed(2)}
                                </text>
                            {/each}

                            <!-- X axis ticks: ~8 evenly-spaced labels along
                                 the X domain. The last index is auto-included
                                 only when it's at least step/2 away from the
                                 previous regular tick; otherwise the
                                 predecessor is dropped and the end-of-axis
                                 label wins. Prevents collisions like
                                 "5900D00" at high LoRA counts. -->
                            {#each chartPoints as p, i (p.lora)}
                                {#if xLabelIndices.has(i)}
                                    <line
                                        x1={xPos(p.x)}
                                        x2={xPos(p.x)}
                                        y1={CHART_PAD_T + chartPlotH}
                                        y2={CHART_PAD_T + chartPlotH + 4}
                                        stroke="rgba(255,255,255,0.2)"
                                    />
                                    <text
                                        x={xPos(p.x)}
                                        y={CHART_PAD_T + chartPlotH + 18}
                                        text-anchor="middle"
                                        font-size="11"
                                        fill="#8a8a9a"
                                        font-family="ui-monospace, monospace"
                                    >
                                        {p.xLabel}
                                    </text>
                                {/if}
                            {/each}

                            <!-- Axes outline -->
                            <line
                                x1={CHART_PAD_L}
                                x2={CHART_PAD_L}
                                y1={CHART_PAD_T}
                                y2={CHART_PAD_T + chartPlotH}
                                stroke="rgba(255,255,255,0.2)"
                            />
                            <line
                                x1={CHART_PAD_L}
                                x2={CHART_PAD_L + chartPlotW}
                                y1={CHART_PAD_T + chartPlotH}
                                y2={CHART_PAD_T + chartPlotH}
                                stroke="rgba(255,255,255,0.2)"
                            />

                            {#if promptSel === 'all'}
                                {#if showMax}
                                    {#if smoothPct > 0}
                                        <path
                                            d={pathFor(chartPoints.map((p) => p.max))}
                                            stroke="#7dd3fc"
                                            stroke-width="1"
                                            fill="none"
                                            opacity="0.22"
                                        />
                                    {/if}
                                    <path
                                        d={pathFor(maxSmoothed)}
                                        stroke="#7dd3fc"
                                        stroke-width="1.5"
                                        fill="none"
                                    />
                                {/if}
                                {#if showP20}
                                    {#if smoothPct > 0}
                                        <path
                                            d={pathFor(chartPoints.map((p) => p.p20))}
                                            stroke="#fcd34d"
                                            stroke-width="1"
                                            fill="none"
                                            stroke-dasharray="6 4"
                                            opacity="0.22"
                                        />
                                    {/if}
                                    <path
                                        d={pathFor(p20Smoothed)}
                                        stroke="#fcd34d"
                                        stroke-width="1.5"
                                        fill="none"
                                        stroke-dasharray="6 4"
                                    />
                                {/if}
                                {#if showMedian}
                                    {#if smoothPct > 0}
                                        <path
                                            d={pathFor(chartPoints.map((p) => p.median))}
                                            stroke="#6ee7b7"
                                            stroke-width="1.25"
                                            fill="none"
                                            opacity="0.22"
                                        />
                                    {/if}
                                    <path
                                        d={pathFor(medianSmoothed)}
                                        stroke="#6ee7b7"
                                        stroke-width="2.5"
                                        fill="none"
                                    />
                                {/if}
                                <!-- Dots for the median series (the eye
                                     follows it). p20 / max stay as lines
                                     to avoid visual clutter. -->
                                {#if showMedian}
                                    {#each chartPoints as p (p.lora)}
                                        {#if p.median != null}
                                            <circle
                                                cx={xPos(p.x)}
                                                cy={yPos(p.median)}
                                                r="2"
                                                fill="#6ee7b7"
                                            />
                                        {/if}
                                    {/each}
                                {/if}
                            {:else}
                                {#if smoothPct > 0}
                                    <path
                                        d={pathFor(chartPoints.map((p) => p.single))}
                                        stroke="#6ee7b7"
                                        stroke-width="1.25"
                                        fill="none"
                                        opacity="0.22"
                                    />
                                {/if}
                                <path
                                    d={pathFor(singleSmoothed)}
                                    stroke="#6ee7b7"
                                    stroke-width="2.5"
                                    fill="none"
                                />
                                {#each chartPoints as p (p.lora)}
                                    {#if p.single != null}
                                        <circle
                                            cx={xPos(p.x)}
                                            cy={yPos(p.single)}
                                            r="2"
                                            fill="#6ee7b7"
                                        />
                                    {/if}
                                {/each}
                            {/if}

                            {#if chartHover}
                                {@const hp = chartPoints[chartHover.idx]}
                                <!-- Hover crosshair -->
                                <line
                                    x1={xPos(hp.x)}
                                    x2={xPos(hp.x)}
                                    y1={CHART_PAD_T}
                                    y2={CHART_PAD_T + chartPlotH}
                                    stroke="rgba(255,255,255,0.25)"
                                    stroke-dasharray="2 3"
                                />
                            {/if}
                        </svg>

                        {#if chartHover}
                            {@const hp = chartPoints[chartHover.idx]}
                            {@const i = chartHover.idx}
                            <div
                                class="mt-2 rounded border border-border bg-bg-1 p-2 text-[11px]"
                            >
                                <div class="font-mono text-fg-muted">{hp.lora}</div>
                                {#if promptSel === 'all'}
                                    <!-- Real (raw) series values. When smoothing is
                                         on we tag the row and add a faint smoothed
                                         row below so both numbers are visible. -->
                                    <div class="mt-1 flex gap-3 tabular-nums">
                                        {#if smoothPct > 0}
                                            <span class="w-12 text-fg-faint">real</span>
                                        {/if}
                                        {#if showMedian}
                                            <span class="text-emerald-300"
                                                >median: <b>{fmtScore(hp.median)}</b></span
                                            >
                                        {/if}
                                        {#if showP20}
                                            <span class="text-amber-300"
                                                >p20: <b>{fmtScore(hp.p20)}</b></span
                                            >
                                        {/if}
                                        {#if showMax}
                                            <span class="text-sky-300"
                                                >max: <b>{fmtScore(hp.max)}</b></span
                                            >
                                        {/if}
                                    </div>
                                    {#if smoothPct > 0}
                                        <div class="mt-0.5 flex gap-3 tabular-nums opacity-65">
                                            <span class="w-12 text-fg-faint">smoothed</span>
                                            {#if showMedian}
                                                <span class="text-emerald-300"
                                                    >median: <b>{fmtScore(medianSmoothed[i])}</b></span
                                                >
                                            {/if}
                                            {#if showP20}
                                                <span class="text-amber-300"
                                                    >p20: <b>{fmtScore(p20Smoothed[i])}</b></span
                                                >
                                            {/if}
                                            {#if showMax}
                                                <span class="text-sky-300"
                                                    >max: <b>{fmtScore(maxSmoothed[i])}</b></span
                                                >
                                            {/if}
                                        </div>
                                    {/if}
                                {:else}
                                    <div class="mt-1 flex gap-3 tabular-nums text-emerald-300">
                                        {#if smoothPct > 0}
                                            <span class="w-12 text-fg-faint">real</span>
                                        {/if}
                                        <span>score: <b>{fmtScore(hp.single)}</b></span>
                                    </div>
                                    {#if smoothPct > 0}
                                        <div
                                            class="mt-0.5 flex gap-3 tabular-nums text-emerald-300 opacity-65"
                                        >
                                            <span class="w-12 text-fg-faint">smoothed</span>
                                            <span>score: <b>{fmtScore(singleSmoothed[i])}</b></span>
                                        </div>
                                    {/if}
                                {/if}
                            </div>
                        {/if}
                    {/if}
                </div>
            </div>
        {/if}
    {/if}
</MainPanel>

<svelte:window onkeydown={onKey} />

{#if zoomKey}
    {@const zoomPrompt = data.prompts.find((p) => p.idx === zoomKey!.prompt_idx)}
    {@const zoomRunId = runByLora[zoomKey.lora_display] ?? null}
    {@const zoomRun = zoomRunId != null ? runById.get(zoomRunId) ?? null : null}
    <div
        class="fixed inset-0 z-50 flex flex-col bg-black/92 p-6"
        onclick={closeZoom}
        role="button"
        tabindex="-1"
        aria-label="Close image"
        onkeydown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') closeZoom();
        }}
    >
        <button
            type="button"
            class="absolute right-4 top-4 z-10 text-3xl text-fg-muted hover:text-fg"
            onclick={(e) => {
                e.stopPropagation();
                closeZoom();
            }}
            aria-label="Close"
        >
            ×
        </button>

        <button
            type="button"
            class="absolute left-4 top-1/2 z-10 -translate-y-1/2 px-4 py-3 text-4xl text-fg-muted opacity-60 hover:text-fg hover:opacity-100"
            onclick={(e) => {
                e.stopPropagation();
                step(navMode === 'prompts' ? 'prompt' : 'lora', -1);
            }}
            aria-label={navMode === 'prompts' ? 'Previous prompt' : 'Previous LoRA'}
        >
            ‹
        </button>
        <button
            type="button"
            class="absolute right-4 top-1/2 z-10 -translate-y-1/2 px-4 py-3 text-4xl text-fg-muted opacity-60 hover:text-fg hover:opacity-100"
            onclick={(e) => {
                e.stopPropagation();
                step(navMode === 'prompts' ? 'prompt' : 'lora', +1);
            }}
            aria-label={navMode === 'prompts' ? 'Next prompt' : 'Next LoRA'}
        >
            ›
        </button>

        <div class="mx-auto flex h-full w-full max-w-screen-2xl flex-col items-stretch gap-3">
            <div class="flex min-h-0 flex-1 items-center justify-center">
                {#if zoomCell?.image_url}
                    <button
                        type="button"
                        class="block max-h-full max-w-full"
                        onclick={(e) => e.stopPropagation()}
                        aria-label="Image"
                    >
                        <img
                            src={zoomCell.image_url}
                            alt={`${zoomKey.lora_display} :: ${zoomPrompt?.text}`}
                            class="block max-h-full max-w-full rounded-md object-contain shadow-2xl"
                        />
                    </button>
                {:else}
                    <div
                        class="flex h-64 w-64 items-center justify-center rounded-md border border-border bg-bg-2 text-sm text-fg-faint"
                    >
                        {#if isLive}
                            <div class="flex flex-col items-center gap-2">
                                <svg
                                    class="h-6 w-6 animate-spin"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="2.5"
                                >
                                    <path d="M12 3a9 9 0 1 0 9 9" stroke-linecap="round" />
                                </svg>
                                <span>generating…</span>
                            </div>
                        {:else}
                            <span>missing</span>
                        {/if}
                    </div>
                {/if}
            </div>

            <div class="shrink-0 rounded-md border border-border bg-bg-1 p-3 text-xs">
                <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span class="font-semibold text-accent">{zoomKey.lora_display}</span>
                    <span class="font-mono text-[10px] text-fg-faint">
                        #{zoomPrompt?.idx ?? '?'} · {zoomPrompt?.width}×{zoomPrompt?.height}
                    </span>
                    {#if zoomRun}
                        <span
                            class="rounded border border-border bg-bg-2 px-1.5 py-0.5 text-[10px] text-fg-muted"
                            title={runFullLabel(zoomRun)}
                        >
                            {runShortLabel(zoomRun)}
                        </span>
                    {/if}
                    {#if zoomCell?.face_score != null}
                        <span
                            class="rounded-full px-2 py-0.5 text-[10px] font-semibold tabular-nums {BADGE_CLS[
                                scoreClass(zoomCell.face_score)
                            ]}"
                            title="face↔centroid similarity"
                        >
                            sim {fmtScore(zoomCell.face_score)}
                        </span>
                    {/if}
                    <button
                        type="button"
                        class="rounded-full bg-bg-3 px-2 py-0.5 text-[10px] font-medium text-fg-muted hover:bg-bg-2 hover:text-fg"
                        onclick={(e) => {
                            e.stopPropagation();
                            toggleNavMode();
                        }}
                        title="Toggle ←/→ axis (or press T)"
                    >
                        ←/→ {navMode === 'prompts' ? 'prompts (same LoRA)' : 'LoRAs (same prompt)'}
                    </button>
                    <span class="ml-auto text-[10px] text-fg-faint">
                        ←/→ {navMode} · ↑/↓ {navMode === 'prompts' ? 'loras' : 'prompts'} · T toggle · Esc close
                    </span>
                </div>
                <p class="mt-2 whitespace-pre-wrap text-fg-muted">{zoomPrompt?.text ?? ''}</p>
            </div>
        </div>
    </div>
{/if}
