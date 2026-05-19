"""Self-contained HTML grid renderer.

Generates `index.html` with rows = LoRAs/LoKrs, cols = prompts, click-to-zoom.
If face-recognition scores are present (each row has a `metrics` dict), they
appear as per-cell badges, per-row aggregates, and in the lightbox caption.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from .images import atomic_write


def write_html(output_dir: Path, manifest: dict):
    """Self-contained HTML grid: rows = loras, cols = prompts, click-to-zoom.
    If face-recognition scores are present, they're surfaced as per-cell
    badges, per-row aggregates, and in the lightbox caption."""
    rows = manifest["rows"]
    prompts = manifest["prompts"]
    meta = manifest["meta"]

    face_meta = meta.get("face_recognition") or {}
    face_on = bool(face_meta) and any(r.get("metrics") for r in rows)
    thresholds = face_meta.get("thresholds") or {"good": 0.5, "ok": 0.35}
    th_good = float(thresholds.get("good", 0.5))
    th_ok = float(thresholds.get("ok", 0.35))

    def score_class(s):
        if s is None:
            return "score-na"
        if s >= th_good:
            return "score-good"
        if s >= th_ok:
            return "score-ok"
        return "score-bad"

    def fmt_score(s):
        return "—" if s is None else f"{s:.2f}"

    def cell(img_rel, lora, prompt, p_idx, l_idx, score):
        if img_rel is None:
            return (f'<div class="cell empty" data-lora-idx="{l_idx}" data-prompt-idx="{p_idx}">'
                    f'<span>missing</span></div>')
        badge = ""
        score_attr = ""
        if face_on:
            score_attr = f' data-score="{"" if score is None else f"{score:.4f}"}"'
            badge = (f'<div class="score-badge {score_class(score)}" '
                     f'title="face↔centroid similarity">{fmt_score(score)}</div>')
        return (
            f'<div class="cell" '
            f'data-src="{html.escape(img_rel)}" '
            f'data-lora="{html.escape(lora)}" '
            f'data-prompt="{html.escape(prompt)}" '
            f'data-lora-idx="{l_idx}" '
            f'data-prompt-idx="{p_idx}"'
            f'{score_attr}>'
            f'<img loading="lazy" src="{html.escape(img_rel)}" alt="{html.escape(lora)} :: {html.escape(prompt)}">'
            f'{badge}'
            f'</div>'
        )

    def row_label_html(lora, metrics, l_idx):
        name_html = f'<div class="lora-name" title="{html.escape(lora)}">{html.escape(lora)}</div>'
        if not face_on or not metrics or not metrics.get("n_faces"):
            extra = ""
            if face_on and metrics:
                extra = (f'<div class="row-metrics empty">no face detected '
                         f'({metrics.get("n_total", 0)} img)</div>')
            return f'<div class="row-label" data-row-idx="{l_idx}">{name_html}{extra}</div>'
        med = metrics["median"]
        cls_med = score_class(med)
        cls_p20 = score_class(metrics["p20"])
        cls_max = score_class(metrics["max"])
        m = (
            f'<div class="row-metrics">'
            f'  <div class="metric primary {cls_med}" title="median similarity">'
            f'    <span class="m-val">{metrics["median"]:.2f}</span>'
            f'    <span class="m-lbl">med</span>'
            f'  </div>'
            f'  <div class="metric {cls_p20}" title="20th percentile (worst-quartile floor)">'
            f'    <span class="m-val">{metrics["p20"]:.2f}</span>'
            f'    <span class="m-lbl">p20</span>'
            f'  </div>'
            f'  <div class="metric {cls_max}" title="max similarity in row">'
            f'    <span class="m-val">{metrics["max"]:.2f}</span>'
            f'    <span class="m-lbl">max</span>'
            f'  </div>'
            f'  <div class="metric muted" title="std dev / face count">'
            f'    <span class="m-val">σ {metrics["std"]:.2f}</span>'
            f'    <span class="m-lbl">{metrics["n_faces"]}/{metrics["n_total"]}</span>'
            f'  </div>'
            f'</div>'
        )
        return f'<div class="row-label" data-row-idx="{l_idx}">{name_html}{m}</div>'

    sort_btn_html = (
        '<button class="sort-btn" type="button" '
        'title="Sort rows by this column">↕</button>'
    ) if face_on else ""
    head_title_extra = " — click to toggle for metrics, ↕ to sort rows by this column" if face_on else ""
    header_cells = "".join(
        f'<div class="head prompt-head" data-prompt-idx="{i}" '
        f'title="{html.escape(p)}{head_title_extra}">'
        f'<span class="pill">#{i}</span>'
        f'{sort_btn_html}'
        f'<div class="prompt-text">{html.escape(p)}</div></div>'
        for i, p in enumerate(prompts)
    )

    row_html_parts = []
    for l_idx, row in enumerate(rows):
        lora = row["lora"]
        scores = row.get("scores") or [None] * len(prompts)
        metrics = row.get("metrics")
        cells = "".join(
            cell(row["images"][i], lora, prompts[i], i, l_idx, scores[i] if i < len(scores) else None)
            for i in range(len(prompts))
        )
        row_html_parts.append(row_label_html(lora, metrics, l_idx) + cells)
    rows_html = "".join(row_html_parts)

    n_cols = len(prompts) + 1  # +1 for the lora-label column
    # Cap thumbnails at a sane max so a single-prompt run doesn't stretch the
    # cell across the whole viewport. Lora-label column is also capped.
    label_min = "200px" if face_on else "180px"
    label_max = "240px" if face_on else "220px"
    # Cell sizing: smaller min/max than before to keep typical 5-7-prompt grids
    # within a 1280-1920 viewport. Cells still look ample at 180-300 px.
    grid_template = f"minmax({label_min}, {label_max}) " + " ".join(["minmax(180px, 300px)"] * len(prompts))
    aspect_ratio = f"{meta.get('width', 1)} / {meta.get('height', 1)}"

    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)

    # Legend / header chip describing face-rec status (only when enabled)
    face_legend = ""
    if face_on:
        face_legend = (
            f'<div class="face-legend" title="face↔centroid cosine similarity thresholds">'
            f'  <span class="legend-dot score-good"></span>≥ {th_good:.2f}'
            f'  <span class="legend-dot score-ok"></span>≥ {th_ok:.2f}'
            f'  <span class="legend-dot score-bad"></span>&lt; {th_ok:.2f}'
            f'  <span class="legend-dot score-na"></span>no face'
            f'  <span class="legend-sep"></span>'
            f'  <span class="legend-hint">sort:</span>'
            f'  <button class="legend-btn sort-median-btn" id="flgrid-sort-median" '
            f'type="button" title="Sort LoRAs by row median (click to cycle desc → asc → off)">'
            f'median <span class="dir">↕</span></button>'
            f'  <span class="legend-sep"></span>'
            f'  <button class="legend-btn" id="flgrid-reset-cols" type="button" '
            f'title="Re-enable every column">reset cols</button>'
            f'</div>'
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FLUX.2 LoRA Grid</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #0b0b0f;
    --bg-2: #14141c;
    --fg: #e9e9f0;
    --muted: #8a8aa0;
    --accent: #ff4ecd;
    --accent-2: #6f6bff;
    --border: #23232e;
    --shadow: 0 10px 40px rgba(0,0,0,.5);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg);
                font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, sans-serif; }}
  header {{
    position: sticky; top: 0; z-index: 5;
    padding: 16px 24px; backdrop-filter: blur(14px);
    background: linear-gradient(180deg, rgba(11,11,15,.92), rgba(11,11,15,.65));
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
  }}
  header h1 {{
    font-size: 16px; margin: 0; font-weight: 600; letter-spacing: .2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  header .meta {{ color: var(--muted); font-size: 12px; }}
  header .meta code {{ color: var(--fg); background: var(--bg-2); padding: 2px 6px; border-radius: 4px; }}

  /* No overflow here: setting overflow:auto would turn .grid-wrap into a
     scroll container, which steals sticky positioning away from the
     viewport — making .head and .row-label no longer pin to the visible
     window. We let the body handle scrolling instead (vertical via the
     page, horizontal automatically when the grid is wider than the
     viewport). */
  .grid-wrap {{ padding: 24px; }}
  .grid {{
    display: grid;
    grid-template-columns: {grid_template};
    gap: 10px;
    align-items: stretch;
  }}
  .head {{
    position: sticky; top: var(--header-h, 64px); z-index: 3;
    background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; min-height: 64px;
  }}
  .prompt-head {{ display: flex; flex-direction: column; gap: 6px; }}
  .pill {{
    display: inline-block; align-self: flex-start;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: white; font-weight: 600; font-size: 11px;
    padding: 2px 8px; border-radius: 999px; letter-spacing: .3px;
  }}
  .prompt-text {{
    color: var(--fg); font-size: 12.5px;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .row-label {{
    position: sticky; left: 24px; z-index: 2;
    background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; display: flex; flex-direction: column; gap: 8px;
    justify-content: center;
    word-break: break-all; font-size: 12.5px;
  }}
  .row-label .lora-name {{ font-weight: 600; line-height: 1.3; }}
  .row-metrics {{
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 4px; margin-top: 2px;
  }}
  .row-metrics.empty {{
    display: block; color: var(--muted); font-style: italic;
    font-size: 11px; font-weight: 400;
  }}
  .row-metrics .metric {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 3px 4px; border-radius: 6px; background: rgba(255,255,255,.03);
    border: 1px solid var(--border); min-width: 0;
  }}
  .row-metrics .metric.primary {{ border-width: 1.5px; }}
  .row-metrics .m-val {{ font-size: 13px; font-weight: 700; line-height: 1.1; white-space: nowrap; }}
  .row-metrics .m-lbl {{ font-size: 9.5px; color: var(--muted); letter-spacing: .3px; text-transform: uppercase; margin-top: 1px; }}
  .row-metrics .metric.muted .m-val {{ font-size: 11px; font-weight: 600; color: var(--muted); }}
  .metric.score-good {{ color: #5dd97e; border-color: rgba(93,217,126,.45); background: rgba(93,217,126,.08); }}
  .metric.score-ok   {{ color: #f3c969; border-color: rgba(243,201,105,.45); background: rgba(243,201,105,.08); }}
  .metric.score-bad  {{ color: #ef7676; border-color: rgba(239,118,118,.45); background: rgba(239,118,118,.08); }}
  .metric.score-na   {{ color: #777; }}
  .cell {{
    position: relative; aspect-ratio: {aspect_ratio}; overflow: hidden; border-radius: 10px;
    border: 1px solid var(--border); background: var(--bg-2); cursor: zoom-in;
    transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
  }}
  .score-badge {{
    position: absolute; top: 6px; right: 6px; z-index: 1;
    background: rgba(0,0,0,.72); color: white;
    padding: 3px 8px; border-radius: 999px;
    font-size: 11.5px; font-weight: 700; letter-spacing: .2px;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,.08);
  }}
  .score-badge.score-good {{ background: rgba(35,80,45,.85); color: #b3f5c5; border-color: rgba(93,217,126,.5); }}
  .score-badge.score-ok   {{ background: rgba(74,57,16,.85); color: #ffdf8a; border-color: rgba(243,201,105,.5); }}
  .score-badge.score-bad  {{ background: rgba(72,28,28,.85); color: #ffb1b1; border-color: rgba(239,118,118,.5); }}
  .score-badge.score-na   {{ background: rgba(0,0,0,.55); color: #aaa; }}
  .face-legend {{
    display: flex; align-items: center; gap: 8px; font-size: 11px;
    color: var(--muted); flex-wrap: wrap;
  }}
  .face-legend .legend-dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin: 0 4px 0 8px; border: 1px solid var(--border);
  }}
  .legend-dot.score-good {{ background: #5dd97e; }}
  .legend-dot.score-ok   {{ background: #f3c969; }}
  .legend-dot.score-bad  {{ background: #ef7676; }}
  .legend-dot.score-na   {{ background: #555; }}
  .legend-sep {{
    display: inline-block; width: 1px; height: 14px; background: var(--border);
    margin: 0 6px;
  }}
  .legend-hint {{ font-style: italic; color: var(--muted); }}
  .legend-btn {{
    background: var(--bg-2); color: var(--fg); border: 1px solid var(--border);
    padding: 3px 10px; border-radius: 999px; font-size: 11px; cursor: pointer;
    transition: border-color .15s ease, transform .1s ease;
  }}
  .legend-btn:hover {{ border-color: var(--accent); }}
  .legend-btn:active {{ transform: scale(.97); }}
  /* NOTE: don't override `position` on .prompt-head — it inherits sticky
     from .head and breaking that would unstick the column headers. */
  .prompt-head {{ cursor: pointer; user-select: none; }}
  .prompt-head.disabled {{
    opacity: .35;
    background: rgba(255, 78, 205, .04);
    border-style: dashed;
  }}
  .prompt-head.disabled .prompt-text {{ text-decoration: line-through; }}
  .prompt-head.disabled .pill {{
    background: linear-gradient(90deg, #4a4a55, #3a3a45);
  }}
  .cell.col-disabled {{ opacity: .25; filter: grayscale(.8); }}
  .cell.col-disabled .score-badge {{ opacity: .6; }}

  /* Sort-by-this-column button on each prompt header. Sits in a top-right
     corner of the header cell; click is stop-propagation so it doesn't
     also flip the disable-toggle. */
  .sort-btn {{
    position: absolute; top: 6px; right: 8px; z-index: 4;
    background: rgba(255,255,255,.05); color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px;
    width: 22px; height: 22px; padding: 0;
    font-size: 12px; line-height: 1; cursor: pointer;
    transition: color .15s ease, border-color .15s ease, background .15s ease;
    display: inline-flex; align-items: center; justify-content: center;
  }}
  .sort-btn:hover {{ color: var(--fg); border-color: var(--accent); }}
  .sort-btn.active {{
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: white; border-color: transparent; font-weight: 700;
  }}
  .legend-btn.sort-median-btn.active {{
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: white; border-color: transparent;
  }}
  .legend-btn .dir {{ font-weight: 700; margin-left: 2px; }}
  /* The prompt-head has its own click handler; ensure sort-btn lives ABOVE
     the rest of the header content for clicks. */
  .prompt-head {{ overflow: hidden; }}
  .cell:hover {{
    transform: translateY(-2px);
    border-color: var(--accent);
    box-shadow: 0 8px 30px rgba(255, 78, 205, .15);
  }}
  .cell img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .cell.empty {{
    display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 12px; cursor: default;
  }}
  .corner-spacer {{ /* top-left empty cell of the header row */ }}

  /* Lightbox */
  .lb {{
    position: fixed; inset: 0; background: rgba(5,5,10,.92);
    display: none; align-items: center; justify-content: center;
    z-index: 50; padding: 24px;
  }}
  .lb.open {{ display: flex; }}
  .lb-inner {{
    max-width: 95vw; max-height: 95vh;
    display: flex; flex-direction: column; gap: 12px; align-items: center;
  }}
  .lb img {{
    max-width: 95vw; max-height: 80vh; border-radius: 12px;
    box-shadow: var(--shadow); object-fit: contain; background: #000;
  }}
  .lb-caption {{
    max-width: 95vw; color: var(--fg); font-size: 13px;
    background: var(--bg-2); border: 1px solid var(--border);
    padding: 10px 14px; border-radius: 10px;
  }}
  .lb-caption .lora {{ color: var(--accent); font-weight: 600; }}
  .lb-caption .prompt {{ color: var(--muted); margin-top: 4px; }}
  .lb-score-row {{
    margin-top: 8px; display: flex; align-items: center; gap: 10px;
    font-size: 12px; color: var(--muted);
  }}
  .lb-score-pill {{
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    font-weight: 700; font-size: 13px; border: 1px solid var(--border);
    background: var(--bg);
  }}
  .lb-score-pill.score-good {{ background: rgba(35,80,45,.85); color: #b3f5c5; border-color: rgba(93,217,126,.5); }}
  .lb-score-pill.score-ok   {{ background: rgba(74,57,16,.85); color: #ffdf8a; border-color: rgba(243,201,105,.5); }}
  .lb-score-pill.score-bad  {{ background: rgba(72,28,28,.85); color: #ffb1b1; border-color: rgba(239,118,118,.5); }}
  .lb-score-pill.score-na   {{ background: rgba(0,0,0,.5); color: #aaa; }}
  .lb-close {{
    position: fixed; top: 16px; right: 20px; color: var(--fg);
    font-size: 28px; cursor: pointer; user-select: none; opacity: .7;
  }}
  .lb-close:hover {{ opacity: 1; }}
  .lb-nav {{
    position: fixed; top: 50%; transform: translateY(-50%);
    color: var(--fg); font-size: 36px; cursor: pointer; user-select: none;
    opacity: .5; padding: 12px 18px;
  }}
  .lb-nav:hover {{ opacity: 1; }}
  .lb-nav.prev {{ left: 12px; }}
  .lb-nav.next {{ right: 12px; }}
  .lb-mode {{
    position: fixed; top: 16px; left: 20px; z-index: 60;
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--bg-2); border: 1px solid var(--border);
    color: var(--fg); font-size: 12px; padding: 6px 12px;
    border-radius: 999px; cursor: pointer; user-select: none;
    transition: border-color .15s ease, transform .1s ease;
  }}
  .lb-mode:hover {{ border-color: var(--accent); }}
  .lb-mode:active {{ transform: scale(.97); }}
  .lb-mode .axis {{ font-weight: 600; color: var(--accent); }}
  .lb-mode .kbd {{
    font-size: 10px; color: var(--muted); background: var(--bg);
    padding: 1px 5px; border-radius: 4px; border: 1px solid var(--border);
  }}

  details.meta-panel {{
    margin: 0 24px 24px; background: var(--bg-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 14px;
  }}
  details.meta-panel summary {{ cursor: pointer; color: var(--muted); }}
  details.meta-panel pre {{ color: var(--fg); margin: 8px 0 0; font-size: 12px; overflow: auto; }}
</style>
</head>
<body>
<header>
  <h1>✦ FLUX.2-klein LoRA grid</h1>
  <div class="meta">
    <code>{len(rows)}</code> loras × <code>{len(prompts)}</code> prompts ·
    seed <code>{meta.get("seed")}</code> ·
    <code>{meta.get("width")}×{meta.get("height")}</code>
    {face_legend}
  </div>
</header>

<div class="grid-wrap">
  <div class="grid" style="grid-template-columns: {grid_template};">
    <div class="head corner-spacer"></div>
    {header_cells}
    {rows_html}
  </div>
</div>

<details class="meta-panel">
  <summary>Run metadata</summary>
  <pre>{html.escape(meta_json)}</pre>
</details>

<div class="lb" id="lb">
  <span class="lb-close" id="lb-close">&times;</span>
  <button class="lb-mode" id="lb-mode" type="button" title="Toggle scroll axis (T)">
    <span>scroll axis:</span>
    <span class="axis" id="lb-mode-label">prompts</span>
    <span class="kbd">T</span>
  </button>
  <span class="lb-nav prev" id="lb-prev">&#8249;</span>
  <span class="lb-nav next" id="lb-next">&#8250;</span>
  <div class="lb-inner">
    <img id="lb-img" src="" alt="">
    <div class="lb-caption">
      <div class="lora" id="lb-lora"></div>
      <div class="prompt" id="lb-prompt"></div>
      <div class="lb-score-row" id="lb-score-row" style="display:none;">
        <span class="lb-score-pill" id="lb-score-pill">—</span>
        <span class="lb-score-label">face↔centroid similarity</span>
      </div>
    </div>
  </div>
</div>

<script>
  const N_LORAS = {len(rows)};
  const N_PROMPTS = {len(prompts)};

  // Sticky-column-header offset: the page header's height varies with viewport
  // width (the face-rec legend may wrap). Re-measure on resize so the column
  // headers always stick just below it.
  function syncHeaderHeight() {{
    const h = document.querySelector('header');
    if (!h) return;
    document.documentElement.style.setProperty('--header-h', h.offsetHeight + 'px');
  }}
  syncHeaderHeight();
  window.addEventListener('resize', syncHeaderHeight);

  // Build a 2D lookup of present (non-empty) cells: cellMap["l_p"] -> element
  const cellMap = {{}};
  document.querySelectorAll('.cell').forEach(c => {{
    if (c.classList.contains('empty')) return;
    cellMap[c.dataset.loraIdx + '_' + c.dataset.promptIdx] = c;
  }});

  const lb = document.getElementById('lb');
  const lbImg = document.getElementById('lb-img');
  const lbLora = document.getElementById('lb-lora');
  const lbPrompt = document.getElementById('lb-prompt');
  const lbScoreRow = document.getElementById('lb-score-row');
  const lbScorePill = document.getElementById('lb-score-pill');
  const modeBtn = document.getElementById('lb-mode');
  const modeLabel = document.getElementById('lb-mode-label');

  const FACE_ON = {("true" if face_on else "false")};
  const TH_GOOD = {th_good};
  const TH_OK = {th_ok};
  function scoreClass(s) {{
    if (s === null || s === undefined || s === "" || isNaN(s)) return 'score-na';
    if (s >= TH_GOOD) return 'score-good';
    if (s >= TH_OK)   return 'score-ok';
    return 'score-bad';
  }}

  let curL = 0, curP = 0;
  // 'prompts' = ←/→ change prompt (same lora) — i.e. scroll across the lora's row
  // 'loras'   = ←/→ change lora (same prompt) — i.e. scroll the prompt's column
  let mode = 'prompts';

  function setMode(m) {{
    mode = m;
    modeLabel.textContent = m === 'prompts' ? 'prompts (same LoRA)' : 'LoRAs (same prompt)';
  }}

  function show(l, p) {{
    const c = cellMap[l + '_' + p];
    if (!c) return false;
    curL = l; curP = p;
    lbImg.src = c.dataset.src;
    lbLora.textContent = c.dataset.lora;
    lbPrompt.textContent = '#' + c.dataset.promptIdx + ' · ' + c.dataset.prompt;
    if (FACE_ON) {{
      const raw = c.dataset.score;
      const s = (raw === undefined || raw === '') ? null : parseFloat(raw);
      const cls = scoreClass(s);
      lbScorePill.className = 'lb-score-pill ' + cls;
      lbScorePill.textContent = (s === null) ? 'no face' : s.toFixed(3);
      lbScoreRow.style.display = '';
    }} else {{
      lbScoreRow.style.display = 'none';
    }}
    lb.classList.add('open');
    return true;
  }}
  function close() {{ lb.classList.remove('open'); lbImg.src = ''; }}

  // Walk along one axis with wrap-around, skipping missing cells.
  function step(axis, dir) {{
    const N = axis === 'prompts' ? N_PROMPTS : N_LORAS;
    for (let i = 1; i <= N; i++) {{
      let l = curL, p = curP;
      if (axis === 'prompts') p = ((curP + dir * i) % N + N) % N;
      else                    l = ((curL + dir * i) % N + N) % N;
      if (cellMap[l + '_' + p]) {{ show(l, p); return; }}
    }}
  }}

  document.querySelectorAll('.cell:not(.empty)').forEach(c => {{
    c.addEventListener('click', () => show(parseInt(c.dataset.loraIdx, 10),
                                           parseInt(c.dataset.promptIdx, 10)));
  }});
  document.getElementById('lb-close').addEventListener('click', close);
  document.getElementById('lb-prev').addEventListener('click', e => {{ e.stopPropagation(); step(mode, -1); }});
  document.getElementById('lb-next').addEventListener('click', e => {{ e.stopPropagation(); step(mode, +1); }});
  modeBtn.addEventListener('click', e => {{ e.stopPropagation(); setMode(mode === 'prompts' ? 'loras' : 'prompts'); }});
  lb.addEventListener('click', e => {{ if (e.target === lb) close(); }});

  document.addEventListener('keydown', e => {{
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape')              close();
    else if (e.key === 'ArrowLeft')      step(mode, -1);
    else if (e.key === 'ArrowRight')     step(mode, +1);
    // Up/Down always navigates the *other* axis, regardless of mode
    else if (e.key === 'ArrowUp')        step(mode === 'prompts' ? 'loras' : 'prompts', -1);
    else if (e.key === 'ArrowDown')      step(mode === 'prompts' ? 'loras' : 'prompts', +1);
    else if (e.key === 't' || e.key === 'T') setMode(mode === 'prompts' ? 'loras' : 'prompts');
  }});

  setMode('prompts');

  // -----------------------------------------------------------------
  // Per-column enable/disable toggle (face-rec only).
  // Click on a prompt header to exclude its column from row metrics.
  // State persisted in localStorage so it survives reloads.
  // -----------------------------------------------------------------
  const STORAGE_KEY = 'flgrid_disabled_cols';
  const disabledCols = new Set();
  try {{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) JSON.parse(raw).forEach(i => disabledCols.add(i|0));
  }} catch (e) {{ /* ignore corrupt storage */ }}

  function persistDisabled() {{
    try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify([...disabledCols])); }}
    catch (e) {{}}
  }}

  function quantile(sorted, q) {{
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }}

  function computeMetricsJS(scores) {{
    const total = scores.length;
    const valid = scores.filter(s => s !== null).slice().sort((a, b) => a - b);
    const m = {{ n_total: total, n_faces: valid.length }};
    if (!valid.length) return m;
    const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
    const varv = valid.reduce((s, v) => s + (v - mean) * (v - mean), 0) / valid.length;
    m.mean = mean;
    m.std = Math.sqrt(varv);
    m.median = quantile(valid, 0.5);
    m.p20 = quantile(valid, 0.2);
    m.p80 = quantile(valid, 0.8);
    m.min = valid[0];
    m.max = valid[valid.length - 1];
    return m;
  }}

  function renderRowMetrics(rmEl, m) {{
    if (!m.n_faces) {{
      rmEl.className = 'row-metrics empty';
      rmEl.innerHTML = 'no face detected (' + m.n_total + ' img)';
      return;
    }}
    rmEl.className = 'row-metrics';
    const cMed = scoreClass(m.median),
          cP20 = scoreClass(m.p20),
          cMax = scoreClass(m.max);
    rmEl.innerHTML =
      '<div class="metric primary ' + cMed + '" title="median similarity">' +
        '<span class="m-val">' + m.median.toFixed(2) + '</span>' +
        '<span class="m-lbl">med</span></div>' +
      '<div class="metric ' + cP20 + '" title="20th percentile (worst-quartile floor)">' +
        '<span class="m-val">' + m.p20.toFixed(2) + '</span>' +
        '<span class="m-lbl">p20</span></div>' +
      '<div class="metric ' + cMax + '" title="max similarity in row">' +
        '<span class="m-val">' + m.max.toFixed(2) + '</span>' +
        '<span class="m-lbl">max</span></div>' +
      '<div class="metric muted" title="std dev / face count">' +
        '<span class="m-val">σ ' + m.std.toFixed(2) + '</span>' +
        '<span class="m-lbl">' + m.n_faces + '/' + m.n_total + '</span></div>';
  }}

  function recomputeMetrics() {{
    if (!FACE_ON) return;
    document.querySelectorAll('.row-label').forEach(row => {{
      const lIdx = row.dataset.rowIdx;
      if (lIdx === undefined) return;
      const cells = document.querySelectorAll('.cell[data-lora-idx="' + lIdx + '"]');
      const scores = [];
      cells.forEach(c => {{
        const pIdx = parseInt(c.dataset.promptIdx, 10);
        if (disabledCols.has(pIdx)) return;
        const raw = c.dataset.score;
        if (raw === undefined || raw === '') {{
          // No score data (e.g. missing image). Count toward n_total only if
          // it's an empty placeholder cell; skip otherwise so n_total reflects
          // the actually-considered images.
          if (c.classList.contains('empty')) scores.push(null);
          else scores.push(null);
        }} else {{
          const f = parseFloat(raw);
          scores.push(isNaN(f) ? null : f);
        }}
      }});
      let rm = row.querySelector('.row-metrics');
      if (!rm) {{
        rm = document.createElement('div');
        rm.className = 'row-metrics';
        row.appendChild(rm);
      }}
      renderRowMetrics(rm, computeMetricsJS(scores));
    }});
  }}

  function applyDisabledVisuals() {{
    document.querySelectorAll('.prompt-head').forEach(h => {{
      const p = parseInt(h.dataset.promptIdx, 10);
      h.classList.toggle('disabled', disabledCols.has(p));
    }});
    document.querySelectorAll('.cell').forEach(c => {{
      const p = parseInt(c.dataset.promptIdx, 10);
      c.classList.toggle('col-disabled', disabledCols.has(p));
    }});
  }}

  // ------- Row sort (median or by-column score) ------------------------
  // currentSort = null | {{ field: 'median' | 'col-<N>', dir: 'asc'|'desc' }}
  let currentSort = null;
  try {{
    const raw = localStorage.getItem('flgrid_sort');
    if (raw) currentSort = JSON.parse(raw);
  }} catch (e) {{}}

  // Snapshot original DOM order so we can restore on sort-off.
  const gridEl = document.querySelector('.grid');
  const originalChildren = [...gridEl.children];

  function persistSort() {{
    if (currentSort) localStorage.setItem('flgrid_sort', JSON.stringify(currentSort));
    else localStorage.removeItem('flgrid_sort');
  }}

  function rowSortKey(label) {{
    // Returns the value to sort by for the given row-label element.
    const lIdx = label.dataset.rowIdx;
    if (lIdx === undefined) return null;
    const cells = document.querySelectorAll('.cell[data-lora-idx="' + lIdx + '"]');
    if (currentSort.field === 'median') {{
      const valid = [];
      cells.forEach(c => {{
        const p = parseInt(c.dataset.promptIdx, 10);
        if (disabledCols.has(p)) return;
        const f = parseFloat(c.dataset.score);
        if (!isNaN(f)) valid.push(f);
      }});
      if (!valid.length) return null;
      valid.sort((a, b) => a - b);
      return quantile(valid, 0.5);
    }}
    if (currentSort.field.startsWith('col-')) {{
      const col = parseInt(currentSort.field.slice(4), 10);
      let v = null;
      cells.forEach(c => {{
        if (parseInt(c.dataset.promptIdx, 10) === col) {{
          const f = parseFloat(c.dataset.score);
          v = isNaN(f) ? null : f;
        }}
      }});
      return v;
    }}
    return null;
  }}

  function applySort() {{
    if (!currentSort) {{
      // Restore original order
      originalChildren.forEach(el => gridEl.appendChild(el));
      return;
    }}
    const labels = [...document.querySelectorAll('.row-label')];
    const rows = labels.map(label => {{
      const lIdx = label.dataset.rowIdx;
      const cells = [...document.querySelectorAll('.cell[data-lora-idx="' + lIdx + '"]')];
      return {{ label, cells, value: rowSortKey(label) }};
    }});
    const dir = currentSort.dir === 'asc' ? 1 : -1;
    rows.sort((a, b) => {{
      // Nulls (no face / no score) always fall to the end.
      if (a.value === null && b.value === null) return 0;
      if (a.value === null) return 1;
      if (b.value === null) return -1;
      return (a.value - b.value) * dir;
    }});
    // Reorder by appending in the new sequence. Header row stays first
    // because appendChild moves the row-related children, not the heads.
    rows.forEach(r => {{
      gridEl.appendChild(r.label);
      r.cells.forEach(c => gridEl.appendChild(c));
    }});
  }}

  function cycleSort(field) {{
    // none -> desc -> asc -> none, scoped per field.
    if (!currentSort || currentSort.field !== field) {{
      currentSort = {{ field, dir: 'desc' }};
    }} else if (currentSort.dir === 'desc') {{
      currentSort = {{ field, dir: 'asc' }};
    }} else {{
      currentSort = null;
    }}
    persistSort();
    updateSortIndicators();
    applySort();
  }}

  function updateSortIndicators() {{
    document.querySelectorAll('.sort-btn').forEach(b => {{
      b.classList.remove('active');
      b.textContent = '↕';
    }});
    const medBtn = document.getElementById('flgrid-sort-median');
    if (medBtn) {{
      medBtn.classList.remove('active');
      medBtn.querySelector('.dir').textContent = '↕';
    }}
    if (!currentSort) return;
    const arrow = currentSort.dir === 'asc' ? '↑' : '↓';
    if (currentSort.field === 'median' && medBtn) {{
      medBtn.classList.add('active');
      medBtn.querySelector('.dir').textContent = arrow;
    }} else if (currentSort.field.startsWith('col-')) {{
      const col = currentSort.field.slice(4);
      const heads = document.querySelectorAll('.prompt-head[data-prompt-idx="' + col + '"] .sort-btn');
      heads.forEach(b => {{ b.classList.add('active'); b.textContent = arrow; }});
    }}
  }}

  if (FACE_ON) {{
    document.querySelectorAll('.prompt-head').forEach(h => {{
      h.addEventListener('click', (e) => {{
        // Clicks on the sort-btn (or any descendant) should NOT toggle disable.
        if (e.target.closest('.sort-btn')) return;
        const p = parseInt(h.dataset.promptIdx, 10);
        if (disabledCols.has(p)) disabledCols.delete(p);
        else disabledCols.add(p);
        applyDisabledVisuals();
        recomputeMetrics();
        persistDisabled();
        // When sorting by median, the median value depends on which columns
        // are enabled — re-sort to reflect the change.
        if (currentSort && currentSort.field === 'median') applySort();
      }});
    }});

    document.querySelectorAll('.sort-btn').forEach(btn => {{
      btn.addEventListener('click', (e) => {{
        e.stopPropagation();
        const head = btn.closest('.prompt-head');
        const col = head ? head.dataset.promptIdx : null;
        if (col !== null) cycleSort('col-' + col);
      }});
    }});

    const medianSortBtn = document.getElementById('flgrid-sort-median');
    if (medianSortBtn) {{
      medianSortBtn.addEventListener('click', () => cycleSort('median'));
    }}

    const resetBtn = document.getElementById('flgrid-reset-cols');
    if (resetBtn) {{
      resetBtn.addEventListener('click', () => {{
        disabledCols.clear();
        applyDisabledVisuals();
        recomputeMetrics();
        persistDisabled();
        if (currentSort && currentSort.field === 'median') applySort();
      }});
    }}

    applyDisabledVisuals();
    recomputeMetrics();
    updateSortIndicators();
    applySort();
  }}
</script>
</body>
</html>
"""

    out = output_dir / "index.html"
    atomic_write(out, html_doc)


def write_html_inline(output_dir: Path, run: dict) -> None:
    """Standalone HTML for a single run, with inline JSON manifest.

    Input `run` is a snapshot from `glt.db.fetch_run` enriched with
    `test_name`. Cells embed per-cell `aspect-ratio: W / H` inline style
    so a grid with mixed prompt shapes renders correctly. The full run
    dict is embedded as `<script type="application/json" id="manifest">`
    so the page is self-describing and re-tooling later doesn't require
    parsing the DOM.

    Light feature parity vs the legacy `write_html`: grid + click-to-zoom
    lightbox + face score badges + per-row metrics. The sort / disable-
    column / column-cycle features stay legacy-only (will port if needed).
    """
    test_name = run.get("test_name") or f"test-{run.get('test_id')}"
    rows = run["rows"]
    cells = run["cells"]
    face_meta = run.get("face_meta") or {}
    face_on = bool(face_meta) and any(r.get("metrics") for r in rows)
    thresholds = face_meta.get("thresholds") or {"good": 0.5, "ok": 0.35}
    th_good = float(thresholds.get("good", 0.5))
    th_ok = float(thresholds.get("ok", 0.35))

    # Build the (lora_idx, prompt_idx) → cell dict, and the prompts list
    # (one entry per prompt_idx, taking the first row's text — every row
    # has the same expanded prompts since the test definition is fixed).
    cells_by_pos: dict[tuple[int, int], dict] = {
        (c["lora_idx"], c["prompt_idx"]): c for c in cells
    }
    prompt_idxs = sorted({c["prompt_idx"] for c in cells})
    n_prompts = (max(prompt_idxs) + 1) if prompt_idxs else 0
    prompts_meta: list[dict] = []
    for p in range(n_prompts):
        # Find the first row that has this prompt; safe because we
        # pre-populate every (lora, prompt) at run start.
        first_cell = next(
            (c for c in cells if c["prompt_idx"] == p),
            None,
        )
        if first_cell is None:
            prompts_meta.append({"text": "", "width": 1024, "height": 1024})
        else:
            prompts_meta.append({
                "text": first_cell["prompt_text"],
                "width": first_cell["prompt_width"],
                "height": first_cell["prompt_height"],
            })

    def score_class(s):
        if s is None:
            return "score-na"
        if s >= th_good:
            return "score-good"
        if s >= th_ok:
            return "score-ok"
        return "score-bad"

    def fmt_score(s):
        return "—" if s is None else f"{s:.2f}"

    def render_cell(row_idx: int, lora_display: str, p_idx: int) -> str:
        c = cells_by_pos.get((row_idx, p_idx))
        if c is None or not c.get("image_filename"):
            return (
                f'<div class="cell empty" data-lora-idx="{row_idx}" '
                f'data-prompt-idx="{p_idx}"><span>missing</span></div>'
            )
        img_rel = c["image_filename"]
        score = c.get("face_score")
        # aspect-ratio style per cell — supports mixed shapes within a row.
        aspect = f'{c["prompt_width"]} / {c["prompt_height"]}'
        badge = ""
        score_attr = ""
        if face_on:
            score_attr = f' data-score="{"" if score is None else f"{score:.4f}"}"'
            badge = (
                f'<div class="score-badge {score_class(score)}" '
                f'title="face↔centroid similarity">{fmt_score(score)}</div>'
            )
        return (
            f'<div class="cell" style="aspect-ratio: {aspect};" '
            f'data-src="{html.escape(img_rel)}" '
            f'data-lora="{html.escape(lora_display)}" '
            f'data-prompt="{html.escape(c["prompt_text"])}" '
            f'data-lora-idx="{row_idx}" '
            f'data-prompt-idx="{p_idx}"'
            f'{score_attr}>'
            f'<img loading="lazy" src="{html.escape(img_rel)}" '
            f'alt="{html.escape(lora_display)} :: {html.escape(c["prompt_text"])}">'
            f'{badge}'
            f'</div>'
        )

    def render_row_label(row: dict) -> str:
        lora = row["lora_display"]
        l_idx = row["lora_idx"]
        m = row.get("metrics")
        name_html = (
            f'<div class="lora-name" title="{html.escape(lora)}">'
            f'{html.escape(lora)}</div>'
        )
        if not face_on or not m or not m.get("n_faces"):
            extra = ""
            if face_on and m:
                extra = (
                    f'<div class="row-metrics empty">no face detected '
                    f'({m.get("n_total", 0)} img)</div>'
                )
            return (
                f'<div class="row-label" data-row-idx="{l_idx}">'
                f'{name_html}{extra}</div>'
            )
        cls_med = score_class(m["median"])
        cls_p20 = score_class(m["p20"])
        cls_max = score_class(m["max"])
        metrics_html = (
            f'<div class="row-metrics">'
            f'  <div class="metric primary {cls_med}" title="median similarity">'
            f'    <span class="m-val">{m["median"]:.2f}</span>'
            f'    <span class="m-lbl">med</span></div>'
            f'  <div class="metric {cls_p20}" title="20th percentile">'
            f'    <span class="m-val">{m["p20"]:.2f}</span>'
            f'    <span class="m-lbl">p20</span></div>'
            f'  <div class="metric {cls_max}" title="max">'
            f'    <span class="m-val">{m["max"]:.2f}</span>'
            f'    <span class="m-lbl">max</span></div>'
            f'  <div class="metric muted" title="std / face count">'
            f'    <span class="m-val">σ {m["std"]:.2f}</span>'
            f'    <span class="m-lbl">{m["n_faces"]}/{m["n_total"]}</span></div>'
            f'</div>'
        )
        return (
            f'<div class="row-label" data-row-idx="{l_idx}">'
            f'{name_html}{metrics_html}</div>'
        )

    header_cells = "".join(
        f'<div class="head prompt-head" data-prompt-idx="{i}" '
        f'title="{html.escape(p["text"])} ({p["width"]}×{p["height"]})">'
        f'<span class="pill">#{i}</span>'
        f'<div class="prompt-dims">{p["width"]}×{p["height"]}</div>'
        f'<div class="prompt-text">{html.escape(p["text"])}</div></div>'
        for i, p in enumerate(prompts_meta)
    )

    row_html_parts = []
    for row in sorted(rows, key=lambda r: r["lora_idx"]):
        cells_html = "".join(
            render_cell(row["lora_idx"], row["lora_display"], p)
            for p in range(n_prompts)
        )
        row_html_parts.append(render_row_label(row) + cells_html)
    rows_html = "".join(row_html_parts)

    # Inline JSON for any future tooling — the full run snapshot.
    inline_json = json.dumps(run, ensure_ascii=False, indent=2)

    grid_template = "minmax(200px, 240px) " + " ".join(
        ["minmax(180px, 300px)"] * n_prompts
    )

    face_legend = ""
    if face_on:
        face_legend = (
            f'<div class="face-legend">'
            f'  <span class="legend-dot score-good"></span>≥ {th_good:.2f}'
            f'  <span class="legend-dot score-ok"></span>≥ {th_ok:.2f}'
            f'  <span class="legend-dot score-bad"></span>&lt; {th_ok:.2f}'
            f'  <span class="legend-dot score-na"></span>no face'
            f'</div>'
        )

    cfg = run.get("config") or {}
    started = run.get("started_at") or "—"
    finished = run.get("finished_at") or "—"
    status = run.get("status") or "—"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(test_name)} · run {run["id"]} · FLUX.2 grid</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #0b0b0f; --bg-2: #14141c; --fg: #e9e9f0; --muted: #8a8aa0;
    --accent: #ff4ecd; --accent-2: #6f6bff;
    --border: #23232e; --shadow: 0 10px 40px rgba(0,0,0,.5);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg);
                font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, sans-serif; }}
  header {{
    position: sticky; top: 0; z-index: 5;
    padding: 14px 24px; backdrop-filter: blur(14px);
    background: linear-gradient(180deg, rgba(11,11,15,.92), rgba(11,11,15,.65));
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  }}
  header h1 {{
    font-size: 16px; margin: 0; font-weight: 600; letter-spacing: .2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  header .meta {{ color: var(--muted); font-size: 12px; }}
  header .meta code {{ color: var(--fg); background: var(--bg-2); padding: 2px 6px; border-radius: 4px; }}
  .status-pill {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
  }}
  .status-running {{ background: rgba(243,201,105,.15); color: #f3c969; }}
  .status-completed {{ background: rgba(93,217,126,.15); color: #5dd97e; }}
  .status-failed {{ background: rgba(239,118,118,.15); color: #ef7676; }}
  .status-cancelled {{ background: rgba(138,138,160,.15); color: var(--muted); }}

  .grid-wrap {{ padding: 24px; }}
  .grid {{
    display: grid; grid-template-columns: {grid_template};
    gap: 10px; align-items: stretch;
  }}
  .head {{
    position: sticky; top: var(--header-h, 64px); z-index: 3;
    background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; min-height: 64px;
  }}
  .prompt-head {{ display: flex; flex-direction: column; gap: 4px; }}
  .pill {{
    display: inline-block; align-self: flex-start;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: white; font-weight: 600; font-size: 11px;
    padding: 2px 8px; border-radius: 999px; letter-spacing: .3px;
  }}
  .prompt-dims {{ color: var(--muted); font-size: 10.5px; font-family: ui-monospace, monospace; }}
  .prompt-text {{
    color: var(--fg); font-size: 12.5px;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .row-label {{
    position: sticky; left: 24px; z-index: 2;
    background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; display: flex; flex-direction: column; gap: 8px;
    justify-content: center; word-break: break-all; font-size: 12.5px;
  }}
  .row-label .lora-name {{ font-weight: 600; line-height: 1.3; }}
  .row-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px; }}
  .row-metrics.empty {{ display: block; color: var(--muted); font-style: italic; font-size: 11px; }}
  .row-metrics .metric {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 3px 4px; border-radius: 6px; background: rgba(255,255,255,.03);
    border: 1px solid var(--border); min-width: 0;
  }}
  .row-metrics .metric.primary {{ border-width: 1.5px; }}
  .row-metrics .m-val {{ font-size: 13px; font-weight: 700; line-height: 1.1; white-space: nowrap; }}
  .row-metrics .m-lbl {{ font-size: 9.5px; color: var(--muted); letter-spacing: .3px; text-transform: uppercase; margin-top: 1px; }}
  .row-metrics .metric.muted .m-val {{ font-size: 11px; font-weight: 600; color: var(--muted); }}
  .metric.score-good {{ color: #5dd97e; border-color: rgba(93,217,126,.45); background: rgba(93,217,126,.08); }}
  .metric.score-ok   {{ color: #f3c969; border-color: rgba(243,201,105,.45); background: rgba(243,201,105,.08); }}
  .metric.score-bad  {{ color: #ef7676; border-color: rgba(239,118,118,.45); background: rgba(239,118,118,.08); }}
  .metric.score-na   {{ color: #777; }}

  .cell {{
    position: relative; overflow: hidden; border-radius: 10px;
    border: 1px solid var(--border); background: var(--bg-2); cursor: zoom-in;
    transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
  }}
  .cell:hover {{
    transform: translateY(-2px);
    border-color: var(--accent); box-shadow: 0 8px 30px rgba(255,78,205,.15);
  }}
  .cell img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .cell.empty {{
    display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 12px; cursor: default;
    aspect-ratio: 1 / 1;
  }}
  .score-badge {{
    position: absolute; top: 6px; right: 6px; z-index: 1;
    background: rgba(0,0,0,.72); color: white;
    padding: 3px 8px; border-radius: 999px;
    font-size: 11.5px; font-weight: 700; letter-spacing: .2px;
    backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,.08);
  }}
  .score-badge.score-good {{ background: rgba(35,80,45,.85); color: #b3f5c5; border-color: rgba(93,217,126,.5); }}
  .score-badge.score-ok   {{ background: rgba(74,57,16,.85); color: #ffdf8a; border-color: rgba(243,201,105,.5); }}
  .score-badge.score-bad  {{ background: rgba(72,28,28,.85); color: #ffb1b1; border-color: rgba(239,118,118,.5); }}
  .score-badge.score-na   {{ background: rgba(0,0,0,.55); color: #aaa; }}

  .face-legend {{ display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--muted); flex-wrap: wrap; }}
  .face-legend .legend-dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin: 0 4px 0 8px; border: 1px solid var(--border);
  }}
  .legend-dot.score-good {{ background: #5dd97e; }}
  .legend-dot.score-ok   {{ background: #f3c969; }}
  .legend-dot.score-bad  {{ background: #ef7676; }}
  .legend-dot.score-na   {{ background: #555; }}

  .lb {{
    position: fixed; inset: 0; background: rgba(5,5,10,.92);
    display: none; align-items: center; justify-content: center;
    z-index: 50; padding: 24px;
  }}
  .lb.open {{ display: flex; }}
  .lb-inner {{ max-width: 95vw; max-height: 95vh; display: flex; flex-direction: column; gap: 12px; align-items: center; }}
  .lb img {{
    max-width: 95vw; max-height: 80vh; border-radius: 12px;
    box-shadow: var(--shadow); object-fit: contain; background: #000;
  }}
  .lb-caption {{
    max-width: 95vw; color: var(--fg); font-size: 13px;
    background: var(--bg-2); border: 1px solid var(--border);
    padding: 10px 14px; border-radius: 10px;
  }}
  .lb-caption .lora {{ color: var(--accent); font-weight: 600; }}
  .lb-caption .prompt {{ color: var(--muted); margin-top: 4px; }}
  .lb-close {{
    position: fixed; top: 16px; right: 20px; color: var(--fg);
    font-size: 28px; cursor: pointer; user-select: none; opacity: .7;
  }}
  .lb-close:hover {{ opacity: 1; }}
  .lb-nav {{
    position: fixed; top: 50%; transform: translateY(-50%);
    color: var(--fg); font-size: 36px; cursor: pointer; user-select: none;
    opacity: .5; padding: 12px 18px;
  }}
  .lb-nav:hover {{ opacity: 1; }}
  .lb-nav.prev {{ left: 12px; }}
  .lb-nav.next {{ right: 12px; }}

  details.meta-panel {{
    margin: 0 24px 24px; background: var(--bg-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 14px;
  }}
  details.meta-panel summary {{ cursor: pointer; color: var(--muted); }}
  details.meta-panel pre {{ color: var(--fg); margin: 8px 0 0; font-size: 11.5px; overflow: auto; max-height: 400px; }}
</style>
</head>
<body>
<header>
  <h1>✦ {html.escape(test_name)} · <span class="status-pill status-{status}">{status}</span></h1>
  <div class="meta">
    run <code>#{run["id"]}</code> ·
    <code>{len(rows)}</code> loras × <code>{n_prompts}</code> prompts ·
    seed <code>{cfg.get("seed", "—")}</code> ·
    res <code>{cfg.get("resolution", "—")}</code> ·
    trigger <code>{html.escape(cfg.get("trigger") or "—")}</code> ·
    started <code>{started}</code> · finished <code>{finished}</code>
    {face_legend}
  </div>
</header>

<div class="grid-wrap">
  <div class="grid">
    <div class="head corner-spacer"></div>
    {header_cells}
    {rows_html}
  </div>
</div>

<details class="meta-panel">
  <summary>Run JSON (inline)</summary>
  <pre>{html.escape(inline_json)}</pre>
</details>

<script type="application/json" id="manifest">{inline_json}</script>

<div class="lb" id="lb">
  <span class="lb-close" id="lb-close">&times;</span>
  <span class="lb-nav prev" id="lb-prev">&#8249;</span>
  <span class="lb-nav next" id="lb-next">&#8250;</span>
  <div class="lb-inner">
    <img id="lb-img" src="" alt="">
    <div class="lb-caption">
      <div class="lora" id="lb-lora"></div>
      <div class="prompt" id="lb-prompt"></div>
    </div>
  </div>
</div>

<script>
  function syncHeaderHeight() {{
    const h = document.querySelector('header');
    if (!h) return;
    document.documentElement.style.setProperty('--header-h', h.offsetHeight + 'px');
  }}
  syncHeaderHeight();
  window.addEventListener('resize', syncHeaderHeight);

  const N_LORAS = {len(rows)};
  const N_PROMPTS = {n_prompts};
  const cellMap = {{}};
  document.querySelectorAll('.cell').forEach(c => {{
    if (c.classList.contains('empty')) return;
    cellMap[c.dataset.loraIdx + '_' + c.dataset.promptIdx] = c;
  }});

  const lb = document.getElementById('lb');
  const lbImg = document.getElementById('lb-img');
  const lbLora = document.getElementById('lb-lora');
  const lbPrompt = document.getElementById('lb-prompt');
  let curL = 0, curP = 0;

  function show(l, p) {{
    const c = cellMap[l + '_' + p];
    if (!c) return false;
    curL = l; curP = p;
    lbImg.src = c.dataset.src;
    lbLora.textContent = c.dataset.lora;
    lbPrompt.textContent = '#' + p + ' · ' + c.dataset.prompt;
    lb.classList.add('open');
    return true;
  }}
  function close() {{ lb.classList.remove('open'); lbImg.src = ''; }}
  function step(axis, dir) {{
    const N = axis === 'prompts' ? N_PROMPTS : N_LORAS;
    for (let i = 1; i <= N; i++) {{
      let l = curL, p = curP;
      if (axis === 'prompts') p = ((curP + dir * i) % N + N) % N;
      else                    l = ((curL + dir * i) % N + N) % N;
      if (cellMap[l + '_' + p]) {{ show(l, p); return; }}
    }}
  }}
  document.querySelectorAll('.cell:not(.empty)').forEach(c => {{
    c.addEventListener('click', () => show(parseInt(c.dataset.loraIdx, 10),
                                           parseInt(c.dataset.promptIdx, 10)));
  }});
  document.getElementById('lb-close').addEventListener('click', close);
  document.getElementById('lb-prev').addEventListener('click', e => {{ e.stopPropagation(); step('prompts', -1); }});
  document.getElementById('lb-next').addEventListener('click', e => {{ e.stopPropagation(); step('prompts', +1); }});
  lb.addEventListener('click', e => {{ if (e.target === lb) close(); }});
  document.addEventListener('keydown', e => {{
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape')          close();
    else if (e.key === 'ArrowLeft')  step('prompts', -1);
    else if (e.key === 'ArrowRight') step('prompts', +1);
    else if (e.key === 'ArrowUp')    step('loras', -1);
    else if (e.key === 'ArrowDown')  step('loras', +1);
  }});
</script>
</body>
</html>
"""

    out = output_dir / "index.html"
    atomic_write(out, html_doc)
