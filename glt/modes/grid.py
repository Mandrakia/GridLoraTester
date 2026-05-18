"""`glt --grid` — main grid generation mode.

Orchestrates the whole flow:
  1. resolve the test definition (CLI args OR --test-id from DB) and
     expand per-prompt {text, width, height} via the `[trigger]` /
     `[W:H]` substitution
  2. discover adapters in the input dir(s)
  3. open the SQLite embeds cache + check whether every (expanded) prompt
     is already cached → lets us skip Qwen3 load entirely on a full hit
  4. build the FLUX-2 Klein pipeline (FP8 / offload / sage / compile)
  5. encode missing prompts, then **radical** unload of text_encoder +
     tokenizer (frees RAM and VRAM for the diffusion pass)
  6. load sticky base LoRAs (config.json `base_loras`)
  7. create a fresh `test_runs` row (status=running), pre-populate rows
     and cells so the dashboard sees the skeleton immediately
  8. per LoRA, per shape group: denoise → decode → save → score, with
     incremental UPDATEs on test_run_cells so progress is observable
  9. write the standalone HTML (inline JSON manifest) and flip the run
     to status=completed
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import db as glt_db
from ..adapters.discovery import (
    discover_loras,
    lora_display_name,
    parse_lora_step,
    safe_stem,
)
from ..adapters.lokr import LoKrHookManager, is_lokr_file, load_lokr_data
from ..adapters.lora import load_base_loras
from ..analysis.face import FaceScorer
from ..analysis.metrics import compute_row_metrics
from ..generation.decode import batch_decode_latents
from ..generation.latents import generate_latents_batched
from ..generation.prompt_cache import (
    compute_te_model_id,
    open_cache as open_embed_cache,
)
from ..generation.prompt_template import expand_prompts
from ..generation.prompts import all_prompts_cached, encode_prompts
from ..output.html import write_html_inline
from ..output.images import save_image
from ..pipeline.attention import (
    print_sage_attention_stats,
    sage_attention_stats_snapshot,
)
from ..pipeline.build import (
    QUANT_CHOICES,
    build_te_pipeline,
    build_transformer_pipeline,
)
from ..pipeline.int8_convrot import (
    backup_quantized_baseline,
    load_lora_into_quantized,
    restore_quantized_baseline,
)
from ..pipeline.loaders import DEFAULT_MODEL_ID as MODEL_ID
from ..utils.config import load_config, load_prompts as legacy_load_prompts


_DTYPE_CHOICES = ["bf16", "fp16", "fp8", "fp8_e4m3fn", "fp8_e5m2", "fp8_mixed"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LoRA x prompt grid with FLUX.2-klein-9B (FP8).",
        add_help=False,
    )
    # Mode dispatch ------------------------------------------------------
    # DB mode is the primary path: --test-id pulls the full definition
    # (prompts/trigger/resolution/etc) from ui/data/glt.db. Legacy CLI mode
    # is kept for ad-hoc experiments without going through the dashboard.
    parser.add_argument(
        "--test-id", type=int, default=None,
        help="DB mode: id of the test row in `tests`. Implies --db. Loads "
             "prompts, trigger, resolution, batch_size, quant, offload, and "
             "advanced flags from the row. Per-arg overrides on the CLI take "
             "precedence over the stored values.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to ui/data/glt.db. Required with --test-id.",
    )
    parser.add_argument(
        "--tests-root", default=None,
        help="DB mode: output root. Run dir is <root>/<test_name>/run_<id>/. "
             "Defaults to the `tests_root` row in `settings`.",
    )

    parser.add_argument(
        "input_dir", nargs="*",
        help="LoRA folder(s). Overrides the test's lora_path in DB mode. "
             "Required in legacy CLI mode. Pass multiple to merge into one "
             "grid — files sorted (basename, step) across all sources.",
    )
    parser.add_argument(
        "-prompts", "--prompts-file", dest="prompts_file", default=None,
        help="Legacy CLI mode: text file with one prompt per line. Ignored "
             "in DB mode (prompts come from the test's prompt_set_id).",
    )
    parser.add_argument(
        "-o", dest="output_dir", default=None,
        help="Legacy CLI mode: output directory. Ignored in DB mode "
             "(derived from --tests-root + test name + run id).",
    )

    # Per-prompt expansion knobs (override DB if set) --------------------
    parser.add_argument(
        "--trigger", default=None,
        help="LoRA trigger word substituted into [trigger] placeholders in "
             "every prompt. Empty string keeps placeholders. Overrides DB.",
    )
    parser.add_argument(
        "--resolution", default=None,
        help="Target image area expressed as megapixels: '1MP' (1024^2), "
             "'2MP' (~1448^2), '1.5MP'. The per-prompt [W:H] tag picks the "
             "aspect ratio; W and H are rounded to the model's stride (16). "
             "Overrides DB.",
    )

    # Diffusion + format knobs ------------------------------------------
    # Defaults are None so we can distinguish "user passed an explicit
    # value" from "use what's in tests.advanced_json". `_resolve_test_params`
    # falls back to the canonical defaults when both CLI and DB are unset.
    parser.add_argument("-s", dest="seed", type=int, default=None,
                        help="Seed (default 42, overrides DB advanced.seed)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default 4 — Klein is a few-step model)")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Guidance scale (default 1.0)")
    parser.add_argument("--lora-scale", type=float, default=None,
                        help="LoRA strength (default 1.0)")
    parser.add_argument("--format", choices=["png", "jpg"], default=None,
                        help="Output image format (default png)")
    parser.add_argument("--jpg-quality", type=int, default=92,
                        help="JPEG quality if --format jpg")

    # Model + dtypes ----------------------------------------------------
    parser.add_argument("--model-path", default=MODEL_ID,
                        help=f"HF repo ID or local diffusers dir. Default: {MODEL_ID}")
    parser.add_argument("--transformer-file", default=None,
                        help="Path to a .safetensors with the transformer (FLUX-2 ComfyUI-style).")
    parser.add_argument("--vae-file", default=None,
                        help="Path to a .safetensors with the FLUX-2 VAE.")
    parser.add_argument("--qwen-file", default=None,
                        help="Path to a .safetensors with Qwen3 weights (override the HF cache copy).")
    parser.add_argument("--transformer-dtype", choices=_DTYPE_CHOICES, default="bf16")
    parser.add_argument("--vae-dtype", choices=_DTYPE_CHOICES, default="bf16")
    parser.add_argument("--qwen-dtype", choices=_DTYPE_CHOICES, default=None,
                        help="Qwen3 dtype during encode (Qwen is unloaded after). bf16=16GB, "
                             "fp8_e4m3fn=8GB for smaller cards. Defaults to advanced.qwen_dtype "
                             "from the DB (or 'bf16' if unset).")
    parser.add_argument("--quant", choices=list(QUANT_CHOICES), default="auto",
                        help="Transformer quantization. 'auto' picks per GPU: Ampere -> "
                             "int8_convrot, Ada/Hopper -> fp8_weight.")

    # LoRA + attention + compile ----------------------------------------
    # Bool flags use store_const+default=None so "user didn't pass it" is
    # distinguishable from "user explicitly set it to False" — required
    # to know when to defer to the DB advanced value.
    parser.add_argument("--preload-loras", action="store_const", const=True, default=None,
                        help="Preload every LoRA as a named adapter (fp8 paths only — meaningless for int8_convrot).")
    parser.add_argument("--sage-attention", action="store_const", const=True, default=None,
                        help="Use SageAttention for the joint attention (~30-40%% speedup on Ada/Hopper).")
    parser.add_argument("--compile-mode", choices=["on", "auto", "off"], default=None,
                        help="torch.compile(transformer). 'on' (default for new tests) compiles "
                             "every run — disk cache makes 2nd+ run cheap. 'auto' enables only "
                             "when n_loras x n_prompts >= n_shapes x 8 (warm-cache break-even). "
                             "'off' skips compile. When unset, the DB tests.compile_mode wins.")

    parser.add_argument("--no-html", action="store_true", help="Skip writing index.html")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=0,
                        help="Prompts per pipe() call within each LoRA and same shape group. "
                             "0 (default) = full batch of all same-shape prompts.")
    parser.add_argument("--min-step", "--minStep", dest="min_step", type=int, default=None,
                        help="Skip LoRAs whose parsed step < this value. Default 0 = no filter.")
    parser.add_argument("--config", default=None,
                        help="Path to config.json defining sticky base LoRAs / face_recognition.")
    parser.add_argument("--skip-face-recognition", "--no-face", dest="skip_face",
                        action="store_const", const=True, default=None,
                        help="Skip face-similarity scoring for this run.")
    parser.add_argument("--shift", type=float, default=None,
                        help="Override scheduler shift / compute_empirical_mu. "
                             "ComfyUI uses 2.02 for FLUX-2.")
    # ComfyUI-parity noise is forced-on. The previous toggle was the
    # constant source of "why doesn't my grid match Comfy?" confusion;
    # there's no good reason to ever disable it. Kept in config_snapshot
    # for historical clarity.
    parser.add_argument("--debug-dump", dest="debug_dump", default=None,
                        help="Directory to dump intermediate tensors for parity debugging.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the work without loading the model.")
    parser.add_argument("--help", action="help", help="Show this help and exit")
    return parser.parse_args()


def _maybe_patch_shift(shift: float | None) -> None:
    """Monkey-patch every Flux2 pipeline module's compute_empirical_mu so
    the scheduler uses a fixed mu instead of the empirical estimate."""
    if shift is None:
        return
    try:
        import diffusers.pipelines.flux2.pipeline_flux2_klein as _klein
        import diffusers.pipelines.flux2.pipeline_flux2 as _flux2
        import diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint as _klein_ip
        import diffusers.pipelines.flux2.pipeline_flux2_klein_kv as _klein_kv
        _shift = float(shift)
        _patched = lambda image_seq_len, num_steps: _shift
        for _mod in (_klein, _flux2, _klein_ip, _klein_kv):
            if hasattr(_mod, "compute_empirical_mu"):
                _mod.compute_empirical_mu = _patched
        print(f"[shift] override compute_empirical_mu → fixed mu = {shift}")
    except ImportError as e:
        print(f"[shift] override failed: {e}")


def _resolve_test_params(args, test_def):
    """CLI args override DB test_def, which overrides hardcoded defaults.

    For most fields the argparse default is None so 'user didn't pass it'
    is distinguishable from a real value. Two exceptions:
      - `args.quant`: default 'auto' (not None), so we treat 'auto' as
        "use DB value if any".
      - `args.batch_size`: default 0 (interpreted as 'full batch'),
        also treated as "use DB value if any".
    """
    pick = lambda cli, db, default: cli if cli is not None else (db if db is not None else default)
    advanced = (test_def.get("advanced") if test_def else None) or {}

    quant_cli = args.quant if args.quant != "auto" else None
    batch_cli = args.batch_size if args.batch_size else None

    return {
        # Top-level test columns ------------------------------------------
        "trigger": pick(args.trigger, test_def["trigger"] if test_def else None, ""),
        "resolution": pick(args.resolution, test_def["resolution"] if test_def else None, "1MP"),
        "batch_size": pick(batch_cli, test_def["batch_size"] if test_def else None, 0),
        "quant": pick(quant_cli, test_def["quant"] if test_def else None, "auto"),
        "compile_mode": pick(args.compile_mode, test_def.get("compile_mode") if test_def else None, "on"),

        # Fields stored in tests.advanced_json — historically only the CLI
        # was reading these, so a UI-only user got their advanced settings
        # silently ignored. Resolve all of them properly now.
        "seed":            pick(args.seed,          advanced.get("seed"),           42),
        "steps":           pick(args.steps,         advanced.get("steps"),          4),
        "guidance":        pick(args.guidance,      advanced.get("guidance"),       1.0),
        "lora_scale":      pick(args.lora_scale,    advanced.get("lora_scale"),     1.0),
        "format":          pick(args.format,        advanced.get("format"),         "png"),
        "min_step":        pick(args.min_step,      advanced.get("min_step"),       0),
        "shift":           pick(args.shift,         advanced.get("shift"),          None),
        "sage_attention":  pick(args.sage_attention, advanced.get("sage_attention"), False),
        "preload_loras":   pick(args.preload_loras, advanced.get("preload_loras"),  False),
        "skip_face":       pick(args.skip_face,     advanced.get("skip_face"),      False),
        "qwen_dtype":      pick(args.qwen_dtype,    advanced.get("qwen_dtype"),     "bf16"),
    }


# torch.compile break-even: warm-cache warmup ~12 s/shape, steady gain
# ~1.5 s/image vs no-compile. The 8 below comes from 12 / 1.5 ≈ 8 images
# per shape needed to pay back the warmup. Conservative (some grids may
# be cold-cache on first run = ~47 s/shape, which needs ~32 imgs/shape).
COMPILE_BREAKEVEN_IMAGES_PER_SHAPE = 8


def _resolve_compile_enabled(
    mode: str, n_loras: int, n_prompts: int, n_shapes: int,
) -> bool:
    """Decide whether torch.compile is on for this run.

    `mode='on'`  → always True
    `mode='off'` → always False
    `mode='auto'`→ rentable iff n_loras * n_prompts >= n_shapes * THRESHOLD
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto
    if n_shapes <= 0:
        return False
    total_images = n_loras * n_prompts
    threshold = n_shapes * COMPILE_BREAKEVEN_IMAGES_PER_SHAPE
    decision = total_images >= threshold
    print(
        f"[compile:auto] n_loras={n_loras} * n_prompts={n_prompts} = {total_images} images, "
        f"n_shapes={n_shapes} * {COMPILE_BREAKEVEN_IMAGES_PER_SHAPE} = {threshold} threshold "
        f"-> compile={'on' if decision else 'off'}"
    )
    return decision


def run(args=None):
    """Entry point: parse CLI args and run the grid."""
    if args is None:
        args = parse_args()

    # ---- Load test definition (DB mode) ----
    db_conn = None
    test_def = None
    if args.test_id is not None:
        if not args.db:
            sys.exit("[error] --test-id requires --db pointing at ui/data/glt.db")
        db_conn = glt_db.connect(args.db)
        try:
            test_def = glt_db.load_test_def(db_conn, args.test_id)
        except LookupError as e:
            sys.exit(f"[error] {e}")
        except RuntimeError as e:
            sys.exit(f"[error] {e}")
        print(f"[db] loaded test #{args.test_id}: {test_def['name']}  "
              f"({test_def['prompts_source']}, {len(test_def['prompts'])} prompt(s))")

    effective = _resolve_test_params(args, test_def)
    trigger = effective["trigger"]
    resolution = effective["resolution"]
    batch_size_cli = effective["batch_size"]

    # Apply the resolved scheduler shift now that we know the value (CLI
    # OR advanced.shift OR None). Was previously called on `args.shift`
    # which ignored the DB advanced field.
    _maybe_patch_shift(effective["shift"])

    # CLI --quant overrides DB tests.quant overrides 'auto'. `auto`
    # gets resolved against the GPU's compute capability later, by
    # `build_transformer_pipeline()`.
    quant_resolved = (effective["quant"] or "auto").strip() or "auto"

    # ---- Resolve input dirs + prompts ----
    if args.input_dir:
        input_dirs = [Path(d).resolve() for d in args.input_dir]
    elif test_def:
        input_dirs = [Path(test_def["lora_path"]).resolve()]
    else:
        sys.exit("[error] input_dir(s) required (positional) in legacy CLI mode")

    for d in input_dirs:
        if not d.is_dir():
            sys.exit(f"[error] input_dir not found: {d}")

    if test_def:
        raw_prompts = test_def["prompts"]
    else:
        if not args.prompts_file:
            sys.exit("[error] -prompts <file> required in legacy CLI mode")
        prompts_path = Path(args.prompts_file).resolve()
        if not prompts_path.is_file():
            sys.exit(f"[error] prompts file not found: {prompts_path}")
        raw_prompts = legacy_load_prompts(prompts_path)

    # ---- Expand prompts: [trigger] + [W:H] → concrete (text, w, h) ----
    expanded = expand_prompts(raw_prompts, trigger=trigger, resolution=resolution)
    distinct_shapes = sorted({(ep.width, ep.height) for ep in expanded})
    print(f"[expand] trigger={trigger!r}  resolution={resolution}  "
          f"{len(expanded)} prompt(s) across {len(distinct_shapes)} distinct shape(s)")
    for w, h in distinct_shapes:
        n = sum(1 for ep in expanded if (ep.width, ep.height) == (w, h))
        print(f"  - {w}x{h}: {n} prompt(s)")

    # ---- Discover LoRAs ----
    loras = discover_loras(input_dirs)
    min_step = int(effective["min_step"] or 0)
    if min_step > 0:
        before = len(loras)
        loras = [l for l in loras if parse_lora_step(l)[1] >= min_step]
        print(f"[filter] min_step {min_step}: kept {len(loras)}/{before} LoRA(s)")
        if not loras:
            sys.exit(f"[error] all LoRAs filtered out (no step >= {min_step})")

    if len(input_dirs) > 1:
        print(f"[plan] inputs: {len(input_dirs)} folders → "
              f"{' | '.join(str(d) for d in input_dirs)}")
    print(f"[plan] {len(loras)} lora(s) × {len(expanded)} prompt(s) = "
          f"{len(loras) * len(expanded)} image(s)")
    for lora in loras:
        _, step = parse_lora_step(lora)
        step_str = "final" if step == math.inf else f"step={step}"
        print(f"  - {lora_display_name(lora)}  ({step_str})")

    config = load_config(Path(args.config) if args.config else None)
    base_lora_entries = config.get("base_loras", [])

    # Face scoring: in DB mode (--test-id), prefer the centroid stored in
    # the `centroids` table for the test's dataset scope. Falls back to
    # config.json's face_recognition section in legacy CLI mode. The DB
    # path is what makes scoring "by default" for UI-launched runs — no
    # config.json required, no centroid CLI arg required.
    face_cfg: dict = {}
    if db_conn is not None and args.test_id is not None:
        db_face_cfg = glt_db.load_face_cfg_for_test(db_conn, args.test_id)
        if db_face_cfg is not None:
            face_cfg = db_face_cfg
            # Try to surface the centroid dim cheaply (decode the b64
            # length / 4 bytes per fp32 — 512 for buffalo_l). Skip if
            # the cfg shape is weird; the real shape will be logged by
            # `FaceScorer._load_centroid` when it actually parses.
            import base64 as _b64
            try:
                blob = _b64.b64decode(face_cfg.get("centroid_b64") or "")
                dim = max(1, len(blob) // 4)
            except Exception:
                dim = "?"
            print(f"[face] using centroid from DB (dim={dim}) for test #{args.test_id}")
    if not face_cfg:
        face_cfg = config.get("face_recognition") or {}
    face_enabled = bool(face_cfg) and (face_cfg.get("centroid") or face_cfg.get("centroid_b64"))
    if effective["skip_face"] and face_enabled:
        print("[face] skip_face: skipping face scoring this run")
        face_enabled = False
    elif not face_enabled:
        print("[face] no centroid available — face scoring disabled "
              "(analyze the dataset to compute one, or pass --config with face_recognition)")

    if args.dry_run:
        for lora in loras:
            stem = safe_stem(lora)
            print(f"  - {lora_display_name(lora)}")
            for p_idx, ep in enumerate(expanded):
                print(f"    [{p_idx}] {ep.width}x{ep.height}  ar={ep.ar_tag!r:>6}  "
                      f"::  {ep.text[:60]}")
        return

    # Resolve compile mode against the actual workload — auto needs the
    # final loras list + the distinct shape count to compute break-even.
    n_distinct_shapes = len({(ep.width, ep.height) for ep in expanded})
    compile_enabled = _resolve_compile_enabled(
        effective["compile_mode"], len(loras), len(expanded), n_distinct_shapes,
    )

    # Compile + FP8 quant on Ampere (sm < 89) crashes Inductor — Triton
    # tries to lower fp8e4nv ops but sm_80/86 don't have native FP8 ops:
    #   ValueError: type fp8e4nv not supported in this architecture.
    # FP8 weight-only on Ampere is already a no-op for compute (emulated
    # via INT8 IMMA under the hood), so disabling compile here costs the
    # user nothing — and int8_convrot would be strictly better anyway.
    # Detect resolved quant ahead of time (auto → recommended) so the
    # check is precise: int8_convrot on Ampere keeps compile on.
    from ..pipeline.build import resolve_quant as _resolve_quant_str
    resolved_quant_for_check = _resolve_quant_str(quant_resolved) if quant_resolved != "auto" \
        else _resolve_quant_str("auto")
    if compile_enabled and resolved_quant_for_check in ("fp8_weight", "fp8_dynamic"):
        import torch as _torch
        if _torch.cuda.is_available():
            cap = _torch.cuda.get_device_capability(0)
            if cap[0] == 8 and cap[1] < 9:
                print(
                    f"[compile] disabling compile: fp8 quant + Ampere (sm_{cap[0]}{cap[1]}) "
                    f"crashes Inductor (fp8e4nv not supported in this architecture). "
                    f"Switch to --quant int8_convrot for a real 2x speedup on Ampere; "
                    f"FP8 weight-only on your GPU has no compute speedup anyway."
                )
                compile_enabled = False

    # ---- Compile-transformer + preload + LoKr guards ----
    if compile_enabled and effective["preload_loras"] and len(loras) > 1:
        sys.exit(
            "[fatal] compile + --preload-loras with multiple LoRAs is broken (fp8 path):\n"
            "        torch.compile bakes the active LoRA into the peft adapter graph,\n"
            "        set_adapters can't swap it. Drop one of the two, or use\n"
            "        --quant int8_convrot which doesn't go through peft adapters."
        )
    n_lokr = sum(1 for l in loras if is_lokr_file(l))
    if n_lokr:
        print(f"[lokr] detected {n_lokr}/{len(loras)} LoKr file(s) — will route via forward hooks")
    if n_lokr and effective["preload_loras"]:
        sys.exit("[fatal] --preload-loras incompatible with LoKr files (hooks vs peft adapters).")
    if n_lokr and compile_enabled:
        sys.exit("[fatal] compile incompatible with LoKr files (hooks not traced).")
    # int8_convrot bakes LoRA into weights via in-place .copy_() — that
    # preserves Parameter identity, so compile survives the swap. But
    # `--preload-loras` is meaningless here (we don't keep peft adapters
    # around), so silently ignore it.
    if quant_resolved == "int8_convrot" and effective["preload_loras"]:
        print("[int8_convrot] --preload-loras is a no-op in this mode (LoRAs are baked).")
    if n_lokr and quant_resolved == "int8_convrot":
        sys.exit("[fatal] LoKr is not supported in quant=int8_convrot (hooks vs baked INT8).")

    # ---- Phase 1: encode prompts (Qwen alone on GPU) ----
    embed_cache = open_embed_cache()
    te_model_id = compute_te_model_id(
        args.model_path, args.qwen_file, effective["qwen_dtype"],
    )
    full_cache_hit = all_prompts_cached(
        embed_cache, [ep.text for ep in expanded], te_model_id,
    ) and not args.qwen_file
    if full_cache_hit:
        print("[embeds:cache] full hit → skipping Qwen3 load entirely")
        prompt_embeds_list = encode_prompts(
            None, [ep.text for ep in expanded], embed_cache, te_model_id,
        )
    else:
        pipe_te = build_te_pipeline(
            model_path=args.model_path,
            qwen_dtype=effective["qwen_dtype"],
            qwen_file=args.qwen_file,
        )
        prompt_embeds_list = encode_prompts(
            pipe_te, [ep.text for ep in expanded], embed_cache, te_model_id,
        )
        # Radically free Qwen before loading the transformer.
        pipe_te.text_encoder = None
        pipe_te.tokenizer = None
        del pipe_te
        gc.collect()
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        print("[encode] Qwen unloaded — transformer phase starts with full VRAM")

    # ---- Phase 2: transformer + VAE on GPU ----
    pipe = build_transformer_pipeline(
        quant=quant_resolved,
        model_path=args.model_path,
        transformer_file=args.transformer_file,
        transformer_dtype=args.transformer_dtype,
        vae_file=args.vae_file,
        vae_dtype=args.vae_dtype,
        compile_transformer=compile_enabled,
        sage_attention=bool(effective["sage_attention"]),
    )
    is_int8_convrot = getattr(pipe, "_glt_quant", None) == "int8_convrot"

    # ---- Sticky base LoRAs ----
    # In int8_convrot mode, peft can't wrap our quantized layers, so we
    # bake base LoRAs straight into the INT8 weights. The backup snapshot
    # is taken AFTER baking so per-row test LoRAs restore to "baseline +
    # base LoRAs", keeping base LoRA application a one-time cost.
    if is_int8_convrot:
        base_names: list[str] = []
        base_weights: list[float] = []
        base_meta: list[dict] = []
        if base_lora_entries:
            from safetensors.torch import load_file as _load_lora_sd
            print(f"[base-lora] baking {len(base_lora_entries)} base LoRA(s) into INT8 transformer")
            for i, entry in enumerate(base_lora_entries):
                path = entry.get("path")
                weight = float(entry.get("weight", 1.0))
                nick = entry.get("name", f"base_{i}")
                if not path:
                    print(f"  [base-lora] entry {i} has no 'path', skipping")
                    continue
                p = Path(path)
                if not p.exists():
                    print(f"  [base-lora] file not found: {p}, skipping")
                    continue
                sd = _load_lora_sd(str(p))
                rep = load_lora_into_quantized(pipe.transformer, sd, alpha_scale=weight)
                base_names.append(nick)
                base_weights.append(weight)
                base_meta.append({
                    "name": nick, "path": str(p), "weight": weight,
                    "int8_convrot_applied": rep["applied"],
                    "int8_convrot_unmatched": rep["unmatched_key"],
                })
                print(f"  [base-lora] baked {p.name} (weight={weight}, applied={rep['applied']})")
        # Backup the *post-base* state so per-row LoRA swap restores
        # "baseline + base LoRAs" without re-baking the base.
        n_bk = backup_quantized_baseline(pipe.transformer)
        print(f"[int8_convrot] baseline backup saved for {n_bk} module(s) "
              "(LoRA swap restore point includes base LoRAs)")
    else:
        base_names, base_weights, base_meta = load_base_loras(pipe, base_lora_entries)

    scorer = None
    face_meta_run = None
    if face_enabled:
        try:
            scorer = FaceScorer(face_cfg)
            face_meta_run = {
                "model_name": scorer.model_name,
                "providers": scorer.providers,
                "centroid_dim": int(scorer.centroid.shape[0]),
                "thresholds": scorer.thresholds,
            }
        except Exception as e:
            print(f"[face] disabled: {e}")
            scorer = None

    # ---- Create the run row + resolve output_dir ----
    run_id: int | None = None
    if db_conn is not None:
        config_snapshot = {
            "trigger": trigger,
            "resolution": resolution,
            "seed": effective["seed"],
            "steps": effective["steps"],
            "guidance": effective["guidance"],
            "lora_scale": effective["lora_scale"],
            "format": effective["format"],
            "quant": quant_resolved,
            "compile_mode": effective["compile_mode"],
            "compile_enabled": compile_enabled,
            "batch_size": batch_size_cli,
            "shift": effective["shift"],
            "comfyui_noise": True,
            "sage_attention": bool(effective["sage_attention"]),
            "preload_loras": bool(effective["preload_loras"]),
            "min_step": min_step,
            "skip_face": bool(effective["skip_face"]),
            "model_path": args.model_path,
            "transformer_file": args.transformer_file,
            "transformer_dtype": args.transformer_dtype,
            "vae_file": args.vae_file,
            "vae_dtype": args.vae_dtype,
            "qwen_file": args.qwen_file,
            "qwen_dtype": effective["qwen_dtype"],
        }
        run_id = glt_db.create_run(db_conn, args.test_id, config_snapshot)
        glt_db.set_run_base_loras(db_conn, run_id, base_meta or [])
        if face_meta_run:
            glt_db.set_run_face_meta(db_conn, run_id, face_meta_run)

        tests_root = args.tests_root or test_def["tests_root"]
        if not tests_root:
            glt_db.finish_run(db_conn, run_id, "failed",
                              error="tests_root not configured (pass --tests-root or set in DB settings)")
            sys.exit("[error] tests_root is required (pass --tests-root or set in DB settings)")
        output_dir = Path(tests_root).resolve() / test_def["name"] / f"run_{run_id}"
    else:
        if not args.output_dir:
            sys.exit("[error] -o <output_dir> required in legacy CLI mode")
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output_dir = {output_dir}"
          + (f"  (run_id={run_id})" if run_id is not None else ""))

    # ---- Pre-populate rows + cells (DB mode) ----
    if db_conn is not None:
        assert run_id is not None
        for lora_idx, lora in enumerate(loras):
            glt_db.upsert_row(db_conn, run_id, lora_idx, lora_display_name(lora))
            for prompt_idx, ep in enumerate(expanded):
                glt_db.upsert_cell(
                    db_conn, run_id, lora_idx, prompt_idx,
                    ep.text, ep.width, ep.height,
                )

    # ---- Optional preload of every LoRA as a named peft adapter ----
    adapter_name_for: dict[int, str | None] = {}
    if effective["preload_loras"] and not args.dry_run:
        import torch as _torch_preload
        print(f"\n[preload] loading {len(loras)} LoRA(s) as persistent adapters")
        preload_t0 = time.time()
        n_loaded = 0
        for lora_idx, lora in enumerate(loras):
            name = f"adapter_{lora_idx}"
            try:
                pipe.load_lora_weights(str(lora), adapter_name=name)
                adapter_name_for[lora_idx] = name
                n_loaded += 1
                if n_loaded % 10 == 0 or n_loaded == len(loras):
                    msg = f"  [preload] {n_loaded}/{len(loras)} loaded"
                    if _torch_preload.cuda.is_available():
                        msg += f"  vram={_torch_preload.cuda.memory_allocated()/1024**3:.2f}GB"
                    print(msg)
            except Exception as e:
                print(f"  [preload error] {lora_display_name(lora)}: {e}")
                adapter_name_for[lora_idx] = None
        print(f"[preload] {n_loaded}/{len(loras)} in {time.time() - preload_t0:.1f}s")

    save_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="save")

    try:
        for lora_idx, lora in enumerate(loras):
            stem = safe_stem(lora)
            print(f"\n[{lora_idx + 1}/{len(loras)}] lora: {lora_display_name(lora)}")

            adapter_name: str | None = None
            lokr_mgr = None
            is_lokr = is_lokr_file(lora)

            if is_lokr:
                if is_int8_convrot:
                    print("  [warn] LoKr files are not supported in int8_convrot mode "
                          "(hooks vs INT8 baked weights), skipping")
                    continue
                lokr_mgr = LoKrHookManager(pipe.transformer)
                try:
                    transformer_device = next(pipe.transformer.parameters()).device
                    lokr_data = load_lokr_data(lora, device=transformer_device, dtype=None)
                    n_hooks = lokr_mgr.apply(lokr_data, strength=effective["lora_scale"])
                    if n_hooks == 0:
                        raise RuntimeError("no hooks attached (all submodule lookups failed)")
                    print(f"  [lokr] attached {n_hooks} forward hook(s) "
                          f"(strength={effective["lora_scale"]})")
                except Exception as e:
                    print(f"  [warn] failed to apply lokr ({e}), skipping LoRA")
                    if lokr_mgr is not None:
                        lokr_mgr.remove()
                    continue
            elif is_int8_convrot:
                # INT8 ConvRot path: restore baseline (= post-base-LoRA
                # state) then bake the test LoRA directly into the
                # quantized weights. ~1-2s per swap, no recompile, no
                # extra runtime branches in forward.
                try:
                    from safetensors.torch import load_file as _load_lora_sd
                    swap_t0 = time.time()
                    restore_quantized_baseline(pipe.transformer)
                    sd = _load_lora_sd(str(lora))
                    rep = load_lora_into_quantized(
                        pipe.transformer, sd, alpha_scale=effective["lora_scale"],
                    )
                    del sd
                    swap_t = time.time() - swap_t0
                    print(f"  [int8_convrot:lora] baked {rep['applied']}/"
                          f"{rep['total_pairs']} pair(s), swap_time={swap_t:.2f}s")
                    if rep["unmatched_key"] or rep["missing_module"]:
                        print(f"  [warn] {rep['unmatched_key']} unmatched + "
                              f"{rep['missing_module']} missing target keys; "
                              "may have been trained against a different layout")
                except Exception as e:
                    print(f"  [warn] failed to bake int8_convrot lora ({e}), skipping")
                    continue
            elif effective["preload_loras"]:
                adapter_name = adapter_name_for.get(lora_idx)
                if adapter_name is None:
                    print("  [warn] LoRA was not preloaded (load failed earlier), skipping")
                    continue
            else:
                adapter_name = f"adapter_{lora_idx}"
                try:
                    pipe.load_lora_weights(str(lora), adapter_name=adapter_name)
                except Exception as e:
                    print(f"  [warn] failed to load lora ({e}), skipping")
                    continue

            if not is_lokr and not is_int8_convrot:
                try:
                    all_names = base_names + [adapter_name]
                    all_weights = base_weights + [effective["lora_scale"]]
                    pipe.set_adapters(all_names, adapter_weights=all_weights)
                except Exception:
                    pass

            sage_row_stats0 = sage_attention_stats_snapshot()

            # ---- Group prompts by shape, then batch within each shape ----
            shape_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for prompt_idx, ep in enumerate(expanded):
                shape_groups[(ep.width, ep.height)].append(prompt_idx)

            MAX_ATTEMPTS = 5
            latents: list[object] = [None] * len(expanded)
            denoise_t0 = time.time()
            for (w, h), prompt_indices in shape_groups.items():
                batch_size = batch_size_cli if batch_size_cli > 0 else len(prompt_indices)
                batch_size = max(1, min(batch_size, len(prompt_indices)))
                for chunk_start in range(0, len(prompt_indices), batch_size):
                    chunk = prompt_indices[chunk_start:chunk_start + batch_size]
                    B = len(chunk)
                    chunk_embeds = [prompt_embeds_list[i] for i in chunk]
                    t0 = time.time()
                    last_err = None
                    absorbed: list[str] = []
                    batch_latents = None
                    for attempt in range(MAX_ATTEMPTS):
                        try:
                            batch_latents = generate_latents_batched(
                                pipe, chunk_embeds,
                                width=w, height=h,
                                steps=effective["steps"], guidance=effective["guidance"],
                                seed=effective["seed"],
                                comfyui_noise=True,
                                debug_dump=args.debug_dump,
                            )
                            break
                        except Exception as e:
                            last_err = e
                            absorbed.append(type(e).__name__)
                    if batch_latents is not None:
                        for i, p_idx in enumerate(chunk):
                            latents[p_idx] = batch_latents[i:i + 1]
                        dt = time.time() - t0
                        tag = f"denoise={dt:.1f}s (batch={B} @ {w}x{h})"
                        if absorbed:
                            tag += f" (absorbed {len(absorbed)}: {','.join(absorbed)})"
                        print(f"  {tag}")
                    else:
                        print(f"  [error] batch {chunk} @ {w}x{h} failed after "
                              f"{MAX_ATTEMPTS} attempts: {last_err}")
            denoise_t = time.time() - denoise_t0

            decode_t0 = time.time()
            pil_images = batch_decode_latents(pipe, latents, chunk_size=4)
            decode_t = time.time() - decode_t0
            for i in range(len(latents)):
                latents[i] = None

            # ---- Save + score + DB updates ----
            scores: list[float | None] = [None] * len(expanded)
            new_count = 0
            face_t0 = time.time()
            for p_idx in range(len(expanded)):
                pil = pil_images[p_idx]
                if pil is None:
                    if db_conn is not None and run_id is not None:
                        glt_db.set_cell_image(db_conn, run_id, lora_idx, p_idx, None)
                    continue
                out_name = f"{stem}_{p_idx}.{effective['format']}"
                save_pool.submit(save_image, pil, output_dir / out_name,
                                 effective["format"], args.jpg_quality)
                new_count += 1
                if db_conn is not None and run_id is not None:
                    glt_db.set_cell_image(db_conn, run_id, lora_idx, p_idx, out_name)
                if scorer is not None:
                    s = scorer.score_pil(pil)
                    scores[p_idx] = s
                    if db_conn is not None and run_id is not None:
                        glt_db.set_cell_score(db_conn, run_id, lora_idx, p_idx, s)
            face_t = (time.time() - face_t0) if scorer is not None else 0.0

            if scorer is not None:
                metrics = compute_row_metrics(scores)
                if db_conn is not None and run_id is not None:
                    glt_db.set_row_metrics(db_conn, run_id, lora_idx, metrics)
                m = metrics
                if m["n_faces"]:
                    print(f"  [face] median={m['median']:.3f} p20={m['p20']:.3f} "
                          f"faces={m['n_faces']}/{m['n_total']}")
                else:
                    print(f"  [face] no faces detected ({m['n_total']} img)")

            print(f"  [timing] denoise={denoise_t:.1f}s decode={decode_t:.1f}s"
                  + (f" face={face_t:.1f}s" if scorer is not None else "")
                  + f" ({new_count} new image(s))")
            print_sage_attention_stats(label="row", since=sage_row_stats0)

            # ---- Unload LoRA ----
            if is_int8_convrot:
                # No explicit unload: the next row will `restore_quantized_baseline`
                # before baking the new LoRA. Skipping the restore here avoids
                # a redundant CPU->GPU copy when iterating.
                pass
            elif is_lokr:
                if lokr_mgr is not None:
                    lokr_mgr.remove()
                gc.collect()
            elif not effective["preload_loras"]:
                try:
                    pipe.delete_adapters(adapter_name)
                except Exception as _e:
                    if base_names:
                        print(f"  [warn] delete_adapters failed ({_e}); base LoRAs kept intact")
                    else:
                        try:
                            pipe.unload_lora_weights()
                        except Exception:
                            pass
                gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except ImportError:
                    pass

        print("\n[save] waiting for background image saves to finish")
        save_pool.shutdown(wait=True)

        # ---- Standalone HTML (DB mode) ----
        if db_conn is not None and run_id is not None and not args.no_html:
            run_snapshot = glt_db.fetch_run(db_conn, run_id)
            run_snapshot["test_name"] = test_def["name"] if test_def else "ad-hoc"
            write_html_inline(output_dir, run_snapshot)
            print(f"[html] {output_dir / 'index.html'}")

        if db_conn is not None and run_id is not None:
            glt_db.finish_run(db_conn, run_id, "completed")
        print_sage_attention_stats()
        print("\n[done]")
    except KeyboardInterrupt:
        if db_conn is not None and run_id is not None:
            glt_db.finish_run(db_conn, run_id, "cancelled", error="KeyboardInterrupt")
        raise
    except Exception as e:
        save_pool.shutdown(wait=False)
        if db_conn is not None and run_id is not None:
            glt_db.finish_run(db_conn, run_id, "failed", error=f"{type(e).__name__}: {e}")
        raise
