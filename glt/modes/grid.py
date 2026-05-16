"""`glt <input_dirs> -prompts <p>` — main grid generation mode.

Orchestrates the whole flow:
  1. discover adapters in the input dir(s)
  2. plan the per-row image set (idempotence: skip cells already on disk)
  3. build the FLUX pipeline (FP8 / offload / sage / compile)
  4. encode prompts once, reuse for every row
  5. load sticky base LoRAs (config.json `base_loras`)
  6. per row: load adapter → denoise → decode → save → score → write artifacts
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
from ..generation.prompts import encode_prompts, prompts_embeds_cache_path
from ..output.images import save_image
from ..output.manifest import write_grid_artifacts
from ..pipeline.attention import (
    print_sage_attention_stats,
    sage_attention_stats_snapshot,
)
from ..pipeline.build import build_pipeline
from ..pipeline.loaders import DEFAULT_MODEL_ID as MODEL_ID
from ..utils.config import load_config, load_prompts


def parse_args():
    # disable default -h / --help so we can use -h for height
    parser = argparse.ArgumentParser(
        description="Generate a LoRA x prompt grid with FLUX.2-klein-9B (FP8).",
        add_help=False,
    )
    parser.add_argument("input_dir", nargs="+",
                        help="Directory or directories containing .safetensors LoRA files. "
                             "Pass multiple to merge them into one grid — files are sorted "
                             "by (basename, step) across all sources, and output filenames "
                             "are prefixed with the source dir to avoid clashes when two "
                             "folders have files with the same name.")
    parser.add_argument("-prompts", dest="prompts", required=True, help="Text file with one prompt per line")
    parser.add_argument("-w", dest="width", type=int, default=1024, help="Image width (default 1024)")
    parser.add_argument("-h", dest="height", type=int, default=1024, help="Image height (default 1024)")
    parser.add_argument("-o", dest="output_dir", required=True, help="Output directory")
    parser.add_argument("-s", dest="seed", type=int, default=42, help="Seed (default 42)")
    parser.add_argument("-minStep", "--min-step", dest="min_step", type=int, default=0,
                        help="Skip LoRAs whose parsed step < this value. Files without a "
                             "trailing _<digits> are treated as 'final' and always kept.")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default 4 — klein is a few-step model)")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale (default 1.0)")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength (default 1.0)")
    parser.add_argument("--format", choices=["png", "jpg"], default="png", help="Output image format")
    parser.add_argument("--jpg-quality", type=int, default=92, help="JPEG quality if --format jpg")
    parser.add_argument("--model-path", default=MODEL_ID,
                        help=f"HF repo ID OR local diffusers-format directory for the pipeline. "
                             f"Default: {MODEL_ID} (downloads from HF Hub to HF cache). "
                             f"Set to a local path like /workspace/models/flux2-klein-9b to avoid "
                             f"re-downloading on fresh pods. Also acts as the source of "
                             f"config/tokenizer when individual components below are overridden.")
    parser.add_argument("--transformer-file", default=None,
                        help="Path to a .safetensors with the transformer (FLUX-2 ComfyUI-style). "
                             "Replaces only the transformer; other components come from --model-path.")
    parser.add_argument("--text-encoder-file", default=None,
                        help="Path to a .safetensors with Qwen3 text encoder weights. "
                             "Config + tokenizer are still pulled from --model-path.")
    parser.add_argument("--vae-file", default=None,
                        help="Path to a .safetensors with the FLUX-2 VAE.")
    # ComfyUI-style mixed setups are common: e.g. fp8 transformer + bf16 TE
    # + bf16 VAE. Each component has its own dtype flag. When a component is
    # loaded from FP8 source, the matching --fp8-mode quantization for that
    # component is auto-skipped to avoid double-quantization.
    # "fp8" is a friendly alias for "fp8_e4m3fn" (the variant used by 99% of
    # community ComfyUI checkpoints for FLUX inference; e5m2 has wider range
    # but slightly worse accuracy, used mostly for training).
    _DTYPE_CHOICES = ["bf16", "fp16", "fp8", "fp8_e4m3fn", "fp8_e5m2", "fp8_mixed"]
    parser.add_argument("--transformer-dtype", choices=_DTYPE_CHOICES, default="bf16",
                        help="Storage dtype of --transformer-file (or the transformer subfolder of "
                             "--model-path when no override). 'bf16' (default) = compatible with "
                             "--fp8-mode torchao quant. 'fp8_e4m3fn'/'fp8_e5m2' = checkpoint is "
                             "already FP8 (ComfyUI style); torchao is auto-skipped. "
                             "'fp8_mixed' = mixed fp8/bf16 layers (loaded as-is).")
    parser.add_argument("--text-encoder-dtype", choices=_DTYPE_CHOICES, default="bf16",
                        help="Storage dtype of --text-encoder-file (or text_encoder subfolder of "
                             "--model-path). Qwen3 is usually bf16 in community packs.")
    parser.add_argument("--vae-dtype", choices=_DTYPE_CHOICES, default="bf16",
                        help="Storage dtype of --vae-file (or vae subfolder of --model-path). "
                             "VAE is ~300 MB so bf16 is fine — fp8 saves negligible VRAM.")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 quantization (use bfloat16)")
    parser.add_argument("--fp8-backend", choices=["torchao", "quanto"], default="torchao",
                        help="FP8 backend (default torchao — no JIT compile, no CUDA toolkit needed)")
    parser.add_argument("--fp8-mode", choices=["weight-only", "dynamic"], default="weight-only",
                        help="torchao FP8 quantization mode. 'weight-only' (default, safe) stores "
                             "weights in FP8 but computes in bf16 — saves VRAM, no compute speedup. "
                             "'dynamic' also quantizes activations and uses real FP8 tensor cores "
                             "— ~1.5-2x faster on Hopper/Ada (H100, L40S, 4090), but no-op (or slower) "
                             "on Ampere (3090, A100). Quality may degrade slightly; validate before use.")
    parser.add_argument("--preload-loras", action="store_true",
                        help="Load ALL LoRAs as named adapters at startup and switch the active one "
                             "via set_adapters() per row (instead of load/unload). Eliminates per-row "
                             "LoRA loading overhead and enables stable torch.compile. Costs extra VRAM "
                             "(~200-500MB per LoRA × N) — only viable on 40GB+ GPUs for large grids.")
    parser.add_argument("--sage-attention", action="store_true",
                        help="Replace torch SDPA with SageAttention for the transformer's "
                             "joint (text+image) attention. Significant speedup on FLUX "
                             "(~30-40%%) on Ada/Hopper/Blackwell. Falls back to original "
                             "SDPA when the call has a mask, is_causal, or dropout (so the "
                             "Qwen3 text encoder isn't affected). Requires `pip install sageattention`.")
    parser.add_argument("--compile-transformer", action="store_true",
                        help="torch.compile(pipe.transformer, mode='reduce-overhead'). +30-50%% throughput "
                             "on stable shapes. WARNING: incompatible with --preload-loras when there are "
                             "multiple LoRAs in the grid — the compiled graph bakes in the first adapter "
                             "and set_adapters() can't swap it (script refuses this combination). Useful "
                             "for single-LoRA inference or when --batch-size 0 with one prompt. Best on "
                             "Hopper/Ada/Blackwell.")
    offload_group = parser.add_mutually_exclusive_group()
    offload_group.add_argument("--no-offload", action="store_true",
                        help="Keep ALL components on GPU (fastest, needs ~24GB VRAM with FP8)")
    offload_group.add_argument("--offload-text-encoder", action="store_true",
                        help="Transformer/VAE on GPU, text encoder paged to RAM (best on 24GB)")
    parser.add_argument("--no-html", action="store_true", help="Skip writing the HTML grid")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate images even if they already exist on disk "
                             "(default: skip prompts whose output is already there, "
                             "skip loading the LoRA entirely if its whole row is done)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=0,
                        help="Process N prompts per pipe() call within each LoRA. "
                             "0 (default) = full batch of all missing prompts. "
                             "Each batch item uses an identically-seeded generator "
                             "so results are bit-identical to batch=1.")
    parser.add_argument("--config", default=None,
                        help="Path to config.json defining sticky 'base' LoRAs that "
                             "stay loaded throughout the run (see README). Defaults "
                             "to ./config.json next to this script if it exists.")
    parser.add_argument("--skip-face-recognition", "--no-face", dest="skip_face",
                        action="store_true",
                        help="Don't load insightface or compute face-similarity scores "
                             "during this run, even if config.json has face_recognition. "
                             "Useful to keep the generation pass focused on diffusion only "
                             "(no CUDA-context contention with onnxruntime). You can run "
                             "`--rescore` separately afterwards to add the scores.")
    parser.add_argument("--shift", type=float, default=None,
                        help="Override the scheduler's mu/shift parameter. Diffusers' Klein "
                             "pipeline defaults to compute_empirical_mu(image_seq_len, num_steps) "
                             "≈ 2.28 for 832×1216 + 4 steps. ComfyUI hardcodes shift=2.02 for "
                             "FLUX-2 (cf supported_models.py:771). Pass --shift 2.02 to monkey-"
                             "patch compute_empirical_mu and match ComfyUI exactly. Combined with "
                             "--comfyui-noise, gives full sampler+noise parity with ComfyUI.")
    parser.add_argument("--comfyui-noise", action="store_true",
                        help="Use CPU torch generators for the initial noise — the way "
                             "ComfyUI's KSampler does (each prompt = torch.manual_seed(seed) "
                             "then torch.randn on CPU, moved to GPU). Same seed integer "
                             "yields identical noise to a single-prompt ComfyUI run. Without "
                             "this flag, generators are on CUDA, which gives a completely "
                             "different RNG sequence even with the same seed.")
    parser.add_argument("--force-rescore", action="store_true",
                        help="Recompute face-similarity scores from scratch. Without this "
                             "flag (default), existing per-cell scores in manifest.json are "
                             "preserved: only newly-generated images get fresh scores, and "
                             "rescore mode skips rows that are already fully scored.")
    parser.add_argument("--debug-dump", dest="debug_dump", default=None,
                        help="Directory to dump intermediate tensors for ComfyUI parity "
                             "debugging: noise_initial.pt (latents from prepare_latents, "
                             "shape [B, h*w, 128]), prompt_embeds.pt, sigmas.pt (final "
                             "shifted schedule from scheduler), and latents_after_step_N.pt "
                             "for each denoising step. Fires ONCE on the first batch call, "
                             "then deactivates — re-run grid to re-dump. Pair with a Comfy "
                             "side dump at the same seed/dims to bisect divergences.")
    parser.add_argument("--dry-run", action="store_true", help="List the work without loading the model")
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    return parser.parse_args()


def run(args=None):
    """Entry point: parse CLI args (if  is None) and run the grid."""
    if args is None:
        args = parse_args()

    # Optional shift override — monkey-patch compute_empirical_mu in every
    # FLUX-2 pipeline module that defines it. Applied as early as possible so
    # any subsequent pipeline import sees the patched function.
    if args.shift is not None:
        try:
            import diffusers.pipelines.flux2.pipeline_flux2_klein as _klein
            import diffusers.pipelines.flux2.pipeline_flux2 as _flux2
            import diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint as _klein_ip
            import diffusers.pipelines.flux2.pipeline_flux2_klein_kv as _klein_kv
            _shift = float(args.shift)
            _patched = lambda image_seq_len, num_steps: _shift
            for _mod in (_klein, _flux2, _klein_ip, _klein_kv):
                if hasattr(_mod, "compute_empirical_mu"):
                    _mod.compute_empirical_mu = _patched
            print(f"[shift] override compute_empirical_mu → fixed mu = {args.shift} "
                  f"(matches ComfyUI's hardcoded FLUX-2 shift)")
        except ImportError as e:
            print(f"[shift] override failed: {e}")

    # `input_dir` is nargs="+", so always a list. Single dir = list of 1.
    input_dirs = [Path(d).resolve() for d in args.input_dir]
    output_dir = Path(args.output_dir).resolve()
    prompts_path = Path(args.prompts).resolve()

    for d in input_dirs:
        if not d.is_dir():
            sys.exit(f"[error] input_dir not found: {d}")
    if not prompts_path.is_file():
        sys.exit(f"[error] prompts file not found: {prompts_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path)
    loras = discover_loras(input_dirs)
    config = load_config(Path(args.config) if args.config else None)
    base_lora_entries = config.get("base_loras", [])
    face_cfg = config.get("face_recognition") or {}
    # Scorer is constructed later (after pipeline build) so we keep VRAM
    # peaks separated. Here we just resolve whether we'll use it.
    face_enabled = bool(face_cfg) and ("centroid" in face_cfg or "centroid_b64" in face_cfg)
    if args.skip_face and face_enabled:
        print("[face] --skip-face-recognition: skipping face scoring this run "
              "(re-run with --rescore -o <output_dir> --config <config> to add scores later)")
        face_enabled = False

    if args.min_step > 0:
        before = len(loras)
        loras = [l for l in loras if parse_lora_step(l)[1] >= args.min_step]
        print(f"[filter] --min-step {args.min_step}: kept {len(loras)}/{before} LoRA(s)")
        if not loras:
            sys.exit(f"[error] all LoRAs filtered out (no step >= {args.min_step})")

    if len(input_dirs) > 1:
        print(f"[plan] inputs: {len(input_dirs)} folders → {' | '.join(str(d) for d in input_dirs)}")
    print(f"[plan] {len(loras)} lora(s) × {len(prompts)} prompt(s) = {len(loras) * len(prompts)} image(s)")
    print(f"[plan] output -> {output_dir}")
    for lora in loras:
        _, step = parse_lora_step(lora)
        step_str = "final" if step == math.inf else f"step={step}"
        print(f"  - {lora_display_name(lora)}  ({step_str})")

    manifest = {
        "meta": {
            "model": MODEL_ID,
            "width": args.width,
            "height": args.height,
            "seed": args.seed,
            "steps": args.steps,
            "guidance": args.guidance,
            "lora_scale": args.lora_scale,
            "format": args.format,
            "fp8": not args.no_fp8,
            "base_loras": [],  # filled after base LoRAs are loaded
            "face_recognition": None,  # filled if scorer is enabled
        },
        "prompts": prompts,
        "rows": [],
    }

    if args.dry_run:
        for lora in loras:
            stem = safe_stem(lora)
            manifest["rows"].append({
                "lora": lora_display_name(lora),
                "images": [f"{stem}_{i}.{args.format}" for i in range(len(prompts))],
            })
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    # Idempotence planning: figure out per-LoRA which prompts already have a
    # file on disk. A LoRA whose whole row is present is skipped entirely
    # (no load_lora_weights, no encoding cost paid for it).
    lora_plan = []
    for lora in loras:
        stem = safe_stem(lora)
        names = [f"{stem}_{i}.{args.format}" for i in range(len(prompts))]
        if args.force:
            existing = [None] * len(prompts)
            missing_idx = list(range(len(prompts)))
        else:
            existing = [n if (output_dir / n).exists() else None for n in names]
            missing_idx = [i for i, x in enumerate(existing) if x is None]
        lora_plan.append({
            "lora": lora,
            "stem": stem,
            "names": names,
            "existing": existing,
            "missing": missing_idx,
            "is_lokr": is_lokr_file(lora),
        })

    total_missing = sum(len(p["missing"]) for p in lora_plan)
    fully_done = sum(1 for p in lora_plan if not p["missing"])
    print(f"[plan] {total_missing} image(s) to generate; "
          f"{fully_done}/{len(loras)} LoRA(s) already complete on disk"
          + (" (--force: regenerating everything)" if args.force else ""))

    if total_missing == 0:
        # Nothing to generate — still refresh manifest + HTML in case prompts
        # or display metadata changed. If face-rec is enabled and not in
        # --force-rescore mode, prefer reusing scores from any existing
        # manifest; otherwise re-score from disk.
        scorer = None
        if face_enabled:
            try:
                scorer = FaceScorer(face_cfg)
                manifest["meta"]["face_recognition"] = {
                    "model_name": scorer.model_name,
                    "providers": scorer.providers,
                    "centroid_dim": int(scorer.centroid.shape[0]),
                    "thresholds": scorer.thresholds,
                }
            except Exception as e:
                print(f"[face] disabled: {e}")
                scorer = None

        # Load existing scores from manifest for reuse
        prev_scores_by_lora_early: "dict[str, list]" = {}
        if scorer is not None and not args.force_rescore:
            prev_manifest_path = output_dir / "manifest.json"
            if prev_manifest_path.exists():
                try:
                    prev_m = json.loads(prev_manifest_path.read_text(encoding="utf-8"))
                    for r in prev_m.get("rows", []):
                        if r.get("scores"):
                            prev_scores_by_lora_early[r.get("lora", "")] = list(r["scores"])
                    if prev_scores_by_lora_early:
                        print(f"[face] reusing {len(prev_scores_by_lora_early)} pre-existing row score(s) "
                              f"(override with --force-rescore)")
                except Exception as e:
                    print(f"[face] could not parse manifest for score reuse: {e}")

        for plan in lora_plan:
            display = lora_display_name(plan["lora"])
            row = {"lora": display, "images": plan["existing"]}
            if scorer is not None:
                prev = prev_scores_by_lora_early.get(display)
                if prev is not None and len(prev) == len(prompts) and not args.force_rescore:
                    scores = list(prev)
                    src = "reused"
                else:
                    scores = []
                    for name in plan["existing"]:
                        if name is None:
                            scores.append(None)
                            continue
                        scores.append(scorer.score_path(output_dir / name))
                    src = "scored"
                row["scores"] = scores
                row["metrics"] = compute_row_metrics(scores)
                m = row["metrics"]
                if m["n_faces"]:
                    print(f"  [face/{src}] {display}: median={m['median']:.3f} "
                          f"p20={m['p20']:.3f} faces={m['n_faces']}/{m['n_total']}")
                else:
                    print(f"  [face/{src}] {display}: no faces detected ({m['n_total']} img)")
            manifest["rows"].append(row)
        write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
        print(f"[manifest] {output_dir / 'manifest.json'}")
        print_sage_attention_stats()
        print("\n[done] all images already on disk")
        return

    # Load any pre-existing manifest BEFORE overwriting it with placeholders,
    # so we can preserve face-similarity scores across re-runs (default
    # behaviour — a generate pass shouldn't drop previously computed scores
    # for images that aren't being regenerated). Stash them keyed by display
    # name. Override the whole behaviour with --force-rescore.
    prev_scores_by_lora: "dict[str, list]" = {}
    if not args.force_rescore:
        prev_manifest_path = output_dir / "manifest.json"
        if prev_manifest_path.exists():
            try:
                prev_m = json.loads(prev_manifest_path.read_text(encoding="utf-8"))
                for r in prev_m.get("rows", []):
                    if r.get("scores"):
                        prev_scores_by_lora[r.get("lora", "")] = list(r["scores"])
                if prev_scores_by_lora:
                    print(f"[face] reusing {len(prev_scores_by_lora)} pre-existing row score(s) "
                          f"from {prev_manifest_path.name} (override with --force-rescore)")
            except Exception as e:
                print(f"[face] could not parse existing manifest for score reuse: {e}")

    # Pre-populate manifest with placeholder rows now, so index.html shows
    # the empty grid structure while the pipeline is still loading. Each row
    # is replaced as the corresponding LoRA finishes (see row loop below).
    # Carry over any pre-existing scores so the HTML keeps showing them
    # while new rows are being computed.
    manifest["rows"] = []
    for plan in lora_plan:
        row_entry = {"lora": lora_display_name(plan["lora"]),
                     "images": list(plan["existing"])}
        prev = prev_scores_by_lora.get(lora_display_name(plan["lora"]))
        if prev is not None and len(prev) == len(prompts):
            row_entry["scores"] = list(prev)
            row_entry["metrics"] = compute_row_metrics(prev)
        manifest["rows"].append(row_entry)
    write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
    print(f"[init] wrote skeleton manifest+index.html — open {output_dir / 'index.html'} "
          f"to watch progress as rows complete")

    if args.no_offload:
        offload_mode = "none"
    elif args.offload_text_encoder:
        offload_mode = "text-encoder"
    else:
        offload_mode = "full"

    # ---- Guard rail ----------------------------------------------------------
    # torch.compile traces the transformer graph including peft's
    # `for adapter_name in self.active_adapters: ...` loop in TorchaoLoraLinear.
    # That loop is unrolled at compile time with adapter_0 baked in. Subsequent
    # `pipe.set_adapters([adapter_N])` calls change the Python variable but do
    # NOT reach the cached compiled graph → every row silently produces the
    # output of the FIRST LoRA, with normal-looking 0.2s denoise timings.
    # This combination is therefore refused outright.
    if args.compile_transformer and args.preload_loras and len(loras) > 1:
        sys.exit(
            "[fatal] --compile-transformer + --preload-loras with multiple LoRAs is broken:\n"
            "        torch.compile bakes the active LoRA adapter into the compiled graph;\n"
            "        set_adapters() can't swap it at runtime, so every row generates the\n"
            "        SAME images (those of the first LoRA).\n"
            "        Pick one:\n"
            "          • drop --compile-transformer (recommended for multi-LoRA grids)\n"
            "          • drop --preload-loras (forces re-load + recompile per row — very slow)"
        )

    # LoKr adapters apply via forward hooks (not peft adapters), so they're
    # incompatible with --preload-loras (which preloads named peft adapters)
    # and with --compile-transformer (hooks aren't part of the traced graph).
    n_lokr = sum(1 for p in lora_plan if p["is_lokr"])
    if n_lokr:
        print(f"[lokr] detected {n_lokr}/{len(loras)} LoKr file(s) — will route via forward hooks")
    if n_lokr and args.preload_loras:
        sys.exit(
            "[fatal] --preload-loras is incompatible with LoKr files: LoKrs apply via\n"
            "        forward hooks, not named peft adapters. Drop --preload-loras."
        )
    if n_lokr and args.compile_transformer:
        sys.exit(
            "[fatal] --compile-transformer is incompatible with LoKr files: forward hooks\n"
            "        aren't included in the compiled graph, so the LoKr delta would be\n"
            "        silently dropped. Drop --compile-transformer."
        )

    # Check the prompt-embeds cache BEFORE loading the pipeline so we can
    # skip Qwen3 entirely on a cache hit. Only safe to skip when the caller
    # provided a real text encoder source (the default HF model_path), not
    # a --text-encoder-file with custom dtype (those need re-encoding).
    embeds_cache = prompts_embeds_cache_path(prompts_path)
    embeds_cache_hit = embeds_cache.exists() and not args.text_encoder_file
    if embeds_cache_hit:
        print(f"[embeds:cache] pre-check hit → will skip Qwen3 load entirely  ({embeds_cache})")

    pipe = build_pipeline(
        use_fp8=not args.no_fp8,
        offload_mode=offload_mode,
        backend=args.fp8_backend,
        fp8_mode=args.fp8_mode,
        compile_transformer=args.compile_transformer,
        sage_attention=args.sage_attention,
        model_path=args.model_path,
        transformer_file=args.transformer_file,
        transformer_dtype=args.transformer_dtype,
        text_encoder_file=args.text_encoder_file,
        text_encoder_dtype=args.text_encoder_dtype,
        vae_file=args.vae_file,
        vae_dtype=args.vae_dtype,
        skip_text_encoder=embeds_cache_hit,
    )

    prompt_embeds_list = encode_prompts(pipe, prompts, offload_mode=offload_mode,
                                         prompts_path=prompts_path)

    # Sticky base LoRAs (always applied on top of each test LoRA, never unloaded).
    base_names, base_weights, base_meta = load_base_loras(pipe, base_lora_entries)
    manifest["meta"]["base_loras"] = base_meta

    # Face-similarity scorer (optional). Created after FLUX so we don't fight
    # over the same CUDA init burst; insightface allocates a few hundred MB
    # which fits alongside FLUX on most setups.
    scorer = None
    if face_enabled:
        try:
            scorer = FaceScorer(face_cfg)
            manifest["meta"]["face_recognition"] = {
                "model_name": scorer.model_name,
                "providers": scorer.providers,
                "centroid_dim": int(scorer.centroid.shape[0]),
                "thresholds": scorer.thresholds,
            }
        except Exception as e:
            print(f"[face] disabled: {e}")
            scorer = None

    # Optional: preload every test LoRA as its own named adapter UP FRONT,
    # then switch the active one per row via set_adapters(). Avoids the
    # per-row load_lora_weights cost (~1-3s/LoRA) and — critically — keeps
    # the transformer graph stable so torch.compile doesn't retrace.
    if args.preload_loras and not args.dry_run:
        import torch as _torch_preload
        loras_to_preload = [p for p in lora_plan if p["missing"]]
        print(f"\n[preload] loading {len(loras_to_preload)} LoRA(s) as persistent adapters "
              f"(VRAM cost ~200-500MB each)")
        preload_t0 = time.time()
        n_loaded = 0
        for lora_idx, plan in enumerate(lora_plan):
            if not plan["missing"]:
                plan["adapter_name"] = None
                continue
            adapter_name = f"adapter_{lora_idx}"
            try:
                pipe.load_lora_weights(str(plan["lora"]), adapter_name=adapter_name)
                plan["adapter_name"] = adapter_name
                n_loaded += 1
                if n_loaded % 10 == 0 or n_loaded == len(loras_to_preload):
                    msg = f"  [preload] {n_loaded}/{len(loras_to_preload)} loaded"
                    if _torch_preload.cuda.is_available():
                        msg += f"  vram={_torch_preload.cuda.memory_allocated()/1024**3:.2f}GB"
                    print(msg)
            except Exception as e:
                print(f"  [preload error] {lora_display_name(plan['lora'])}: {e}")
                plan["adapter_name"] = None
        print(f"[preload] {n_loaded}/{len(loras_to_preload)} loaded in {time.time()-preload_t0:.1f}s")

    save_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="save")

    for lora_idx, plan in enumerate(lora_plan):
        lora = plan["lora"]
        stem = plan["stem"]
        missing = plan["missing"]
        existing = plan["existing"]

        if not missing:
            print(f"\n[{lora_idx + 1}/{len(loras)}] lora: {lora_display_name(lora)}  "
                  f"[skip] {len(prompts)}/{len(prompts)} images already on disk")
            row = {"lora": lora_display_name(lora), "images": existing}
            if scorer is not None:
                # All images already on disk for this LoRA. Reuse prev scores
                # if we have them and --force-rescore isn't set; otherwise
                # score from disk.
                prev = prev_scores_by_lora.get(lora_display_name(lora))
                if prev is not None and len(prev) == len(prompts) and not args.force_rescore:
                    scores = list(prev)
                    row["scores"] = scores
                    row["metrics"] = compute_row_metrics(scores)
                    m = row["metrics"]
                    if m["n_faces"]:
                        print(f"  [face] reusing prev manifest scores: "
                              f"median={m['median']:.3f} faces={m['n_faces']}/{m['n_total']}")
                    else:
                        print(f"  [face] reusing prev manifest scores (no faces)")
                    manifest["rows"][lora_idx] = row
                    write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
                    continue

                t0 = time.time()
                scores = []
                for name in existing:
                    if name is None:
                        scores.append(None)
                        continue
                    scores.append(scorer.score_path(output_dir / name))
                row["scores"] = scores
                row["metrics"] = compute_row_metrics(scores)
                m = row["metrics"]
                if m["n_faces"]:
                    print(f"  [face] median={m['median']:.3f} p20={m['p20']:.3f} "
                          f"faces={m['n_faces']}/{m['n_total']} ({time.time()-t0:.1f}s)")
                else:
                    print(f"  [face] no faces detected ({m['n_total']} img)")
            manifest["rows"][lora_idx] = row
            write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
            continue

        n_skipped = len(prompts) - len(missing)
        skip_tag = f" (skipping {n_skipped} already done)" if n_skipped else ""
        print(f"\n[{lora_idx + 1}/{len(loras)}] lora: {lora_display_name(lora)}{skip_tag}")

        adapter_name = None
        lokr_mgr = None
        if plan["is_lokr"]:
            # LoKr path: register forward hooks instead of loading a peft adapter.
            lokr_mgr = LoKrHookManager(pipe.transformer)
            try:
                # Keep w1/w2 in their native dtype (bf16) on the transformer's
                # device. We avoid casting to the transformer's param dtype
                # because under --fp8 that's fp8_e4m3fn, which is not a valid
                # compute dtype for the two LoKr matmuls.
                transformer_device = next(pipe.transformer.parameters()).device
                lokr_data = load_lokr_data(
                    lora,
                    device=transformer_device,
                    dtype=None,
                )
                n_hooks = lokr_mgr.apply(lokr_data, strength=args.lora_scale)
                if n_hooks == 0:
                    raise RuntimeError("no hooks attached (all submodule lookups failed)")
                print(f"  [lokr] attached {n_hooks} forward hook(s) "
                      f"(strength={args.lora_scale})")
            except Exception as e:
                print(f"  [warn] failed to apply lokr ({e}), skipping")
                lokr_mgr.remove()
                manifest["rows"][lora_idx] = {"lora": lora_display_name(lora), "images": existing}
                write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
                continue
            # Sticky base LoRAs (peft adapters) stay active via set_adapters;
            # the LoKr hooks compose on top of their output.
        elif args.preload_loras:
            adapter_name = plan.get("adapter_name")
            if adapter_name is None:
                print(f"  [warn] LoRA was not preloaded (load failed earlier), skipping")
                manifest["rows"][lora_idx] = {"lora": lora_display_name(lora), "images": existing}
                write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
                continue
        else:
            adapter_name = f"adapter_{lora_idx}"
            try:
                pipe.load_lora_weights(str(lora), adapter_name=adapter_name)
            except Exception as e:
                print(f"  [warn] failed to load lora ({e}), skipping")
                manifest["rows"][lora_idx] = {"lora": lora_display_name(lora), "images": existing}
                write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
                continue

        # Activate the right peft adapter combination. In preload mode this
        # is the ONLY per-row LoRA op. Skipped for LoKr (no peft adapter to
        # activate — base sticky LoRAs are already active from earlier setup).
        if not plan["is_lokr"]:
            try:
                all_names = base_names + [adapter_name]
                all_weights = base_weights + [args.lora_scale]
                pipe.set_adapters(all_names, adapter_weights=all_weights)
            except Exception:
                pass  # older diffusers may not have set_adapters
        # NOTE: retry logic in the prompt loop below absorbs the well-known
        # peft+torchao first-forward(s) device mismatch (1-2 failed pipe()
        # calls per LoRA load). No separate warmup step needed.
        sage_row_stats0 = sage_attention_stats_snapshot()

        # Phase 1 — denoise ONLY the missing prompts, in batches of
        # `args.batch_size` (0 = single batch covering everything missing).
        # Retry up to 5 times per batch: the first forwards after a fresh
        # load_lora_weights are known to hit a peft+torchao device hiccup;
        # state self-corrects after a couple absorbed failures.
        MAX_ATTEMPTS = 5
        latents: list[object] = [None] * len(prompts)
        batch_size = args.batch_size if args.batch_size > 0 else len(missing)
        batch_size = max(1, min(batch_size, len(missing)))
        denoise_t0 = time.time()
        for chunk_start in range(0, len(missing), batch_size):
            chunk = missing[chunk_start:chunk_start + batch_size]
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
                        width=args.width, height=args.height,
                        steps=args.steps, guidance=args.guidance,
                        seed=args.seed,
                        comfyui_noise=args.comfyui_noise,
                        debug_dump=args.debug_dump,
                    )
                    break
                except Exception as e:
                    last_err = e
                    absorbed.append(type(e).__name__)
            if batch_latents is not None:
                # Slice the batched output back into individual [1, C, H, W] latents
                for i, p_idx in enumerate(chunk):
                    latents[p_idx] = batch_latents[i:i + 1]
                dt = time.time() - t0
                tag = f"denoise={dt:.1f}s (batch={B})"
                if absorbed:
                    tag += f" (absorbed {len(absorbed)}: {','.join(absorbed)})"
                print(f"  [batch {chunk[0] + 1}..{chunk[-1] + 1}/{len(prompts)}] {tag}")
            else:
                print(f"  [error] batch {chunk} failed after {MAX_ATTEMPTS} attempts: {last_err}")
                for p_idx in chunk:
                    latents[p_idx] = None
        denoise_t = time.time() - denoise_t0

        # Phase 2 — batch VAE decode for the slots we generated.
        decode_t0 = time.time()
        pil_images = batch_decode_latents(pipe, latents, chunk_size=4)
        decode_t = time.time() - decode_t0
        for i in range(len(latents)):
            latents[i] = None

        # Phase 3 — async save the new images, keep existing names where present.
        row_images: list[str | None] = list(existing)
        new_count = 0
        for p_idx in missing:
            pil = pil_images[p_idx]
            if pil is None:
                row_images[p_idx] = None
                continue
            out_name = plan["names"][p_idx]
            save_pool.submit(save_image, pil, output_dir / out_name, args.format, args.jpg_quality)
            row_images[p_idx] = out_name
            new_count += 1

        # Phase 3b — face similarity scoring (optional). Score policy:
        #   - newly generated cells (pil_images[p] is set) → always re-score
        #     (the image is fresh, prev score would refer to a different image)
        #   - skipped cells (image already on disk):
        #       · --force-rescore        → score from disk
        #       · prev manifest has score → reuse (most common case)
        #       · no prev score          → score from disk
        row_entry = {"lora": lora_display_name(lora), "images": row_images}
        face_t = 0.0
        if scorer is not None:
            face_t0 = time.time()
            prev = prev_scores_by_lora.get(lora_display_name(lora))
            if prev is not None and len(prev) == len(prompts) and not args.force_rescore:
                scores: list = list(prev)
            else:
                scores = [None] * len(prompts)
            for p_idx in range(len(prompts)):
                if pil_images[p_idx] is not None:
                    # newly generated → fresh PIL gets a fresh score
                    scores[p_idx] = scorer.score_pil(pil_images[p_idx])
                elif scores[p_idx] is None and row_images[p_idx] is not None:
                    # no prev score for this cell, but image on disk → score it
                    scores[p_idx] = scorer.score_path(output_dir / row_images[p_idx])
                # else: keep prev_scores[p_idx] (valid score, or None for missing image)
            row_entry["scores"] = scores
            row_entry["metrics"] = compute_row_metrics(scores)
            face_t = time.time() - face_t0

        print(f"  [timing] denoise={denoise_t:.1f}s decode={decode_t:.1f}s"
              + (f" face={face_t:.1f}s" if scorer is not None else "")
              + f" (saves running async, {new_count} new image(s))")
        print_sage_attention_stats(label="row", since=sage_row_stats0)
        if scorer is not None:
            m = row_entry["metrics"]
            if m["n_faces"]:
                print(f"  [face] median={m['median']:.3f} p20={m['p20']:.3f} "
                      f"mean={m['mean']:.3f} std={m['std']:.3f} "
                      f"faces={m['n_faces']}/{m['n_total']}")
            else:
                print(f"  [face] no faces detected ({m['n_total']} img)")
        manifest["rows"][lora_idx] = row_entry
        write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)

        if plan["is_lokr"]:
            # Remove the LoKr forward hooks; the small w1/w2 tensors get freed
            # once the hook closures fall out of scope.
            if lokr_mgr is not None:
                lokr_mgr.remove()
                lokr_mgr = None
            gc.collect()
        elif not args.preload_loras:
            try:
                pipe.delete_adapters(adapter_name)
            except Exception as _e:
                if base_names:
                    # Don't fall back to unload_lora_weights when sticky base LoRAs
                    # are loaded — it would wipe them too.
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

    # Final write — manifest and HTML are already up to date because each
    # row update did its own atomic write, but rewriting once at the very
    # end guarantees a consistent on-disk state regardless of any in-loop
    # exception path.
    write_grid_artifacts(output_dir, manifest, write_html_flag=not args.no_html)
    print(f"[manifest] {output_dir / 'manifest.json'}")
    print_sage_attention_stats()

    print("\n[done]")
