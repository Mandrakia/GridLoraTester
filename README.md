# GridLoraTester (`glt`)

Generate side-by-side comparison grids for **LoRA** and **LoKr** adapters on
**FLUX.2** (and compatible Flux-family) diffusion models, then audit identity
consistency with a face-recognition–based scorer.

> One row per adapter, one column per prompt. The output is a self-contained
> HTML dashboard (`index.html`) you can open straight from disk.

## Highlights

- **LoRA + LoKr** adapter support out of the box (LoKr applied through forward
  hooks, no full-tensor materialization → fits in 24 GB VRAM).
- **FP8 quantization** via `torchao` (weight-only or dynamic) — runs FLUX.2-9B
  on a single 24 GB GPU.
- **ComfyUI parity mode** (`--comfyui-noise`) for bit-exact noise reproduction
  vs a ComfyUI reference run, useful when validating numerical changes.
- **Idempotent generation**: a row whose images are already on disk is
  skipped; partial rows resume.
- **Face-similarity scoring** (InsightFace + ArcFace) with per-row median/p20
  metrics and optional centroid centering.
- **Live-updating HTML dashboard** — refresh the browser to see rows appear
  one by one.

## Layout

```
GridLoraTester/
├── requirements.txt
├── config.example.json
└── glt/
    ├── __main__.py     # `python -m glt …`
    ├── cli.py          # argparse + mode dispatch
    ├── modes/          # grid generation, centroid computation, rescore
    ├── pipeline/       # model build, FP8 quantize, offload, sage attention
    ├── adapters/       # LoRA / LoKr loaders + discovery
    ├── generation/     # prompt encoding, latent denoise, VAE decode
    ├── analysis/       # face scoring + per-row metrics
    ├── output/         # manifest, HTML dashboard, image saving
    └── utils/          # config loading, CUDA lib preload
```

No `pip install` step — clone the repo and run `python -m glt`.

## Quick start

```bash
git clone https://github.com/mandrakia/GridLoraTester.git
cd GridLoraTester

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m glt path/to/loras \
    -prompts prompts.txt \
    -w 1024 -h 1024 \
    -o output_grid \
    -s 42
```

Open `output_grid/index.html` once the first row finishes.

## Modes

| Command | Purpose |
|---|---|
| `python -m glt <loras> -prompts …` | Generate a LoRA × prompt grid |
| `python -m glt --compute-centroid <dir>` | Average face embeddings into a centroid (write to JSON / paste into `config.json`) |
| `python -m glt --rescore <out_dir>` | Re-score an existing grid with the current centroid, without regenerating images |

See `python -m glt --help` for the full flag list.

## License

MIT — see `LICENSE`.
