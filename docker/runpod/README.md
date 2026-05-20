# GridLoraTester + ai-toolkit — RunPod image

A single image that mirrors the RunPod "ai-toolkit template" (FileBrowser +
exposed SSH + ai-toolkit) and adds the **GridLoraTester (GLT)** dashboard.
ai-toolkit trains LoRAs; GLT grid-tests them — and they **share** the same
dataset and output folders, so a freshly-trained LoRA is testable in GLT with
zero copying.

## Services / ports

| Port | Service | Notes |
|---|---|---|
| 3000 | **GLT** dashboard (SvelteKit) | the grid tester |
| 8675 | **ai-toolkit** UI (Next.js) | training |
| 8080 | **FileBrowser** | web file manager for `/workspace` |
| 22 | **SSH** | only when `$PUBLIC_KEY` is set |

## Storage layout

`/workspace` is the RunPod **persistent network volume** (paid). Only the
irreplaceable lives there. Everything re-fetchable goes to `/opt/scratch`
(ephemeral container disk: free, faster, wiped on pod destroy).

| Path | Persisted? | ai-toolkit key | GLT setting |
|---|---|---|---|
| `/workspace/datasets` | ✅ volume | `DATASETS_FOLDER` | `dataset_root` |
| `/workspace/outputs` | ✅ volume | `TRAINING_FOLDER` | `lora_root` |
| `/workspace/grids` | ✅ volume | — | `tests_root` |
| `/workspace/data/glt.db` | ✅ volume | — | GLT DB |
| `/workspace/aitk_db.db` | ✅ volume | — | ai-toolkit DB |
| `/opt/scratch/{huggingface,insightface,torch,torchinductor,int8_convrot,latents}` | ❌ ephemeral | — | all caches |

> **Pick Container Disk ≥ 80 GB.** The HF cache (FLUX-2 Klein inference +
> Klein-base training + Qwen3 text encoder) can reach 50 GB+. Container Disk
> is included in the hourly price — bigger is free.

## Build

Build context is the **repo root**, not this folder:

```bash
docker build -f docker/runpod/Dockerfile -t <you>/glt-runpod:latest .
```

The image runs on any RunPod GPU from Ampere to Blackwell out of the box —
torch/onnxruntime wheels cover them all. SageAttention is off by default
(`BUILD_SAGE=0`) and isn't needed for the int8_convrot path. If you do want it,
narrow the arch list to your rented GPU for a fast build:

```bash
#   Ampere 8.0/8.6 · Ada 8.9 · Hopper 9.0 · Blackwell 10.0 (B200) / 12.0 (5090)
docker build -f docker/runpod/Dockerfile -t <you>/glt-runpod:latest \
  --build-arg BUILD_SAGE=1 --build-arg TORCH_CUDA_ARCH_LIST=8.9 .
```

Nothing is built or installed at pod boot — all npm/pip builds are baked into
image layers. Boot = ssh keygen + filebrowser init (first boot) + idempotent
settings seed + launch.

## RunPod template settings

- **Container Image:** `<you>/glt-runpod:latest`
- **Container Disk:** ≥ 80 GB
- **Volume Mount Path:** `/workspace`
- **Expose HTTP Ports:** `3000,8675,8080`
- **Expose TCP Ports:** `22`

### Environment variables

| Var | Purpose |
|---|---|
| `HF_TOKEN` | required for the first FLUX-2-Klein download (gated repo); also seeded into ai-toolkit |
| `PUBLIC_KEY` | your SSH public key — enables `ssh root@<pod>` |
| `GLT_PASSWORD` | password for the **GLT dashboard** (:3000). When set, the app requires sign-in via a `/login` page (session cookie); pages redirect there and API/data requests 401 until you authenticate. Unset = no auth. **Strongly recommended on RunPod** — see below. |
| `FILEBROWSER_ADMIN_PASSWORD` | FileBrowser admin password (default `adminadmin12`). **Note the spelling — no underscore between FILE and BROWSER.** |
| `AI_TOOLKIT_AUTH` | ai-toolkit UI auth token (unset = no auth) |
| `RUN_MODE` | `both` (default), `glt`, `toolkit`, `cli`, `shell`, … see `start.sh` |

### Securing the dashboard

RunPod exposes every HTTP port through a **public proxy URL**
(`https://<pod-id>-3000.proxy.runpod.net`) — anyone who learns the URL can
reach it. Set `GLT_PASSWORD` so the GLT dashboard requires sign-in: any page
navigation is redirected to a `/login` screen, and API/data/image requests
(`/api/**`, `/connectors/**`, `/tests/output/**`) return `401` until you
authenticate. A successful login sets a 30-day session cookie; the **Log out**
button is in the sidebar.

`GLT_PASSWORD` only covers the GLT dashboard on `:3000`. The other two
services have their own gates — set them too: `AI_TOOLKIT_AUTH` for ai-toolkit
(:8675) and `FILEBROWSER_ADMIN_PASSWORD` for FileBrowser (:8080).

### FileBrowser password caveat

The admin user/password is written **only on first init** (when
`/workspace/.filebrowser/filebrowser.db` doesn't exist). Setting
`FILEBROWSER_ADMIN_PASSWORD` on a later boot of the same volume does nothing.
To rotate it:

```bash
filebrowser -d /workspace/.filebrowser/filebrowser.db users update admin --password NEW
# or delete the fb DB (loses fb settings only, not your files):
rm /workspace/.filebrowser/filebrowser.db
```

## Note on ai-toolkit latent cache

Because `DATASETS_FOLDER=/workspace/datasets` (persistent, shared with GLT),
ai-toolkit's per-job latent cache may land under `/workspace` and consume paid
storage. It's regenerable — delete the `*_latent_cache` dirs to reclaim space.
