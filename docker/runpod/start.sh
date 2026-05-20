#!/usr/bin/env bash
# --------------------------------------------------------------------------
# GridLoraTester + ai-toolkit — RunPod container entrypoint.
#
# Pure runtime wiring — NOTHING is built or installed here (all npm/pip builds
# are baked into image layers at `docker build` time). Boot cost is just:
# ssh keygen, filebrowser DB init (first boot only), idempotent settings seed,
# then exec the already-compiled servers.
#
# Default ($RUN_MODE unset or =both): launches BOTH UIs side-by-side
#   - GLT SvelteKit dashboard on :3000  (node build/index.js)
#   - ostris/ai-toolkit Next.js UI on :8675  (npm run start)
# Either dying takes the container down (tini → docker/runpod restart policy
# can bring everything back).
#
# Other modes for one-shot CLI runs and single-service launches:
#   RUN_MODE=glt       → GLT dashboard only
#   RUN_MODE=toolkit   → ai-toolkit only
#   RUN_MODE=cli       → python -m glt "$@"   (e.g. --grid / --rescore / …)
#   RUN_MODE=toolkit-cmd <cmd...>  → arbitrary command in the toolkit venv
#   RUN_MODE=shell     → interactive bash (GLT venv active)
# --------------------------------------------------------------------------
set -eu

GLT_DIR=/app
GLT_UI_DIR="${GLT_DIR}/ui"
GLT_VENV=/opt/venv
TOOLKIT_DIR=/opt/ai-toolkit
TOOLKIT_VENV="${TOOLKIT_DIR}/venv"

# Helper: activate a venv (modifies PATH + VIRTUAL_ENV).
activate() {
    # shellcheck disable=SC1090,SC1091
    source "$1/bin/activate"
}

# ---------- SSH (style runpod-workers/comfyui-base) ----------------------
# Active only when $PUBLIC_KEY is set. Adds the key to authorized_keys,
# generates host keys if missing, and starts sshd in background.
setup_ssh() {
    if [ -z "${PUBLIC_KEY:-}" ]; then
        echo "[ssh] skipped (set \$PUBLIC_KEY to enable)"
        return 0
    fi
    echo "[ssh] setting up authorized_keys + host keys"
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys

    for t in rsa ecdsa ed25519; do
        if [ ! -f "/etc/ssh/ssh_host_${t}_key" ]; then
            ssh-keygen -t "$t" -f "/etc/ssh/ssh_host_${t}_key" -q -N ''
        fi
    done

    mkdir -p /run/sshd
    /usr/sbin/sshd
    echo "[ssh] sshd listening on :22"
}

# ---------- FileBrowser (web file manager at :8080) ----------------------
# Persists its DB on /workspace so users/settings survive restarts.
# Admin password defaults to "adminadmin12" — override with the env var
# FILEBROWSER_ADMIN_PASSWORD (NOTE: no underscore between FILE and BROWSER).
#
# Caveat: the admin user + password are written ONLY on first init (when the
# DB doesn't exist yet). Changing FILEBROWSER_ADMIN_PASSWORD on a later boot
# of the SAME volume does nothing — the existing DB already has the user. To
# rotate it, either `rm /workspace/.filebrowser/filebrowser.db` (loses fb
# settings only, not your files) or run `filebrowser -d <db> users update
# admin --password NEW`.
start_filebrowser() {
    if [ "${START_FILEBROWSER:-1}" != "1" ]; then
        echo "[filebrowser] skipped (\$START_FILEBROWSER != 1)"
        return 0
    fi
    local FB_DIR=/workspace/.filebrowser
    local DB_FILE="$FB_DIR/filebrowser.db"
    local LOG_FILE="$FB_DIR/filebrowser.log"
    local ADMIN_PASS="${FILEBROWSER_ADMIN_PASSWORD:-adminadmin12}"
    mkdir -p "$FB_DIR"

    if [ ! -f "$DB_FILE" ]; then
        echo "[filebrowser] initializing $DB_FILE"
        filebrowser -d "$DB_FILE" config init
        filebrowser -d "$DB_FILE" config set --address 0.0.0.0 --port 8080 \
            --root /workspace --auth.method=json
        filebrowser -d "$DB_FILE" users add admin "$ADMIN_PASS" --perm.admin
    else
        echo "[filebrowser] reusing existing $DB_FILE"
    fi

    nohup filebrowser -d "$DB_FILE" >>"$LOG_FILE" 2>&1 &
    echo "[filebrowser] listening on :8080  (user=admin, log=$LOG_FILE)"
}

# ---------- GLT settings seed + DB persistence ---------------------------
# Idempotent (INSERT OR IGNORE) — user-edited Settings values are preserved.
# Writes dataset_root / lora_root / tests_root / python_bin so the very first
# page render sees usable paths pointing at the shared /workspace folders.
prepare_glt() {
    mkdir -p /workspace/datasets /workspace/outputs /workspace/grids /workspace/data
    # NODE_PATH points node at the UI's node_modules; the seed lives outside
    # /app/ui, so without it require('better-sqlite3') misses it.
    NODE_PATH="${GLT_UI_DIR}/node_modules" node "${GLT_DIR}/docker/seed-settings.cjs"
}

# ---------- ai-toolkit DB persistence + first-run seed -------------------
# DATASETS_FOLDER + TRAINING_FOLDER point at the SAME /workspace folders GLT
# uses, so trained LoRAs are immediately grid-testable in GLT.
prepare_toolkit_db() {
    local WORKSPACE_DB=/workspace/aitk_db.db
    local APP_DB=/opt/ai-toolkit/aitk_db.db

    if [ -f "$WORKSPACE_DB" ]; then
        rm -f "$APP_DB"
        ln -s "$WORKSPACE_DB" "$APP_DB"
        echo "[toolkit] reusing existing $WORKSPACE_DB"
    else
        # First run on this volume — move the build-time DB to the volume
        # (preserves the Prisma schema baked in at build time).
        if [ -f "$APP_DB" ] && [ ! -L "$APP_DB" ]; then
            mv "$APP_DB" "$WORKSPACE_DB"
        else
            touch "$WORKSPACE_DB"
        fi
        ln -s "$WORKSPACE_DB" "$APP_DB"
        echo "[toolkit] persisted DB at $WORKSPACE_DB"
    fi

    # Latent cache stays ephemeral + fast on the container disk.
    mkdir -p /workspace/datasets /workspace/outputs /opt/scratch/latents
    python3 - "$WORKSPACE_DB" <<'PYSEED'
import os, sqlite3, sys
db = sys.argv[1]
c = sqlite3.connect(db)
c.execute("""CREATE TABLE IF NOT EXISTS Settings (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL
)""")
defaults = {
    "DATASETS_FOLDER": "/workspace/datasets",
    "TRAINING_FOLDER": "/workspace/outputs",
}
hf = os.environ.get("HF_TOKEN")
if hf:
    defaults["HF_TOKEN"] = hf
for k, v in defaults.items():
    row = c.execute("SELECT value FROM Settings WHERE key=?", (k,)).fetchone()
    if row is None:
        c.execute("INSERT INTO Settings (key, value) VALUES (?, ?)", (k, v))
        print(f"[toolkit-seed]   {k} = {v}")
    else:
        print(f"[toolkit-keep]   {k} = {row[0]!r} (already set, not overwriting)")
c.commit()
PYSEED
}

# ---------- run launchers ------------------------------------------------
launch_glt_fg() {
    setup_ssh
    start_filebrowser
    prepare_glt
    cd "$GLT_UI_DIR"
    echo "[glt] node $(node --version)  on http://0.0.0.0:${PORT:-3000}/  (GLT_PASSWORD=${GLT_PASSWORD:+set})"
    exec node build
}

launch_toolkit_fg() {
    setup_ssh
    start_filebrowser
    prepare_toolkit_db
    activate "$TOOLKIT_VENV"
    cd "$TOOLKIT_DIR/ui"
    echo "[toolkit] python: $(python --version 2>&1)  torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "[toolkit] AI_TOOLKIT_AUTH=${AI_TOOLKIT_AUTH:+set} (unset = no auth)"
    exec npm run start
}

launch_both() {
    setup_ssh
    start_filebrowser
    prepare_glt
    prepare_toolkit_db

    # GLT dashboard (SvelteKit prod server) in background. `exec` makes
    # $GLT_PID the actual node PID so SIGTERM propagates directly. When
    # GLT_PASSWORD is set the app is gated behind /login (hooks.server.ts).
    (
        cd "$GLT_UI_DIR"
        exec node build
    ) &
    GLT_PID=$!

    # ai-toolkit (Next.js + TS worker; `npm run start` keeps both alive).
    (
        activate "$TOOLKIT_VENV"
        cd "$TOOLKIT_DIR/ui"
        exec npm run start
    ) &
    TOOLKIT_PID=$!

    echo "[start] GLT dashboard: pid=$GLT_PID  on http://0.0.0.0:${PORT:-3000}/  (GLT_PASSWORD=${GLT_PASSWORD:+set})"
    echo "[start] toolkit UI:    pid=$TOOLKIT_PID  on http://0.0.0.0:8675/  (AI_TOOLKIT_AUTH=${AI_TOOLKIT_AUTH:+set})"

    cleanup() {
        echo "[start] received signal — stopping both services"
        kill -TERM "$GLT_PID" "$TOOLKIT_PID" 2>/dev/null || true
        wait 2>/dev/null || true
        exit 0
    }
    trap cleanup TERM INT

    # If either service dies, take the other down too so the restart policy
    # can bring the whole pod back consistently.
    EXIT_CODE=0
    wait -n "$GLT_PID" "$TOOLKIT_PID" || EXIT_CODE=$?
    echo "[start] one service exited (code=$EXIT_CODE) — stopping the other"
    kill -TERM "$GLT_PID" "$TOOLKIT_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    exit $EXIT_CODE
}

# ---------- main dispatch ------------------------------------------------
echo "[start] GridLoraTester + ai-toolkit container"
echo "[start] HF_HOME=${HF_HOME:-/root/.cache/huggingface}  (caches → /opt/scratch, NOT /workspace)"

mkdir -p /workspace
cd /workspace

case "${RUN_MODE:-both}" in
    both|all|"")
        echo "[start] mode=both — GLT on :3000 + ai-toolkit on :8675"
        launch_both
        ;;
    glt|ui|UI)
        echo "[start] mode=glt — GLT dashboard on :3000 (ai-toolkit NOT started)"
        launch_glt_fg
        ;;
    toolkit|ai-toolkit)
        echo "[start] mode=toolkit — ai-toolkit on :8675 (GLT NOT started)"
        launch_toolkit_fg
        ;;
    cli|generate)
        activate "$GLT_VENV"
        cd "$GLT_DIR"
        echo "[start] mode=cli — python -m glt $*"
        exec python -u -m glt "$@"
        ;;
    toolkit-cmd|toolkit-cli)
        activate "$TOOLKIT_VENV"
        cd "$TOOLKIT_DIR"
        echo "[start] mode=toolkit-cmd — $*"
        exec "$@"
        ;;
    shell|bash)
        setup_ssh
        start_filebrowser
        activate "$GLT_VENV"
        echo "[start] mode=shell — interactive bash (GLT venv)"
        exec bash -l
        ;;
    *)
        # Unknown mode → treat as literal command (GLT venv active).
        activate "$GLT_VENV"
        echo "[start] mode=$RUN_MODE (literal exec) $*"
        exec "$RUN_MODE" "$@"
        ;;
esac
