#!/bin/sh
set -e

cd /app/ui

# Preseed workspace paths on first boot (INSERT OR IGNORE — user-edited values
# in the DB are never clobbered).
# NODE_PATH points node at the UI's node_modules; the seed script lives outside
# /app/ui, so without it `require('better-sqlite3')` walks up from /app/docker
# and misses it.
NODE_PATH=/app/ui/node_modules node /app/docker/seed-settings.cjs

# Hand off to the SvelteKit prod server. adapter-node emits ui/build/index.js;
# exec replaces PID 1 so tini handles signals (Ctrl-C, docker stop).
exec node build
