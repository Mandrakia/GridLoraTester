# `glt-ui` — GridLoraTester dashboard

SvelteKit + Tailwind + SQLite dashboard for browsing datasets, LoRA families,
and grid-test results, and for driving new test runs.

## Stack

| | |
|---|---|
| Framework | SvelteKit (adapter-node) |
| Language | TypeScript |
| Styling | TailwindCSS |
| Storage | SQLite via `better-sqlite3` (file: `data/glt.db`) |

## Layout

```
ui/
├── src/
│   ├── lib/
│   │   ├── components/  Sidebar etc.
│   │   └── server/      DB client + settings + datasets listing (Node-only)
│   └── routes/
│       ├── +layout.svelte    sidebar + main split
│       ├── datasets/         list of dataset subfolders
│       └── settings/         dataset / tests / lora roots
└── data/                runtime DB (gitignored)
```

## Quick start

```bash
cd ui
npm install
npm run dev
```

Open <http://localhost:5273>. First-run :

1. Open **Settings**, enter the three root paths (dataset / tests / lora).
2. Reload **Datasets** to see the subfolders of your `dataset_root`.

## Settings storage

Settings live in `data/glt.db` (SQLite). Override the DB path with the
`GLT_DB_PATH` env var.

## Production

```bash
npm run build
node build  # serves on :3000 by default
```

Set `GLT_PASSWORD` to gate the dashboard: `hooks.server.ts` then redirects
page navigations to a `/login` screen and 401s API/data requests until you
sign in (session cookie). Unset = open instance. Set it whenever the port is
reachable beyond `localhost`.
