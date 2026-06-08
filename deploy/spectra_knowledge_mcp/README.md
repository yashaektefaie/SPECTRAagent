# SPECTRA Knowledge MCP

This is a read-only MCP server for SPECTRA protocol memory and saved findings.
It is deliberately separate from the SPECTRA audit/executor MCP.

Hosted endpoint:

```text
https://spectra.yashaektefaie.com/mcp
```

## What It Exposes

Resources:

- `spectra://protocol/current`
- `spectra://findings/index`
- `spectra://models/index`
- `spectra://runs/index`
- `spectra://artifacts/index`

Tools:

- `list_spectra_models`
- `list_spectra_findings`
- `search_spectra_findings`
- `get_spectra_finding`
- `list_spectra_runs`
- `get_spectra_run`
- `list_spectra_artifacts`
- `get_spectra_artifact`
- `get_spectra_protocol`
- `suggest_next_spectra_move`

There are no tools that run audits, launch agents, call models, download
datasets, mutate artifacts, or prepare sessions.

## Artifact Policy

`data/store.json` is tracked because it contains compact structured findings,
claim boundaries, and artifact ids. The actual curated run artifacts under
`data/artifacts/` are not tracked in Git; they live on the hosted MCP instance
or external storage. See `ARTIFACTS.md`.

## Static Website

The human-readable site at `/` is generated from `data/store.json`:

```sh
python build_site.py
```

The generated files live under `site/`. Caddy serves the website for normal
browser paths and forwards `/mcp` to the private FastMCP process on
`127.0.0.1:8000`.

## Run

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python server.py --transport streamable-http --host 127.0.0.1 --port 8000
```

For local MCP clients over stdio:

```sh
python server.py --transport stdio
```
