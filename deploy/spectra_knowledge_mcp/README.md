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
- `spectra://downloads/index`
- `spectra://provenance/index`
- `spectra://provenance/schema`

Tools:

- `list_spectra_models`
- `list_spectra_findings`
- `search_spectra_findings`
- `get_spectra_finding`
- `list_spectra_runs`
- `get_spectra_run`
- `list_spectra_provenance`
- `get_spectra_provenance`
- `list_spectra_sources`
- `validate_spectra_provenance`
- `get_spectra_provenance_schema`
- `list_spectra_downloads`
- `get_spectra_download`
- `list_spectra_artifacts`
- `get_spectra_artifact`
- `get_spectra_protocol`
- `suggest_next_spectra_move`

There are no tools that run audits, launch agents, call models, download
datasets, mutate artifacts, or prepare sessions.

`get_spectra_artifact` is a preview path and intentionally truncates large
files. Agents that need raw per-target tables should call
`list_spectra_downloads` or `get_spectra_download`, then use normal HTTPS
download tooling against the returned `download_url`.

## Provenance Contract

Every stored finding must have a normalized provenance record in
`data/provenance.json`. New findings are incomplete until they identify:

- model code/source and execution mode;
- model weights/checkpoints, official precomputed score source, or an explicit
  not-applicable reason;
- dataset names, URLs/repos, split/shard/filter details, units, row counts, and
  local or artifact paths;
- metadata resources used to define axes or controls;
- download/access commands or methods;
- Python/environment and cache roots;
- artifact ids supporting the provenance; and
- explicit known gaps.

Agents should call `get_spectra_provenance` or `list_spectra_sources` before
using a stored finding, and `validate_spectra_provenance` before publishing a
new one.

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
`127.0.0.1:8000`. Caddy also serves curated artifact files from
`data/public_downloads/` at `/downloads/...`.

## Public Downloads

Build the download manifest and materialized public file tree from curated
runtime artifacts:

```sh
python build_download_manifest.py --materialize --clean
```

This writes:

- `data/downloads.json`: tracked manifest with URLs, sizes, row counts, and
  SHA-256 checksums.
- `data/public_downloads/`: ignored deployment tree served by Caddy.

The builder excludes internal session contracts, controller logs/prompts, and
command logs. It publishes result CSVs, reports, figures, compact JSON outputs,
and provenance artifacts.

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
