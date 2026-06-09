# SPECTRA Knowledge MCP

This is an MCP server for SPECTRA protocol memory, saved findings, and
authenticated contributed-finding submissions. Canonical findings remain
reviewed knowledge; submissions land in a pending queue and do not mutate the
canonical store.

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
- `spectra://submissions/schema`

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
- `get_spectra_submission_schema`
- `validate_spectra_submission`
- `submit_spectra_finding`
- `get_spectra_submission_status`
- `list_spectra_submissions`
- `list_spectra_artifacts`
- `get_spectra_artifact`
- `get_spectra_protocol`
- `suggest_next_spectra_move`

There are no tools that run audits, launch agents, call models, download
datasets, prepare sessions, or mutate canonical finding artifacts. The only
write path is the authenticated pending-submission queue.

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

## Contributed Finding Queue

External agents can prepare a SPECTRA finding bundle, validate it, and submit it
for review. Submission creates:

```text
data/submissions/pending/<submission_id>/
  manifest.json
  submission.json
  finding.json
  provenance.json
  downloads.json
  artifact_manifest.json
  audit_card.md
```

Submission does not update `data/store.json`, `data/provenance.json`, or
`data/downloads.json`. A maintainer or CI process must review and promote the
bundle before it becomes canonical.

Unauthenticated clients may inspect the schema and validate a bundle:

```json
{
  "tool": "get_spectra_submission_schema",
  "arguments": {}
}
```

```json
{
  "tool": "validate_spectra_submission",
  "arguments": {
    "submission": {
      "title": "Example finding",
      "submitter": {
        "name": "Agent name",
        "contact": "email-or-handle",
        "agent": "Claude Code / Codex / other MCP client"
      },
      "finding": {},
      "provenance": {},
      "downloads": {
        "records": []
      },
      "artifact_manifest": {
        "records": []
      }
    }
  }
}
```

Submitting requires `SPECTRA_SUBMISSION_TOKEN` on the server and the same token
as the `auth_token` tool argument:

```json
{
  "tool": "submit_spectra_finding",
  "arguments": {
    "auth_token": "<shared submission token>",
    "submission": {
      "...": "full validated submission bundle"
    }
  }
}
```

Large scored CSVs, figures, and result bundles should not be uploaded through
MCP. Put them somewhere retrievable and include `download_url`, `sha256`,
`bytes`, and `rows` in the submission.

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

## Submission Token

Create a deployment-local `.env` file:

```sh
SPECTRA_SUBMISSION_TOKEN=<long-random-token>
```

The tracked systemd unit reads this file through `EnvironmentFile=-.../.env`.
The file is ignored by Git. If the token is absent, `submit_spectra_finding`,
`get_spectra_submission_status`, and `list_spectra_submissions` are disabled,
while schema and validation remain available.

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
