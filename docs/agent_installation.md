# Installing SPECTRA For Agent Use

SPECTRA can be used as a local MCP server so Claude, Cursor, Codex, and other
MCP-capable agents can discover SPECTRA tools and respond to `/spectra ...`
requests.

## Install

From PyPI, once released:

```bash
pipx install "spectrae[mcp]"
```

From GitHub:

```bash
pipx install "spectrae[mcp] @ git+https://github.com/yashaektefaie/SPECTRAagent.git"
```

For local development from a checkout:

```bash
pip install -e ".[mcp]"
```

Then check the install:

```bash
spectra-doctor
```

Set a scratch root when audits may create large files:

```bash
export SPECTRA_SCRATCH_ROOT="$HOME/spectra_runs"
```

On shared clusters, use a large scratch filesystem instead.

## Register With Claude Code

Use the local stdio MCP server:

```bash
claude mcp add spectra -- spectra-mcp serve --transport stdio
```

If your client expects a JSON configuration, use:

```json
{
  "mcpServers": {
    "spectra": {
      "command": "spectra-mcp",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

## Register With Codex

Install the packaged Codex skill:

```bash
spectra install-codex-skill
```

Or copy the repo skill at:

```text
.agents/skills/spectra/SKILL.md
```

The skill tells Codex to treat `/spectra ...` as a SPECTRA run and to call the
local CLI/MCP tools instead of writing an ad hoc explanation.

## Register With Cursor Or Other MCP Clients

Use the same stdio server command:

```json
{
  "command": "spectra-mcp",
  "args": ["serve", "--transport", "stdio"]
}
```

## Optional HTTP Server

For a hosted endpoint:

```bash
spectra-mcp serve --transport streamable-http --host 0.0.0.0 --port 8000
```

The Dockerfile runs the same command.

## Common Commands

Human-friendly autonomous audit:

```bash
spectra ask "Should I use this model for this dataset?" \
  --paper ./paper.pdf \
  --model ./checkpoint_or_repo \
  --dataset ./dataset
```

Core SPECTRA curve mode:

```bash
spectra audit \
  --eval eval_predictions.csv \
  --out spectra_curve \
  --target-col y_true \
  --pred-col y_pred \
  --axis-col max_train_similarity \
  --axis-type similarity
```

Search the registries:

```bash
spectra similarity-definitions suggest \
  --dataset-description "protein mutational fitness sequences" \
  --task-description "predict variant fitness"

spectra similarity-computation suggest \
  --dataset-description "100k protein sequences" \
  --similarity-definition "sequence identity"
```

## What `/spectra` Means

SPECTRA supports three user-facing modes:

- General SPECTRA mode: compute a spectral performance curve from predictions
  and a similarity/distance axis.
- Applicability mode: answer whether a model should be used for a specific
  dataset or task.
- Discovery mode: investigate where and why a model generalizes or fails.

The MCP server exposes the registries, prompts, planning tools, and audit tools
needed for all three modes.
