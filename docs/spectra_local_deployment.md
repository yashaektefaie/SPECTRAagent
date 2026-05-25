# Local SPECTRA Deployment

SPECTRA is installed locally as an editable package in the `pgt` micromamba
environment and exposed through stable wrappers on the user PATH.

## Commands

```bash
spectra --help
spectra ask "<question>" --paper /path/to/paper.pdf --model /path/to/model --dataset /path/to/data
```

The wrapper lives at:

```text
/home/unix/yektefai/bin/spectra
```

It runs:

```text
micromamba run -n pgt spectra
```

and exports `TMPDIR=/ewsc/$USER/tmp`.

## Codex Integration

Fresh Codex sessions can use `/spectra ...` through the global Codex skill:

```text
/home/unix/yektefai/.codex/skills/spectra/SKILL.md
```

The skill tells Codex to treat `/spectra` as a request to run `spectra ask`,
infer obvious parameters, preserve user constraints, avoid `/tmp`, and report
the final SPECTRA artifacts.

SPECTRA is also registered as a local stdio MCP server:

```text
codex mcp get spectra
```

with command:

```text
/home/unix/yektefai/bin/spectra-mcp
```

## Current Scope

This is a local Codex deployment. It is not an always-on hosted HTTP endpoint.
The practical invocation is a Codex user message such as `/spectra assess ...`,
which triggers the skill, or a direct shell command using `spectra ask`.
