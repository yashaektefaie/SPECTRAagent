# Reproducing the SPECTRA Agent

This repository contains the package-backed SPECTRA agent workflow used by Codex.

## Environment

Use the conda environment file when RDKit-backed molecular split construction is needed:

```bash
micromamba create -f environment.yml
micromamba activate spectra-agent
```

If you already use the shared `pgt` environment, install this checkout editable:

```bash
micromamba run -n pgt pip install -e .
```

## Codex Skill

Install or symlink the SPECTRA skill into your Codex skill directory:

```bash
mkdir -p "$CODEX_HOME/skills"
ln -s "$(pwd)/codex/skills/spectra" "$CODEX_HOME/skills/spectra"
```

The skill instructs Codex to call `spectra ask` for `/spectra` requests.

## Smoke Tests

Routing only:

```bash
spectra ask "construct SPECTRA splits for this molecular dataset" \
  --dataset tutorials/example_mol.ipynb \
  --out /ewsc/$USER/spectra_runs/smoke \
  --dry-run
```

Unit tests:

```bash
python -m unittest tests.test_agent_orchestrator tests.test_cli_ask_routing
```

## Expected Routing

`spectra ask` automatically chooses the mode:

- Split construction requests route to focused SPECTRA split mode and do not launch the general multi-agent audit loop.
- Model, paper, checkpoint, or foundation-model generalization requests route to the autonomous audit loop.

Focused molecular split mode writes:

- `similarity_definition_selection.json`
- `similarity_computation_selection.json`
- `pairwise_similarity/`
- `property_similarity_graph/`
- `split_assignments/`
- `diagnostics/`
- `baseline_validation/`
- `split_construction_report.md`
