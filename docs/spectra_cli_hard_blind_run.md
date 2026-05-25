# Package-Backed SPECTRA CLI Run

Date: 2026-05-11

## Command

This run used the molecule adapter to define one similarity graph. The framework
level interface is broader: an agent can provide any pairwise similarity graph
or measured spectral axis, and SPECTRA computes the curve from that graph.

```bash
python -m spectrae.cli audit \
  --mode adapter \
  --domain molecules \
  --train /ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/train.csv \
  --eval /ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/eval_predictions.csv \
  --out /ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit \
  --smiles-col smiles \
  --train-target-col y \
  --eval-target-col y_true \
  --pred-col y_pred \
  --sample-id-col sample_id
```

The installed console-script form is:

```bash
spectra audit --domain molecules ...
```

## Artifacts

The run wrote:

- `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/audit_card.json`
- `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/split_stats.csv`
- `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/performance_by_distance.csv`
- `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/eval_with_distance.csv`
- `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/spectral_curve.svg`
- `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/report.md`

## Result

Aggregate held-out performance:

- n: `1440`
- MAE: `0.155118`
- RMSE: `0.192916`
- R2: `0.154806`
- bias: `-0.114425`

Performance degraded as mean maximum train-set Morgan Tanimoto similarity
decreased:

| Subset | n | Mean max train similarity | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| all eval | 1440 | 0.499135 | 0.155118 | 0.192916 | -0.114425 |
| max similarity <= 0.8 | 1398 | 0.487465 | 0.157806 | 0.195294 | -0.117299 |
| max similarity <= 0.7 | 1283 | 0.464177 | 0.165585 | 0.201941 | -0.124128 |
| max similarity <= 0.6 | 1095 | 0.432405 | 0.176970 | 0.212350 | -0.135466 |
| max similarity <= 0.5 | 831 | 0.394577 | 0.191319 | 0.225832 | -0.145558 |

Overlap validation status: `valid`.

AUSPC: `-0.208112` as negative-RMSE area over normalized overlap novelty.

The command was rerun on the same inputs and produced identical hashes for
`audit_card.json`, `split_stats.csv`, `performance_by_distance.csv`, and
`report.md`.

The same curve was also reproduced through the generic axis interface by using
the adapter-produced `max_train_similarity` column as an externally supplied
spectral axis:

```bash
spectra audit \
  --eval /ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit/eval_with_distance.csv \
  --out /ewsc/yektefai/spectra_assets/spectra_cli_generic_axis_audit \
  --domain generic \
  --scientific-unit molecule \
  --target-col y_true \
  --pred-col y_pred \
  --axis-col max_train_similarity \
  --axis-type similarity \
  --axis-name externally_defined_train_similarity \
  --unit-col sample_id \
  --thresholds 1.0,0.8,0.7,0.6,0.5
```

This second run writes generic artifacts such as `performance_by_axis.csv` and
`eval_with_axis.csv`.

## Interpretation

This is now a package-backed SPECTRA audit rather than an agent-written ad hoc
analysis. The molecule adapter is only one way to define a similarity axis; the
generic SPECTRA interface accepts any agent/domain-defined pairwise similarity
graph or measured spectral axis. The same input files and command produce
standardized artifacts that an agent can cite. This supports the architecture:

> SPECTRA is the deterministic audit engine. `/spectra` is the agent-native
> interface to that engine.
