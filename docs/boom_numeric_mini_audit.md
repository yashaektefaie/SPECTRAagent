# BOOM Numeric Mini-Audit

Date: 2026-05-07

## Purpose

This is the first measured `/spectra`-style computational pilot on BOOM. It is
not a full BOOM reproduction and not a full SPECTRA resplitting run. It tests a
smaller question:

> Within the BOOM 10k density OOD split, does a lightweight model perform worse
> as chemical train-test overlap decreases?

## Inputs

Assets:

- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/10k_dft_density_data.csv`
- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/10k_dft_hof_data.csv`
- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/10k_dft_data_with_ood_splits.csv`

Runner:

- `spectrae/benchmark_runners/boom_numeric_mini_audit.py`

Model:

- `RandomForestRegressor`
- Morgan radius-2 fingerprints, 1024 bits
- BOOM 10k density train split

Novelty axis:

- Maximum Morgan Tanimoto similarity from each test molecule to the BOOM
  training set.

## Command

```sh
PYTHONPATH=/ewsc/yektefai/spectra_assets/boom_pilot/repos/boom \
/ewsc/yektefai/envs/envs/boltz/bin/python \
  spectrae/benchmark_runners/boom_numeric_mini_audit.py \
  --output-dir /ewsc/yektefai/spectra_assets/boom_numeric_pilot \
  --split-file /ewsc/yektefai/spectra_assets/boom_numeric_pilot/10k_dft_data_with_ood_splits.csv \
  --n-estimators 120
```

## Artifacts

- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/audit_card.json`
- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/split_stats.csv`
- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/performance_by_overlap.csv`
- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/spectral_curve.svg`
- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/report.md`

## Results

| Subset | n | Mean Max Tanimoto | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| ID test | 440 | 0.5595 | 0.0443 | 0.0338 |
| OOD, all | 1000 | 0.4726 | 0.2296 | 0.2085 |
| OOD, max Tanimoto <= 0.8 | 978 | 0.4632 | 0.2317 | 0.2110 |
| OOD, max Tanimoto <= 0.7 | 931 | 0.4489 | 0.2354 | 0.2149 |
| OOD, max Tanimoto <= 0.6 | 823 | 0.4230 | 0.2434 | 0.2231 |
| OOD, max Tanimoto <= 0.5 | 653 | 0.3900 | 0.2534 | 0.2330 |

AUSPC:

- Metric: negative RMSE area
- Novelty axis: `1 - normalized_mean_max_tanimoto`
- Value: `-0.24096112142647133`

## Interpretation

The measured result is directionally consistent with the `/spectra` hypothesis:

- BOOM ID error is much lower than BOOM OOD error.
- Within BOOM OOD molecules, lower train-test chemical overlap corresponds to
  higher RMSE for this lightweight baseline.

This converts the earlier audit-quality pilot into a first numeric
performance-over-overlap curve.

## Limitations

- This is a post-hoc OOD test-subset curve, not generated SPECTRA train/test
  splits.
- The training set is fixed.
- Only one property task and one lightweight baseline were evaluated.
- The next rigorous step is to generate overlap-controlled train/test splits,
  retrain or consistently reuse model protocols, and compare multiple models.
