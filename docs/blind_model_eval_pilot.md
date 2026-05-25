# Blind Model-Generalizability Pilot

Date: 2026-05-11

## Goal

This pilot tested the stronger hypothesis:

> Without seeing the BOOM paper, can agents discover BOOM-like generalization
> failure from a trained molecular model and its evaluation artifacts?

The agent-visible bundle did not mention BOOM and did not expose ID/OOD labels.
It included:

- training molecules and target values,
- held-out predictions,
- per-sample errors,
- maximum Morgan Tanimoto similarity from each held-out molecule to the training set.

## Conditions

Vanilla:

- Asked to evaluate whether the model generalizes.
- Received no `/spectra` terminology.

`/spectra`:

- Asked to evaluate generalization using `/spectra`.
- Required to identify novelty axes, property graphs, overlap validation, and
  performance as a function of measured novelty.

## Result

Both agents discovered the key failure mode.

| Condition | Score |
| --- | ---: |
| Vanilla | 14 / 14 |
| `/spectra` | 14 / 14 |

The vanilla agent computed aggregate error, compared train and evaluation target
distributions, found that 1000 evaluation molecules lay outside the training
target range, used `max_train_tanimoto`, and showed that error increased as
nearest-training similarity decreased.

The `/spectra` agent produced a more explicitly reusable audit structure:

- novelty axes,
- property graphs,
- overlap validation,
- performance-vs-novelty bins,
- target-support analysis,
- combined structure/target novelty analysis,
- a clear validity statement for the spectral claim.

## Interpretation

This pilot supports the idea that BOOM-like generalization failures can be
discovered without showing the BOOM paper. It does not support the stronger
claim that a vanilla agent cannot do it.

The task was probably too easy because the agent-visible artifacts included:

- `abs_error`,
- `squared_error`,
- `max_train_tanimoto`,
- `y_true`.

These columns made the failure mode statistically visible even without
`/spectra`.

## Next Harder Blind Test

The next blind benchmark should remove some scaffolding:

1. Give train data and held-out raw predictions only.
2. Hide `abs_error` and `squared_error`.
3. Hide `max_train_tanimoto`.
4. Require agents to construct or request novelty metrics from SMILES.
5. Score whether `/spectra` causes the agent to build those novelty metrics and
   report performance-over-novelty curves.

That harder setup better tests whether `/spectra` changes the agent's behavior,
rather than whether an agent can analyze already-computed novelty columns.
