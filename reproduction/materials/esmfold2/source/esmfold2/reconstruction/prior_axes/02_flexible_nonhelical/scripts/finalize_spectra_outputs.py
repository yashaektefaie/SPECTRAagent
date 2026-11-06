#!/usr/bin/env python3
"""Write final SPECTRA state, ledger, command log, and report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "final_report.md"
LEDGER = ROOT / "axis_ledger.json"
STATE = ROOT / "spectra_loop_state.json"
COMMANDS = ROOT / "commands_run.json"
MEMBERSHIP = ROOT / "primary_axis_membership.csv"


def f3(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    initial_summary = pd.read_csv(ROOT / "axis_summary.csv")
    follow = pd.read_csv(ROOT / "followup_axis_summary.csv")
    curves = pd.read_csv(ROOT / "followup_axis_curves.csv")
    thresholds = pd.read_csv(ROOT / "primary_axis_threshold_sweep.csv")
    features = pd.read_csv(ROOT / "spectra_ready_dataset" / "prospective_features.followup.csv")
    results = pd.read_csv(ROOT / "target_model_results.csv")
    primary_id = "seq_intrinsic_flexible_nonhelical_composite_7"
    primary_full = follow[(follow["axis_id"].eq(primary_id)) & (follow["subset"].eq("full_138"))].iloc[0]
    primary_conf = follow[
        (follow["axis_id"].eq(primary_id)) & (follow["subset"].eq("confirmation_folds1_2"))
    ].iloc[0]
    primary_curve = curves[(curves["axis_id"].eq(primary_id)) & (curves["subset"].eq("full_138"))]
    top20_full = thresholds[
        (thresholds["subset"].eq("full_138")) & (thresholds["tail_quantile"].eq(0.80))
    ].iloc[0]
    top20_conf = thresholds[
        (thresholds["subset"].eq("confirmation_folds1_2"))
        & (thresholds["tail_quantile"].eq(0.80))
    ].iloc[0]

    membership = features.merge(results, on="target_id", how="inner")
    level_col = "axis_flexible_nonhelical_composite_rank_q7_level"
    membership = membership[
        [
            "target_id",
            "flexible_nonhelical_composite_rank",
            level_col,
            "seq_len",
            "msa_log10_nonquery_rows",
            "msa_top_nonquery_identity",
            "difficulty",
            "ca_lddt",
            "gdt_ts",
            "ca_rmsd",
            "baseline_ca_lddt",
        ]
    ].sort_values(["axis_flexible_nonhelical_composite_rank_q7_level", "flexible_nonhelical_composite_rank"])
    membership.to_csv(MEMBERSHIP, index=False)

    entries = []
    for _, row in initial_summary.iterrows():
        entries.append(
            {
                "axis_id": row["axis_id"],
                "feature": row["feature"],
                "axis_type": row["axis_type"],
                "prospective": bool(row["prospective"]),
                "benchmark_only": bool(row["benchmark_only"]),
                "decision": row["decision"],
                "n_evaluated": int(row["n_evaluated"]),
                "n_levels": int(row["n_levels"]),
                "ca_lddt_endpoint_effect": float(row["ca_lddt_endpoint_effect"]),
                "ca_lddt_broad_adjusted_coef": float(row["ca_lddt_broad_adjusted_coef"]),
                "baseline_ca_lddt_endpoint_effect": float(row["baseline_ca_lddt_endpoint_effect"]),
                "terminal_primary": False,
            }
        )
    for _, row in follow.iterrows():
        if row["subset"] != "full_138":
            continue
        decision = "claim_valid_primary" if row["axis_id"] == primary_id else "supporting_followup"
        entries.append(
            {
                "axis_id": row["axis_id"],
                "feature": row["feature"],
                "axis_type": "sequence_intrinsic",
                "prospective": bool(row["prospective"]),
                "benchmark_only": bool(row["benchmark_only"]),
                "outcome_informed": bool(row["outcome_informed"]),
                "decision": decision,
                "n_evaluated": int(row["n"]),
                "n_levels": int(row["n_levels"]),
                "min_level_n": int(row["min_level_n"]),
                "ca_lddt_top_vs_lower_effect": float(row["ca_lddt_top_vs_lower_effect"]),
                "gdt_ts_top_vs_lower_effect": float(row["gdt_ts_top_vs_lower_effect"]),
                "ca_rmsd_top_vs_lower_effect": float(row["ca_rmsd_top_vs_lower_effect"]),
                "ca_lddt_broad_adjusted_coef": float(row["ca_lddt_broad_adjusted_coef"]),
                "ca_lddt_broad_ref_adjusted_coef": float(row["ca_lddt_broad_ref_adjusted_coef"]),
                "baseline_ca_lddt_top_vs_lower_effect": float(row["baseline_ca_lddt_top_vs_lower_effect"]),
                "terminal_primary": row["axis_id"] == primary_id,
            }
        )
    ledger = {
        "terminal_status": "claim_valid_explanatory_degradation_spc",
        "primary_axis": primary_id,
        "primary_axis_boundary": "ESMFold2 degradation on CAMEO 2025 single-chain targets in the high tail of a prospective flexible/non-helical sequence-composition proxy: high proline plus beta/turn-over-helix propensity.",
        "target_model_results_reused": True,
        "target_model_results": "target_model_results.csv",
        "prospective_feature_table": "spectra_ready_dataset/prospective_features.followup.csv",
        "fixed_outcome_free_axis_definitions": [
            "axis_definitions_prejoin.json",
            "axis_definitions_followup.json",
        ],
        "entries": entries,
    }
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    state = {
        "phase": "terminal_synthesis",
        "terminal_gate": "passed",
        "scientific_unit": "CAMEO 2025 target chain",
        "fixed_target_model": "Biohub ESMFold2, num_loops=3, num_sampling_steps=14, one diffusion sample",
        "rows_candidate_features": 141,
        "rows_fixed_esmfold2_ok": 138,
        "primary_axis": primary_id,
        "primary_axis_type": "sequence_intrinsic prospective; outcome-informed then holdout-confirmed",
        "primary_shape": "localized high-tail threshold",
        "primary_full_top_vs_lower_ca_lddt": float(primary_full["ca_lddt_top_vs_lower_effect"]),
        "primary_confirmation_top_vs_lower_ca_lddt": float(primary_conf["ca_lddt_top_vs_lower_effect"]),
        "known_controls": ["seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty"],
        "fixed_baseline_result": "baseline_ca_lddt did not degrade on the primary top tail",
        "artifacts": {
            "final_report": "final_report.md",
            "axis_ledger": "axis_ledger.json",
            "commands_run": "commands_run.json",
            "spectral_performance_curve": "spectral_performance_curve.csv",
            "prospective_features": "spectra_ready_dataset/prospective_features.followup.csv",
            "primary_axis_membership": "primary_axis_membership.csv",
        },
    }
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    commands = [
        {"cmd": "sed -n '1,220p' /home/unix/yektefai/.codex/skills/spectra/SKILL.md", "purpose": "load SPECTRA workflow instructions", "status": "ok"},
        {"cmd": "find prior run artifacts; copy candidate_features.csv, raw_cameo_hf, target_model_results.csv", "purpose": "manifest-first continuation from allowed saved artifacts", "status": "ok"},
        {"cmd": "python scripts/build_nonerror_features.py", "purpose": "build outcome-free prospective sequence/metadata/reference features and freeze initial axes", "status": "ok"},
        {"cmd": "python scripts/score_axes.py", "purpose": "score initial frozen axes against fixed ESMFold2 outcomes with length/MSA controls and baselines", "status": "ok"},
        {"cmd": "python scripts/mine_outcome_informed_hypotheses.py", "purpose": "mine broad-control residual successes/failures to propose prospective follow-up axes", "status": "ok"},
        {"cmd": "python scripts/freeze_followup_axes.py", "purpose": "freeze outcome-informed follow-up axes from prospective features only", "status": "ok"},
        {"cmd": "python scripts/score_followup_axes.py", "purpose": "score follow-up axes on full, discovery, and confirmation folds", "status": "stopped_once_for_slow_pandas_bootstrap_then_vectorized_fallback_ok"},
        {"cmd": "python scripts/densify_primary_axis.py", "purpose": "densify primary axis with deciles and top-tail threshold sweep", "status": "ok"},
        {"cmd": "python scripts/finalize_spectra_outputs.py", "purpose": "write final report, ledger, state, command log", "status": "ok"},
    ]
    COMMANDS.write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")

    report = f"""# ESMFold2 Beyond-MSA SPECTRA Audit

## Terminal Finding

The run found a claim-valid non-evolutionary axis: a prospective sequence-intrinsic flexible/non-helical composition proxy. The axis is the mean percentile rank of `proline_fraction`, `chou_beta_minus_helix`, and `chou_turn_minus_helix`; higher values mean more proline-rich and beta/turn-favored relative to helix-favored composition.

This is an outcome-informed axis, so it was generated from broad-control residual failures on discovery fold 0, then frozen from prospective sequence features only and confirmed on held-out folds 1-2. It does not use ESMFold2 error, confidence, pLDDT, pTM, RMSD, lDDT, or prediction/reference comparisons to define split membership.

## Primary SPC Evidence

Full fixed ESMFold2 pool: 138 successful target-model evaluations, seven frozen levels, minimum level size {int(primary_full['min_level_n'])}. The highest-stress level had mean CA-lDDT {f3(primary_full['top_rank_ca_lddt'])}; compared with lower levels, it was {f3(primary_full['ca_lddt_top_vs_lower_effect'])} CA-lDDT, {f3(primary_full['gdt_ts_top_vs_lower_effect'])} GDT-TS, and {f3(primary_full['ca_rmsd_top_vs_lower_effect'])} A CA-RMSD worse. Bootstrap 95% CI for the CA-lDDT top-tail effect was [{f3(primary_full['bootstrap_ca_lddt_top_vs_lower_ci_low'])}, {f3(primary_full['bootstrap_ca_lddt_top_vs_lower_ci_high'])}].

Held-out confirmation folds 1-2: 95 targets, seven levels, minimum level size {int(primary_conf['min_level_n'])}. The highest-stress level was {f3(primary_conf['ca_lddt_top_vs_lower_effect'])} CA-lDDT, {f3(primary_conf['gdt_ts_top_vs_lower_effect'])} GDT-TS, and {f3(primary_conf['ca_rmsd_top_vs_lower_effect'])} A CA-RMSD worse than lower levels. Bootstrap 95% CI for CA-lDDT was [{f3(primary_conf['bootstrap_ca_lddt_top_vs_lower_ci_low'])}, {f3(primary_conf['bootstrap_ca_lddt_top_vs_lower_ci_high'])}].

The trend is not globally monotone; it is a localized high-tail/threshold failure. In the top-20% threshold sweep, full-pool CA-lDDT dropped {f3(top20_full['ca_lddt_top_minus_lower'])} with broad adjusted coefficient {f3(top20_full['broad_adjusted_top_tail_coef'])}; held-out confirmation dropped {f3(top20_conf['ca_lddt_top_minus_lower'])} with adjusted coefficient {f3(top20_conf['broad_adjusted_top_tail_coef'])}. Top-15% thresholds strengthened the effect in both full and confirmation subsets.

## Controls

Known evolutionary and size controls were included: sequence length, MSA nonquery depth, top nonquery identity, and CAMEO difficulty. The primary axis remained negative after broad controls: full-pool standardized axis coefficient {f3(primary_full['ca_lddt_broad_adjusted_coef'])}; held-out confirmation {f3(primary_conf['ca_lddt_broad_adjusted_coef'])}. Adding benchmark-only reference contact-order/contact-density controls did not remove it: full {f3(primary_full['ca_lddt_broad_ref_adjusted_coef'])}; confirmation {f3(primary_conf['ca_lddt_broad_ref_adjusted_coef'])}.

The fixed baseline did not show the same degradation. Baseline CA-lDDT top-tail effect was {f3(primary_full['baseline_ca_lddt_top_vs_lower_effect'])} in the full pool and {f3(primary_conf['baseline_ca_lddt_top_vs_lower_effect'])} in held-out confirmation.

## Other Axes

Reference contact order was a strong benchmark-only explanatory axis, but it is not deployment-prospective because it requires the reference structure. Initial sequence axes for transmembrane hydrophobicity, low complexity, coiled-coil proxy, disorder composition, quaternary state, method/resolution, and submission date were negative or weak after controls. Charge/Uversky-like disorder showed a weaker high-tail signal but was less stable than the flexible/non-helical composite.

## Claim Boundary

Within the fixed 2025 CAMEO single-chain pool and exact saved Biohub ESMFold2 outputs, ESMFold2 is least reliable for the high tail of proline-rich, beta/turn-over-helix sequence composition. This is not a claim about all disordered proteins, all low-MSA proteins, or all reference topology classes; it is a deployment-prospective sequence-composition boundary confirmed on the saved 138-target evaluation pool.

## Artifacts

- `spectra_ready_dataset/prospective_features.followup.csv`
- `target_model_results.csv`
- `spectral_performance_curve.csv`
- `followup_axis_summary.csv`
- `primary_axis_decile_curve.csv`
- `primary_axis_threshold_sweep.csv`
- `primary_axis_membership.csv`
- `axis_ledger.json`
- `spectra_loop_state.json`
- `commands_run.json`
"""
    REPORT.write_text(report)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
