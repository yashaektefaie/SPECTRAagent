#!/usr/bin/env python3
"""Score frozen non-evolutionary SPECTRA axes against fixed ESMFold2 outcomes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "spectra_ready_dataset" / "prospective_features.csv"
RESULTS_CSV = ROOT / "target_model_results.csv"
AXIS_JSON = ROOT / "axis_definitions_prejoin.json"
LEDGER_JSON = ROOT / "axis_ledger.json"
CURVES_CSV = ROOT / "spectral_curves_all_axes.csv"
SUMMARY_CSV = ROOT / "axis_summary.csv"
STATE_JSON = ROOT / "spectra_loop_state.json"

PRIMARY_METRIC = "ca_lddt"
SECONDARY_METRICS = ["gdt_ts", "ca_rmsd"]
HIGHER_IS_BETTER = {"ca_lddt": True, "gdt_ts": True, "ca_rmsd": False}
BASELINE_FOR = {
    "ca_lddt": "baseline_ca_lddt",
    "gdt_ts": "baseline_gdt_ts",
    "ca_rmsd": "baseline_ca_rmsd",
}
CONTROL_COLS_CORE = ["seq_len", "msa_log10_nonquery_rows"]
CONTROL_COLS_BROAD = [
    "seq_len",
    "msa_log10_nonquery_rows",
    "msa_top_nonquery_identity",
    "difficulty",
]


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def ols_coef(df: pd.DataFrame, y_col: str, stress_col: str, controls: list[str]) -> dict[str, float]:
    cols = [y_col, stress_col] + controls
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(15, len(cols) * 4):
        return {"n": len(sub), "coef": float("nan"), "partial_corr": float("nan")}
    y = sub[y_col].to_numpy(dtype=float)
    x_stress = zscore(sub[stress_col]).to_numpy(dtype=float)
    x_controls = [zscore(sub[c]).to_numpy(dtype=float) for c in controls]
    X = np.column_stack([np.ones(len(sub)), x_stress] + x_controls)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # Partial correlation by residualizing y and stress on controls.
    C = np.column_stack([np.ones(len(sub))] + x_controls)
    by, *_ = np.linalg.lstsq(C, y, rcond=None)
    bs, *_ = np.linalg.lstsq(C, x_stress, rcond=None)
    y_res = y - C @ by
    s_res = x_stress - C @ bs
    if np.std(y_res) == 0 or np.std(s_res) == 0:
        pcorr = float("nan")
    else:
        pcorr = float(np.corrcoef(y_res, s_res)[0, 1])
    return {"n": int(len(sub)), "coef": float(beta[1]), "partial_corr": pcorr}


def stress_rank_for_axis(df: pd.DataFrame, axis: dict[str, object]) -> pd.Series:
    level_col = axis["level_column"]
    raw = df[level_col]
    if axis["feature"] == "quaternary_state":
        return pd.to_numeric(raw, errors="coerce")
    levels = pd.to_numeric(raw, errors="coerce")
    max_level = levels.max()
    if axis.get("stress_direction") == "low":
        return max_level - levels
    return levels


def monotonicity(level_df: pd.DataFrame, metric: str) -> float:
    if len(level_df) < 3:
        return float("nan")
    return float(level_df["stress_rank"].corr(level_df[metric], method="spearman"))


def endpoint_effect(level_df: pd.DataFrame, metric: str) -> float:
    if len(level_df) < 2:
        return float("nan")
    ordered = level_df.sort_values("stress_rank")
    return float(ordered.iloc[-1][metric] - ordered.iloc[0][metric])


def classify_axis(summary: dict[str, object]) -> str:
    if not summary["split_valid"]:
        return "invalid"
    effect = summary["ca_lddt_endpoint_effect"]
    broad = summary["ca_lddt_broad_adjusted_coef"]
    rho = summary["ca_lddt_level_spearman"]
    baseline_effect = summary["baseline_ca_lddt_endpoint_effect"]
    n_min = summary["min_level_n"]
    if not np.isfinite(effect) or n_min < 8:
        return "invalid"
    target_degrades = effect <= -0.08 and broad <= -0.025
    monotone_or_threshold = rho <= -0.55 or summary["ca_lddt_worst_level_drop"] <= -0.10
    baseline_not_same = (not np.isfinite(baseline_effect)) or baseline_effect > effect * 0.5
    if target_degrades and monotone_or_threshold and baseline_not_same:
        return "valid_candidate"
    if effect <= -0.04 or broad <= -0.015 or summary["ca_lddt_worst_level_drop"] <= -0.08:
        return "weak"
    return "negative"


def main() -> None:
    features = pd.read_csv(FEATURE_CSV)
    results = pd.read_csv(RESULTS_CSV)
    axes = json.loads(AXIS_JSON.read_text())["axes"]
    merged = features.merge(results, on="target_id", how="inner", validate="one_to_one")
    merged = merged[merged["status"].eq("ok")].copy()
    merged["log_seq_len"] = np.log10(pd.to_numeric(merged["seq_len"], errors="coerce"))

    curve_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    ledger_entries: list[dict[str, object]] = []

    for axis in axes:
        if axis.get("status") != "frozen":
            ledger_entries.append({**axis, "decision": "invalid"})
            continue
        level_col = axis["level_column"]
        if level_col not in merged.columns:
            ledger_entries.append({**axis, "decision": "invalid", "reason": "missing_level_col"})
            continue
        work = merged.copy()
        work["stress_rank"] = stress_rank_for_axis(work, axis)
        work = work.dropna(subset=["stress_rank", PRIMARY_METRIC])
        grouped = []
        for stress_rank, g in work.groupby("stress_rank", dropna=True):
            row = {
                "axis_id": axis["axis_id"],
                "feature": axis["feature"],
                "axis_type": axis["type"],
                "prospective": axis["prospective"],
                "benchmark_only": axis["benchmark_only"],
                "stress_direction": axis["stress_direction"],
                "stress_rank": float(stress_rank),
                "n": int(len(g)),
                "feature_median": float(pd.to_numeric(g[axis["feature"]], errors="coerce").median()),
                "seq_len_median": float(pd.to_numeric(g["seq_len"], errors="coerce").median()),
                "msa_log10_nonquery_rows_median": float(
                    pd.to_numeric(g["msa_log10_nonquery_rows"], errors="coerce").median()
                ),
                "difficulty_mean": float(pd.to_numeric(g["difficulty"], errors="coerce").mean()),
            }
            for metric in [PRIMARY_METRIC] + SECONDARY_METRICS:
                row[metric] = float(pd.to_numeric(g[metric], errors="coerce").mean())
                row[f"{metric}_median"] = float(pd.to_numeric(g[metric], errors="coerce").median())
                baseline = BASELINE_FOR[metric]
                row[baseline] = float(pd.to_numeric(g[baseline], errors="coerce").mean())
            grouped.append(row)
            curve_rows.append(row)

        level_df = pd.DataFrame(grouped).sort_values("stress_rank")
        split_counts = level_df["n"].tolist()
        measured_order = level_df["feature_median"].is_monotonic_increasing
        if axis.get("stress_direction") == "low":
            measured_order = level_df["feature_median"].is_monotonic_decreasing
        split_valid = len(level_df) >= 3 and min(split_counts) >= 8 and bool(measured_order)
        low_perf = float(level_df.iloc[0][PRIMARY_METRIC]) if len(level_df) else float("nan")
        level_df["ca_lddt_drop_from_low_stress"] = level_df[PRIMARY_METRIC] - low_perf
        worst_drop = float(level_df["ca_lddt_drop_from_low_stress"].min()) if len(level_df) else float("nan")

        core = ols_coef(work, PRIMARY_METRIC, "stress_rank", CONTROL_COLS_CORE)
        broad = ols_coef(work, PRIMARY_METRIC, "stress_rank", CONTROL_COLS_BROAD)
        gdt_broad = ols_coef(work, "gdt_ts", "stress_rank", CONTROL_COLS_BROAD)
        rmsd_broad = ols_coef(work, "ca_rmsd", "stress_rank", CONTROL_COLS_BROAD)
        baseline_core = ols_coef(work, "baseline_ca_lddt", "stress_rank", CONTROL_COLS_CORE)
        endpoint = endpoint_effect(level_df, PRIMARY_METRIC)
        baseline_endpoint = endpoint_effect(level_df, "baseline_ca_lddt")
        summary = {
            "axis_id": axis["axis_id"],
            "feature": axis["feature"],
            "axis_type": axis["type"],
            "prospective": bool(axis["prospective"]),
            "benchmark_only": bool(axis["benchmark_only"]),
            "stress_direction": axis["stress_direction"],
            "n_evaluated": int(len(work)),
            "n_levels": int(len(level_df)),
            "min_level_n": int(min(split_counts)) if split_counts else 0,
            "split_valid": bool(split_valid),
            "measured_order_ok": bool(measured_order),
            "ca_lddt_endpoint_effect": endpoint,
            "ca_lddt_worst_level_drop": worst_drop,
            "ca_lddt_level_spearman": monotonicity(level_df, PRIMARY_METRIC),
            "gdt_ts_endpoint_effect": endpoint_effect(level_df, "gdt_ts"),
            "ca_rmsd_endpoint_effect": endpoint_effect(level_df, "ca_rmsd"),
            "baseline_ca_lddt_endpoint_effect": baseline_endpoint,
            "baseline_ca_lddt_level_spearman": monotonicity(level_df, "baseline_ca_lddt"),
            "ca_lddt_core_adjusted_coef": core["coef"],
            "ca_lddt_core_partial_corr": core["partial_corr"],
            "ca_lddt_broad_adjusted_coef": broad["coef"],
            "ca_lddt_broad_partial_corr": broad["partial_corr"],
            "gdt_ts_broad_adjusted_coef": gdt_broad["coef"],
            "ca_rmsd_broad_adjusted_coef": rmsd_broad["coef"],
            "baseline_ca_lddt_core_adjusted_coef": baseline_core["coef"],
            "low_stress_ca_lddt": float(level_df.iloc[0][PRIMARY_METRIC]) if len(level_df) else float("nan"),
            "high_stress_ca_lddt": float(level_df.iloc[-1][PRIMARY_METRIC]) if len(level_df) else float("nan"),
            "stress_level_counts": ",".join(str(x) for x in split_counts),
        }
        summary["decision"] = classify_axis(summary)
        summaries.append(summary)
        ledger_entries.append(
            {
                **axis,
                "decision": summary["decision"],
                "n_evaluated": summary["n_evaluated"],
                "split_valid": summary["split_valid"],
                "measured_order_ok": summary["measured_order_ok"],
                "ca_lddt_endpoint_effect": summary["ca_lddt_endpoint_effect"],
                "ca_lddt_broad_adjusted_coef": summary["ca_lddt_broad_adjusted_coef"],
                "baseline_ca_lddt_endpoint_effect": summary["baseline_ca_lddt_endpoint_effect"],
                "classification_note": (
                    "Prospective axis can support deployment-facing claim"
                    if axis["prospective"] and not axis["benchmark_only"]
                    else "Benchmark-only/explanatory axis, not terminal deployment claim"
                ),
            }
        )

    curve_df = pd.DataFrame(curve_rows).sort_values(["axis_id", "stress_rank"])
    summary_df = pd.DataFrame(summaries).sort_values(
        ["decision", "ca_lddt_endpoint_effect"], ascending=[True, True]
    )
    curve_df.to_csv(CURVES_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    ledger = {
        "run_root": str(ROOT),
        "feature_table": str(FEATURE_CSV.relative_to(ROOT)),
        "target_model_results": str(RESULTS_CSV.relative_to(ROOT)),
        "target_model_results_reused": True,
        "outcome_free_axis_freeze": str(AXIS_JSON.relative_to(ROOT)),
        "target_metric": PRIMARY_METRIC,
        "control_cols_core": CONTROL_COLS_CORE,
        "control_cols_broad": CONTROL_COLS_BROAD,
        "entries": ledger_entries,
    }
    LEDGER_JSON.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    state = {
        "phase": "initial_axis_scoring",
        "rows_prospective_feature_table": int(len(features)),
        "rows_fixed_target_model_results": int(len(results)),
        "rows_joined_ok": int(len(merged)),
        "axes_scored": int(len(summaries)),
        "valid_candidate_axes": summary_df[summary_df["decision"].eq("valid_candidate")][
            "axis_id"
        ].tolist(),
        "weak_axes": summary_df[summary_df["decision"].eq("weak")]["axis_id"].tolist(),
        "negative_axes": summary_df[summary_df["decision"].eq("negative")]["axis_id"].tolist(),
        "next_step": "Densify promising prospective axes; if none pass, mine success/failure residuals for frozen follow-up axes.",
    }
    STATE_JSON.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(summary_df.to_string(index=False))
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
