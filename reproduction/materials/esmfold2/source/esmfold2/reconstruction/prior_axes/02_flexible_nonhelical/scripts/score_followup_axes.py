#!/usr/bin/env python3
"""Score and densify frozen follow-up axes, including holdout confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "spectra_ready_dataset" / "prospective_features.followup.csv"
RESULTS_CSV = ROOT / "target_model_results.csv"
AXIS_JSON = ROOT / "axis_definitions_followup.json"
CURVES_CSV = ROOT / "followup_axis_curves.csv"
SUMMARY_CSV = ROOT / "followup_axis_summary.csv"
CONTROL_JSON = ROOT / "control_analysis.json"

CONTROL_CORE = ["seq_len", "msa_log10_nonquery_rows"]
CONTROL_BROAD = ["seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty"]
CONTROL_BROAD_REF = CONTROL_BROAD + ["relative_contact_order_ca8", "contacts_per_residue"]
METRICS = ["ca_lddt", "gdt_ts", "ca_rmsd"]
BASELINE_FOR = {
    "ca_lddt": "baseline_ca_lddt",
    "gdt_ts": "baseline_gdt_ts",
    "ca_rmsd": "baseline_ca_rmsd",
}


def fold_for_target(target_id: str) -> int:
    return int(hashlib.sha256(target_id.encode()).hexdigest(), 16) % 3


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


def stress_rank_for_axis(df: pd.DataFrame, axis: dict[str, object]) -> pd.Series:
    levels = pd.to_numeric(df[axis["level_column"]], errors="coerce")
    if axis.get("stress_direction") == "low":
        return levels.max() - levels
    return levels


def ols_coef(df: pd.DataFrame, y_col: str, stress_col: str, controls: list[str]) -> dict[str, float]:
    cols = [y_col, stress_col] + controls
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(15, len(cols) * 4):
        return {"n": int(len(sub)), "coef": float("nan"), "partial_corr": float("nan")}
    y = sub[y_col].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(len(sub)), zscore(sub[stress_col]).to_numpy(dtype=float)]
        + [zscore(sub[c]).to_numpy(dtype=float) for c in controls]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"n": int(len(sub)), "coef": float(beta[1])}


def level_curve(work: pd.DataFrame, axis: dict[str, object], subset_name: str) -> pd.DataFrame:
    rows = []
    for stress_rank, g in work.groupby("stress_rank"):
        row = {
            "subset": subset_name,
            "axis_id": axis["axis_id"],
            "feature": axis["feature"],
            "stress_rank": float(stress_rank),
            "n": int(len(g)),
            "feature_median": float(pd.to_numeric(g[axis["feature"]], errors="coerce").median()),
            "seq_len_median": float(pd.to_numeric(g["seq_len"], errors="coerce").median()),
            "msa_log10_nonquery_rows_median": float(
                pd.to_numeric(g["msa_log10_nonquery_rows"], errors="coerce").median()
            ),
            "msa_top_nonquery_identity_median": float(
                pd.to_numeric(g["msa_top_nonquery_identity"], errors="coerce").median()
            ),
            "difficulty_mean": float(pd.to_numeric(g["difficulty"], errors="coerce").mean()),
            "reference_contact_order_median": float(
                pd.to_numeric(g["relative_contact_order_ca8"], errors="coerce").median()
            ),
            "reference_contacts_per_residue_median": float(
                pd.to_numeric(g["contacts_per_residue"], errors="coerce").median()
            ),
        }
        for metric in METRICS:
            row[metric] = float(pd.to_numeric(g[metric], errors="coerce").mean())
            row[f"{metric}_median"] = float(pd.to_numeric(g[metric], errors="coerce").median())
            baseline = BASELINE_FOR[metric]
            row[baseline] = float(pd.to_numeric(g[baseline], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stress_rank")


def effect_stats(curve: pd.DataFrame, metric: str) -> dict[str, float]:
    if len(curve) < 3:
        return {"endpoint": float("nan"), "top_vs_lower": float("nan"), "top_vs_middle": float("nan"), "spearman": float("nan")}
    ordered = curve.sort_values("stress_rank")
    top_rank = ordered["stress_rank"].max()
    low_rank = ordered["stress_rank"].min()
    top = ordered[ordered["stress_rank"].eq(top_rank)]
    low = ordered[ordered["stress_rank"].eq(low_rank)]
    lower = ordered[ordered["stress_rank"].lt(top_rank)]
    middle = ordered[(ordered["stress_rank"].gt(low_rank)) & (ordered["stress_rank"].lt(top_rank))]
    return {
        "endpoint": float(top[metric].iloc[0] - low[metric].iloc[0]),
        "top_vs_lower": float(top[metric].iloc[0] - np.average(lower[metric], weights=lower["n"])),
        "top_vs_middle": float(top[metric].iloc[0] - np.average(middle[metric], weights=middle["n"])) if len(middle) else float("nan"),
        "spearman": float(ordered["stress_rank"].corr(ordered[metric], method="spearman")),
    }


def row_level_effect(work: pd.DataFrame, metric: str) -> float:
    top_rank = work["stress_rank"].max()
    top = work[work["stress_rank"].eq(top_rank)]
    lower = work[work["stress_rank"].lt(top_rank)]
    if len(top) < 5 or len(lower) < 10:
        return float("nan")
    return float(pd.to_numeric(top[metric], errors="coerce").mean() - pd.to_numeric(lower[metric], errors="coerce").mean())


def bootstrap_effect(work: pd.DataFrame, metric: str, n_boot: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(20260531)
    vals = []
    arr = work[["stress_rank", metric]].dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 30:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    stress = arr[:, 0]
    metric_values = arr[:, 1]
    top_rank = np.nanmax(stress)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_stress = stress[idx]
        sample_metric = metric_values[idx]
        top = sample_stress == top_rank
        lower = sample_stress < top_rank
        if top.sum() < 5 or lower.sum() < 10:
            continue
        vals.append(float(sample_metric[top].mean() - sample_metric[lower].mean()))
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return {
        "mean": float(vals.mean()) if len(vals) else float("nan"),
        "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else float("nan"),
        "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else float("nan"),
    }


def score_axis_subset(df: pd.DataFrame, axis: dict[str, object], subset_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    work = df.copy()
    work["stress_rank"] = stress_rank_for_axis(work, axis)
    work = work.dropna(subset=["stress_rank", "ca_lddt"])
    curve = level_curve(work, axis, subset_name)
    lddt = effect_stats(curve, "ca_lddt")
    gdt = effect_stats(curve, "gdt_ts")
    rmsd = effect_stats(curve, "ca_rmsd")
    baseline = effect_stats(curve, "baseline_ca_lddt")
    core = ols_coef(work, "ca_lddt", "stress_rank", CONTROL_CORE)
    broad = ols_coef(work, "ca_lddt", "stress_rank", CONTROL_BROAD)
    broad_ref = ols_coef(work, "ca_lddt", "stress_rank", CONTROL_BROAD_REF)
    boot_lddt = bootstrap_effect(work, "ca_lddt")
    boot_gdt = bootstrap_effect(work, "gdt_ts")
    counts = curve["n"].tolist()
    measured_order = curve["feature_median"].is_monotonic_increasing
    if axis.get("stress_direction") == "low":
        measured_order = curve["feature_median"].is_monotonic_decreasing
    summary = {
        "subset": subset_name,
        "axis_id": axis["axis_id"],
        "feature": axis["feature"],
        "origin": axis["origin"],
        "prospective": bool(axis["prospective"]),
        "benchmark_only": bool(axis["benchmark_only"]),
        "outcome_informed": bool(axis.get("outcome_informed", False)),
        "n": int(len(work)),
        "n_levels": int(len(curve)),
        "min_level_n": int(min(counts)) if counts else 0,
        "level_counts": ",".join(str(x) for x in counts),
        "measured_order_ok": bool(measured_order),
        "ca_lddt_endpoint_effect": lddt["endpoint"],
        "ca_lddt_top_vs_lower_effect": lddt["top_vs_lower"],
        "ca_lddt_top_vs_middle_effect": lddt["top_vs_middle"],
        "ca_lddt_level_spearman": lddt["spearman"],
        "gdt_ts_top_vs_lower_effect": gdt["top_vs_lower"],
        "ca_rmsd_top_vs_lower_effect": rmsd["top_vs_lower"],
        "baseline_ca_lddt_top_vs_lower_effect": baseline["top_vs_lower"],
        "baseline_ca_lddt_level_spearman": baseline["spearman"],
        "ca_lddt_core_adjusted_coef": core["coef"],
        "ca_lddt_broad_adjusted_coef": broad["coef"],
        "ca_lddt_broad_ref_adjusted_coef": broad_ref["coef"],
        "bootstrap_ca_lddt_top_vs_lower_mean": boot_lddt["mean"],
        "bootstrap_ca_lddt_top_vs_lower_ci_low": boot_lddt["ci_low"],
        "bootstrap_ca_lddt_top_vs_lower_ci_high": boot_lddt["ci_high"],
        "bootstrap_gdt_top_vs_lower_mean": boot_gdt["mean"],
        "bootstrap_gdt_top_vs_lower_ci_low": boot_gdt["ci_low"],
        "bootstrap_gdt_top_vs_lower_ci_high": boot_gdt["ci_high"],
        "low_rank_ca_lddt": float(curve.sort_values("stress_rank").iloc[0]["ca_lddt"]) if len(curve) else float("nan"),
        "top_rank_ca_lddt": float(curve.sort_values("stress_rank").iloc[-1]["ca_lddt"]) if len(curve) else float("nan"),
    }
    return curve, summary


def main() -> None:
    features = pd.read_csv(FEATURE_CSV)
    results = pd.read_csv(RESULTS_CSV)
    axes = json.loads(AXIS_JSON.read_text())["axes"]
    df = features.merge(results, on="target_id", how="inner", validate="one_to_one")
    df = df[df["status"].eq("ok")].copy()
    df["discovery_fold"] = df["target_id"].map(fold_for_target)
    subsets = {
        "full_138": df,
        "discovery_fold0": df[df["discovery_fold"].eq(0)],
        "confirmation_folds1_2": df[df["discovery_fold"].isin([1, 2])],
    }
    curves = []
    summaries = []
    for axis in axes:
        for subset_name, subset in subsets.items():
            curve, summary = score_axis_subset(subset, axis, subset_name)
            curves.append(curve)
            summaries.append(summary)
    curve_df = pd.concat(curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    curve_df.to_csv(CURVES_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    control = {
        "control_sets": {
            "core": CONTROL_CORE,
            "broad": CONTROL_BROAD,
            "broad_ref_benchmark_only": CONTROL_BROAD_REF,
        },
        "bootstrap_iterations": 1000,
        "confirmation_definition": "folds 1 and 2 from sha256(target_id) % 3; fold 0 used only for outcome-informed hypothesis generation",
        "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
        "curves_csv": str(CURVES_CSV.relative_to(ROOT)),
    }
    CONTROL_JSON.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n")
    print(summary_df.sort_values(["subset", "ca_lddt_top_vs_lower_effect"]).to_string(index=False))


if __name__ == "__main__":
    main()
