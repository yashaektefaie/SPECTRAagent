#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_frozen_charge_axis import apply_levels, bootstrap_diff, finite_float, greedy_match, q_thresholds  # noqa: E402


AXES = [
    {
        "name": "rg_len13_norm",
        "feature": "rg_len13_norm",
        "description": "reference radius of gyration normalized by length^(1/3); high means expanded reference geometry",
    },
    {
        "name": "shape_anisotropy",
        "feature": "shape_anisotropy",
        "description": "largest covariance eigenvalue fraction of reference CA coordinates; high means elongated/anisotropic geometry",
    },
    {
        "name": "relative_contact_order",
        "feature": "relative_contact_order_ca8",
        "description": "mean sequence separation of CA contacts under 8A normalized by length; high means more nonlocal contact topology",
    },
]


def curve_for_feature(df: pd.DataFrame, feature: str, level_col: str, subset: str, axis_name: str) -> pd.DataFrame:
    work = df[df[level_col].notna() & df["ca_lddt"].notna() & df[feature].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work[level_col] = work[level_col].astype(int)
    group = (
        work.groupby(level_col)
        .agg(
            n=("target_id", "nunique"),
            feature_min=(feature, "min"),
            feature_mean=(feature, "mean"),
            feature_max=(feature, "max"),
            seq_len_mean=("seq_len", "mean"),
            msa_log10_nonquery_rows_mean=("msa_log10_nonquery_rows", "mean"),
            msa_top_nonquery_identity_mean=("msa_top_nonquery_identity", "mean"),
            difficulty_mean=("difficulty", "mean"),
            flexible_nonhelical_composite_rank_mean=("flexible_nonhelical_composite_rank", "mean"),
            ca_lddt_mean=("ca_lddt", "mean"),
            ca_lddt_median=("ca_lddt", "median"),
            gdt_ts_mean=("gdt_ts", "mean"),
            baseline_ca_lddt_mean=("baseline_ca_lddt", "mean"),
            target_minus_baseline_mean=("target_minus_baseline_ca_lddt", "mean"),
        )
        .reset_index()
        .rename(columns={level_col: "axis_level"})
    )
    group.insert(0, "axis_name", axis_name)
    group.insert(1, "feature", feature)
    group.insert(2, "subset", subset)
    group.insert(3, "level_schema", level_col)
    return group


def summarize(curve: pd.DataFrame) -> dict:
    if curve.empty or curve["axis_level"].nunique() < 3:
        return {"valid": False, "reason": "fewer than three nonempty levels"}
    low = curve.sort_values("axis_level").iloc[0]
    high = curve.sort_values("axis_level").iloc[-1]
    return {
        "valid": True,
        "levels": int(curve["axis_level"].nunique()),
        "n": int(curve["n"].sum()),
        "min_level_n": int(curve["n"].min()),
        "easy_level": int(low["axis_level"]),
        "hard_level": int(high["axis_level"]),
        "easy_feature_mean": finite_float(low["feature_mean"]),
        "hard_feature_mean": finite_float(high["feature_mean"]),
        "easy_ca_lddt_mean": finite_float(low["ca_lddt_mean"]),
        "hard_ca_lddt_mean": finite_float(high["ca_lddt_mean"]),
        "ca_lddt_drop_easy_minus_hard": finite_float(low["ca_lddt_mean"] - high["ca_lddt_mean"]),
        "baseline_ca_lddt_drop_easy_minus_hard": finite_float(low["baseline_ca_lddt_mean"] - high["baseline_ca_lddt_mean"]),
        "target_minus_baseline_drop_easy_minus_hard": finite_float(low["target_minus_baseline_mean"] - high["target_minus_baseline_mean"]),
        "level_vs_ca_lddt_spearman": finite_float(curve["axis_level"].rank().corr(curve["ca_lddt_mean"].rank())),
        "level_vs_baseline_spearman": finite_float(curve["axis_level"].rank().corr(curve["baseline_ca_lddt_mean"].rank())),
        "level_vs_msa_spearman": finite_float(curve["axis_level"].rank().corr(curve["msa_log10_nonquery_rows_mean"].rank())),
        "level_vs_flexible_rank_spearman": finite_float(curve["axis_level"].rank().corr(curve["flexible_nonhelical_composite_rank_mean"].rank())),
        "level_vs_length_spearman": finite_float(curve["axis_level"].rank().corr(curve["seq_len_mean"].rank())),
        "level_vs_difficulty_spearman": finite_float(curve["axis_level"].rank().corr(curve["difficulty_mean"].rank())),
    }


def regression_axis(df: pd.DataFrame, feature: str) -> dict:
    controls = [
        "log_seq_len",
        "msa_log10_nonquery_rows",
        "msa_top_nonquery_identity",
        "difficulty",
        "flexible_nonhelical_composite_rank",
    ]
    cols = ["ca_lddt", feature] + [c for c in controls if c in df.columns]
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(work) < len(cols) + 8:
        return {"valid": False, "n": int(len(work))}
    y = work["ca_lddt"].to_numpy(dtype=float)
    names = [feature] + [c for c in controls if c in work.columns]
    xs = []
    kept = []
    for col in names:
        x = work[col].to_numpy(dtype=float)
        sd = x.std()
        if not np.isfinite(sd) or sd == 0:
            continue
        xs.append((x - x.mean()) / sd)
        kept.append(col)
    X = np.column_stack([np.ones(len(work))] + xs)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float((resid @ resid) / max(1, dof))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "valid": True,
        "n": int(len(work)),
        "coef_axis_per_1sd": float(beta[1]),
        "se_axis_per_1sd": float(se[1]),
        "t_axis": float(beta[1] / se[1]) if se[1] else None,
        "controls": kept[1:],
    }


def main() -> None:
    root = Path(".").resolve()
    df = pd.read_csv(root / "spectra_ready_dataset/combined_feature_outcome_table.csv").copy()
    df["target_minus_baseline_ca_lddt"] = df["ca_lddt"] - df["baseline_ca_lddt"]
    if "log_seq_len" not in df.columns:
        df["log_seq_len"] = np.log10(df["seq_len"].clip(lower=1))
    discovery = df[(df["dataset"].eq("cameo")) & (df["hash_fold10"] <= 5)].copy()
    subsets = {
        "cameo_discovery_hash0_5": df[(df["dataset"].eq("cameo")) & (df["hash_fold10"] <= 5)].copy(),
        "cameo_confirmation_hash6_9": df[(df["dataset"].eq("cameo")) & (df["hash_fold10"] >= 6)].copy(),
        "cameo_full_default": df[df["dataset"].eq("cameo")].copy(),
        "cameo_full_default_excluding_known_flexible_q7_level6": df[
            df["dataset"].eq("cameo") & df["axis_flexible_nonhelical_composite_rank_q7_level"].ne(6)
        ].copy(),
        "cameo_scaled_top_tail_best_ptm": df[df["dataset"].eq("cameo_scaled")].copy(),
        "external_rcsb_same_bucket_scaled_best_ptm": df[df["dataset"].eq("external_rcsb")].copy(),
    }
    all_curves = []
    audits = {}
    for axis in AXES:
        name = axis["name"]
        feature = axis["feature"]
        thresholds = {
            "q7": q_thresholds(discovery[feature].dropna(), 7),
            "q10": q_thresholds(discovery[feature].dropna(), 10),
        }
        for q in [7, 10]:
            df[f"frozen_{name}_q{q}_level"] = apply_levels(df[feature], thresholds[f"q{q}"])
        summaries = {}
        for subset_name, subset_raw in subsets.items():
            subset = df.loc[subset_raw.index].copy()
            summaries[subset_name] = {}
            for q in [7, 10]:
                col = f"frozen_{name}_q{q}_level"
                curve = curve_for_feature(subset, feature, col, subset_name, name)
                if not curve.empty:
                    all_curves.append(curve)
                summaries[subset_name][f"q{q}"] = summarize(curve)
                summaries[subset_name][f"q{q}_top_vs_low_bootstrap_ca_lddt"] = bootstrap_diff(
                    subset.rename(columns={col: "axis_level"}), "axis_level", {q - 1}, {0, 1}, "ca_lddt"
                )
                summaries[subset_name][f"q{q}_top_vs_low_bootstrap_baseline_ca_lddt"] = bootstrap_diff(
                    subset.rename(columns={col: "axis_level"}), "axis_level", {q - 1}, {0, 1}, "baseline_ca_lddt"
                )
        cameo = df[df["dataset"].eq("cameo")].copy()
        pairs7, match7 = greedy_match(
            cameo.rename(columns={f"frozen_{name}_q7_level": "axis_level"}),
            "axis_level",
            high_levels={6},
            eligible_low_levels={0, 1, 2, 3, 4},
            controls=["log_seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"],
        )
        pairs10, match10 = greedy_match(
            cameo.rename(columns={f"frozen_{name}_q10_level": "axis_level"}),
            "axis_level",
            high_levels={9},
            eligible_low_levels={0, 1, 2, 3, 4, 5, 6, 7},
            controls=["log_seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"],
        )
        pairs7.to_csv(root / f"frozen_{name}_matched_pairs_q7.csv", index=False)
        pairs10.to_csv(root / f"frozen_{name}_matched_pairs_q10.csv", index=False)
        audits[name] = {
            "axis": axis,
            "frozen_from": "CAMEO discovery hash folds 0-5; reference-derived benchmark-only feature; no target-model outcome/confidence used for membership",
            "thresholds": thresholds,
            "subset_summaries": summaries,
            "regression_controls_cameo_full": regression_axis(cameo, feature),
            "regression_controls_cameo_excluding_known_flexible_q7_level6": regression_axis(
                cameo[cameo["axis_flexible_nonhelical_composite_rank_q7_level"].ne(6)], feature
            ),
            "matched_pair_q7_top_vs_lower_axis_cameo": match7,
            "matched_pair_q10_top_vs_lower_axis_cameo": match10,
        }
    curves_df = pd.concat(all_curves, ignore_index=True, sort=False)
    curves_df.to_csv(root / "frozen_reference_topology_axis_curves.csv", index=False)
    out = {
        "benchmark_only_note": "These axes use reference structure metadata. They are valid for benchmark generalizability auditing, not for deploy-time sequence-only pre-screening.",
        "audits": audits,
        "closure_gate_assessment": {
            "passes": False,
            "reason": "controller must inspect stability and external portability before final synthesis",
        },
    }
    (root / "frozen_reference_topology_axes_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
