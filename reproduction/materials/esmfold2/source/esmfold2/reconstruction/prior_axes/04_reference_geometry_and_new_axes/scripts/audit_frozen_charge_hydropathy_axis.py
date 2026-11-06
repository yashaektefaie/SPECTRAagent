#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_frozen_charge_axis import (  # noqa: E402
    apply_levels,
    bootstrap_diff,
    curve_for,
    finite_float,
    greedy_match,
    q_thresholds,
    regression,
    summarize_curve,
)


def z_from_reference(values: pd.Series, ref: pd.Series) -> pd.Series:
    mu = float(pd.to_numeric(ref, errors="coerce").mean())
    sd = float(pd.to_numeric(ref, errors="coerce").std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (pd.to_numeric(values, errors="coerce") - mu) / sd


def regression_for_axis(df: pd.DataFrame, axis_col: str) -> dict:
    controls = [
        "log_seq_len",
        "msa_log10_nonquery_rows",
        "msa_top_nonquery_identity",
        "difficulty",
        "flexible_nonhelical_composite_rank",
    ]
    cols = ["ca_lddt", axis_col] + [c for c in controls if c in df.columns]
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(work) < len(cols) + 8:
        return {"valid": False, "n": int(len(work))}
    y = work["ca_lddt"].to_numpy(dtype=float)
    names = [axis_col] + [c for c in controls if c in work.columns]
    xs = []
    kept = []
    for col in names:
        x = work[col].to_numpy(dtype=float)
        sd = x.std()
        if sd == 0 or not np.isfinite(sd):
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
        "axis_col": axis_col,
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
    df["frozen_charge_hydropathy_imbalance"] = z_from_reference(
        df["abs_charge_per_residue"], discovery["abs_charge_per_residue"]
    ) - z_from_reference(df["hydropathy_norm_uversky"], discovery["hydropathy_norm_uversky"])
    thresholds = {
        "q7": q_thresholds(df.loc[discovery.index, "frozen_charge_hydropathy_imbalance"], 7),
        "q10": q_thresholds(df.loc[discovery.index, "frozen_charge_hydropathy_imbalance"], 10),
    }
    for q in [7, 10]:
        df[f"frozen_charge_hydropathy_q{q}_level"] = apply_levels(
            df["frozen_charge_hydropathy_imbalance"], thresholds[f"q{q}"]
        )
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
    curves = []
    summaries = {}
    for subset_name, subset in subsets.items():
        summaries[subset_name] = {}
        for q in [7, 10]:
            col = f"frozen_charge_hydropathy_q{q}_level"
            # Reuse the curve helper by temporarily exposing the audited feature
            # as abs_charge_per_residue for feature range columns.
            tmp = subset.copy()
            tmp["abs_charge_per_residue"] = tmp["frozen_charge_hydropathy_imbalance"]
            c = curve_for(tmp, col, subset_name)
            if not c.empty:
                c = c.rename(
                    columns={
                        "abs_charge_min": "axis_min",
                        "abs_charge_mean": "axis_mean",
                        "abs_charge_max": "axis_max",
                    }
                )
                curves.append(c)
            summaries[subset_name][f"q{q}"] = summarize_curve(
                c.rename(columns={"axis_min": "abs_charge_min", "axis_mean": "abs_charge_mean", "axis_max": "abs_charge_max"})
            )
            high_levels = {q - 1}
            low_levels = {0, 1}
            summaries[subset_name][f"q{q}_top_vs_low_bootstrap_ca_lddt"] = bootstrap_diff(subset, col, high_levels, low_levels, "ca_lddt")
            summaries[subset_name][f"q{q}_top_vs_low_bootstrap_baseline_ca_lddt"] = bootstrap_diff(
                subset, col, high_levels, low_levels, "baseline_ca_lddt"
            )
    curves_df = pd.concat(curves, ignore_index=True, sort=False)
    curves_df.to_csv(root / "frozen_charge_hydropathy_axis_curves.csv", index=False)

    cameo = df[df["dataset"].eq("cameo")].copy()
    pairs7, match7 = greedy_match(
        cameo,
        "frozen_charge_hydropathy_q7_level",
        high_levels={6},
        eligible_low_levels={0, 1, 2, 3, 4},
        controls=["log_seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"],
    )
    pairs10, match10 = greedy_match(
        cameo,
        "frozen_charge_hydropathy_q10_level",
        high_levels={9},
        eligible_low_levels={0, 1, 2, 3, 4, 5, 6, 7},
        controls=["log_seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"],
    )
    pairs7.to_csv(root / "frozen_charge_hydropathy_axis_matched_pairs_q7.csv", index=False)
    pairs10.to_csv(root / "frozen_charge_hydropathy_axis_matched_pairs_q10.csv", index=False)

    audit = {
        "axis": {
            "name": "charge_hydropathy_imbalance",
            "feature": "z(abs_net_charge_per_residue) - z(hydropathy_norm_uversky), with z-scores frozen from CAMEO discovery folds",
            "definition": "sequence-only high-charge/low-hydropathy imbalance; higher levels are farther from low-charge or hydrophobic sequence composition",
            "frozen_from": "CAMEO discovery hash folds 0-5; no target-model outcome/confidence or reference comparison used for membership",
            "thresholds": thresholds,
        },
        "subset_summaries": summaries,
        "regression_controls_cameo_full": regression_for_axis(cameo, "frozen_charge_hydropathy_imbalance"),
        "regression_controls_cameo_excluding_known_flexible_q7_level6": regression_for_axis(
            cameo[cameo["axis_flexible_nonhelical_composite_rank_q7_level"].ne(6)],
            "frozen_charge_hydropathy_imbalance",
        ),
        "matched_pair_q7_top_vs_lower_axis_cameo": match7,
        "matched_pair_q10_top_vs_lower_axis_cameo": match10,
        "closure_gate_assessment": {
            "passes": False,
            "reason": "audit metrics generated; controller must inspect stability, matching, and confounding before final synthesis",
        },
    }
    (root / "frozen_charge_hydropathy_axis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
