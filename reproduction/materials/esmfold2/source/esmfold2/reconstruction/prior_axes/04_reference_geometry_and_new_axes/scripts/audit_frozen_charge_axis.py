#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLS = [
    "log_seq_len",
    "msa_log10_nonquery_rows",
    "msa_top_nonquery_identity",
    "difficulty",
    "flexible_nonhelical_composite_rank",
    "hydropathy_norm_uversky",
]


def finite_float(x):
    try:
        if x is None or not np.isfinite(x):
            return None
        return float(x)
    except TypeError:
        return None


def q_thresholds(values: pd.Series, q: int) -> list[float]:
    probs = [i / q for i in range(1, q)]
    return [float(v) for v in np.nanquantile(values.astype(float), probs)]


def apply_levels(values: pd.Series, thresholds: list[float]) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=x.index)
    mask = x.notna()
    out.loc[mask] = np.searchsorted(np.asarray(thresholds), x.loc[mask].to_numpy(dtype=float), side="right")
    return out


def curve_for(df: pd.DataFrame, level_col: str, subset: str) -> pd.DataFrame:
    work = df[df[level_col].notna() & df["ca_lddt"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work[level_col] = work[level_col].astype(int)
    group = (
        work.groupby(level_col)
        .agg(
            n=("target_id", "nunique"),
            abs_charge_min=("abs_charge_per_residue", "min"),
            abs_charge_mean=("abs_charge_per_residue", "mean"),
            abs_charge_max=("abs_charge_per_residue", "max"),
            net_charge_mean=("net_charge_per_residue", "mean"),
            fcr_mean=("fraction_charged_residues_fcr", "mean"),
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
        .rename(columns={level_col: "charge_level"})
    )
    group.insert(0, "subset", subset)
    group.insert(1, "level_schema", level_col)
    return group


def summarize_curve(curve: pd.DataFrame) -> dict:
    if curve.empty or curve["charge_level"].nunique() < 3:
        return {"valid": False, "reason": "fewer than three nonempty levels"}
    low = curve.sort_values("charge_level").iloc[0]
    high = curve.sort_values("charge_level").iloc[-1]
    return {
        "valid": True,
        "levels": int(curve["charge_level"].nunique()),
        "n": int(curve["n"].sum()),
        "min_level_n": int(curve["n"].min()),
        "easy_level": int(low["charge_level"]),
        "hard_level": int(high["charge_level"]),
        "easy_abs_charge_mean": finite_float(low["abs_charge_mean"]),
        "hard_abs_charge_mean": finite_float(high["abs_charge_mean"]),
        "easy_ca_lddt_mean": finite_float(low["ca_lddt_mean"]),
        "hard_ca_lddt_mean": finite_float(high["ca_lddt_mean"]),
        "ca_lddt_drop_easy_minus_hard": finite_float(low["ca_lddt_mean"] - high["ca_lddt_mean"]),
        "baseline_ca_lddt_drop_easy_minus_hard": finite_float(low["baseline_ca_lddt_mean"] - high["baseline_ca_lddt_mean"]),
        "target_minus_baseline_drop_easy_minus_hard": finite_float(
            (low["target_minus_baseline_mean"] - high["target_minus_baseline_mean"])
        ),
        "level_vs_ca_lddt_spearman": finite_float(curve["charge_level"].rank().corr(curve["ca_lddt_mean"].rank())),
        "level_vs_baseline_spearman": finite_float(curve["charge_level"].rank().corr(curve["baseline_ca_lddt_mean"].rank())),
        "level_vs_msa_spearman": finite_float(curve["charge_level"].rank().corr(curve["msa_log10_nonquery_rows_mean"].rank())),
        "level_vs_flexible_rank_spearman": finite_float(
            curve["charge_level"].rank().corr(curve["flexible_nonhelical_composite_rank_mean"].rank())
        ),
        "level_vs_length_spearman": finite_float(curve["charge_level"].rank().corr(curve["seq_len_mean"].rank())),
        "level_vs_difficulty_spearman": finite_float(curve["charge_level"].rank().corr(curve["difficulty_mean"].rank())),
    }


def bootstrap_diff(df: pd.DataFrame, group_col: str, high_levels: set[int], low_levels: set[int], metric: str, n_boot: int = 5000) -> dict:
    work = df[df[group_col].notna() & df[metric].notna()].copy()
    work[group_col] = work[group_col].astype(int)
    high = work[work[group_col].isin(high_levels)][metric].to_numpy(dtype=float)
    low = work[work[group_col].isin(low_levels)][metric].to_numpy(dtype=float)
    if len(high) < 3 or len(low) < 3:
        return {"n_high": int(len(high)), "n_low": int(len(low)), "mean_low_minus_high": None, "ci95": None}
    rng = np.random.default_rng(20260531)
    obs = float(np.mean(low) - np.mean(high))
    boots = []
    for _ in range(n_boot):
        boots.append(float(np.mean(rng.choice(low, size=len(low), replace=True)) - np.mean(rng.choice(high, size=len(high), replace=True))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "mean_low_minus_high": obs,
        "ci95": [float(lo), float(hi)],
    }


def regression(df: pd.DataFrame) -> dict:
    cols = ["ca_lddt", "abs_charge_per_residue"] + [c for c in CONTROLS if c in df.columns]
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(work) < len(cols) + 8:
        return {"valid": False, "n": int(len(work))}
    y = work["ca_lddt"].to_numpy(dtype=float)
    xs = []
    names = ["abs_charge_per_residue"]
    for col in names + [c for c in CONTROLS if c in work.columns]:
        x = work[col].to_numpy(dtype=float)
        sd = x.std()
        if sd == 0 or not np.isfinite(sd):
            continue
        xs.append((x - x.mean()) / sd)
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
        "coef_abs_charge_per_1sd": float(beta[1]),
        "se_abs_charge_per_1sd": float(se[1]),
        "t_abs_charge": float(beta[1] / se[1]) if se[1] else None,
        "controls": [c for c in CONTROLS if c in work.columns],
    }


def greedy_match(df: pd.DataFrame, level_col: str, high_levels: set[int], eligible_low_levels: set[int], controls: list[str]) -> tuple[pd.DataFrame, dict]:
    work = df[df[level_col].notna()].copy()
    work[level_col] = work[level_col].astype(int)
    ctrl = [c for c in controls if c in work.columns and work[c].notna().sum() > 5]
    for c in ctrl:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=ctrl + ["ca_lddt", "baseline_ca_lddt", "abs_charge_per_residue"])
    high = work[work[level_col].isin(high_levels)].copy()
    low = work[work[level_col].isin(eligible_low_levels)].copy()
    if len(high) < 3 or len(low) < 3 or not ctrl:
        return pd.DataFrame(), {"valid": False, "n_high": int(len(high)), "n_low_pool": int(len(low))}
    means = work[ctrl].mean()
    sds = work[ctrl].std().replace(0, 1)
    high = high.sort_values("abs_charge_per_residue", ascending=False)
    used = set()
    pairs = []
    for _, h in high.iterrows():
        candidates = low[~low["target_id"].isin(used)]
        if candidates.empty:
            break
        hz = ((h[ctrl] - means) / sds).to_numpy(dtype=float)
        cz = ((candidates[ctrl] - means) / sds).to_numpy(dtype=float)
        dist = np.sqrt(((cz - hz) ** 2).sum(axis=1))
        j = int(np.argmin(dist))
        l = candidates.iloc[j]
        used.add(l["target_id"])
        pairs.append(
            {
                "high_target_id": h["target_id"],
                "low_target_id": l["target_id"],
                "match_distance": float(dist[j]),
                "high_abs_charge": float(h["abs_charge_per_residue"]),
                "low_abs_charge": float(l["abs_charge_per_residue"]),
                "high_ca_lddt": float(h["ca_lddt"]),
                "low_ca_lddt": float(l["ca_lddt"]),
                "high_baseline_ca_lddt": float(h["baseline_ca_lddt"]),
                "low_baseline_ca_lddt": float(l["baseline_ca_lddt"]),
                "high_flexible_rank": finite_float(h.get("flexible_nonhelical_composite_rank")),
                "low_flexible_rank": finite_float(l.get("flexible_nonhelical_composite_rank")),
                "high_msa_log10": finite_float(h.get("msa_log10_nonquery_rows")),
                "low_msa_log10": finite_float(l.get("msa_log10_nonquery_rows")),
                "high_seq_len": finite_float(h.get("seq_len")),
                "low_seq_len": finite_float(l.get("seq_len")),
                "high_difficulty": finite_float(h.get("difficulty")),
                "low_difficulty": finite_float(l.get("difficulty")),
            }
        )
    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pairs_df, {"valid": False, "reason": "no pairs"}
    diffs = pairs_df["low_ca_lddt"] - pairs_df["high_ca_lddt"]
    base_diffs = pairs_df["low_baseline_ca_lddt"] - pairs_df["high_baseline_ca_lddt"]
    summary = {
        "valid": True,
        "n_pairs": int(len(pairs_df)),
        "mean_low_minus_high_ca_lddt": float(diffs.mean()),
        "median_low_minus_high_ca_lddt": float(diffs.median()),
        "mean_low_minus_high_baseline_ca_lddt": float(base_diffs.mean()),
        "mean_target_minus_baseline_pair_effect": float((diffs - base_diffs).mean()),
        "mean_match_distance": float(pairs_df["match_distance"].mean()),
        "controls": ctrl,
    }
    return pairs_df, summary


def main() -> None:
    root = Path(".").resolve()
    df = pd.read_csv(root / "spectra_ready_dataset/combined_feature_outcome_table.csv")
    df["target_minus_baseline_ca_lddt"] = df["ca_lddt"] - df["baseline_ca_lddt"]
    if "log_seq_len" not in df.columns:
        df["log_seq_len"] = np.log10(df["seq_len"].clip(lower=1))

    discovery = df[(df["dataset"].eq("cameo")) & (df["hash_fold10"] <= 5)].copy()
    thresholds = {
        "charge_q7_discovery_thresholds": q_thresholds(discovery["abs_charge_per_residue"], 7),
        "charge_q10_discovery_thresholds": q_thresholds(discovery["abs_charge_per_residue"], 10),
    }
    for q, th in [(7, thresholds["charge_q7_discovery_thresholds"]), (10, thresholds["charge_q10_discovery_thresholds"])]:
        df[f"frozen_abs_charge_q{q}_level"] = apply_levels(df["abs_charge_per_residue"], th)

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
            col = f"frozen_abs_charge_q{q}_level"
            c = curve_for(subset, col, subset_name)
            if not c.empty:
                curves.append(c)
            summaries[subset_name][f"q{q}"] = summarize_curve(c)
            high_levels = {q - 1}
            low_levels = {0, 1}
            summaries[subset_name][f"q{q}_top_vs_low_bootstrap_ca_lddt"] = bootstrap_diff(subset, col, high_levels, low_levels, "ca_lddt")
            summaries[subset_name][f"q{q}_top_vs_low_bootstrap_baseline_ca_lddt"] = bootstrap_diff(
                subset, col, high_levels, low_levels, "baseline_ca_lddt"
            )
    curves_df = pd.concat(curves, ignore_index=True, sort=False)
    curves_df.to_csv(root / "frozen_charge_axis_curves.csv", index=False)

    cameo = df[df["dataset"].eq("cameo")].copy()
    reg_all = regression(cameo)
    reg_not_flex6 = regression(cameo[cameo["axis_flexible_nonhelical_composite_rank_q7_level"].ne(6)])
    pairs7, match7 = greedy_match(
        cameo,
        "frozen_abs_charge_q7_level",
        high_levels={6},
        eligible_low_levels={0, 1, 2, 3, 4},
        controls=["log_seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"],
    )
    pairs10, match10 = greedy_match(
        cameo,
        "frozen_abs_charge_q10_level",
        high_levels={9},
        eligible_low_levels={0, 1, 2, 3, 4, 5, 6, 7},
        controls=["log_seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"],
    )
    pairs7.to_csv(root / "frozen_charge_axis_matched_pairs_q7.csv", index=False)
    pairs10.to_csv(root / "frozen_charge_axis_matched_pairs_q10.csv", index=False)

    audit = {
        "axis": {
            "name": "charge_imbalance_abs_net_charge_per_residue",
            "feature": "abs_charge_per_residue",
            "definition": "absolute net sequence charge per residue computed from amino-acid sequence only; higher levels are farther along the prospective charge-imbalance axis",
            "frozen_from": "CAMEO discovery hash folds 0-5; no ESMFold2 outcome, confidence, or prediction/reference comparison used for membership",
            "thresholds": thresholds,
            "known_nonterminal_controls": [
                "MSA/evolutionary support",
                "sequence length",
                "CAMEO difficulty",
                "known flexible/non-helical composite",
                "hydropathy",
                "dataset/source",
            ],
        },
        "subset_summaries": summaries,
        "regression_controls_cameo_full": reg_all,
        "regression_controls_cameo_excluding_known_flexible_q7_level6": reg_not_flex6,
        "matched_pair_q7_top_vs_lower_charge_cameo": match7,
        "matched_pair_q10_top_vs_lower_charge_cameo": match10,
        "closure_gate_assessment": {
            "passes": False,
            "reason": "audit metrics generated; controller must inspect stability and confounding before final synthesis",
        },
    }
    (root / "frozen_charge_axis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
