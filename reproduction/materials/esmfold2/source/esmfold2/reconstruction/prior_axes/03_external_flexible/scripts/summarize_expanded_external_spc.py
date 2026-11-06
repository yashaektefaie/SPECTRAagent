#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "spectra_ready_dataset" / "candidate_features.csv"
RESULTS = ROOT / "target_model_results.scaled_samples.csv"


def best_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ok = df[(df["status"].eq("ok")) & (df["sample_index"] >= 0)].copy()
    ok = ok.sort_values(["target_id", col, "sample_index"], ascending=[True, False, True])
    return ok.groupby("target_id", as_index=False).head(1)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = 7, reps: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"mean": float(np.nan), "ci_low": float(np.nan), "ci_high": float(np.nan)}
    vals = []
    for _ in range(reps):
        vals.append(float(rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return x * 0
    return (x - x.mean()) / sd


def regression_coef(df: pd.DataFrame) -> dict[str, float]:
    sub = df[["ca_lddt", "flexible_nonhelical_composite_rank", "seq_len", "resolution"]].dropna().copy()
    if len(sub) < 10:
        return {"n": int(len(sub)), "axis_coef": float("nan")}
    y = sub["ca_lddt"].to_numpy(float)
    x = pd.DataFrame(
        {
            "intercept": 1.0,
            "axis_z": zscore(sub["flexible_nonhelical_composite_rank"]),
            "seq_len_z": zscore(sub["seq_len"]),
            "resolution_z": zscore(sub["resolution"]),
        }
    ).to_numpy(float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    resid = y - pred
    dof = max(1, len(y) - x.shape[1])
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    return {
        "n": int(len(sub)),
        "axis_coef": float(coef[1]),
        "axis_se": float(se[1]),
        "axis_t": float(coef[1] / se[1]) if se[1] else float("nan"),
        "seq_len_coef": float(coef[2]),
        "resolution_coef": float(coef[3]),
    }


def summarize(selection_name: str, target: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = features.merge(target, on="target_id", how="left", suffixes=("", "_metric"))
    df = df[df["status"].eq("ok") & df["ca_lddt"].notna()].copy()
    group = (
        df.groupby(["external_axis_level", "external_axis_level_name"])
        .agg(
            n=("target_id", "count"),
            entries=("entry_id", "nunique"),
            rank_mean=("flexible_nonhelical_composite_rank", "mean"),
            rank_min=("flexible_nonhelical_composite_rank", "min"),
            rank_max=("flexible_nonhelical_composite_rank", "max"),
            seq_len_mean=("seq_len", "mean"),
            seq_len_median=("seq_len", "median"),
            resolution_median=("resolution", "median"),
            ca_lddt_mean=("ca_lddt", "mean"),
            ca_lddt_median=("ca_lddt", "median"),
            gdt_ts_mean=("gdt_ts", "mean"),
            ca_rmsd_median=("ca_rmsd", "median"),
            ptm_mean=("ptm", "mean"),
            plddt_mean=("plddt_mean", "mean"),
        )
        .reset_index()
        .sort_values("external_axis_level")
    )
    top = df[df["external_axis_level"].eq(4)]["ca_lddt"].to_numpy(float)
    lower = df[df["external_axis_level"].lt(4)]["ca_lddt"].to_numpy(float)
    low = df[df["external_axis_level"].eq(0)]["ca_lddt"].to_numpy(float)
    rho_level = spearmanr(df["external_axis_level"], df["ca_lddt"], nan_policy="omit")
    rho_rank = spearmanr(df["flexible_nonhelical_composite_rank"], df["ca_lddt"], nan_policy="omit")
    effect = {
        "selection": selection_name,
        "n": int(len(df)),
        "levels": int(group["external_axis_level"].nunique()),
        "top_level_n": int(len(top)),
        "lower_levels_n": int(len(lower)),
        "top_minus_lower_ca_lddt": float(np.nanmean(top) - np.nanmean(lower)),
        "top_minus_low_ca_lddt": float(np.nanmean(top) - np.nanmean(low)),
        "top_minus_lower_gdt_ts": float(
            df[df["external_axis_level"].eq(4)]["gdt_ts"].mean()
            - df[df["external_axis_level"].lt(4)]["gdt_ts"].mean()
        ),
        "top_minus_lower_ca_rmsd_median": float(
            df[df["external_axis_level"].eq(4)]["ca_rmsd"].median()
            - df[df["external_axis_level"].lt(4)]["ca_rmsd"].median()
        ),
        "spearman_level_vs_ca_lddt_rho": float(rho_level.correlation),
        "spearman_level_vs_ca_lddt_p": float(rho_level.pvalue),
        "spearman_rank_vs_ca_lddt_rho": float(rho_rank.correlation),
        "spearman_rank_vs_ca_lddt_p": float(rho_rank.pvalue),
        "bootstrap_top_minus_lower_ca_lddt": bootstrap_diff(top, lower),
        "length_resolution_adjusted_regression": regression_coef(df),
    }
    return group, effect


def main() -> None:
    features = pd.read_csv(FEATURES)
    results = pd.read_csv(RESULTS)
    selections = {
        "best_by_ptm": best_by(results, "ptm"),
        "best_by_plddt": best_by(results, "plddt_mean"),
        "oracle_best_ca_lddt": best_by(results, "ca_lddt"),
        "first_sample": results[(results["sample_index"].eq(0)) & results["status"].eq("ok")].copy(),
    }

    all_groups = []
    effects = []
    for name, target in selections.items():
        group, effect = summarize(name, target, features)
        group.insert(0, "selection", name)
        all_groups.append(group)
        effects.append(effect)

    group_df = pd.concat(all_groups, ignore_index=True)
    group_df.to_csv(ROOT / "expanded_external_spc_by_level.csv", index=False)
    pd.DataFrame(effects).to_json(ROOT / "expanded_external_effects.json", orient="records", indent=2)

    primary = effects[0]
    report = [
        "# Expanded External RCSB Flexible/Non-Helical SPC",
        "",
        "Frozen axis: `seq_intrinsic_flexible_nonhelical_composite_7`, scored as `flexible_nonhelical_composite_rank` against the original CAMEO feature distribution.",
        "",
        "The external panel uses five prospective rank bands and exact Biohub ESMFold2 inference-time scaling (`num_loops=10`, `num_sampling_steps=68`, `num_diffusion_samples=5`). Target-level summaries use best-by-pTM unless noted.",
        "",
        "## Best-by-pTM Level Curve",
        "",
        group_df[group_df["selection"].eq("best_by_ptm")][
            [
                "external_axis_level",
                "external_axis_level_name",
                "n",
                "rank_mean",
                "seq_len_median",
                "ca_lddt_mean",
                "gdt_ts_mean",
                "ca_rmsd_median",
                "ptm_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Primary Effect",
        "",
        f"Top frozen high-tail minus all lower levels CA-lDDT: {primary['top_minus_lower_ca_lddt']:.3f}.",
        f"Top frozen high-tail minus lowest level CA-lDDT: {primary['top_minus_low_ca_lddt']:.3f}.",
        f"Top frozen high-tail minus lower levels GDT-TS: {primary['top_minus_lower_gdt_ts']:.3f}.",
        f"Spearman rank vs CA-lDDT: rho={primary['spearman_rank_vs_ca_lddt_rho']:.3f}, p={primary['spearman_rank_vs_ca_lddt_p']:.3g}.",
        f"Bootstrap top-minus-lower CA-lDDT mean={primary['bootstrap_top_minus_lower_ca_lddt']['mean']:.3f}, 95% CI [{primary['bootstrap_top_minus_lower_ca_lddt']['ci_low']:.3f}, {primary['bootstrap_top_minus_lower_ca_lddt']['ci_high']:.3f}].",
        f"Length/resolution adjusted axis coefficient={primary['length_resolution_adjusted_regression']['axis_coef']:.3f}, t={primary['length_resolution_adjusted_regression']['axis_t']:.2f}.",
        "",
        "## Artifacts",
        "",
        "- `expanded_external_panel_manifest.json`",
        "- `spectra_ready_dataset/candidate_features.csv`",
        "- `target_model_results.scaled_samples.csv`",
        "- `expanded_external_spc_by_level.csv`",
        "- `expanded_external_effects.json`",
    ]
    (ROOT / "expanded_external_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(primary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
