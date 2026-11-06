#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_envelope_fusion_axis_confirmation_20260602")
FEATURES = ROOT / "spectra_ready_dataset" / "surface_lowhelical_confirmation_candidate_features.csv"
SAMPLES = ROOT / "target_model_results.surface_lowhelical_confirmation.scaled_samples.csv"
TARGET_LEVEL = ROOT / "target_model_results.surface_lowhelical_confirmation.target_level.csv"
CURVES = ROOT / "analysis_outputs" / "surface_lowhelical_confirmation_level_curves.csv"
EFFECTS = ROOT / "analysis_outputs" / "surface_lowhelical_confirmation_effects.json"
REPORT = ROOT / "surface_lowhelical_confirmation_report.md"


def finite_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def finite_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else float("nan")


def select_rows(samples: pd.DataFrame, policy: str) -> pd.DataFrame:
    ok = samples[samples["status"].eq("ok")].copy()
    for col in ["ca_lddt", "ptm", "plddt_ca_mean", "sample_index"]:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    selected = []
    for _target, part in ok.groupby("target_id", sort=False):
        finite = part[part["ca_lddt"].notna()].copy()
        if finite.empty:
            selected.append(part.iloc[0])
            continue
        if policy == "best_by_ptm":
            selected.append(finite.sort_values(["ptm", "ca_lddt"], ascending=[False, False]).iloc[0])
        elif policy == "best_by_plddt":
            selected.append(finite.sort_values(["plddt_ca_mean", "ca_lddt"], ascending=[False, False]).iloc[0])
        elif policy == "oracle_best_ca_lddt":
            selected.append(finite.sort_values(["ca_lddt"], ascending=False).iloc[0])
        elif policy == "first_sample":
            selected.append(finite.sort_values(["sample_index"]).iloc[0])
        else:
            raise ValueError(policy)
    out = pd.DataFrame(selected)
    out["selection"] = policy
    return out


def bootstrap_diff(df: pd.DataFrame, mask: pd.Series, metric: str = "ca_lddt", rng_seed: int = 0, n_boot: int = 5000) -> dict:
    rng = np.random.default_rng(rng_seed)
    top = pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna().to_numpy(float)
    rest = pd.to_numeric(df.loc[~mask, metric], errors="coerce").dropna().to_numpy(float)
    if len(top) < 2 or len(rest) < 2:
        return {"n_top": int(len(top)), "n_rest": int(len(rest)), "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    vals = [
        float(rng.choice(top, len(top), replace=True).mean() - rng.choice(rest, len(rest), replace=True).mean())
        for _ in range(n_boot)
    ]
    arr = np.asarray(vals)
    return {
        "n_top": int(len(top)),
        "n_rest": int(len(rest)),
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def add_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = (
        out["organism"].fillna("").astype(str)
        + " "
        + out["entity_description"].fillna("").astype(str)
        + " "
        + out["title"].fillna("").astype(str)
        + " "
        + out["keywords"].fillna("").astype(str)
    ).str.lower()
    out["subtype_e2_e"] = text.str.contains(r"\b(e2|envelope protein e|envelope protein|protein e)\b", regex=True)
    out["subtype_e1_g"] = text.str.contains(r"\b(e1|glycoprotein g|attachment glycoprotein|fusion glycoprotein)\b", regex=True)
    out["subtype_gp1"] = text.str.contains(r"\bgp1\b", regex=True)
    out["subtype_gp2"] = text.str.contains(r"\bgp2\b", regex=True)
    out["subtype_classII_virus"] = text.str.contains(r"\b(chikungunya|semliki|yellow fever|zika|dengue|alphavirus|flavivirus)\b", regex=True)
    out["subtype_classII_e_or_e2"] = out["subtype_e2_e"] & out["subtype_classII_virus"]
    return out


def summarize_policy(features: pd.DataFrame, samples: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    chosen = select_rows(samples, policy)
    merged = features.merge(chosen, on="target_id", how="left", suffixes=("", "_result"))
    merged = add_subtypes(merged)
    for col in [
        "surface_lowhelical_score",
        "ca_lddt",
        "gdt_ts",
        "ca_rmsd",
        "baseline_ca_lddt",
        "baseline_gdt_ts",
        "baseline_ca_rmsd",
        "seq_len",
        "resolution",
        "ptm",
        "surface_lowhelical_disulfide_dense",
        "surface_lowhelical_low_tm_proxy",
        "tm_helix_proxy_calc",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    levels = []
    for level, part in merged.groupby("surface_lowhelical_score"):
        levels.append(
            {
                "selection": policy,
                "surface_lowhelical_score": int(level),
                "n": int(part["ca_lddt"].notna().sum()),
                "panel_n": int(len(part)),
                "seq_len_median": finite_median(part["seq_len"]),
                "resolution_median": finite_median(part["resolution"]),
                "ca_lddt_mean": finite_mean(part["ca_lddt"]),
                "gdt_ts_mean": finite_mean(part["gdt_ts"]),
                "ca_rmsd_median": finite_median(part["ca_rmsd"]),
                "baseline_ca_lddt_mean": finite_mean(part["baseline_ca_lddt"]),
                "ptm_mean": finite_mean(part["ptm"]),
                "classII_e_or_e2_n": int(part["subtype_classII_e_or_e2"].sum()),
                "e1_g_n": int(part["subtype_e1_g"].sum()),
                "gp1_n": int(part["subtype_gp1"].sum()),
                "gp2_n": int(part["subtype_gp2"].sum()),
            }
        )
    score = pd.to_numeric(merged["surface_lowhelical_score"], errors="coerce")
    ca = pd.to_numeric(merged["ca_lddt"], errors="coerce")
    valid = score.notna() & ca.notna()
    rho, p = spearmanr(score[valid], ca[valid]) if valid.sum() > 2 else (np.nan, np.nan)
    top = merged["surface_lowhelical_score"].eq(2)
    classII = merged["subtype_classII_e_or_e2"]
    effect = {
        "selection": policy,
        "n_panel": int(len(merged)),
        "n_finite": int(merged["ca_lddt"].notna().sum()),
        "top_minus_lower_ca_lddt": finite_mean(merged.loc[top, "ca_lddt"]) - finite_mean(merged.loc[~top, "ca_lddt"]),
        "top_minus_level0_ca_lddt": finite_mean(merged.loc[top, "ca_lddt"]) - finite_mean(merged.loc[merged["surface_lowhelical_score"].eq(0), "ca_lddt"]),
        "top_minus_lower_gdt_ts": finite_mean(merged.loc[top, "gdt_ts"]) - finite_mean(merged.loc[~top, "gdt_ts"]),
        "top_minus_lower_ca_rmsd_median": finite_median(merged.loc[top, "ca_rmsd"]) - finite_median(merged.loc[~top, "ca_rmsd"]),
        "top_minus_lower_baseline_ca_lddt": finite_mean(merged.loc[top, "baseline_ca_lddt"]) - finite_mean(merged.loc[~top, "baseline_ca_lddt"]),
        "bootstrap_top_minus_lower_ca_lddt": bootstrap_diff(merged, top),
        "spearman_score_vs_ca_lddt": {"rho": float(rho), "p": float(p)},
        "classII_e_or_e2_minus_rest_ca_lddt": finite_mean(merged.loc[classII, "ca_lddt"]) - finite_mean(merged.loc[~classII, "ca_lddt"]),
        "classII_e_or_e2_bootstrap_ca_lddt": bootstrap_diff(merged, classII),
        "e1_g_minus_e2_classII_ca_lddt": finite_mean(merged.loc[merged["subtype_e1_g"], "ca_lddt"]) - finite_mean(merged.loc[classII, "ca_lddt"]),
    }
    return pd.DataFrame(levels), effect, merged


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def main() -> None:
    (ROOT / "analysis_outputs").mkdir(exist_ok=True)
    features = pd.read_csv(FEATURES)
    samples = pd.read_csv(SAMPLES)
    curves = []
    effects = []
    targets = []
    primary_merged = None
    for policy in ["best_by_ptm", "best_by_plddt", "oracle_best_ca_lddt", "first_sample"]:
        curve, effect, merged = summarize_policy(features, samples, policy)
        curves.append(curve)
        effects.append(effect)
        targets.append(merged.assign(selection=policy))
        if policy == "best_by_ptm":
            primary_merged = merged
    pd.concat(curves, ignore_index=True).to_csv(CURVES, index=False)
    pd.concat(targets, ignore_index=True).to_csv(TARGET_LEVEL, index=False)
    EFFECTS.write_text(json.dumps(effects, indent=2, sort_keys=True) + "\n")

    assert primary_merged is not None
    primary = effects[0]
    primary_curve = curves[0]
    ci = primary["bootstrap_top_minus_lower_ca_lddt"]
    ci2 = primary["classII_e_or_e2_bootstrap_ca_lddt"]
    worst = primary_merged.sort_values("ca_lddt").head(12)
    lines = [
        "# Surface-Lowhelical Confirmation Result",
        "",
        "Frozen axis: viral surface, non-capsid/non-enzyme proteins with disulfide density and lower helical/TM proxy.",
        "",
        "- Level 0: viral non-surface, capsid, or enzyme controls",
        "- Level 1: viral surface proteins without both disulfide density and low helical/TM proxy",
        "- Level 2: viral surface proteins with disulfide_capacity_per100 >= 1.2 and tm_helix_proxy_calc < 2.55",
        "",
        "## Best-by-pTM Level Curve",
        "",
        "| Level | n | CA-lDDT | GDT-TS | median CA-RMSD | baseline CA-lDDT | median length | classII E/E2 n | E1/G n | GP1 n | GP2 n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _idx, row in primary_curve.iterrows():
        lines.append(
            f"| {int(row['surface_lowhelical_score'])} | {int(row['n'])} | {fmt(row['ca_lddt_mean'])} | {fmt(row['gdt_ts_mean'])} | {fmt(row['ca_rmsd_median'])} | {fmt(row['baseline_ca_lddt_mean'])} | {fmt(row['seq_len_median'], 1)} | {int(row['classII_e_or_e2_n'])} | {int(row['e1_g_n'])} | {int(row['gp1_n'])} | {int(row['gp2_n'])} |"
        )
    lines += [
        "",
        "## Primary Effects",
        "",
        f"- Level 2 vs lower CA-lDDT: {primary['top_minus_lower_ca_lddt']:+.4f}; bootstrap 95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]",
        f"- Level 2 vs level 0 CA-lDDT: {primary['top_minus_level0_ca_lddt']:+.4f}",
        f"- Level 2 vs lower GDT-TS: {primary['top_minus_lower_gdt_ts']:+.4f}",
        f"- Level 2 vs lower median CA-RMSD: {primary['top_minus_lower_ca_rmsd_median']:+.2f} A",
        f"- Baseline CA-lDDT movement: {primary['top_minus_lower_baseline_ca_lddt']:+.4f}",
        f"- Spearman level vs CA-lDDT: rho={primary['spearman_score_vs_ca_lddt']['rho']:+.4f}, p={primary['spearman_score_vs_ca_lddt']['p']:.4g}",
        "",
        "## Sharper Subtype Contrast",
        "",
        f"- Class-II-like E/E2/envelope-E subtype vs rest CA-lDDT: {primary['classII_e_or_e2_minus_rest_ca_lddt']:+.4f}; bootstrap 95% CI [{ci2['ci_low']:+.4f}, {ci2['ci_high']:+.4f}]",
        f"- E1/G/fusion subtype mean minus class-II E/E2 subtype mean: {primary['e1_g_minus_e2_classII_ca_lddt']:+.4f}",
        "",
        "## Worst Best-by-pTM Targets",
        "",
        "| target | level | CA-lDDT | GDT-TS | RMSD | length | organism | description |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _idx, row in worst.iterrows():
        lines.append(
            f"| {row['target_id']} | {int(row['surface_lowhelical_score'])} | {fmt(row['ca_lddt'])} | {fmt(row['gdt_ts'])} | {fmt(row['ca_rmsd'], 2)} | {int(row['seq_len'])} | {row['organism']} | {row['entity_description']} |"
        )
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(primary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
