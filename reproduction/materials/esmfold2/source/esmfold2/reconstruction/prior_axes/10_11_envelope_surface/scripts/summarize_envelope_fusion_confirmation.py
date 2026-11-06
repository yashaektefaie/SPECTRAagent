#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_envelope_fusion_axis_confirmation_20260602")
FEATURES = ROOT / "spectra_ready_dataset" / "envelope_fusion_confirmation_candidate_features.csv"
SAMPLES = ROOT / "target_model_results.envelope_fusion_confirmation.scaled_samples.csv"
TARGET_LEVEL = ROOT / "target_model_results.envelope_fusion_confirmation.target_level.csv"
CURVES = ROOT / "analysis_outputs" / "envelope_fusion_confirmation_level_curves.csv"
EFFECTS = ROOT / "analysis_outputs" / "envelope_fusion_confirmation_effects.json"
REPORT = ROOT / "envelope_fusion_confirmation_report.md"


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


def bootstrap_top_minus_lower(df: pd.DataFrame, rng_seed: int = 0, n_boot: int = 5000) -> dict:
    rng = np.random.default_rng(rng_seed)
    top = df[df["envelope_fusion_score"].eq(2)]["ca_lddt"].dropna().to_numpy(float)
    lower = df[~df["envelope_fusion_score"].eq(2)]["ca_lddt"].dropna().to_numpy(float)
    if len(top) < 2 or len(lower) < 2:
        return {
            "n_top": int(len(top)),
            "n_lower": int(len(lower)),
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    draws = [
        float(rng.choice(top, len(top), replace=True).mean() - rng.choice(lower, len(lower), replace=True).mean())
        for _ in range(n_boot)
    ]
    arr = np.asarray(draws)
    return {
        "n_top": int(len(top)),
        "n_lower": int(len(lower)),
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def ols_axis_coef(df: pd.DataFrame, controls: list[str]) -> dict:
    work = df.copy()
    work["axis_top"] = work["envelope_fusion_score"].eq(2).astype(float)
    work["log_len"] = np.log(pd.to_numeric(work["seq_len"], errors="coerce").clip(lower=1))
    work["method_em"] = work["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).astype(float)
    y = pd.to_numeric(work["ca_lddt"], errors="coerce")
    xcols = ["axis_top"] + controls
    for col in xcols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    mat = work[xcols]
    mask = y.notna() & mat.notna().all(axis=1)
    yv = y[mask].to_numpy(float)
    xv = np.column_stack([np.ones(mask.sum()), mat[mask].to_numpy(float)])
    if len(yv) <= xv.shape[1] + 1:
        return {"n": int(len(yv)), "coef": float("nan"), "se": float("nan"), "t": float("nan"), "controls": controls}
    beta = np.linalg.lstsq(xv, yv, rcond=None)[0]
    resid = yv - xv @ beta
    dof = len(yv) - xv.shape[1]
    try:
        cov = (resid @ resid / dof) * np.linalg.inv(xv.T @ xv)
        se = float(np.sqrt(np.diag(cov))[1])
        t = float(beta[1] / se) if se else float("nan")
    except Exception:
        se = float("nan")
        t = float("nan")
    return {"n": int(len(yv)), "coef": float(beta[1]), "se": se, "t": t, "controls": controls}


def summarize_selection(features: pd.DataFrame, samples: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    chosen = select_rows(samples, policy)
    merged = features.merge(chosen, on="target_id", how="left", suffixes=("", "_result"))
    for col in [
        "envelope_fusion_score",
        "ca_lddt",
        "gdt_ts",
        "ca_rmsd",
        "baseline_ca_lddt",
        "baseline_gdt_ts",
        "baseline_ca_rmsd",
        "seq_len",
        "resolution",
        "disulfide_capacity_per100",
        "tm_helix_proxy_calc",
        "nglyc_motif_count",
        "ptm",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["method_em"] = merged["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).astype(int)
    levels = []
    for level, part in merged.groupby("envelope_fusion_score"):
        levels.append(
            {
                "selection": policy,
                "envelope_fusion_score": int(level),
                "n": int(part["ca_lddt"].notna().sum()),
                "panel_n": int(len(part)),
                "strict_proxy_n": int(part["strict_envelope_fusion_sequence_proxy"].astype(str).str.lower().isin(["true", "1"]).sum()),
                "seq_len_median": finite_median(part["seq_len"]),
                "resolution_median": finite_median(part["resolution"]),
                "em_n": int(part["method_em"].sum()),
                "cys_fraction_mean": finite_mean(part["cys_fraction"]),
                "disulfide_capacity_per100_mean": finite_mean(part["disulfide_capacity_per100"]),
                "tm_helix_proxy_mean": finite_mean(part["tm_helix_proxy_calc"]),
                "nglyc_motif_mean": finite_mean(part["nglyc_motif_count"]),
                "ca_lddt_mean": finite_mean(part["ca_lddt"]),
                "gdt_ts_mean": finite_mean(part["gdt_ts"]),
                "ca_rmsd_median": finite_median(part["ca_rmsd"]),
                "baseline_ca_lddt_mean": finite_mean(part["baseline_ca_lddt"]),
                "ptm_mean": finite_mean(part["ptm"]),
            }
        )
    top = merged[merged["envelope_fusion_score"].eq(2)]
    lower = merged[~merged["envelope_fusion_score"].eq(2)]
    score = pd.to_numeric(merged["envelope_fusion_score"], errors="coerce")
    ca = pd.to_numeric(merged["ca_lddt"], errors="coerce")
    valid = score.notna() & ca.notna()
    rho, p = spearmanr(score[valid], ca[valid]) if valid.sum() > 2 else (np.nan, np.nan)
    effect = {
        "selection": policy,
        "n_panel": int(len(merged)),
        "n_finite": int(merged["ca_lddt"].notna().sum()),
        "n_top": int(top["ca_lddt"].notna().sum()),
        "n_lower": int(lower["ca_lddt"].notna().sum()),
        "top_minus_lower_ca_lddt": finite_mean(top["ca_lddt"]) - finite_mean(lower["ca_lddt"]),
        "top_minus_level0_ca_lddt": finite_mean(top["ca_lddt"]) - finite_mean(merged[merged["envelope_fusion_score"].eq(0)]["ca_lddt"]),
        "top_minus_lower_gdt_ts": finite_mean(top["gdt_ts"]) - finite_mean(lower["gdt_ts"]),
        "top_minus_lower_ca_rmsd_median": finite_median(top["ca_rmsd"]) - finite_median(lower["ca_rmsd"]),
        "top_minus_lower_baseline_ca_lddt": finite_mean(top["baseline_ca_lddt"]) - finite_mean(lower["baseline_ca_lddt"]),
        "spearman_score_vs_ca_lddt": {"rho": float(rho), "p": float(p)},
        "bootstrap_top_minus_lower_ca_lddt": bootstrap_top_minus_lower(merged),
        "adjusted_length_resolution_method": ols_axis_coef(merged, ["log_len", "resolution", "method_em"]),
        "adjusted_length_only": ols_axis_coef(merged, ["log_len"]),
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
    target_tables = []
    curves = []
    effects = []
    primary_merged = None
    for policy in ["best_by_ptm", "best_by_plddt", "oracle_best_ca_lddt", "first_sample"]:
        curve, effect, merged = summarize_selection(features, samples, policy)
        curves.append(curve)
        effects.append(effect)
        target_tables.append(merged.assign(selection=policy))
        if policy == "best_by_ptm":
            primary_merged = merged
    pd.concat(target_tables, ignore_index=True).to_csv(TARGET_LEVEL, index=False)
    pd.concat(curves, ignore_index=True).to_csv(CURVES, index=False)
    EFFECTS.write_text(json.dumps(effects, indent=2, sort_keys=True) + "\n")

    assert primary_merged is not None
    primary = effects[0]
    primary_curve = curves[0]
    worst = primary_merged.sort_values("ca_lddt", ascending=True).head(10)
    ci = primary["bootstrap_top_minus_lower_ca_lddt"]
    adj = primary["adjusted_length_resolution_method"]
    closure = "exploratory"
    if (
        primary["n_top"] >= 12
        and primary["top_minus_lower_ca_lddt"] < -0.08
        and ci["ci_high"] < 0
        and abs(primary["top_minus_lower_baseline_ca_lddt"]) < 0.03
    ):
        closure = "claim-valid for this frozen proxy axis"
    elif primary["top_minus_lower_ca_lddt"] < -0.05:
        closure = "weak or exploratory"
    else:
        closure = "not supported"

    lines = [
        "# Envelope/Fusion Confirmation Result",
        "",
        "Frozen axis: `envelope_fusion_score = disulfide_dense + membrane_lowglyc_proxy`.",
        "",
        "- `disulfide_dense`: floor(cysteine_count/2) per 100 aa >= 1.2",
        "- `membrane_lowglyc_proxy`: `tm_helix_proxy_calc >= 1.8` and N-glycosylation motif count <= 2",
        "- Level 0: neither condition; level 1: one condition; level 2: both conditions",
        "- Scope: viral/envelope RCSB polymer entities, release year >= 2024, excluding prior scored target IDs and exact sequences",
        "",
        f"Auditor status: **{closure}**.",
        "",
        "## Best-by-pTM Level Curve",
        "",
        "| Level | n finite | CA-lDDT | GDT-TS | median CA-RMSD | baseline CA-lDDT | median length | EM n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _idx, row in primary_curve.iterrows():
        lines.append(
            f"| {int(row['envelope_fusion_score'])} | {int(row['n'])} | {fmt(row['ca_lddt_mean'])} | {fmt(row['gdt_ts_mean'])} | {fmt(row['ca_rmsd_median'])} | {fmt(row['baseline_ca_lddt_mean'])} | {fmt(row['seq_len_median'], 1)} | {int(row['em_n'])} |"
        )
    lines += [
        "",
        "## Primary Effect",
        "",
        f"- Top level vs lower CA-lDDT: {primary['top_minus_lower_ca_lddt']:+.4f}",
        f"- Top level vs level 0 CA-lDDT: {primary['top_minus_level0_ca_lddt']:+.4f}",
        f"- Bootstrap 95% CI: [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]",
        f"- Top level vs lower GDT-TS: {primary['top_minus_lower_gdt_ts']:+.4f}",
        f"- Top level vs lower median CA-RMSD: {primary['top_minus_lower_ca_rmsd_median']:+.4f} A",
        f"- Baseline CA-lDDT movement: {primary['top_minus_lower_baseline_ca_lddt']:+.4f}",
        f"- Spearman score vs CA-lDDT: rho={primary['spearman_score_vs_ca_lddt']['rho']:+.4f}, p={primary['spearman_score_vs_ca_lddt']['p']:.4g}",
        f"- Adjusted top-level coefficient after log length, resolution, EM method: {adj['coef']:+.4f} (t={adj['t']:+.2f}, n={adj['n']})",
        "",
        "## Worst Best-by-pTM Targets",
        "",
        "| target | level | CA-lDDT | GDT-TS | RMSD | length | organism | description |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _idx, row in worst.iterrows():
        lines.append(
            f"| {row['target_id']} | {int(row['envelope_fusion_score'])} | {fmt(row['ca_lddt'])} | {fmt(row['gdt_ts'])} | {fmt(row['ca_rmsd'], 2)} | {int(row['seq_len'])} | {row['organism']} | {row['entity_description']} |"
        )
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(primary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
