#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_envelope_fusion_axis_confirmation_20260602")
TARGET_LEVEL = ROOT / "target_model_results.envelope_fusion_confirmation.target_level.csv"
OUT_CANDIDATES = ROOT / "analysis_outputs" / "envelope_subaxis_discovery_candidates.csv"
OUT_REPORT = ROOT / "analysis_outputs" / "envelope_subaxis_discovery_report.md"
OUT_FEATURED = ROOT / "analysis_outputs" / "envelope_subaxis_feature_table.csv"


def norm_text(*values: object) -> str:
    return " ".join("" if pd.isna(v) else str(v).lower() for v in values)


def contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def finite_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def finite_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else float("nan")


def bootstrap_diff(df: pd.DataFrame, mask: pd.Series, metric: str, rng_seed: int = 1, n_boot: int = 5000) -> dict:
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = [norm_text(a, b, c, d) for a, b, c, d in zip(out["organism"], out["entity_description"], out["title"], out["keywords"])]
    text = out["text"]
    out["surface_keyword"] = text.map(lambda s: contains(s, r"\b(envelope|spike|glycoprotein|fusion|gp1|gp2|gp350|protein e)\b"))
    out["capsid_or_enzyme_keyword"] = text.map(lambda s: contains(s, r"\b(capsid|protease|helicase|polymerase|nsp5|ns2b|ns3)\b"))
    out["virion_vlp_context"] = text.map(lambda s: contains(s, r"\b(virion|viron|vlp|virus-like|virus like|particle|inactivated|vaccine|asymmetric unit|prme|spike)\b"))
    out["alphavirus_keyword"] = text.map(lambda s: contains(s, r"\b(semliki|equine encephalitis|alphavirus)\b"))
    out["flavivirus_keyword"] = text.map(lambda s: contains(s, r"\b(tick-borne|yellow fever|japanese encephalitis|dengue|zika|langat|deer tick|orthoflavivirus|flavivirus|powassan)\b"))
    out["class_ii_virus_keyword"] = out["alphavirus_keyword"] | out["flavivirus_keyword"]
    out["class_ii_surface_name"] = text.map(
        lambda s: contains(
            s,
            r"\b(envelope protein e|envelope protein|e glycoprotein|spike glycoprotein e1|spike glycoprotein e2|envelope glycoprotein e2|structural polyprotein|genome polyprotein)\b",
        )
    )
    out["e2_or_flavi_e_name"] = text.map(lambda s: contains(s, r"\b(e2|envelope protein e|envelope protein|e glycoprotein)\b"))
    out["fusion_f_or_s_name"] = text.map(lambda s: contains(s, r"\b(fusion glycoprotein f1|fusion glycoprotein f2|spike protein s1|spike protein s2|spike protein s2')\b"))
    out["gp1_gp2_g_name"] = text.map(lambda s: contains(s, r"\b(gp1|gp2|glycoprotein g|gp350)\b"))
    out["disulfide_dense"] = pd.to_numeric(out["disulfide_capacity_per100"], errors="coerce").ge(1.2)
    out["low_helical_surface_proxy"] = pd.to_numeric(out["tm_helix_proxy_calc"], errors="coerce").lt(2.55)
    out["very_high_hydrophobic_window"] = pd.to_numeric(out["hydrophobic_window21_max_frac_calc"], errors="coerce").ge(0.66)
    out["few_nglyc"] = pd.to_numeric(out["nglyc_motif_count"], errors="coerce").le(3)
    out["long_chain"] = pd.to_numeric(out["seq_len"], errors="coerce").ge(350)
    out["classII_beta_surface_seq"] = (
        out["surface_keyword"]
        & out["class_ii_virus_keyword"]
        & out["disulfide_dense"]
        & out["low_helical_surface_proxy"]
        & out["few_nglyc"]
        & ~out["capsid_or_enzyme_keyword"]
    )
    out["classII_E_E2_keyword"] = out["class_ii_virus_keyword"] & out["class_ii_surface_name"] & ~out["capsid_or_enzyme_keyword"]
    out["classII_E_E2_lowhelical"] = out["classII_E_E2_keyword"] & out["low_helical_surface_proxy"]
    out["classII_E_E2_lowhelical_assembly"] = out["classII_E_E2_lowhelical"] & out["virion_vlp_context"]
    out["classII_E_E2_lowhelical_hydrophobic"] = out["classII_E_E2_lowhelical"] & out["very_high_hydrophobic_window"]
    out["surface_not_capsid_or_enzyme"] = out["surface_keyword"] & ~out["capsid_or_enzyme_keyword"]
    out["surface_disulfide_lowhelical"] = out["surface_keyword"] & out["disulfide_dense"] & out["low_helical_surface_proxy"] & ~out["capsid_or_enzyme_keyword"]
    return out


def summarize_axis(df: pd.DataFrame, axis: str) -> dict:
    mask = df[axis].fillna(False).astype(bool)
    top = df[mask]
    rest = df[~mask]
    if len(top) == 0 or len(rest) == 0:
        return {"axis": axis, "n_top": int(len(top)), "n_rest": int(len(rest)), "usable": False}
    ca_diff = finite_mean(top["ca_lddt"]) - finite_mean(rest["ca_lddt"])
    gdt_diff = finite_mean(top["gdt_ts"]) - finite_mean(rest["gdt_ts"])
    rmsd_diff = finite_median(top["ca_rmsd"]) - finite_median(rest["ca_rmsd"])
    baseline_diff = finite_mean(top["baseline_ca_lddt"]) - finite_mean(rest["baseline_ca_lddt"])
    ci = bootstrap_diff(df, mask, "ca_lddt")
    return {
        "axis": axis,
        "usable": True,
        "n_top": int(mask.sum()),
        "n_rest": int((~mask).sum()),
        "top_ca_lddt_mean": finite_mean(top["ca_lddt"]),
        "rest_ca_lddt_mean": finite_mean(rest["ca_lddt"]),
        "top_minus_rest_ca_lddt": ca_diff,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "top_minus_rest_gdt_ts": gdt_diff,
        "top_minus_rest_ca_rmsd_median": rmsd_diff,
        "top_minus_rest_baseline_ca_lddt": baseline_diff,
        "top_seq_len_median": finite_median(top["seq_len"]),
        "rest_seq_len_median": finite_median(rest["seq_len"]),
        "top_examples": "; ".join(top.sort_values("ca_lddt").head(5)["target_id"].tolist()),
        "best_top_examples": "; ".join(top.sort_values("ca_lddt", ascending=False).head(5)["target_id"].tolist()),
    }


def main() -> None:
    df = pd.read_csv(TARGET_LEVEL)
    df = df[df["selection"].eq("best_by_ptm")].copy()
    for col in ["ca_lddt", "gdt_ts", "ca_rmsd", "baseline_ca_lddt", "seq_len"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_features(df)
    axes = [
        "surface_keyword",
        "surface_not_capsid_or_enzyme",
        "virion_vlp_context",
        "class_ii_virus_keyword",
        "class_ii_surface_name",
        "e2_or_flavi_e_name",
        "fusion_f_or_s_name",
        "gp1_gp2_g_name",
        "classII_E_E2_keyword",
        "classII_E_E2_lowhelical",
        "classII_E_E2_lowhelical_assembly",
        "classII_E_E2_lowhelical_hydrophobic",
        "classII_beta_surface_seq",
        "surface_disulfide_lowhelical",
        "very_high_hydrophobic_window",
        "long_chain",
    ]
    results = [summarize_axis(df, axis) for axis in axes]
    res = pd.DataFrame(results).sort_values(["top_minus_rest_ca_lddt", "n_top"], ascending=[True, False])
    OUT_CANDIDATES.parent.mkdir(exist_ok=True)
    res.to_csv(OUT_CANDIDATES, index=False)
    df.drop(columns=["text"]).to_csv(OUT_FEATURED, index=False)

    lines = [
        "# Envelope Subaxis Discovery",
        "",
        "This is outcome-informed discovery, not claim closure. Any promoted axis must be frozen and confirmed on new or held-out evidence.",
        "",
        "## Candidate Axes",
        "",
        "| axis | n top | CA-lDDT diff | CI low | CI high | GDT diff | RMSD median diff | baseline diff | worst top examples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _idx, row in res.iterrows():
        if not bool(row.get("usable", False)):
            continue
        lines.append(
            f"| {row['axis']} | {int(row['n_top'])} | {row['top_minus_rest_ca_lddt']:+.4f} | {row['ci_low']:+.4f} | {row['ci_high']:+.4f} | {row['top_minus_rest_gdt_ts']:+.4f} | {row['top_minus_rest_ca_rmsd_median']:+.2f} | {row['top_minus_rest_baseline_ca_lddt']:+.4f} | {row['top_examples']} |"
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(res.head(8).to_dict(orient="records"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
