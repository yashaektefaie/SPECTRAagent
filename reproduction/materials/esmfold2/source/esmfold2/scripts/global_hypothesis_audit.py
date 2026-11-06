#!/usr/bin/env python3
"""Aggregate saved ESMFold2 results and recheck SPECTRA hypotheses globally.

This does not run ESMFold2. It normalizes existing target-level outputs,
computes prospective sequence/metadata features, and measures whether prior
hypotheses explain CA-lDDT degradation over all available scored evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ewsc/yektefai/spectra_runs")
OUT = Path("/ewsc/yektefai/spectra_runs/esmfold2_global_hypothesis_audit_20260602")
DATA_OUT = OUT / "spectra_ready_dataset"
ANALYSIS_OUT = OUT / "analysis_outputs"
FEATURE_CACHE: dict[Path, pd.DataFrame | None] = {}


AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWYV")
ACIDIC = set("DE")
BASIC = set("KRH")
CHARGED = ACIDIC | BASIC
DISORDER_PROMOTING = set("ARGQSEKP")
ORDER_PROMOTING = set("CWYFILV")

KD = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

CHOU_HELIX = {
    "A": 1.45,
    "C": 0.77,
    "D": 0.98,
    "E": 1.53,
    "F": 1.12,
    "G": 0.53,
    "H": 1.24,
    "I": 1.00,
    "K": 1.07,
    "L": 1.34,
    "M": 1.20,
    "N": 0.73,
    "P": 0.59,
    "Q": 1.17,
    "R": 0.79,
    "S": 0.79,
    "T": 0.82,
    "V": 1.14,
    "W": 1.14,
    "Y": 0.61,
}
CHOU_BETA = {
    "A": 0.97,
    "C": 1.30,
    "D": 0.80,
    "E": 0.26,
    "F": 1.28,
    "G": 0.81,
    "H": 0.71,
    "I": 1.60,
    "K": 0.74,
    "L": 1.22,
    "M": 1.67,
    "N": 0.65,
    "P": 0.62,
    "Q": 1.23,
    "R": 0.90,
    "S": 0.72,
    "T": 1.20,
    "V": 1.65,
    "W": 1.19,
    "Y": 1.29,
}
CHOU_TURN = {
    "A": 0.66,
    "C": 1.19,
    "D": 1.46,
    "E": 0.74,
    "F": 0.60,
    "G": 1.56,
    "H": 0.95,
    "I": 0.47,
    "K": 1.01,
    "L": 0.59,
    "M": 0.60,
    "N": 1.56,
    "P": 1.52,
    "Q": 0.98,
    "R": 0.95,
    "S": 1.43,
    "T": 0.96,
    "V": 0.50,
    "W": 0.96,
    "Y": 1.14,
}


def safe_float(v) -> float:
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def clean_seq(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return re.sub(r"[^A-Z]", "", str(v).upper())


def hash_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def frac(seq: str, chars: set[str]) -> float:
    if not seq:
        return np.nan
    return sum(1 for c in seq if c in chars) / len(seq)


def mean_prop(seq: str, table: dict[str, float]) -> float:
    vals = [table[c] for c in seq if c in table]
    return float(np.mean(vals)) if vals else np.nan


def max_window_fraction(seq: str, chars: set[str] | None = None, width: int = 21, mode: str = "set") -> float:
    if not seq:
        return np.nan
    if len(seq) < width:
        window = seq
        if chars is None:
            return Counter(window).most_common(1)[0][1] / len(window)
        return sum(1 for c in window if c in chars) / len(window)
    vals = []
    for i in range(0, len(seq) - width + 1):
        window = seq[i : i + width]
        if chars is None:
            vals.append(Counter(window).most_common(1)[0][1] / width)
        elif mode == "set":
            vals.append(sum(1 for c in window if c in chars) / width)
    return float(max(vals)) if vals else np.nan


def max_kd_window(seq: str, width: int = 21) -> float:
    vals = [KD[c] for c in seq if c in KD]
    if not vals:
        return np.nan
    if len(vals) < width:
        return float(np.mean(vals))
    return float(max(np.mean(vals[i : i + width]) for i in range(len(vals) - width + 1)))


def nglyc_count(seq: str) -> int:
    return len(re.findall(r"N[^P][ST][^P]", seq))


def data_family(path: Path, row: pd.Series) -> str:
    p = str(path).lower()
    source = str(row.get("source", "")).lower()
    panel = str(row.get("result_panel", row.get("feature_panel", ""))).lower()
    if "cameo" in source or "cameo" in panel:
        return "cameo"
    if "envelope_fusion_axis" in p or "viral_disulfide" in p:
        return "viral_rcsb"
    if "external" in p:
        return "external_rcsb"
    if "charge_axis" in p or "all_scored" in p or "all_evidence" in p:
        return "fresh_rcsb_general"
    if "single_controller" in p or "beyond_evolutionary_support" in p:
        return "cameo_or_cameo_followup"
    if "controller_loop" in p or "closuregate" in p or "claimvalid" in p:
        return "early_controller"
    if str(row.get("entry_id", "")) or str(row.get("rcsb_entity_id", "")):
        return "rcsb"
    return "unknown"


def setting_label(row: pd.Series) -> str:
    loops = safe_float(row.get("num_loops", np.nan))
    steps = safe_float(row.get("num_sampling_steps", np.nan))
    samples = safe_float(row.get("num_diffusion_samples", np.nan))
    if np.isfinite(loops) and np.isfinite(steps) and np.isfinite(samples):
        return f"loops{int(loops)}_steps{int(steps)}_samples{int(samples)}"
    s = str(row.get("inference_setting", "")).strip()
    return s if s and s.lower() != "nan" else "unknown_setting"


def budget_score(row: pd.Series) -> float:
    loops = safe_float(row.get("num_loops", np.nan))
    steps = safe_float(row.get("num_sampling_steps", np.nan))
    samples = safe_float(row.get("num_diffusion_samples", np.nan))
    loops = loops if np.isfinite(loops) else 1
    steps = steps if np.isfinite(steps) else 1
    samples = samples if np.isfinite(samples) else 1
    return loops * steps * samples


def result_files() -> list[Path]:
    files = []
    for p in ROOT.rglob("*.csv"):
        lower = str(p).lower()
        if "esmfold2" not in lower:
            continue
        if "target_model_results" not in p.name:
            continue
        if "partial" in p.name.lower():
            continue
        if "/model_outputs/" in lower:
            continue
        files.append(p)
    return sorted(files)


def nearby_feature_files(path: Path) -> list[Path]:
    dataset_dirs = []
    for base in [path.parent, path.parent.parent, path.parent.parent.parent]:
        if base == base.parent:
            continue
        ds = base / "spectra_ready_dataset"
        if ds.exists():
            dataset_dirs.append(ds)
    seen = set()
    files = []
    patterns = [
        "candidate_prospective_features*.csv",
        "candidate_features*.csv",
        "prospective_features.followup.csv",
        "prospective_features.csv",
    ]
    for ds in dataset_dirs:
        for pat in patterns:
            for f in ds.glob(pat):
                if f in seen:
                    continue
                seen.add(f)
                files.append(f)
    return files


def read_feature_file(path: Path) -> pd.DataFrame | None:
    if path in FEATURE_CACHE:
        return FEATURE_CACHE[path]
    try:
        feat = pd.read_csv(path, low_memory=False)
    except Exception:
        FEATURE_CACHE[path] = None
        return None
    if "target_id" not in feat.columns or "sequence" not in feat.columns:
        FEATURE_CACHE[path] = None
        return None
    keep = [
        "target_id",
        "sequence",
        "cameo_target",
        "submission_date",
        "difficulty",
        "quaternary_state",
        "method",
        "resolution",
        "msa_rows",
        "msa_nonquery_rows",
        "msa_log10_nonquery_rows",
        "msa_top_nonquery_identity",
        "source",
        "feature_panel",
        "result_panel",
    ]
    keep = [c for c in keep if c in feat.columns]
    feat = feat[keep].drop_duplicates("target_id", keep="first").copy()
    FEATURE_CACHE[path] = feat
    return feat


def merge_nearby_features(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "target_id" not in df.columns:
        return df
    needs_sequence = "sequence" not in df.columns or df["sequence"].map(clean_seq).str.len().eq(0).all()
    if not needs_sequence:
        return df
    out = df.copy()
    for feat_path in nearby_feature_files(path):
        feat = read_feature_file(feat_path)
        if feat is None:
            continue
        before = out["sequence"].map(clean_seq).str.len().gt(0).sum() if "sequence" in out.columns else 0
        merged = out.merge(feat, on="target_id", how="left", suffixes=("", "_feat"))
        for col in feat.columns:
            if col == "target_id":
                continue
            feat_col = col if col not in out.columns else f"{col}_feat"
            if feat_col not in merged.columns:
                continue
            if col not in merged.columns:
                merged[col] = merged[feat_col]
            else:
                missing = merged[col].isna() | merged[col].astype(str).str.lower().isin(["", "nan", "none"])
                merged.loc[missing, col] = merged.loc[missing, feat_col]
        merged = merged[[c for c in merged.columns if not c.endswith("_feat")]].copy()
        out = merged
        after = out["sequence"].map(clean_seq).str.len().gt(0).sum() if "sequence" in out.columns else 0
        if after > before:
            break
    return out


def normalize_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"SKIP read error {path}: {exc}")
        return None
    if "ca_lddt" not in df.columns:
        return None
    if "selection" in df.columns:
        sel = df["selection"].fillna("best_by_ptm").astype(str)
        if (sel == "best_by_ptm").any():
            df = df[sel == "best_by_ptm"].copy()
    if "sequence" not in df.columns and "sequence_upper" in df.columns:
        df["sequence"] = df["sequence_upper"]
    if "target_id" not in df.columns:
        for col in ["cameo_target", "rcsb_instance_id", "entry_id"]:
            if col in df.columns:
                df["target_id"] = df[col].astype(str)
                break
    if "target_id" not in df.columns:
        df["target_id"] = [f"{path.stem}_{i}" for i in range(len(df))]
    df = merge_nearby_features(df, path)

    if "sample_index" in df.columns:
        key_cols = ["target_id"]
        if "sequence" in df.columns:
            key_cols.append("sequence")
        sort_cols = []
        if "ptm" in df.columns:
            df["_ptm_sort"] = pd.to_numeric(df["ptm"], errors="coerce").fillna(-1)
            sort_cols.append("_ptm_sort")
        if "plddt_mean" in df.columns:
            df["_plddt_sort"] = pd.to_numeric(df["plddt_mean"], errors="coerce").fillna(-1)
            sort_cols.append("_plddt_sort")
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False).drop_duplicates(key_cols, keep="first")
        else:
            df = df.drop_duplicates(key_cols, keep="first")
        df["selection"] = "best_by_ptm"

    df = df.copy()
    df["result_file"] = str(path)
    df["run_dir"] = str(path.parent)
    df["run_name"] = next((part for part in path.parts if part.startswith("esmfold2_")), path.parent.name)
    seq = df["sequence"].map(clean_seq) if "sequence" in df.columns else pd.Series([""] * len(df), index=df.index)
    df["sequence_clean"] = seq
    df["sequence_hash_calc"] = seq.map(hash_seq)
    df["ca_lddt"] = pd.to_numeric(df["ca_lddt"], errors="coerce")
    for col in ["gdt_ts", "ca_rmsd", "baseline_ca_lddt", "ptm", "plddt_mean", "seq_len", "resolution"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["setting_label"] = df.apply(setting_label, axis=1)
    df["budget_score"] = df.apply(budget_score, axis=1)
    df["data_family"] = df.apply(lambda r: data_family(path, r), axis=1)
    return df


def build_all_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest = []
    for path in result_files():
        df = normalize_file(path)
        if df is None or df.empty:
            continue
        finite = df["ca_lddt"].notna().sum()
        if finite == 0:
            continue
        frames.append(df)
        manifest.append({"path": str(path), "rows_after_file_normalization": int(len(df)), "finite_ca_lddt": int(finite)})
    if not frames:
        raise RuntimeError("No ESMFold2 result files with finite ca_lddt found.")
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = all_df[all_df["ca_lddt"].notna()].copy()
    all_df = all_df[all_df["sequence_clean"].str.len() > 0].copy()
    all_df["_ca_round"] = all_df["ca_lddt"].round(6)
    all_df["_gdt_round"] = pd.to_numeric(all_df.get("gdt_ts", np.nan), errors="coerce").round(6)
    all_df["_rmsd_round"] = pd.to_numeric(all_df.get("ca_rmsd", np.nan), errors="coerce").round(6)
    all_df = all_df.drop_duplicates(
        ["target_id", "sequence_hash_calc", "setting_label", "_ca_round", "_gdt_round", "_rmsd_round"],
        keep="first",
    ).copy()
    all_df = all_df.drop(columns=["_ca_round", "_gdt_round", "_rmsd_round"], errors="ignore")

    sort_cols = ["budget_score", "setting_label", "run_name"]
    primary = all_df.sort_values(sort_cols, ascending=[False, False, False]).drop_duplicates("sequence_hash_calc", keep="first")
    primary = primary.copy()
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return all_df, primary


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seqs = out["sequence_clean"].fillna("")
    out["calc_seq_len"] = seqs.str.len()
    out["aa20_fraction"] = seqs.map(lambda s: frac(s, AA20))
    out["cys_fraction_calc"] = seqs.map(lambda s: frac(s, {"C"}))
    out["disulfide_capacity_per100_calc"] = seqs.map(lambda s: math.floor(s.count("C") / 2) / len(s) * 100 if s else np.nan)
    out["cys_window30_max_fraction"] = seqs.map(lambda s: max_window_fraction(s, {"C"}, 30))
    out["proline_fraction_calc"] = seqs.map(lambda s: frac(s, {"P"}))
    out["gly_pro_fraction_calc"] = seqs.map(lambda s: frac(s, {"G", "P"}))
    out["hydrophobic_fraction_calc"] = seqs.map(lambda s: frac(s, HYDROPHOBIC))
    out["hydrophobic_window21_max_frac_calc2"] = seqs.map(lambda s: max_window_fraction(s, HYDROPHOBIC, 21))
    out["tm_helix_proxy_calc2"] = seqs.map(lambda s: max_kd_window(s, 21))
    out["acidic_fraction_calc"] = seqs.map(lambda s: frac(s, ACIDIC))
    out["basic_fraction_calc"] = seqs.map(lambda s: frac(s, BASIC))
    out["charged_fraction_calc"] = seqs.map(lambda s: frac(s, CHARGED))
    out["abs_charge_per_residue_calc"] = out.apply(
        lambda r: abs(r["basic_fraction_calc"] - r["acidic_fraction_calc"]) if np.isfinite(r["basic_fraction_calc"]) else np.nan,
        axis=1,
    )
    out["net_charge_per_residue_calc"] = out["basic_fraction_calc"] - out["acidic_fraction_calc"]
    out["low_complexity_window30_max_fraction_calc"] = seqs.map(lambda s: max_window_fraction(s, None, 30))
    out["nglyc_motif_count_calc"] = seqs.map(nglyc_count)
    out["nglyc_motif_per100_calc"] = out["nglyc_motif_count_calc"] / out["calc_seq_len"].replace(0, np.nan) * 100
    out["disorder_promoting_fraction_calc"] = seqs.map(lambda s: frac(s, DISORDER_PROMOTING))
    out["order_promoting_fraction_calc"] = seqs.map(lambda s: frac(s, ORDER_PROMOTING))
    out["disorder_minus_order_fraction_calc"] = out["disorder_promoting_fraction_calc"] - out["order_promoting_fraction_calc"]
    out["chou_helix_mean_calc"] = seqs.map(lambda s: mean_prop(s, CHOU_HELIX))
    out["chou_beta_mean_calc"] = seqs.map(lambda s: mean_prop(s, CHOU_BETA))
    out["chou_turn_mean_calc"] = seqs.map(lambda s: mean_prop(s, CHOU_TURN))
    out["chou_beta_minus_helix_calc"] = out["chou_beta_mean_calc"] - out["chou_helix_mean_calc"]
    out["chou_turn_minus_helix_calc"] = out["chou_turn_mean_calc"] - out["chou_helix_mean_calc"]
    out["kd_gravy_calc"] = seqs.map(lambda s: mean_prop(s, KD))
    out["uversky_disorder_score_calc"] = 2.785 * out["abs_charge_per_residue_calc"] - 1.151 * ((out["kd_gravy_calc"] + 4.5) / 9.0)

    for col in ["proline_fraction_calc", "chou_beta_minus_helix_calc", "chou_turn_minus_helix_calc"]:
        out[col + "_pct"] = out[col].rank(pct=True)
    out["flexible_nonhelical_composite_calc"] = out[
        ["proline_fraction_calc_pct", "chou_beta_minus_helix_calc_pct", "chou_turn_minus_helix_calc_pct"]
    ].mean(axis=1)

    text_cols = []
    for col in ["organism", "entity_description", "title", "keywords", "query_hits"]:
        if col in out.columns:
            text_cols.append(out[col].fillna("").astype(str))
    if text_cols:
        text = text_cols[0]
        for col in text_cols[1:]:
            text = text + " " + col
    else:
        text = pd.Series([""] * len(out), index=out.index)
    text = text.str.lower()
    out["metadata_text"] = text
    out["metadata_text_available"] = text.str.strip().str.len() > 0
    out["tax_virus_or_phage_calc"] = text.str.contains(
        r"virus|viral|phage|virion|flavivirus|alphavirus|dengue|zika|chikungunya|encephalitis|yellow fever|semiliki|semliki|ebola|marburg",
        regex=True,
    )
    out["tax_parasite_or_nonhuman_surface_calc"] = text.str.contains(
        r"parasite|plasmodium|trypanosoma|leishmania|helminth|tick-borne|mosquito|nonhuman", regex=True
    )
    out["surface_keyword_calc"] = text.str.contains(
        r"\b(envelope|spike|glycoprotein|fusion|gp1|gp2|gp350|protein e|e1|e2|surface)\b", regex=True
    )
    out["capsid_or_enzyme_keyword_calc"] = text.str.contains(
        r"capsid|polymerase|protease|non-structural|nonstructural|nsp|helicase|methyltransferase|nucleoprotein|matrix|tegument|replicase|enzyme",
        regex=True,
    )
    out["classii_e_e2_envelopeE_calc"] = text.str.contains(
        r"\b(e2 glycoprotein|glycoprotein e2|envelope glycoprotein e2|envelope protein e\b|protein e\b|envelope protein\b)",
        regex=True,
    ) & ~text.str.contains(
        r"\bgp1\b|\bgp2\b|\bspike\b|fusion protein|\bglycoprotein g\b|capsid|polymerase|protease|non-structural|nonstructural|nucleoprotein|matrix|helicase",
        regex=True,
    )
    out["gp1_calc"] = text.str.contains(r"\bgp1\b", regex=True)
    out["gp2_calc"] = text.str.contains(r"\bgp2\b", regex=True)
    out["e1_or_g_surface_calc"] = text.str.contains(r"\be1\b|glycoprotein g|protein g\b", regex=True)
    out["disulfide_dense_calc"] = out["disulfide_capacity_per100_calc"] >= 1.2
    out["low_tm_proxy_calc"] = out["tm_helix_proxy_calc2"] < 2.55
    out["membrane_lowglyc_proxy_calc"] = (out["tm_helix_proxy_calc2"] >= 1.8) & (out["nglyc_motif_count_calc"] <= 2)
    out["viral_disulfide_score_calc"] = out["tax_virus_or_phage_calc"].astype(int) + out["disulfide_dense_calc"].astype(int)
    out["strict_envelope_fusion_score_calc"] = out["disulfide_dense_calc"].astype(int) + out["membrane_lowglyc_proxy_calc"].astype(int)
    out["surface_lowhelical_score_calc"] = out["surface_keyword_calc"].astype(int) + (
        out["disulfide_dense_calc"] & out["low_tm_proxy_calc"]
    ).astype(int)
    out["surface_disulfide_lowhelical_calc"] = out["surface_keyword_calc"] & out["disulfide_dense_calc"] & out["low_tm_proxy_calc"]
    out["classii_e_e2_lowhelical_calc"] = out["classii_e_e2_envelopeE_calc"] & out["low_tm_proxy_calc"]
    return out


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(float(np.mean(aa) - np.mean(bb)))
    arr = np.array(vals)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def eval_top_lower(df: pd.DataFrame, name: str, top_mask: pd.Series, eligible_mask: pd.Series, stress_values: pd.Series | None = None) -> dict:
    d = df[eligible_mask & df["ca_lddt"].notna()].copy()
    if d.empty:
        return {"axis": name, "status": "not_evaluable", "n": 0}
    top = top_mask.reindex(d.index).fillna(False).astype(bool)
    lower = ~top
    n_top = int(top.sum())
    n_lower = int(lower.sum())
    if n_top < 5 or n_lower < 5:
        status = "too_small"
    else:
        status = "evaluated"
    top_vals = d.loc[top, "ca_lddt"].astype(float).to_numpy()
    lower_vals = d.loc[lower, "ca_lddt"].astype(float).to_numpy()
    diff = float(np.mean(top_vals) - np.mean(lower_vals)) if n_top and n_lower else np.nan
    boot_mean, ci_low, ci_high = bootstrap_ci(top_vals, lower_vals) if n_top >= 2 and n_lower >= 2 else (np.nan, np.nan, np.nan)
    base_diff = np.nan
    if "baseline_ca_lddt" in d.columns:
        base = pd.to_numeric(d["baseline_ca_lddt"], errors="coerce")
        if base.loc[top].notna().sum() >= 2 and base.loc[lower].notna().sum() >= 2:
            base_diff = float(base.loc[top].mean() - base.loc[lower].mean())
    rho = np.nan
    if stress_values is not None:
        s = stress_values.reindex(d.index)
        ok = s.notna() & d["ca_lddt"].notna()
        if ok.sum() >= 8 and s.loc[ok].nunique() > 1:
            rho = float(s.loc[ok].rank().corr(d.loc[ok, "ca_lddt"].rank()))
    return {
        "axis": name,
        "status": status,
        "n": int(len(d)),
        "n_top": n_top,
        "n_lower": n_lower,
        "top_ca_lddt_mean": float(np.mean(top_vals)) if n_top else np.nan,
        "lower_ca_lddt_mean": float(np.mean(lower_vals)) if n_lower else np.nan,
        "top_minus_lower_ca_lddt": diff,
        "bootstrap_mean": boot_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "top_minus_lower_baseline_ca_lddt": base_diff,
        "spearman_stress_vs_ca_lddt": rho,
        "data_families": ";".join(f"{k}:{v}" for k, v in d["data_family"].value_counts().sort_index().items()),
        "settings": ";".join(f"{k}:{v}" for k, v in d["setting_label"].value_counts().sort_index().items()),
    }


def level_curve(df: pd.DataFrame, axis: str, levels: pd.Series, eligible: pd.Series) -> list[dict]:
    rows = []
    d = df[eligible & df["ca_lddt"].notna()].copy()
    lv = levels.reindex(d.index)
    for level, idx in lv.dropna().groupby(lv.dropna()).groups.items():
        sub = d.loc[idx]
        rows.append(
            {
                "axis": axis,
                "level": level,
                "n": int(len(sub)),
                "stress_mean": float(level) if isinstance(level, (int, float, np.integer, np.floating)) else np.nan,
                "ca_lddt_mean": float(sub["ca_lddt"].mean()),
                "ca_lddt_median": float(sub["ca_lddt"].median()),
                "gdt_ts_mean": float(pd.to_numeric(sub.get("gdt_ts", np.nan), errors="coerce").mean()),
                "ca_rmsd_median": float(pd.to_numeric(sub.get("ca_rmsd", np.nan), errors="coerce").median()),
                "baseline_ca_lddt_mean": float(pd.to_numeric(sub.get("baseline_ca_lddt", np.nan), errors="coerce").mean()),
            }
        )
    return rows


def qlevel(series: pd.Series, q: int = 5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    ok = s.notna()
    if ok.sum() < q * 4 or s.loc[ok].nunique() < q:
        return out
    try:
        out.loc[ok] = pd.qcut(s.loc[ok], q=q, labels=False, duplicates="drop")
    except ValueError:
        return out
    return out


def slice_effects(df: pd.DataFrame, axis: str, top_mask: pd.Series, eligible: pd.Series) -> list[dict]:
    rows = []
    d = df[eligible & df["ca_lddt"].notna()].copy()
    if d.empty:
        return rows
    top_all = top_mask.reindex(d.index).fillna(False).astype(bool)
    for fam, idx in d.groupby("data_family").groups.items():
        sub = d.loc[idx]
        top = top_all.reindex(sub.index).fillna(False)
        n_top = int(top.sum())
        n_lower = int((~top).sum())
        if len(sub) < 12 or n_top < 3 or n_lower < 3:
            continue
        rows.append(
            {
                "axis": axis,
                "slice": fam,
                "n": int(len(sub)),
                "n_top": n_top,
                "n_lower": n_lower,
                "top_minus_lower_ca_lddt": float(sub.loc[top, "ca_lddt"].mean() - sub.loc[~top, "ca_lddt"].mean()),
                "top_mean": float(sub.loc[top, "ca_lddt"].mean()),
                "lower_mean": float(sub.loc[~top, "ca_lddt"].mean()),
            }
        )
    return rows


def evaluate_hypotheses(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    slices = []

    continuous = [
        ("length_high", "calc_seq_len"),
        ("high_cysteine_fraction", "cys_fraction_calc"),
        ("high_disulfide_capacity", "disulfide_capacity_per100_calc"),
        ("high_cys_window30", "cys_window30_max_fraction"),
        ("high_proline_fraction", "proline_fraction_calc"),
        ("high_gly_pro_fraction", "gly_pro_fraction_calc"),
        ("flexible_nonhelical_composite", "flexible_nonhelical_composite_calc"),
        ("high_abs_charge_per_residue", "abs_charge_per_residue_calc"),
        ("high_acidic_fraction", "acidic_fraction_calc"),
        ("high_charged_fraction", "charged_fraction_calc"),
        ("high_hydrophobic_window", "hydrophobic_window21_max_frac_calc2"),
        ("high_tm_helix_proxy", "tm_helix_proxy_calc2"),
        ("high_uversky_disorder_score", "uversky_disorder_score_calc"),
        ("high_low_complexity_window", "low_complexity_window30_max_fraction_calc"),
        ("high_nglyc_per100", "nglyc_motif_per100_calc"),
    ]
    for name, col in continuous:
        s = pd.to_numeric(df[col], errors="coerce")
        eligible = s.notna()
        if eligible.sum() >= 20:
            threshold = s.loc[eligible].quantile(0.80)
            top = s >= threshold
            row = eval_top_lower(df, name, top, eligible, s)
            row["dataset"] = label
            row["axis_kind"] = "continuous_top20"
            row["threshold"] = float(threshold) if np.isfinite(threshold) else np.nan
            rows.append(row)
            levels = qlevel(s, 5)
            curves.extend(level_curve(df, name, levels, eligible))
            slices.extend(slice_effects(df, name, top, eligible))

    metadata_eligible = df["metadata_text_available"].fillna(False).astype(bool)
    binary_axes = [
        ("virus_or_phage_metadata", "tax_virus_or_phage_calc", metadata_eligible),
        ("metadata_surface_keyword", "surface_keyword_calc", metadata_eligible),
        ("surface_not_capsid_or_enzyme", "surface_keyword_calc", metadata_eligible & ~df["capsid_or_enzyme_keyword_calc"].fillna(False)),
        ("classII_E_E2_envelopeE", "classii_e_e2_envelopeE_calc", metadata_eligible),
        ("classII_E_E2_lowhelical", "classii_e_e2_lowhelical_calc", metadata_eligible),
        ("surface_disulfide_lowhelical", "surface_disulfide_lowhelical_calc", metadata_eligible),
        ("viral_disulfide_dense", "disulfide_dense_calc", metadata_eligible & df["tax_virus_or_phage_calc"].fillna(False)),
        ("gp1_metadata", "gp1_calc", metadata_eligible),
        ("gp2_metadata", "gp2_calc", metadata_eligible),
        ("e1_or_g_surface_metadata", "e1_or_g_surface_calc", metadata_eligible),
    ]
    for name, col, eligible in binary_axes:
        top = df[col].fillna(False).astype(bool)
        row = eval_top_lower(df, name, top, eligible, top.astype(float))
        row["dataset"] = label
        row["axis_kind"] = "binary_metadata"
        rows.append(row)
        curves.extend(level_curve(df, name, top.astype(int), eligible))
        slices.extend(slice_effects(df, name, top, eligible))

    level_axes = [
        ("viral_disulfide_score", "viral_disulfide_score_calc", metadata_eligible),
        ("strict_envelope_fusion_score", "strict_envelope_fusion_score_calc", metadata_eligible),
        ("surface_lowhelical_score", "surface_lowhelical_score_calc", metadata_eligible),
    ]
    for name, col, eligible in level_axes:
        levels = pd.to_numeric(df[col], errors="coerce")
        if eligible.sum() < 10:
            continue
        top = levels == levels.loc[eligible].max()
        row = eval_top_lower(df, name, top, eligible, levels)
        row["dataset"] = label
        row["axis_kind"] = "metadata_level_top"
        rows.append(row)
        curves.extend(level_curve(df, name, levels, eligible))
        slices.extend(slice_effects(df, name, top, eligible))

    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(slices)


def summarize_findings(effects: pd.DataFrame, primary_effects: pd.DataFrame, slice_df: pd.DataFrame, all_df: pd.DataFrame, primary: pd.DataFrame) -> str:
    def fmt(x, nd=3):
        if x is None or not np.isfinite(x):
            return "NA"
        return f"{x:.{nd}f}"

    primary_sorted = primary_effects.sort_values("top_minus_lower_ca_lddt").copy()
    lines = []
    lines.append("# ESMFold2 Global Hypothesis Recheck")
    lines.append("")
    lines.append("This audit aggregates saved ESMFold2 target-level outputs only; no new ESMFold2 inference was run.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Unique scored evaluation rows after exact result de-duplication: {len(all_df)}")
    lines.append(f"- Exact-sequence primary rows: {len(primary)}")
    lines.append("- Primary rows choose the highest inference budget per exact sequence, then de-duplicate exact sequences.")
    lines.append("- MSA-depth is not treated as a global axis because many RCSB panels lack comparable MSA fields.")
    lines.append("")
    lines.append("Primary exact-sequence rows by data family:")
    for k, v in primary["data_family"].value_counts().sort_index().items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Strongest Negative Effects On Primary Exact-Sequence Table")
    lines.append("")
    lines.append("| axis | n | top n | lower n | top CA-lDDT | lower CA-lDDT | diff | 95% CI | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for _, r in primary_sorted.head(12).iterrows():
        lines.append(
            f"| {r['axis']} | {int(r.get('n', 0))} | {int(r.get('n_top', 0))} | {int(r.get('n_lower', 0))} | "
            f"{fmt(r.get('top_ca_lddt_mean'))} | {fmt(r.get('lower_ca_lddt_mean'))} | {fmt(r.get('top_minus_lower_ca_lddt'))} | "
            f"[{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}] | {r.get('status', '')} |"
        )
    lines.append("")
    lines.append("## Slice Consistency Notes")
    lines.append("")
    for axis in [
        "classII_E_E2_envelopeE",
        "high_cysteine_fraction",
        "high_disulfide_capacity",
        "flexible_nonhelical_composite",
        "high_abs_charge_per_residue",
        "strict_envelope_fusion_score",
        "surface_lowhelical_score",
    ]:
        sdf = slice_df[slice_df["axis"] == axis].copy()
        if sdf.empty:
            continue
        neg = int((sdf["top_minus_lower_ca_lddt"] < -0.03).sum())
        pos = int((sdf["top_minus_lower_ca_lddt"] > 0.03).sum())
        lines.append(f"- {axis}: {neg} negative slices, {pos} positive/reversal slices among {len(sdf)} evaluable slices.")
        for _, r in sdf.sort_values("slice").iterrows():
            lines.append(
                f"  - {r['slice']}: n={int(r['n'])}, top={int(r['n_top'])}, diff={fmt(r['top_minus_lower_ca_lddt'])}"
            )
    lines.append("")
    lines.append("## Auditor Interpretation")
    lines.append("")
    lines.append(
        "Across all saved evidence, no single non-MSA axis cleanly explains degradation everywhere. "
        "The strongest current all-evidence pattern is a localized metadata-defined class-II viral E/E2/envelope-E weakness "
        "inside rows where that metadata exists. Sequence-only broad proxies such as cysteine/disulfide and flexible/non-helical "
        "composition remain useful risk signals, but their effects weaken or reverse across source slices."
    )
    lines.append("")
    lines.append("Artifacts:")
    lines.append("")
    lines.append("- `spectra_ready_dataset/all_esmfold2_scored_evaluations.csv`")
    lines.append("- `spectra_ready_dataset/primary_exact_sequence_dedup.csv`")
    lines.append("- `analysis_outputs/global_hypothesis_effects_all_evaluations.csv`")
    lines.append("- `analysis_outputs/global_hypothesis_effects_primary_sequence.csv`")
    lines.append("- `analysis_outputs/global_hypothesis_slice_effects_primary_sequence.csv`")
    lines.append("- `analysis_outputs/global_hypothesis_level_curves_primary_sequence.csv`")
    return "\n".join(lines) + "\n"


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    all_df, primary = build_all_results()
    all_df = add_features(all_df)
    primary = add_features(primary)

    all_df.to_csv(DATA_OUT / "all_esmfold2_scored_evaluations.csv", index=False)
    primary.to_csv(DATA_OUT / "primary_exact_sequence_dedup.csv", index=False)

    effects_all, curves_all, slices_all = evaluate_hypotheses(all_df, "all_unique_evaluations")
    effects_primary, curves_primary, slices_primary = evaluate_hypotheses(primary, "primary_exact_sequence")

    effects_all.to_csv(ANALYSIS_OUT / "global_hypothesis_effects_all_evaluations.csv", index=False)
    effects_primary.to_csv(ANALYSIS_OUT / "global_hypothesis_effects_primary_sequence.csv", index=False)
    curves_all.to_csv(ANALYSIS_OUT / "global_hypothesis_level_curves_all_evaluations.csv", index=False)
    curves_primary.to_csv(ANALYSIS_OUT / "global_hypothesis_level_curves_primary_sequence.csv", index=False)
    slices_all.to_csv(ANALYSIS_OUT / "global_hypothesis_slice_effects_all_evaluations.csv", index=False)
    slices_primary.to_csv(ANALYSIS_OUT / "global_hypothesis_slice_effects_primary_sequence.csv", index=False)

    report = summarize_findings(effects_all, effects_primary, slices_primary, all_df, primary)
    (OUT / "global_hypothesis_audit_report.md").write_text(report, encoding="utf-8")

    summary = {
        "n_unique_evaluations": int(len(all_df)),
        "n_primary_exact_sequence": int(len(primary)),
        "primary_by_family": {str(k): int(v) for k, v in primary["data_family"].value_counts().sort_index().items()},
        "top_primary_effects": effects_primary.sort_values("top_minus_lower_ca_lddt")
        .head(10)[["axis", "n", "n_top", "top_minus_lower_ca_lddt", "ci_low", "ci_high"]]
        .to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
