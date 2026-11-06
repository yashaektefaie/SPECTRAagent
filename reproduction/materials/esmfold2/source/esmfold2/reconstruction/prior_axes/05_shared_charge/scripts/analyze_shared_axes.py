#!/usr/bin/env python3
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_shared_axis_cameo_rcsb_20260601T173500Z")

PATHS = {
    "cameo_default_features": Path("/ewsc/yektefai/spectra_runs/esmfold2_single_controller_final_run/spectra_ready_dataset/candidate_features.csv"),
    "cameo_default_results": Path("/ewsc/yektefai/spectra_runs/esmfold2_single_controller_final_run/target_model_results.csv"),
    "cameo_default_axis_ledger": Path("/ewsc/yektefai/spectra_runs/esmfold2_single_controller_final_run/axis_ledger.json"),
    "cameo_followup_features": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/spectra_ready_dataset/prospective_features.followup.csv"),
    "cameo_followup_results": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/target_model_results.csv"),
    "cameo_followup_summary": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/followup_axis_summary.csv"),
    "cameo_followup_axis_ledger": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/axis_ledger.json"),
    "cameo_scaled_results": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/inference_time_scaling_10loops68steps_5samples_full_top_tail_matched/target_model_results.scaled_samples.csv"),
    "rcsb_orig_features": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/external_rcsb_same_bucket_expansion_20260531/spectra_ready_dataset/candidate_features.csv"),
    "rcsb_orig_results": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/external_rcsb_same_bucket_expansion_20260531/target_model_results.scaled_samples.csv"),
    "rcsb_exp_features": Path("/ewsc/yektefai/spectra_runs/esmfold2_external_flexible_axis_expanded_20260601T170049Z/spectra_ready_dataset/candidate_features.csv"),
    "rcsb_exp_results": Path("/ewsc/yektefai/spectra_runs/esmfold2_external_flexible_axis_expanded_20260601T170049Z/target_model_results.scaled_samples.csv"),
    "rcsb_exp_effects": Path("/ewsc/yektefai/spectra_runs/esmfold2_external_flexible_axis_expanded_20260601T170049Z/expanded_external_effects.json"),
    "rcsb_exp_report": Path("/ewsc/yektefai/spectra_runs/esmfold2_external_flexible_axis_expanded_20260601T170049Z/expanded_external_report.md"),
    "rcsb_pool": Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/external_rcsb_same_bucket_expansion_20260531/rcsb_external_candidate_pool.csv"),
}


AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVY")
FLEXIBLE = set("GPSNQD")
AROMATIC = set("FWY")
TINY = set("ACDGST")
CHARGED = set("DEKRH")
ACIDIC = set("DE")
BASIC = set("KRH")

PROHIBITED_AXIS_SUBSTRINGS = (
    "aligned",
    "lddt",
    "gdt",
    "rmsd",
    "error",
    "ptm",
    "plddt",
    "confidence",
    "alignment",
    "median_error",
    "p90",
    "baseline",
    "wall_sec",
    "sample_index",
    "seed",
    "num_loops",
    "num_sampling_steps",
    "num_diffusion_samples",
    "status",
)

NON_SHARED_PRIMARY_FEATURES = {
    "msa_rows",
    "msa_nonquery_rows",
    "msa_log10_nonquery_rows",
    "msa_top_nonquery_identity",
    "msa_median_identity",
    "msa_q90_identity",
    "msa_frac_id_ge_0_9",
    "msa_frac_id_ge_0_7",
    "msa_frac_cov_ge_0_7",
    "difficulty",
    "quaternary_state",
    "monomeric_model_count",
    "ref_ca_residues",
    "ref_polymer_residues",
    "query_ref_identity",
    "pdb_ca_count",
    "ca_resolved_fraction",
    "rg",
    "rg_len13_norm",
    "end_to_end_distance",
    "end_to_end_over_rg",
    "shape_anisotropy",
    "contact_count_ca8_seqsep3",
    "contacts_per_residue",
    "contact_density_ca8_seqsep3",
    "relative_contact_order_ca8",
    "long_contacts_per_residue",
    "long_contact_fraction",
    "medium_contact_fraction",
    "local_contact_fraction",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_dirs() -> None:
    (ROOT / "spectra_ready_dataset").mkdir(exist_ok=True)
    (ROOT / "analysis_outputs").mkdir(exist_ok=True)


def finite_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sequence_features(seq) -> dict:
    seq = str(seq or "").upper()
    residues = [c for c in seq if c.isalpha()]
    n = len(residues)
    if n == 0:
        return {
            "calc_seq_len": np.nan,
            "calc_seq_entropy": np.nan,
            "calc_seq_entropy_norm": np.nan,
            "calc_seq_max_aa_fraction": np.nan,
            "calc_seq_unknown_fraction": np.nan,
            "calc_seq_hydrophobic_fraction": np.nan,
            "calc_seq_gly_pro_fraction": np.nan,
            "calc_seq_flexible_fraction": np.nan,
            "calc_seq_tiny_fraction": np.nan,
            "calc_seq_aromatic_fraction": np.nan,
            "calc_seq_charged_fraction": np.nan,
            "calc_seq_abs_charge_per_residue": np.nan,
            "calc_seq_max_homopolymer_run_norm": np.nan,
            "calc_seq_dipeptide_repeat_fraction": np.nan,
        }
    counts = {aa: residues.count(aa) for aa in set(residues)}
    probs = np.array(list(counts.values()), dtype=float) / n
    entropy = float(-(probs * np.log2(probs)).sum())
    max_run = 1
    cur = 1
    for i in range(1, n):
        if residues[i] == residues[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    di = [seq[i : i + 2] for i in range(0, max(0, n - 1), 2)]
    if di:
        di_counts = pd.Series(di).value_counts(normalize=True)
        di_repeat = float(di_counts.iloc[0])
    else:
        di_repeat = np.nan
    charge = sum(1 for c in residues if c in BASIC) - sum(1 for c in residues if c in ACIDIC)
    return {
        "calc_seq_len": n,
        "calc_seq_entropy": entropy,
        "calc_seq_entropy_norm": entropy / math.log2(20),
        "calc_seq_max_aa_fraction": max(counts.values()) / n,
        "calc_seq_unknown_fraction": sum(1 for c in residues if c not in AA20) / n,
        "calc_seq_hydrophobic_fraction": sum(1 for c in residues if c in HYDROPHOBIC) / n,
        "calc_seq_gly_pro_fraction": sum(1 for c in residues if c in set("GP")) / n,
        "calc_seq_flexible_fraction": sum(1 for c in residues if c in FLEXIBLE) / n,
        "calc_seq_tiny_fraction": sum(1 for c in residues if c in TINY) / n,
        "calc_seq_aromatic_fraction": sum(1 for c in residues if c in AROMATIC) / n,
        "calc_seq_charged_fraction": sum(1 for c in residues if c in CHARGED) / n,
        "calc_seq_abs_charge_per_residue": abs(charge) / n,
        "calc_seq_max_homopolymer_run_norm": max_run / n,
        "calc_seq_dipeptide_repeat_fraction": di_repeat,
    }


def add_common_metadata(df: pd.DataFrame, source: str, feature_panel: str) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    out["feature_panel"] = feature_panel
    if "sequence" in out.columns:
        seq_features = pd.DataFrame([sequence_features(s) for s in out["sequence"]], index=out.index)
        for col in seq_features.columns:
            out[col] = seq_features[col]
        out["sequence_hash"] = out["sequence"].fillna("").map(sha256_text)
    else:
        out["sequence_hash"] = np.nan
    if source == "CAMEO":
        out["entry_id"] = out.get("ref_pdb_id", pd.Series(index=out.index, dtype=object))
        date_col = "submission_date"
    else:
        date_col = "release_date"
    if date_col in out.columns:
        dt = pd.to_datetime(out[date_col], errors="coerce", utc=True)
        out["date_ordinal"] = dt.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    else:
        out["date_ordinal"] = np.nan
    if "method" in out.columns:
        method = out["method"].fillna("").astype(str).str.upper()
        out["is_xray_common"] = method.str.contains("X-RAY|XRAY|DIFFRACTION", regex=True).astype(float)
        out["is_em_common"] = method.str.contains("ELECTRON|EM", regex=True).astype(float)
        out["is_nmr_common"] = method.str.contains("NMR", regex=True).astype(float)
    return out


def best_by_ptm(samples: pd.DataFrame, source_panel_col: str | None = None) -> pd.DataFrame:
    df = samples.copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("ok").eq("ok")].copy()
    df["ptm_sort"] = pd.to_numeric(df.get("ptm"), errors="coerce").fillna(-np.inf)
    sort_cols = ["target_id", "ptm_sort"]
    ascending = [True, False]
    if source_panel_col and source_panel_col in df.columns:
        df["_panel_pref"] = df[source_panel_col].map({"expanded_107": 1, "original_34": 0}).fillna(0)
        sort_cols.append("_panel_pref")
        ascending.append(False)
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")
        ascending.append(True)
    df = df.sort_values(sort_cols, ascending=ascending)
    out = df.groupby("target_id", as_index=False).head(1).drop(columns=["ptm_sort"], errors="ignore")
    return out


def load_combined_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cameo_features = add_common_metadata(read_csv(PATHS["cameo_followup_features"]), "CAMEO", "followup_full_141")
    cameo_features = cameo_features.drop_duplicates("target_id", keep="first")

    cameo_default_results = read_csv(PATHS["cameo_followup_results"])
    cameo_default_results = cameo_default_results[cameo_default_results["status"].eq("ok")].copy()
    cameo_default = cameo_features.merge(cameo_default_results, on="target_id", how="inner", suffixes=("", "_perf"))
    cameo_default["inference_setting"] = "default_single_sample_full_pool"
    cameo_default["selection_policy"] = "single_default"
    cameo_default["result_panel"] = "cameo_default_full"

    cameo_scaled_samples = read_csv(PATHS["cameo_scaled_results"])
    cameo_scaled_best = best_by_ptm(cameo_scaled_samples)
    cameo_scaled = cameo_features.merge(cameo_scaled_best, on="target_id", how="inner", suffixes=("", "_perf"))
    cameo_scaled["inference_setting"] = "scaled_10loops68steps_5samples_best_by_ptm_top_tail_matched"
    cameo_scaled["selection_policy"] = "best_by_ptm"
    cameo_scaled["result_panel"] = "cameo_scaled_top_tail_matched"

    rcsb_orig_features = add_common_metadata(read_csv(PATHS["rcsb_orig_features"]), "RCSB", "original_34")
    rcsb_exp_features = add_common_metadata(read_csv(PATHS["rcsb_exp_features"]), "RCSB", "expanded_107")
    rcsb_features = pd.concat([rcsb_orig_features, rcsb_exp_features], ignore_index=True, sort=False)
    rcsb_features["_panel_pref"] = rcsb_features["feature_panel"].map({"expanded_107": 1, "original_34": 0}).fillna(0)
    rcsb_features = (
        rcsb_features.sort_values(["target_id", "_panel_pref"])
        .drop_duplicates("target_id", keep="last")
        .drop(columns=["_panel_pref"], errors="ignore")
    )

    rcsb_orig_results = read_csv(PATHS["rcsb_orig_results"])
    rcsb_orig_results["result_panel"] = "original_34"
    rcsb_exp_results = read_csv(PATHS["rcsb_exp_results"])
    rcsb_exp_results["result_panel"] = "expanded_107"
    rcsb_samples = pd.concat([rcsb_orig_results, rcsb_exp_results], ignore_index=True, sort=False)
    rcsb_best = best_by_ptm(rcsb_samples, "result_panel")
    rcsb = rcsb_features.merge(rcsb_best, on="target_id", how="inner", suffixes=("", "_perf"))
    rcsb["source"] = "RCSB"
    rcsb["inference_setting"] = "scaled_10loops68steps_5samples_best_by_ptm_combined_rcsb"
    rcsb["selection_policy"] = "best_by_ptm"

    combined = pd.concat([cameo_default, cameo_scaled, rcsb], ignore_index=True, sort=False)
    target_cols = [
        "source",
        "target_id",
        "inference_setting",
        "selection_policy",
        "result_panel",
        "sample_index",
        "status",
        "error",
        "ptm",
        "plddt_mean",
        "plddt_ca_mean",
        "alignment_identity",
        "alignment_coverage",
        "aligned_ca",
        "ca_lddt",
        "ca_rmsd",
        "gdt_ts",
        "baseline_ca_lddt",
        "baseline_gdt_ts",
        "baseline_ca_rmsd",
    ]
    target_results = combined[[c for c in target_cols if c in combined.columns]].copy()
    manifest = {
        "rows": {
            "cameo_default": int(len(cameo_default)),
            "cameo_scaled_best_by_ptm": int(len(cameo_scaled)),
            "rcsb_combined_best_by_ptm": int(len(rcsb)),
            "combined": int(len(combined)),
        },
        "unique_targets": {
            "cameo_default": int(cameo_default["target_id"].nunique()),
            "cameo_scaled_best_by_ptm": int(cameo_scaled["target_id"].nunique()),
            "rcsb_combined_best_by_ptm": int(rcsb["target_id"].nunique()),
        },
        "input_paths": {k: str(v) for k, v in PATHS.items()},
    }
    return combined, target_results, manifest


def numeric_candidate_features(df: pd.DataFrame) -> list[str]:
    cameo = df[df["source"].eq("CAMEO") & df["inference_setting"].eq("default_single_sample_full_pool")]
    rcsb = df[df["source"].eq("RCSB")]
    features = []
    metadata = {
        "resolution",
        "date_ordinal",
        "is_xray_common",
        "is_em_common",
        "is_nmr_common",
    }
    for col in df.columns:
        low = col.lower()
        if any(tok in low for tok in PROHIBITED_AXIS_SUBSTRINGS):
            continue
        if col in NON_SHARED_PRIMARY_FEATURES:
            continue
        if low.startswith("axis_"):
            continue
        if col.startswith("default_") or col.endswith("_perf"):
            continue
        if col in {
            "source",
            "target_id",
            "sequence",
            "pdb_path",
            "msa_path",
            "ref_pdb_chain",
            "ref_pdb_id",
            "ref_chain",
            "actual_ref_chain",
            "chain_mapping_note",
            "cameo_target",
            "submission_date",
            "release_date",
            "method",
            "organism",
            "polymer_type",
            "rcsb_instance_id",
            "entry_id",
            "entity_id",
            "asym_id",
            "auth_asym_id",
            "feature_panel",
            "result_panel",
            "inference_setting",
            "selection_policy",
            "sequence_hash",
            "pdb_ca_parse_status",
            "external_axis_level_name",
            "scaling_group",
        }:
            continue
        s_all = pd.to_numeric(df[col], errors="coerce")
        if s_all.notna().sum() == 0:
            continue
        c = pd.to_numeric(cameo[col], errors="coerce") if col in cameo.columns else pd.Series(dtype=float)
        r = pd.to_numeric(rcsb[col], errors="coerce") if col in rcsb.columns else pd.Series(dtype=float)
        if c.notna().sum() < 20 or r.notna().sum() < 20:
            continue
        if c.nunique(dropna=True) < 3 or r.nunique(dropna=True) < 3:
            if col not in metadata:
                continue
        features.append(col)
    return sorted(set(features))


def spearman(x, y) -> tuple[float | None, float | None]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 4 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return None, None
    if stats is not None:
        rho, p = stats.spearmanr(a[mask], b[mask])
        return finite_float(rho), finite_float(p)
    rho = a[mask].rank().corr(b[mask].rank())
    return finite_float(rho), None


def assign_levels(values: pd.Series, direction: str, n_levels: int) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    stress = x.copy()
    if direction == "low":
        stress = -stress
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = stress.notna()
    if mask.sum() < n_levels * 3 or stress[mask].nunique() < 3:
        return out
    try:
        levels = pd.qcut(stress[mask].rank(method="first"), q=n_levels, labels=False, duplicates="drop")
    except ValueError:
        try:
            levels = pd.cut(stress[mask].rank(method="first"), bins=n_levels, labels=False, include_lowest=True)
        except Exception:
            return out
    out.loc[mask] = levels.astype(float)
    return out


def ols_axis_coef(df: pd.DataFrame, feature: str, direction: str, y_col: str, controls: list[str]) -> dict:
    cols = [feature, y_col] + [c for c in controls if c != feature and c in df.columns]
    sub = df[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(12, len(cols) + 4) or sub[feature].nunique() < 3:
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None}
    x_axis = sub[feature].astype(float).copy()
    if direction == "low":
        x_axis = -x_axis
    X_cols = []
    names = []
    for name, arr in [("axis_stress", x_axis)] + [(c, sub[c].astype(float)) for c in controls if c != feature and c in sub.columns]:
        sd = float(np.nanstd(arr, ddof=0))
        if sd == 0 or not math.isfinite(sd):
            continue
        X_cols.append(((arr - float(np.nanmean(arr))) / sd).to_numpy())
        names.append(name)
    if not X_cols:
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None}
    X = np.column_stack([np.ones(len(sub))] + X_cols)
    y = sub[y_col].astype(float).to_numpy()
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        if dof <= 0:
            return {"n": int(len(sub)), "coef": finite_float(beta[1]), "se": None, "t": None}
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        coef = float(beta[1])
        return {"n": int(len(sub)), "coef": coef, "se": se, "t": coef / se if se else None}
    except Exception:
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None}


def bootstrap_top_minus_lower(sub: pd.DataFrame, level_col: str, y_col: str, n_boot: int = 1000, seed: int = 123) -> dict:
    levels = sorted([int(x) for x in sub[level_col].dropna().unique()])
    if len(levels) < 3:
        return {"mean": None, "ci_low": None, "ci_high": None}
    high = sub[sub[level_col].eq(max(levels))][y_col].dropna().to_numpy()
    lower = sub[~sub[level_col].eq(max(levels))][y_col].dropna().to_numpy()
    if len(high) < 5 or len(lower) < 10:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(float(rng.choice(high, len(high), replace=True).mean() - rng.choice(lower, len(lower), replace=True).mean()))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def evaluate_axis(
    df: pd.DataFrame,
    feature: str,
    direction: str,
    n_levels: int = 5,
    source_label: str = "",
    inference_setting: str = "",
    bootstrap: bool = False,
) -> tuple[dict, pd.DataFrame]:
    level_col = f"level_{n_levels}"
    sub = df.copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub[level_col] = assign_levels(sub[feature], direction, n_levels)
    sub = sub[sub[level_col].notna()].copy()
    sub[level_col] = sub[level_col].astype(int)
    levels = sorted(sub[level_col].unique().tolist())
    rows = []
    for level in levels:
        part = sub[sub[level_col].eq(level)]
        rows.append(
            {
                "source": source_label,
                "inference_setting": inference_setting,
                "feature": feature,
                "direction": direction,
                "n_levels_requested": n_levels,
                "level": int(level),
                "n": int(len(part)),
                "feature_mean": finite_float(part[feature].mean()),
                "feature_median": finite_float(part[feature].median()),
                "seq_len_median": finite_float(pd.to_numeric(part.get("seq_len"), errors="coerce").median()) if "seq_len" in part else None,
                "resolution_median": finite_float(pd.to_numeric(part.get("resolution"), errors="coerce").median()) if "resolution" in part else None,
                "ca_lddt_mean": finite_float(pd.to_numeric(part["ca_lddt"], errors="coerce").mean()),
                "gdt_ts_mean": finite_float(pd.to_numeric(part["gdt_ts"], errors="coerce").mean()),
                "ca_rmsd_median": finite_float(pd.to_numeric(part["ca_rmsd"], errors="coerce").median()),
                "baseline_ca_lddt_mean": finite_float(pd.to_numeric(part.get("baseline_ca_lddt"), errors="coerce").mean()) if "baseline_ca_lddt" in part else None,
                "entry_unique": int(part["entry_id"].nunique(dropna=True)) if "entry_id" in part else None,
                "max_entry_count": int(part["entry_id"].value_counts(dropna=True).max()) if "entry_id" in part and part["entry_id"].notna().any() else None,
                "sequence_unique": int(part["sequence_hash"].nunique(dropna=True)) if "sequence_hash" in part else None,
                "max_exact_sequence_count": int(part["sequence_hash"].value_counts(dropna=True).max()) if "sequence_hash" in part and part["sequence_hash"].notna().any() else None,
            }
        )
    curve = pd.DataFrame(rows)
    summary = {
        "source": source_label,
        "inference_setting": inference_setting,
        "feature": feature,
        "direction": direction,
        "n": int(len(sub)),
        "n_levels_requested": int(n_levels),
        "n_levels_observed": int(len(levels)),
        "min_level_n": int(curve["n"].min()) if len(curve) else 0,
        "level_counts": ",".join(str(int(x)) for x in curve["n"]) if len(curve) else "",
        "measured_order_ok": bool(curve["feature_median"].is_monotonic_increasing if direction == "high" else curve["feature_median"].is_monotonic_decreasing) if len(curve) else False,
    }
    if len(levels) >= 2:
        low = curve[curve["level"].eq(min(levels))].iloc[0]
        high = curve[curve["level"].eq(max(levels))].iloc[0]
        lower = sub[~sub[level_col].eq(max(levels))]
        high_rows = sub[sub[level_col].eq(max(levels))]
        summary.update(
            {
                "ca_lddt_endpoint_effect": finite_float(high["ca_lddt_mean"] - low["ca_lddt_mean"]),
                "ca_lddt_top_vs_lower_effect": finite_float(pd.to_numeric(high_rows["ca_lddt"], errors="coerce").mean() - pd.to_numeric(lower["ca_lddt"], errors="coerce").mean()),
                "gdt_ts_endpoint_effect": finite_float(high["gdt_ts_mean"] - low["gdt_ts_mean"]),
                "gdt_ts_top_vs_lower_effect": finite_float(pd.to_numeric(high_rows["gdt_ts"], errors="coerce").mean() - pd.to_numeric(lower["gdt_ts"], errors="coerce").mean()),
                "ca_rmsd_endpoint_effect": finite_float(high["ca_rmsd_median"] - low["ca_rmsd_median"]),
                "baseline_ca_lddt_endpoint_effect": finite_float(high["baseline_ca_lddt_mean"] - low["baseline_ca_lddt_mean"]) if high.get("baseline_ca_lddt_mean") is not None and low.get("baseline_ca_lddt_mean") is not None else None,
                "baseline_ca_lddt_top_vs_lower_effect": finite_float(pd.to_numeric(high_rows.get("baseline_ca_lddt"), errors="coerce").mean() - pd.to_numeric(lower.get("baseline_ca_lddt"), errors="coerce").mean()) if "baseline_ca_lddt" in sub.columns else None,
            }
        )
    rho_feature, p_feature = spearman(sub[feature] if direction == "high" else -sub[feature], sub["ca_lddt"])
    rho_level, p_level = spearman(sub[level_col], sub["ca_lddt"])
    summary["spearman_stress_vs_ca_lddt_rho"] = rho_feature
    summary["spearman_stress_vs_ca_lddt_p"] = p_feature
    summary["spearman_level_vs_ca_lddt_rho"] = rho_level
    summary["spearman_level_vs_ca_lddt_p"] = p_level
    controls = ["seq_len", "resolution"]
    summary["length_resolution_adjusted"] = ols_axis_coef(sub, feature, direction, "ca_lddt", controls)
    cameo_controls = ["seq_len", "resolution", "msa_log10_nonquery_rows", "difficulty"]
    summary["cameo_broad_adjusted"] = ols_axis_coef(sub, feature, direction, "ca_lddt", cameo_controls) if source_label.startswith("CAMEO") else None
    if bootstrap:
        summary["bootstrap_top_vs_lower_ca_lddt"] = bootstrap_top_minus_lower(sub, level_col, "ca_lddt")
        summary["bootstrap_top_vs_lower_gdt_ts"] = bootstrap_top_minus_lower(sub, level_col, "gdt_ts")
    return summary, curve


def choose_discovery_direction(df: pd.DataFrame, feature: str) -> tuple[str, dict, pd.DataFrame]:
    high_summary, high_curve = evaluate_axis(df, feature, "high", 5)
    low_summary, low_curve = evaluate_axis(df, feature, "low", 5)
    high_effect = high_summary.get("ca_lddt_top_vs_lower_effect")
    low_effect = low_summary.get("ca_lddt_top_vs_lower_effect")
    if high_effect is None:
        return "low", low_summary, low_curve
    if low_effect is None:
        return "high", high_summary, high_curve
    if low_effect < high_effect:
        return "low", low_summary, low_curve
    return "high", high_summary, high_curve


def source_views(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "CAMEO_default": combined[combined["source"].eq("CAMEO") & combined["inference_setting"].eq("default_single_sample_full_pool")].copy(),
        "CAMEO_scaled": combined[combined["source"].eq("CAMEO") & combined["inference_setting"].str.contains("scaled_", na=False)].copy(),
        "RCSB_scaled": combined[combined["source"].eq("RCSB")].copy(),
        "RCSB_original": combined[combined["source"].eq("RCSB") & combined["feature_panel"].eq("original_34")].copy(),
        "RCSB_expanded": combined[combined["source"].eq("RCSB") & combined["feature_panel"].eq("expanded_107")].copy(),
    }


def score_shared_axes(combined: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    views = source_views(combined)
    discovery_rows = []
    all_curves = []
    dense_rows = []
    for feature in features:
        for discover_name, confirm_name in [("CAMEO_default", "RCSB_scaled"), ("RCSB_scaled", "CAMEO_default")]:
            disc_df = views[discover_name]
            conf_df = views[confirm_name]
            direction, disc_summary, _ = choose_discovery_direction(disc_df, feature)
            disc_eval, disc_curve = evaluate_axis(
                disc_df,
                feature,
                direction,
                5,
                source_label=discover_name,
                inference_setting=str(disc_df["inference_setting"].iloc[0]) if len(disc_df) else "",
            )
            conf_eval, conf_curve = evaluate_axis(
                conf_df,
                feature,
                direction,
                5,
                source_label=confirm_name,
                inference_setting=str(conf_df["inference_setting"].iloc[0]) if len(conf_df) else "",
            )
            cameo_scaled_eval, cameo_scaled_curve = evaluate_axis(
                views["CAMEO_scaled"],
                feature,
                direction,
                5,
                source_label="CAMEO_scaled",
                inference_setting="scaled_10loops68steps_5samples_best_by_ptm_top_tail_matched",
            )
            rcsb_exp_eval, rcsb_exp_curve = evaluate_axis(
                views["RCSB_expanded"],
                feature,
                direction,
                5,
                source_label="RCSB_expanded",
                inference_setting="scaled_10loops68steps_5samples_best_by_ptm_expanded_only",
            )
            for curve in [disc_curve, conf_curve, cameo_scaled_curve, rcsb_exp_curve]:
                if len(curve):
                    all_curves.append(curve)
            row = {
                "feature": feature,
                "discover_on": discover_name,
                "confirm_on": confirm_name,
                "frozen_direction": direction,
                "discovery_n": disc_eval.get("n"),
                "confirmation_n": conf_eval.get("n"),
                "discovery_min_level_n": disc_eval.get("min_level_n"),
                "confirmation_min_level_n": conf_eval.get("min_level_n"),
                "discovery_order_ok": disc_eval.get("measured_order_ok"),
                "confirmation_order_ok": conf_eval.get("measured_order_ok"),
                "discovery_top_vs_lower_ca_lddt": disc_eval.get("ca_lddt_top_vs_lower_effect"),
                "confirmation_top_vs_lower_ca_lddt": conf_eval.get("ca_lddt_top_vs_lower_effect"),
                "discovery_level_spearman": disc_eval.get("spearman_level_vs_ca_lddt_rho"),
                "confirmation_level_spearman": conf_eval.get("spearman_level_vs_ca_lddt_rho"),
                "discovery_top_vs_lower_gdt_ts": disc_eval.get("gdt_ts_top_vs_lower_effect"),
                "confirmation_top_vs_lower_gdt_ts": conf_eval.get("gdt_ts_top_vs_lower_effect"),
                "discovery_baseline_top_vs_lower_ca_lddt": disc_eval.get("baseline_ca_lddt_top_vs_lower_effect"),
                "confirmation_baseline_top_vs_lower_ca_lddt": conf_eval.get("baseline_ca_lddt_top_vs_lower_effect"),
                "discovery_lr_adjusted_coef": (disc_eval.get("length_resolution_adjusted") or {}).get("coef"),
                "confirmation_lr_adjusted_coef": (conf_eval.get("length_resolution_adjusted") or {}).get("coef"),
                "discovery_lr_adjusted_t": (disc_eval.get("length_resolution_adjusted") or {}).get("t"),
                "confirmation_lr_adjusted_t": (conf_eval.get("length_resolution_adjusted") or {}).get("t"),
                "cameo_broad_adjusted_coef": (disc_eval.get("cameo_broad_adjusted") or {}).get("coef")
                if discover_name.startswith("CAMEO")
                else (conf_eval.get("cameo_broad_adjusted") or {}).get("coef"),
                "cameo_scaled_top_vs_lower_ca_lddt": cameo_scaled_eval.get("ca_lddt_top_vs_lower_effect"),
                "cameo_scaled_n": cameo_scaled_eval.get("n"),
                "rcsb_expanded_top_vs_lower_ca_lddt": rcsb_exp_eval.get("ca_lddt_top_vs_lower_effect"),
                "rcsb_expanded_n": rcsb_exp_eval.get("n"),
            }
            same_degradation = (
                row["discovery_top_vs_lower_ca_lddt"] is not None
                and row["confirmation_top_vs_lower_ca_lddt"] is not None
                and row["discovery_top_vs_lower_ca_lddt"] < 0
                and row["confirmation_top_vs_lower_ca_lddt"] < 0
            )
            practically_meaningful = same_degradation and min(
                abs(row["discovery_top_vs_lower_ca_lddt"]),
                abs(row["confirmation_top_vs_lower_ca_lddt"]),
            ) >= 0.05
            adjusted_same = (
                row["discovery_lr_adjusted_coef"] is not None
                and row["confirmation_lr_adjusted_coef"] is not None
                and row["discovery_lr_adjusted_coef"] < 0
                and row["confirmation_lr_adjusted_coef"] < 0
            )
            if practically_meaningful and adjusted_same:
                row["screen_decision"] = "live_shared_candidate"
            elif same_degradation:
                row["screen_decision"] = "weak_same_sign"
            else:
                row["screen_decision"] = "not_confirmed"
            discovery_rows.append(row)

    summary = pd.DataFrame(discovery_rows)
    if len(summary):
        summary["min_abs_effect_across_sources"] = summary[["discovery_top_vs_lower_ca_lddt", "confirmation_top_vs_lower_ca_lddt"]].abs().min(axis=1)
        summary["sum_effect"] = summary[["discovery_top_vs_lower_ca_lddt", "confirmation_top_vs_lower_ca_lddt"]].sum(axis=1)
    curve_df = pd.concat(all_curves, ignore_index=True, sort=False) if all_curves else pd.DataFrame()

    live = summary[summary["screen_decision"].isin(["live_shared_candidate", "weak_same_sign"])].copy()
    if len(live):
        live = live.sort_values(["screen_decision", "min_abs_effect_across_sources", "sum_effect"], ascending=[True, False, True]).head(12)
    else:
        live = summary.sort_values("sum_effect", ascending=True).head(12)

    for _, row in live.iterrows():
        feature = row["feature"]
        direction = row["frozen_direction"]
        for view_name in ["CAMEO_default", "RCSB_scaled", "CAMEO_scaled", "RCSB_original", "RCSB_expanded"]:
            df = views[view_name]
            for levels in [3, 5, 7, 10]:
                ev, curve = evaluate_axis(
                    df,
                    feature,
                    direction,
                    levels,
                    source_label=view_name,
                    inference_setting=str(df["inference_setting"].iloc[0]) if len(df) else "",
                    bootstrap=levels in (5, 7),
                )
                ev["candidate_origin"] = f"{row['discover_on']}_to_{row['confirm_on']}"
                ev["screen_decision"] = row["screen_decision"]
                dense_rows.append(ev)
    dense = pd.DataFrame(dense_rows)
    return summary, curve_df, dense


def residualize_y(df: pd.DataFrame, y_col: str, controls: list[str]) -> pd.Series:
    cols = [y_col] + [c for c in controls if c in df.columns]
    sub = df[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = sub.dropna().index
    residual = pd.Series(np.nan, index=df.index, dtype=float)
    if len(valid) < 15:
        return y - y.mean()
    X_cols = []
    for c in controls:
        if c not in sub.columns:
            continue
        arr = sub.loc[valid, c].astype(float)
        sd = arr.std(ddof=0)
        if sd and math.isfinite(sd):
            X_cols.append(((arr - arr.mean()) / sd).to_numpy())
    if not X_cols:
        residual.loc[valid] = sub.loc[valid, y_col] - sub.loc[valid, y_col].mean()
        return residual
    X = np.column_stack([np.ones(len(valid))] + X_cols)
    beta, *_ = np.linalg.lstsq(X, sub.loc[valid, y_col].astype(float).to_numpy(), rcond=None)
    residual.loc[valid] = sub.loc[valid, y_col].astype(float).to_numpy() - X @ beta
    return residual


def mine_outcome_informed(combined: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    views = source_views(combined)
    rows = []
    controls_by_view = {
        "CAMEO_default": ["seq_len", "resolution", "msa_log10_nonquery_rows", "difficulty"],
        "RCSB_scaled": ["seq_len", "resolution"],
    }
    for view_name, controls in controls_by_view.items():
        df = views[view_name].copy()
        df["ca_lddt_residual"] = residualize_y(df, "ca_lddt", controls)
        fail_cut = df["ca_lddt_residual"].quantile(0.2)
        success_cut = df["ca_lddt_residual"].quantile(0.8)
        fail = df[df["ca_lddt_residual"].le(fail_cut)]
        succ = df[df["ca_lddt_residual"].ge(success_cut)]
        for feature in features:
            f = pd.to_numeric(df[feature], errors="coerce")
            ff = pd.to_numeric(fail[feature], errors="coerce").dropna()
            ss = pd.to_numeric(succ[feature], errors="coerce").dropna()
            if len(ff) < 5 or len(ss) < 5 or f.nunique(dropna=True) < 3:
                continue
            pooled = f.dropna().std(ddof=0)
            if not pooled or not math.isfinite(pooled):
                continue
            diff_sd = (ff.median() - ss.median()) / pooled
            rho, p = spearman(f, df["ca_lddt_residual"])
            bad_direction = "high" if ff.median() > ss.median() else "low"
            rows.append(
                {
                    "discovery_view": view_name,
                    "feature": feature,
                    "bad_direction": bad_direction,
                    "failure_median": finite_float(ff.median()),
                    "success_median": finite_float(ss.median()),
                    "failure_minus_success_sd": finite_float(diff_sd),
                    "spearman_feature_residual": rho,
                    "spearman_p": p,
                    "n": int(len(df)),
                    "n_failure_tail": int(len(fail)),
                    "n_success_tail": int(len(succ)),
                }
            )
    mined = pd.DataFrame(rows)
    if len(mined):
        mined["abs_sep"] = mined["failure_minus_success_sd"].abs()
        mined = mined.sort_values(["discovery_view", "abs_sep"], ascending=[True, False])
    return mined


def duplicate_diagnostics(combined: pd.DataFrame, dense: pd.DataFrame) -> dict:
    out = {}
    views = source_views(combined)
    for view_name, df in views.items():
        if len(df) == 0:
            continue
        entry_counts = df["entry_id"].value_counts(dropna=True) if "entry_id" in df else pd.Series(dtype=int)
        seq_counts = df["sequence_hash"].value_counts(dropna=True) if "sequence_hash" in df else pd.Series(dtype=int)
        out[view_name] = {
            "n": int(len(df)),
            "unique_targets": int(df["target_id"].nunique()),
            "unique_entries": int(df["entry_id"].nunique(dropna=True)) if "entry_id" in df else None,
            "max_entry_count": int(entry_counts.max()) if len(entry_counts) else None,
            "max_entry_fraction": float(entry_counts.max() / len(df)) if len(entry_counts) else None,
            "unique_exact_sequences": int(df["sequence_hash"].nunique(dropna=True)) if "sequence_hash" in df else None,
            "max_exact_sequence_count": int(seq_counts.max()) if len(seq_counts) else None,
            "max_exact_sequence_fraction": float(seq_counts.max() / len(df)) if len(seq_counts) else None,
        }
    return out


def make_axis_ledger(shared_summary: pd.DataFrame, dense: pd.DataFrame) -> dict:
    ledger = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "shared CAMEO+RCSB prospective axes using saved ESMFold2 audits only",
        "entries": [],
    }
    carried = [
        {
            "axis_id": "cameo_msa_depth",
            "status": "not_primary_shared_axis",
            "prospective": True,
            "reason": "CAMEO-only valid threshold SPC from prior ledger, but RCSB lacks comparable MSA fields.",
        },
        {
            "axis_id": "cameo_flexible_nonhelical_composite",
            "status": "failed_external_expanded_rcsb",
            "prospective": True,
            "reason": "Prior CAMEO-only high-tail degradation did not replicate in the expanded RCSB panel; expanded RCSB best-by-pTM top-minus-lower CA-lDDT was positive.",
        },
        {
            "axis_id": "pre_2021_reference_proximity",
            "status": "not_executable_with_current_shared_tables",
            "prospective": True,
            "reason": "No comparable pre-2021 train/pretraining proximity feature is present for both CAMEO and RCSB scored targets; release/submission date is not a substitute.",
        },
        {
            "axis_id": "reference_topology_geometry",
            "status": "benchmark_only_not_primary_shared",
            "prospective": False,
            "reason": "CAMEO reference geometry axes use held-out/reference structure information and are absent from RCSB features.",
        },
    ]
    ledger["entries"].extend(carried)
    if len(shared_summary):
        for feature, group in shared_summary.groupby("feature"):
            best = group.sort_values("sum_effect", ascending=True).iloc[0]
            status = "failed_shared_confirmation"
            if (group["screen_decision"] == "live_shared_candidate").any():
                status = "weak_or_live_shared_candidate_requires_densification"
            elif (group["screen_decision"] == "weak_same_sign").any():
                status = "weak_same_sign_not_claim_valid"
            ledger["entries"].append(
                {
                    "axis_id": f"shared_{feature}",
                    "feature": feature,
                    "status": status,
                    "prospective": True,
                    "frozen_direction_best_screen": best["frozen_direction"],
                    "discover_on": best["discover_on"],
                    "confirm_on": best["confirm_on"],
                    "discovery_top_vs_lower_ca_lddt": finite_float(best["discovery_top_vs_lower_ca_lddt"]),
                    "confirmation_top_vs_lower_ca_lddt": finite_float(best["confirmation_top_vs_lower_ca_lddt"]),
                    "discovery_lr_adjusted_coef": finite_float(best["discovery_lr_adjusted_coef"]),
                    "confirmation_lr_adjusted_coef": finite_float(best["confirmation_lr_adjusted_coef"]),
                    "screen_decision": best["screen_decision"],
                }
            )
    return ledger


def summarize_closure(shared_summary: pd.DataFrame, dense: pd.DataFrame) -> dict:
    def dict_coef_is_negative(x) -> bool:
        return isinstance(x, dict) and x.get("coef") is not None and x.get("coef") < 0

    def baseline_not_explanatory(frame: pd.DataFrame) -> bool:
        for _, rec in frame.iterrows():
            target = rec.get("ca_lddt_top_vs_lower_effect")
            base = rec.get("baseline_ca_lddt_top_vs_lower_effect")
            if target is None or pd.isna(target) or base is None or pd.isna(base):
                continue
            if base < 0 and abs(base) >= 0.5 * abs(target):
                return False
        return True

    if len(shared_summary) == 0:
        return {
            "terminal_status": "hard_blocker",
            "primary_finding": "No shared numeric prospective features were available after leakage exclusions.",
            "claim_valid_shared_axis": False,
        }
    live = shared_summary[shared_summary["screen_decision"].eq("live_shared_candidate")].copy()
    weak = shared_summary[shared_summary["screen_decision"].eq("weak_same_sign")].copy()
    candidates = pd.concat([live, weak], ignore_index=True, sort=False)
    claim_valid = False
    best = None
    reason = "No claim-valid shared degradation SPC was found in the fixed scored CAMEO/RCSB pools."
    terminal = "no_supported_shared_axis_in_existing_scored_pool"
    if len(candidates):
        candidates = candidates.sort_values(["screen_decision", "sum_effect"], ascending=[True, True])
        best = candidates.iloc[0].to_dict()
        terminal = "exploratory_weak_shared_candidate_needs_new_data"
        reason = (
            "At least one same-sign candidate exists, but densification/controls do not satisfy the "
            "claim-valid shared degradation gate across CAMEO and RCSB."
        )
    supported = []
    if len(live):
        for _, row in live.iterrows():
            d = dense[
                dense["feature"].eq(row["feature"])
                & dense["direction"].eq(row["frozen_direction"])
                & dense["source"].isin(["CAMEO_default", "RCSB_scaled"])
            ]
            q5 = d[d["n_levels_requested"].eq(5)]
            q7 = d[d["n_levels_requested"].eq(7)]
            q10 = d[d["n_levels_requested"].eq(10)]
            cameo_dense = d[d["source"].eq("CAMEO_default") & d["n_levels_requested"].isin([5, 7])]
            if len(q5) >= 2 and len(q7) >= 2:
                stable = (
                    (q5["ca_lddt_top_vs_lower_effect"] < -0.05).all()
                    and (q7["ca_lddt_top_vs_lower_effect"] < -0.05).all()
                    and (len(q10) < 2 or (q10["ca_lddt_top_vs_lower_effect"] < -0.03).all())
                    and (q5["length_resolution_adjusted"].map(lambda x: isinstance(x, dict) and x.get("coef") is not None and x.get("coef") < 0)).all()
                    and (q7["length_resolution_adjusted"].map(lambda x: isinstance(x, dict) and x.get("coef") is not None and x.get("coef") < 0)).all()
                    and (len(cameo_dense) == 0 or cameo_dense["cameo_broad_adjusted"].map(dict_coef_is_negative).all())
                    and baseline_not_explanatory(pd.concat([q5, q7], ignore_index=True))
                )
                if stable:
                    supported.append(row.to_dict())
    if supported:
        claim_valid = True
        terminal = "claim_valid_shared_degradation_spc"
        best = supported[0]
        reason = "A source-heldout prospective axis remained negative and stable after densification and controls."
    return {
        "terminal_status": terminal,
        "primary_finding": reason,
        "claim_valid_shared_axis": claim_valid,
        "best_candidate": best,
        "n_live_candidates": int(len(live)),
        "n_weak_same_sign": int(len(weak)),
        "n_features_screened": int(shared_summary["feature"].nunique()),
        "n_source_heldout_tests": int(len(shared_summary)),
    }


def write_report(state: dict, shared_summary: pd.DataFrame, dense: pd.DataFrame, mined: pd.DataFrame, manifest: dict) -> None:
    closure = state["closure"]
    lines = []
    lines.append("# Shared CAMEO/RCSB ESMFold2 SPECTRA Audit")
    lines.append("")
    lines.append(f"Status: `{closure['terminal_status']}`")
    lines.append("")
    lines.append("## Finding")
    lines.append("")
    lines.append(closure["primary_finding"])
    lines.append("")
    lines.append("This run used only saved target-model outputs. Splits were defined from prospective sequence or metadata features, not from ESMFold2 errors, pTM, pLDDT, RMSD, lDDT, GDT, confidence, or reference/prediction comparisons.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- CAMEO default scored targets: {manifest['unique_targets']['cameo_default']}")
    lines.append(f"- CAMEO scaled best-by-pTM top-tail matched targets: {manifest['unique_targets']['cameo_scaled_best_by_ptm']}")
    lines.append(f"- RCSB scaled best-by-pTM targets: {manifest['unique_targets']['rcsb_combined_best_by_ptm']}")
    lines.append("")
    if closure.get("best_candidate"):
        b = closure["best_candidate"]
        lines.append("## Best Candidate")
        lines.append("")
        lines.append(f"- Feature: `{b.get('feature')}`")
        lines.append(f"- Frozen direction: `{b.get('frozen_direction')}`")
        lines.append(f"- Discovery/confirmation: `{b.get('discover_on')}` -> `{b.get('confirm_on')}`")
        lines.append(f"- Discovery top-vs-lower CA-lDDT: {b.get('discovery_top_vs_lower_ca_lddt'):.4f}")
        lines.append(f"- Confirmation top-vs-lower CA-lDDT: {b.get('confirmation_top_vs_lower_ca_lddt'):.4f}")
        lines.append(f"- Screen decision: `{b.get('screen_decision')}`")
        lines.append("")
    lines.append("## Prior Axis Outcomes Carried Forward")
    lines.append("")
    lines.append("- MSA depth remains CAMEO-only and cannot be the primary shared axis because RCSB lacks comparable MSA fields.")
    lines.append("- The CAMEO flexible/non-helical high-tail result is not confirmed: the expanded RCSB panel showed no degradation for that frozen axis.")
    lines.append("- Reference topology/geometry axes are benchmark-only here because they use reference-structure information and are not shared prospective CAMEO/RCSB features.")
    lines.append("- Pre-2021 reference proximity could not be tested as a shared axis from the saved scored tables.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `spectra_ready_dataset/combined_prospective_performance.csv`")
    lines.append("- `target_model_results.csv`")
    lines.append("- `analysis_outputs/shared_axis_source_heldout_summary.csv`")
    lines.append("- `analysis_outputs/densified_candidate_axes.csv`")
    lines.append("- `analysis_outputs/outcome_informed_feature_mining.csv`")
    lines.append("- `axis_ledger.json`")
    lines.append("- `spectra_loop_state.json`")
    (ROOT / "final_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    combined, target_results, manifest = load_combined_tables()
    features = numeric_candidate_features(combined)

    shared_summary, shared_curves, dense = score_shared_axes(combined, features)
    mined = mine_outcome_informed(combined, features)
    duplicate_info = duplicate_diagnostics(combined, dense)
    closure = summarize_closure(shared_summary, dense)

    combined.to_csv(ROOT / "spectra_ready_dataset" / "combined_prospective_performance.csv", index=False)
    target_results.to_csv(ROOT / "target_model_results.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(ROOT / "analysis_outputs" / "shared_prospective_feature_list.csv", index=False)
    shared_summary.to_csv(ROOT / "analysis_outputs" / "shared_axis_source_heldout_summary.csv", index=False)
    shared_curves.to_csv(ROOT / "analysis_outputs" / "shared_axis_level_curves.csv", index=False)
    dense.to_csv(ROOT / "analysis_outputs" / "densified_candidate_axes.csv", index=False)
    mined.to_csv(ROOT / "analysis_outputs" / "outcome_informed_feature_mining.csv", index=False)

    ledger = make_axis_ledger(shared_summary, dense)
    (ROOT / "axis_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True))

    state = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Find a shared prospective similarity axis yielding consistent spectral degradation across CAMEO and RCSB ESMFold2 audits.",
        "manifest": manifest,
        "features_screened": features,
        "closure": closure,
        "duplicate_diagnostics": duplicate_info,
        "artifacts": {
            "combined_table": "spectra_ready_dataset/combined_prospective_performance.csv",
            "target_model_results": "target_model_results.csv",
            "shared_axis_summary": "analysis_outputs/shared_axis_source_heldout_summary.csv",
            "shared_axis_curves": "analysis_outputs/shared_axis_level_curves.csv",
            "densified_candidate_axes": "analysis_outputs/densified_candidate_axes.csv",
            "outcome_mining": "analysis_outputs/outcome_informed_feature_mining.csv",
            "axis_ledger": "axis_ledger.json",
            "commands_run": "commands_run.json",
        },
        "notes": [
            "All final split memberships are prospective-feature quantiles.",
            "Target-model confidence/performance columns are excluded from candidate-axis construction.",
            "MSA features are retained in the combined table for CAMEO-specific context but excluded from shared primary axes.",
        ],
    }
    (ROOT / "spectra_loop_state.json").write_text(json.dumps(state, indent=2, sort_keys=True))

    commands = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": "python scripts/analyze_shared_axes.py",
            "purpose": "Build combined CAMEO/RCSB table, score shared prospective axes, write SPECTRA artifacts.",
        }
    ]
    (ROOT / "commands_run.json").write_text(json.dumps(commands, indent=2))

    write_report(state, shared_summary, dense, mined, manifest)
    print(json.dumps(closure, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
