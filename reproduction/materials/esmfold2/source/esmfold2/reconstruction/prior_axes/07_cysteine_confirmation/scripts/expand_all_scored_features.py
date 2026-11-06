#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/ewsc/yektefai/spectra_runs")
LATEST = RUNS / "esmfold2_all_evidence_confirmation_loop_20260601T215426Z"
COMBINED_IN = LATEST / "spectra_ready_dataset" / "combined_evidence_table.csv"
RCSB_POOL = (
    RUNS
    / "esmfold2_beyond_evolutionary_support_final_run"
    / "external_rcsb_same_bucket_expansion_20260531"
    / "rcsb_external_candidate_pool.csv"
)

OUT_DATA = ROOT / "spectra_ready_dataset"
OUT_ANALYSIS = ROOT / "analysis_outputs"
OUT_COMBINED = OUT_DATA / "expanded_all_scored_evidence_table.csv"
OUT_POOL = OUT_DATA / "expanded_candidate_pool_features.csv"
OUT_SCREEN = OUT_ANALYSIS / "expanded_all_scored_axis_screen.csv"
OUT_REVIEW = OUT_ANALYSIS / "expanded_candidate_hypothesis_review.csv"
OUT_MINING = OUT_ANALYSIS / "expanded_outcome_mining.csv"
OUT_STATE = ROOT / "spectra_loop_state.json"
OUT_LEDGER = ROOT / "axis_ledger.json"
OUT_COMMANDS = ROOT / "commands_run.json"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("DEKRH")
ACIDIC = set("DE")
BASIC = set("KRH")
FLEXIBLE = set("GPSNQD")
DISORDER_PROMOTING = set("PGSKQEDR")
ORDER_PROMOTING = set("WCMFYILVN")

FAILED_FAMILY_PATTERNS = (
    "charge",
    "acidic",
    "negative_charge",
    "positive_charge",
    "uversky",
    "disorder",
    "dipeptide",
    "repeat",
    "flexible_nonhelical",
    "nonhelical",
)

PROHIBITED_PATTERNS = (
    "lddt",
    "gdt",
    "rmsd",
    "error",
    "ptm",
    "plddt",
    "confidence",
    "baseline",
    "aligned",
    "alignment",
    "wall_sec",
    "sample_index",
    "seed",
    "num_loops",
    "num_sampling_steps",
    "num_diffusion_samples",
    "status",
)

REFERENCE_QUALITY_OR_LABEL_PROXIES = {
    "resolution",
    "reference_quality_stress",
    "rcsb_reference_quality_level",
    "ca_resolved_fraction",
    "pdb_ca_count",
    "ref_ca_residues",
    "ref_polymer_residues",
    "query_ref_identity",
    "seq_ref_len_delta_abs",
    "metadata_ref_len_delta_abs",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def clean_sequence(seq: Any) -> str:
    return "".join(c for c in str(seq or "").upper() if c.isalpha())


def aa_fraction(seq: str, alphabet: set[str]) -> float:
    residues = [c for c in clean_sequence(seq) if c in AA20]
    if not residues:
        return np.nan
    return sum(c in alphabet for c in residues) / len(residues)


def max_run_fraction(seq: str, alphabet: set[str]) -> float:
    residues = [c for c in clean_sequence(seq) if c in AA20]
    if not residues:
        return np.nan
    best = cur = 0
    for aa in residues:
        if aa in alphabet:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best / len(residues)


def nterm_fraction(seq: str, alphabet: set[str], n: int = 30) -> float:
    residues = [c for c in clean_sequence(seq) if c in AA20][:n]
    if not residues:
        return np.nan
    return sum(c in alphabet for c in residues) / len(residues)


def window_stat(seq: str, alphabet: set[str], win: int = 21) -> float:
    residues = [c for c in clean_sequence(seq) if c in AA20]
    if len(residues) < win:
        return np.nan
    return max(sum(c in alphabet for c in residues[i : i + win]) / win for i in range(len(residues) - win + 1))


def polyhis_tag_score(seq: str) -> float:
    s = clean_sequence(seq)
    longest = 0
    for match in re.finditer(r"H{3,}", s):
        longest = max(longest, len(match.group(0)))
    return float(longest)


def antibody_like_score(seq: str) -> float:
    s = clean_sequence(seq)
    if not s:
        return np.nan
    first = s[:35]
    score = 0.0
    if re.search(r"(QVQL|EVQL|DVQL|DIQMT|DIVMT|EIVLT|QIVLT|VLTQ|QLVLT|QSVLT)", first):
        score += 1.0
    if re.search(r"W[GQ][QG].{0,8}G", s):
        score += 0.5
    if re.search(r"YYC|YFC|FG.GT|WGQG", s):
        score += 0.5
    cys = s.count("C")
    if 80 <= len(s) <= 260 and cys >= 2:
        score += 0.5
    if "TVSS" in s or "VTVSS" in s or "WGQGTS" in s:
        score += 0.5
    return score


def sequence_extra_features(seq: Any) -> dict[str, float]:
    s = clean_sequence(seq)
    residues = [c for c in s if c in AA20]
    n = len(residues)
    if n == 0:
        return {
            "seq_kmer4_richness": np.nan,
            "seq_nterm_hydrophobic_frac30": np.nan,
            "seq_hydrophobic_window21_max_frac": np.nan,
            "seq_polyhis_longest_run": np.nan,
            "seq_antibody_like_score": np.nan,
            "seq_gs_linker_fraction": np.nan,
            "seq_disorder_order_balance": np.nan,
            "seq_long_hydrophobic_segment_flag": np.nan,
            "seq_polyhis_tag_flag": np.nan,
            "seq_antibody_like_flag": np.nan,
        }
    kmers = {s[i : i + 4] for i in range(max(0, len(s) - 3)) if set(s[i : i + 4]).issubset(AA20)}
    denom = max(1, len(s) - 3)
    hyd_win = window_stat(s, HYDROPHOBIC, 21)
    polyhis = polyhis_tag_score(s)
    ab = antibody_like_score(s)
    disorder = aa_fraction(s, DISORDER_PROMOTING)
    order = aa_fraction(s, ORDER_PROMOTING)
    return {
        "seq_kmer4_richness": len(kmers) / denom,
        "seq_nterm_hydrophobic_frac30": nterm_fraction(s, HYDROPHOBIC, 30),
        "seq_hydrophobic_window21_max_frac": hyd_win,
        "seq_polyhis_longest_run": polyhis,
        "seq_antibody_like_score": ab,
        "seq_gs_linker_fraction": aa_fraction(s, set("GS")),
        "seq_disorder_order_balance": disorder - order,
        "seq_long_hydrophobic_segment_flag": float(hyd_win >= 0.70) if math.isfinite(hyd_win) else np.nan,
        "seq_polyhis_tag_flag": float(polyhis >= 5),
        "seq_antibody_like_flag": float(ab >= 1.5),
    }


def classify_organism(org: Any) -> dict[str, float]:
    text = str(org or "").lower()
    if not text or text == "nan":
        return {
            "tax_human": np.nan,
            "tax_mammal_nonhuman": np.nan,
            "tax_bacteria": np.nan,
            "tax_virus_or_phage": np.nan,
            "tax_fungi": np.nan,
            "tax_plant": np.nan,
            "tax_synthetic_or_metagenome": np.nan,
            "tax_nonhuman_nonmodel_stress": np.nan,
        }
    human = "homo sapiens" in text
    mammal_terms = ("mus musculus", "macaca", "lama glama", "rattus", "bos taurus", "sus scrofa", "oryctolagus")
    bacteria_terms = (
        "escherichia",
        "staphylococcus",
        "klebsiella",
        "pseudomonas",
        "aliarcobacter",
        "enterococcus",
        "brucella",
        "nitrosospira",
        "legionella",
        "bacter",
    )
    virus = "virus" in text or "phage" in text or "coronavirus" in text or "hiv" in text
    fungi = "saccharomyces" in text or "candida" in text or "fung" in text
    plant = any(tok in text for tok in ("arabidopsis", "oryza", "zea", "cenchrus", "orobanche", "plant"))
    synth = "synthetic" in text or "metagenome" in text or "construct" in text
    bacteria = any(tok in text for tok in bacteria_terms)
    mammal_nonhuman = (not human) and any(tok in text for tok in mammal_terms)
    if human:
        stress = 0.0
    elif mammal_nonhuman or bacteria or fungi:
        stress = 1.0
    elif plant or virus or synth:
        stress = 2.0
    else:
        stress = 1.5
    return {
        "tax_human": float(human),
        "tax_mammal_nonhuman": float(mammal_nonhuman),
        "tax_bacteria": float(bacteria),
        "tax_virus_or_phage": float(virus),
        "tax_fungi": float(fungi),
        "tax_plant": float(plant),
        "tax_synthetic_or_metagenome": float(synth),
        "tax_nonhuman_nonmodel_stress": stress,
    }


def add_method_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    method = out.get("method", pd.Series(index=out.index, dtype=object)).fillna("").astype(str).str.upper()
    out["prov_method_xray"] = method.str.contains("X-RAY|XRAY|DIFFRACTION", regex=True).astype(float)
    out["prov_method_em"] = method.str.contains("ELECTRON|EM", regex=True).astype(float)
    out["prov_method_nmr"] = method.str.contains("NMR", regex=True).astype(float)
    resolution = pd.to_numeric(out.get("resolution"), errors="coerce")
    out["reference_quality_stress"] = resolution
    # A provenance/reference-quality ladder. This is kept in the screen but
    # marked as a likely benchmark confound during review.
    level = pd.Series(np.nan, index=out.index, dtype=float)
    level[(out["prov_method_xray"].eq(1.0)) & (resolution <= 2.0)] = 0
    level[(out["prov_method_xray"].eq(1.0)) & (resolution > 2.0)] = 1
    level[(out["prov_method_em"].eq(1.0)) & (resolution <= 3.0)] = 2
    level[(out["prov_method_em"].eq(1.0)) & (resolution > 3.0)] = 3
    level[out["prov_method_nmr"].eq(1.0)] = 1
    out["rcsb_reference_quality_level"] = level
    return out


def kmer_set(seq: Any, k: int = 4) -> set[str]:
    s = clean_sequence(seq)
    return {s[i : i + k] for i in range(max(0, len(s) - k + 1)) if set(s[i : i + k]).issubset(AA20)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return np.nan
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def add_neighborhood_features(df: pd.DataFrame, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    if "sequence_hash" not in out.columns:
        out["sequence_hash"] = out.get("sequence", "").fillna("").astype(str).str.upper().map(sha256_text)
    scored_sets = [kmer_set(s) for s in out["sequence"]]
    nearest = []
    cluster025 = []
    cluster040 = []
    for i, s in enumerate(scored_sets):
        sims = []
        for j, t in enumerate(scored_sets):
            if i == j:
                continue
            if out["sequence_hash"].iloc[i] == out["sequence_hash"].iloc[j]:
                continue
            sims.append(jaccard(s, t))
        sims = np.array([x for x in sims if math.isfinite(float(x))], dtype=float)
        nearest.append(float(sims.max()) if len(sims) else np.nan)
        cluster025.append(float((sims >= 0.25).sum()))
        cluster040.append(float((sims >= 0.40).sum()))
    out["seq_kmer4_nearest_jaccard_scored_loo"] = nearest
    out["seq_kmer4_cluster_count_j025_scored_loo"] = cluster025
    out["seq_kmer4_cluster_count_j040_scored_loo"] = cluster040
    out["seq_kmer4_isolation_stress_scored_loo"] = 1.0 - pd.to_numeric(out["seq_kmer4_nearest_jaccard_scored_loo"], errors="coerce")

    if pool is None or pool.empty:
        return out

    ref_sets = scored_sets
    ref_hashes = out["sequence_hash"].tolist()
    pool = pool.copy()
    if "sequence_hash" not in pool.columns:
        pool["sequence_hash"] = pool.get("sequence", "").fillna("").astype(str).str.upper().map(sha256_text)
    p_nearest = []
    p_count025 = []
    p_count040 = []
    for seq, h in zip(pool["sequence"], pool["sequence_hash"]):
        s = kmer_set(seq)
        sims = []
        for t, ref_h in zip(ref_sets, ref_hashes):
            if h == ref_h:
                continue
            sims.append(jaccard(s, t))
        sims = np.array([x for x in sims if math.isfinite(float(x))], dtype=float)
        p_nearest.append(float(sims.max()) if len(sims) else np.nan)
        p_count025.append(float((sims >= 0.25).sum()))
        p_count040.append(float((sims >= 0.40).sum()))
    pool["seq_kmer4_nearest_jaccard_to_scored"] = p_nearest
    pool["seq_kmer4_cluster_count_j025_to_scored"] = p_count025
    pool["seq_kmer4_cluster_count_j040_to_scored"] = p_count040
    pool["seq_kmer4_isolation_stress_to_scored"] = 1.0 - pd.to_numeric(pool["seq_kmer4_nearest_jaccard_to_scored"], errors="coerce")
    pool.to_csv(OUT_POOL, index=False)
    return out


def add_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sequence_upper"] = out.get("sequence", "").fillna("").astype(str).str.upper()
    out["sequence_hash"] = out["sequence_upper"].map(sha256_text)
    extras = pd.DataFrame([sequence_extra_features(s) for s in out["sequence_upper"]], index=out.index)
    for col in extras.columns:
        out[col] = extras[col]
    taxa = pd.DataFrame([classify_organism(o) for o in out.get("organism", pd.Series(index=out.index))], index=out.index)
    for col in taxa.columns:
        out[col] = taxa[col]
    out = add_method_features(out)
    return out


def assign_levels(values: pd.Series, direction: str, n_levels: int = 5) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    stress = x.copy()
    if direction == "low":
        stress = -stress
    levels = pd.Series(np.nan, index=values.index, dtype=float)
    mask = stress.notna()
    if mask.sum() < max(15, n_levels * 3) or stress[mask].nunique() < 3:
        return levels
    try:
        q = pd.qcut(stress[mask].rank(method="first"), q=n_levels, labels=False, duplicates="drop")
    except Exception:
        return levels
    levels.loc[mask] = q.astype(float)
    return levels


def spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 5 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return None, None
    if stats is not None:
        rho, p = stats.spearmanr(a[mask], b[mask])
        return finite_float(rho), finite_float(p)
    return finite_float(a[mask].rank().corr(b[mask].rank())), None


def ols_axis_coef(df: pd.DataFrame, feature: str, direction: str, controls: list[str]) -> dict[str, Any]:
    y_col = "ca_lddt"
    cols = [feature, y_col] + [c for c in controls if c in df.columns and c != feature]
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(18, len(cols) + 5) or sub[feature].nunique() < 3:
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None, "controls": controls}
    axis = sub[feature].astype(float)
    if direction == "low":
        axis = -axis
    x_cols = []
    for arr in [axis] + [sub[c].astype(float) for c in controls if c in sub.columns and c != feature]:
        sd = float(np.nanstd(arr, ddof=0))
        if not math.isfinite(sd) or sd == 0:
            continue
        x_cols.append(((arr - float(np.nanmean(arr))) / sd).to_numpy())
    if not x_cols:
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None, "controls": controls}
    X = np.column_stack([np.ones(len(sub))] + x_cols)
    y = sub[y_col].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return {"n": int(len(sub)), "coef": finite_float(beta[1]), "se": None, "t": None, "controls": controls}
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    coef = float(beta[1])
    return {"n": int(len(sub)), "coef": coef, "se": se, "t": coef / se if se else None, "controls": controls}


def bootstrap_diff(high: np.ndarray, low: np.ndarray, seed: int = 123, n_boot: int = 1000) -> dict[str, Any]:
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if len(high) < 8 or len(low) < 20:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_high": int(len(high)), "n_low": int(len(low))}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(float(rng.choice(high, len(high), replace=True).mean() - rng.choice(low, len(low), replace=True).mean()))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
    }


def score_axis(df: pd.DataFrame, feature: str, direction: str, source: str, n_levels: int = 5) -> dict[str, Any] | None:
    sub = df[df["evidence_source"].astype(str).eq(source)].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub[sub["ca_lddt"].notna() & sub[feature].notna()].copy()
    if len(sub) < 30 or sub[feature].nunique() < 3:
        return None
    sub["axis_level"] = assign_levels(sub[feature], direction, n_levels)
    sub = sub[sub["axis_level"].notna()].copy()
    if sub.empty:
        return None
    sub["axis_level"] = sub["axis_level"].astype(int)
    levels = sorted(sub["axis_level"].unique())
    if len(levels) < 3:
        return None
    curve = []
    for level in levels:
        part = sub[sub["axis_level"].eq(level)]
        curve.append(
            {
                "level": int(level),
                "n": int(len(part)),
                "feature_mean": finite_float(part[feature].mean()),
                "feature_median": finite_float(part[feature].median()),
                "ca_lddt_mean": finite_float(part["ca_lddt"].mean()),
                "gdt_ts_mean": finite_float(part["gdt_ts"].mean()),
                "ca_rmsd_median": finite_float(part["ca_rmsd"].median()),
                "baseline_ca_lddt_mean": finite_float(part["baseline_ca_lddt"].mean()) if "baseline_ca_lddt" in part else None,
                "seq_len_median": finite_float(pd.to_numeric(part["seq_len"], errors="coerce").median()) if "seq_len" in part else None,
                "resolution_median": finite_float(pd.to_numeric(part["resolution"], errors="coerce").median()) if "resolution" in part else None,
                "entry_unique": int(part["entry_id"].nunique(dropna=True)) if "entry_id" in part else None,
                "max_entry_count": int(part["entry_id"].value_counts(dropna=True).max()) if "entry_id" in part and part["entry_id"].notna().any() else None,
            }
        )
    curve_df = pd.DataFrame(curve)
    min_level_n = int(curve_df["n"].min())
    high = sub[sub["axis_level"].eq(max(levels))]
    lower = sub[~sub["axis_level"].eq(max(levels))]
    low = sub[sub["axis_level"].eq(min(levels))]
    stress = sub[feature] if direction == "high" else -sub[feature]
    rho, p = spearman(stress, sub["ca_lddt"])
    lr_controls = ["seq_len", "resolution"]
    comp_controls = ["seq_len", "resolution", "calc_seq_abs_charge_per_residue", "fraction_charged_residues_fcr"]
    ca_high = high["ca_lddt"].to_numpy(dtype=float)
    ca_lower = lower["ca_lddt"].to_numpy(dtype=float)
    ca_high = ca_high[np.isfinite(ca_high)]
    ca_lower = ca_lower[np.isfinite(ca_lower)]
    bootstrap_placeholder = {
        "mean": finite_float(np.mean(ca_high) - np.mean(ca_lower)) if len(ca_high) and len(ca_lower) else None,
        "ci_low": None,
        "ci_high": None,
        "n_high": int(len(ca_high)),
        "n_low": int(len(ca_lower)),
    }
    return {
        "source": source,
        "feature": feature,
        "direction": direction,
        "n": int(len(sub)),
        "n_levels": int(len(levels)),
        "min_level_n": min_level_n,
        "level_counts": ",".join(str(int(x)) for x in curve_df["n"]),
        "measured_order_ok": bool(curve_df["feature_median"].is_monotonic_increasing if direction == "high" else curve_df["feature_median"].is_monotonic_decreasing),
        "feature_top_mean": finite_float(high[feature].mean()),
        "feature_low_mean": finite_float(low[feature].mean()),
        "top_minus_lower_ca_lddt": finite_float(high["ca_lddt"].mean() - lower["ca_lddt"].mean()),
        "top_minus_lowest_ca_lddt": finite_float(high["ca_lddt"].mean() - low["ca_lddt"].mean()),
        "top_minus_lower_gdt_ts": finite_float(high["gdt_ts"].mean() - lower["gdt_ts"].mean()),
        "top_minus_lower_ca_rmsd_median": finite_float(high["ca_rmsd"].median() - lower["ca_rmsd"].median()),
        "top_minus_lower_baseline_ca_lddt": finite_float(high["baseline_ca_lddt"].mean() - lower["baseline_ca_lddt"].mean()) if "baseline_ca_lddt" in sub else None,
        "spearman_stress_vs_ca_lddt": rho,
        "spearman_stress_vs_ca_lddt_p": p,
        "bootstrap_mean": bootstrap_placeholder["mean"],
        "bootstrap_ci_low": bootstrap_placeholder["ci_low"],
        "bootstrap_ci_high": bootstrap_placeholder["ci_high"],
        "bootstrap_n_high": bootstrap_placeholder["n_high"],
        "bootstrap_n_low": bootstrap_placeholder["n_low"],
        "adjusted_len_res": ols_axis_coef(sub, feature, direction, lr_controls),
        "adjusted_composition": ols_axis_coef(sub, feature, direction, comp_controls),
    }


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    features = []
    for col in df.columns:
        low = col.lower()
        if any(tok in low for tok in PROHIBITED_PATTERNS):
            continue
        if col.startswith("axis_"):
            continue
        if col in {
            "target_id",
            "cameo_target",
            "sequence",
            "sequence_upper",
            "sequence_hash",
            "pdb_path",
            "msa_path",
            "ref_pdb_chain",
            "ref_pdb_id",
            "ref_chain",
            "actual_ref_chain",
            "chain_mapping_note",
            "method",
            "organism",
            "source",
            "broad_source",
            "evidence_source",
            "evidence_root",
            "target_result_origin",
            "result_panel",
            "feature_panel",
            "inference_setting",
            "selection_policy",
            "entry_id",
            "rcsb_instance_id",
            "entity_id",
            "asym_id",
            "auth_asym_id",
            "polymer_type",
            "release_date",
            "submission_date",
            "pdb_ca_parse_status",
            "structure_format",
            "scaling_group",
        }:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() < 40:
            continue
        if vals.nunique(dropna=True) < 3:
            continue
        features.append(col)
    return sorted(set(features))


def flatten(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {}
        for k, v in rec.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    row[f"{k}_{kk}"] = vv
            else:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def mine_outcomes(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for source, part in df.groupby("evidence_source"):
        part = part[pd.to_numeric(part["ca_lddt"], errors="coerce").notna()].copy()
        if len(part) < 30:
            continue
        y = pd.to_numeric(part["ca_lddt"], errors="coerce")
        fail_cut = float(y.quantile(0.20))
        success_cut = float(y.quantile(0.80))
        fail = part[y <= fail_cut]
        success = part[y >= success_cut]
        for feature in features:
            f = pd.to_numeric(fail[feature], errors="coerce").dropna()
            s = pd.to_numeric(success[feature], errors="coerce").dropna()
            if len(f) < 8 or len(s) < 8:
                continue
            pooled = pd.to_numeric(part[feature], errors="coerce").std()
            if not pooled or not math.isfinite(float(pooled)):
                continue
            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "fail_n": int(len(f)),
                    "success_n": int(len(s)),
                    "fail_mean": float(f.mean()),
                    "success_mean": float(s.mean()),
                    "fail_minus_success": float(f.mean() - s.mean()),
                    "abs_standardized_gap": abs(float(f.mean() - s.mean())) / float(pooled),
                    "fail_cut_ca_lddt": fail_cut,
                    "success_cut_ca_lddt": success_cut,
                }
            )
    return pd.DataFrame(rows).sort_values("abs_standardized_gap", ascending=False)


def family_excluded(feature: str) -> str | None:
    low = feature.lower()
    for pat in FAILED_FAMILY_PATTERNS:
        if pat in low:
            return f"failed_family:{pat}"
    if feature in REFERENCE_QUALITY_OR_LABEL_PROXIES:
        return "reference_quality_or_label_proxy"
    return None


def review_candidates(screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if screen.empty:
        return pd.DataFrame(rows)
    usable = screen[
        screen["measured_order_ok"].eq(True)
        & screen["min_level_n"].ge(8)
        & screen["top_minus_lower_ca_lddt"].notna()
    ].copy()
    for (feature, direction), part in usable.groupby(["feature", "direction"]):
        exclusion = family_excluded(feature)
        neg = part[part["top_minus_lower_ca_lddt"] <= -0.04].copy()
        pos = part[part["top_minus_lower_ca_lddt"] >= 0.03].copy()
        broad_support = sorted(set(neg["source"]) & {"CAMEO_default", "RCSB_scaled_combined"})
        fresh_support = sorted(set(neg["source"]) & {"RCSB_fresh_charge_confirmation", "RCSB_acidic_window_confirmation"})
        reversals = sorted(pos["source"].dropna().astype(str).unique())
        support_sources = sorted(neg["source"].dropna().astype(str).unique())
        adjusted_negative = part[pd.to_numeric(part.get("adjusted_len_res_coef"), errors="coerce").fillna(0) < 0]
        baseline_abs = neg["top_minus_lower_baseline_ca_lddt"].abs().max() if len(neg) else np.nan
        best = part.sort_values("top_minus_lower_ca_lddt").head(1).iloc[0]
        decision = "reject"
        reason = ""
        if exclusion:
            reason = exclusion
        elif reversals:
            reason = "reverses_across_meaningful_slices:" + ",".join(reversals)
        elif len(support_sources) < 2:
            reason = "single_slice_or_cherry_pick"
        elif not broad_support:
            reason = "no_broad_panel_support"
        elif not fresh_support and len(broad_support) < 2:
            reason = "no_fresh_or_second_broad_support"
        elif len(adjusted_negative) < 2:
            reason = "not_stable_after_length_resolution_controls"
        elif math.isfinite(float(baseline_abs)) and baseline_abs > 0.04:
            reason = "baseline_moves_with_axis"
        else:
            decision = "live_requires_freeze_and_confirmation"
            reason = "same_direction_negative_in_multiple_slices"
        rows.append(
            {
                "feature": feature,
                "direction": direction,
                "decision": decision,
                "reason": reason,
                "support_sources": ",".join(support_sources),
                "reversal_sources": ",".join(reversals),
                "best_source": best["source"],
                "best_top_minus_lower_ca_lddt": best["top_minus_lower_ca_lddt"],
                "best_adjusted_len_res_coef": best.get("adjusted_len_res_coef"),
                "min_support_effect": neg["top_minus_lower_ca_lddt"].min() if len(neg) else np.nan,
                "max_reversal_effect": pos["top_minus_lower_ca_lddt"].max() if len(pos) else np.nan,
                "max_abs_baseline_support_effect": baseline_abs,
                "n_support_sources": len(support_sources),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision", "n_support_sources", "min_support_effect"],
        ascending=[True, False, True],
    )


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.load(open(path))
    return default


def write_state_and_ledger(df: pd.DataFrame, screen: pd.DataFrame, review: pd.DataFrame, mining: pd.DataFrame) -> None:
    live = review[review["decision"].eq("live_requires_freeze_and_confirmation")].to_dict(orient="records") if len(review) else []
    state = {
        "created_at_utc": now(),
        "phase": "expanded_all_scored_evidence_feature_discovery",
        "question_mode": "broad_beyond_paper_generalizability_discovery",
        "all_scored_rows": int(len(df)),
        "finite_ca_lddt_rows": int(pd.to_numeric(df["ca_lddt"], errors="coerce").notna().sum()),
        "evidence_source_counts": {str(k): int(v) for k, v in df["evidence_source"].value_counts(dropna=False).items()},
        "artifacts": {
            "expanded_all_scored_evidence_table": str(OUT_COMBINED),
            "expanded_candidate_pool_features": str(OUT_POOL),
            "expanded_all_scored_axis_screen": str(OUT_SCREEN),
            "expanded_candidate_hypothesis_review": str(OUT_REVIEW),
            "expanded_outcome_mining": str(OUT_MINING),
            "prior_combined_evidence_table": str(COMBINED_IN),
        },
        "excluded_axis_families": list(FAILED_FAMILY_PATTERNS),
        "reference_quality_proxy_features": sorted(REFERENCE_QUALITY_OR_LABEL_PROXIES),
        "live_followup_hypotheses": live,
        "closure_gate": {
            "terminal_passed": False,
            "final_report_written": False,
            "reason": "Expanded all-scored discovery is complete; any live hypothesis still requires frozen confirmation on new or explicitly adequate evidence.",
        },
    }
    OUT_STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    prior_ledger = load_json(LATEST / "axis_ledger.json", {"entries": []})
    ledger = {
        "created_at_utc": now(),
        "combined_evidence_table": str(OUT_COMBINED),
        "prior_axis_ledger": str(LATEST / "axis_ledger.json"),
        "duplicate_diagnostics": {
            "n_rows": int(len(df)),
            "unique_targets": int(df["target_id"].nunique()),
            "unique_exact_sequences": int(df["sequence_hash"].nunique()),
            "max_target_count": int(df["target_id"].value_counts().max()),
            "max_exact_sequence_count": int(df["sequence_hash"].value_counts().max()),
        },
        "entries": prior_ledger.get("entries", []),
    }
    ledger["entries"] = [e for e in ledger["entries"] if e.get("axis_id") != "expanded_feature_discovery_review"]
    ledger["entries"].append(
        {
            "axis_id": "expanded_feature_discovery_review",
            "category": "all_scored_evidence_discovery",
            "status": "live_followup_found" if live else "no_claim_valid_candidate_after_expanded_features",
            "prospective": True,
            "outcome_informed": True,
            "frozen_before_target_scoring": False,
            "screen_artifact": str(OUT_SCREEN),
            "review_artifact": str(OUT_REVIEW),
            "outcome_mining_artifact": str(OUT_MINING),
            "derived_feature_families": [
                "taxonomy/provenance heuristics",
                "antibody-like and tag motifs",
                "sequence-neighborhood k-mer homology proxies",
                "hydrophobic segment and linker proxies",
            ],
            "live_followup_hypotheses": live,
            "notes": [
                "Charge, local acidic-window, disorder/Uversky, repeat/dipeptide, and flexible-nonhelical families were excluded because prior fresh or expanded confirmations downgraded them.",
                "Reference-quality proxies such as resolution are screened but not eligible as broad sequence-generalization axes without a narrower benchmark-provenance claim boundary.",
                "Candidate axes with positive reversals in meaningful evidence slices are rejected for broad claim closure.",
            ],
        }
    )
    OUT_LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    commands = load_json(OUT_COMMANDS, [])
    commands.append(
        {
            "command": "/ewsc/yektefai/envs/envs/pgt/bin/python scripts/expand_all_scored_features.py",
            "cwd": str(ROOT),
            "purpose": "Derive additional prospective features and run all-scored-evidence axis discovery after failed acidic-window confirmation.",
            "status": "completed",
        }
    )
    OUT_COMMANDS.write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(COMBINED_IN)
    if "evidence_source" not in df.columns:
        df["evidence_source"] = df.get("source", "unknown")
    df = add_expanded_features(df)

    pool = pd.read_csv(RCSB_POOL) if RCSB_POOL.exists() else pd.DataFrame()
    if not pool.empty:
        pool = add_expanded_features(pool)
    df = add_neighborhood_features(df, pool)
    df.to_csv(OUT_COMBINED, index=False)

    features = candidate_feature_columns(df)
    records: list[dict[str, Any]] = []
    for source in sorted(df["evidence_source"].dropna().astype(str).unique()):
        for feature in features:
            for direction in ("high", "low"):
                rec = score_axis(df, feature, direction, source)
                if rec:
                    records.append(rec)
    screen = flatten(records)
    screen.to_csv(OUT_SCREEN, index=False)
    mining = mine_outcomes(df, features)
    mining.to_csv(OUT_MINING, index=False)
    review = review_candidates(screen)
    review.to_csv(OUT_REVIEW, index=False)
    write_state_and_ledger(df, screen, review, mining)
    live = review[review["decision"].eq("live_requires_freeze_and_confirmation")].to_dict(orient="records") if len(review) else []
    print(
        json.dumps(
            {
                "expanded_rows": int(len(df)),
                "features_screened": int(len(features)),
                "screen_rows": int(len(screen)),
                "live_followup_hypotheses": live[:5],
                "artifacts": {
                    "expanded_table": str(OUT_COMBINED),
                    "screen": str(OUT_SCREEN),
                    "review": str(OUT_REVIEW),
                    "candidate_pool": str(OUT_POOL),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
