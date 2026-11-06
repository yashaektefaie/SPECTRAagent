#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POLAR = set("STNQYC")
BASIC = set("KRH")
ACIDIC = set("DE")
TINY = set("AGS")
DISORDER_PROMOTING = set("PGSENQKRD")
ORDER_PROMOTING = set("WCMFYILV")
KNOWN_CONTROL_FEATURES = [
    "msa_log10_nonquery_rows",
    "msa_top_nonquery_identity",
    "seq_len",
    "difficulty",
    "flexible_nonhelical_composite_rank",
]


def stable_fold(value: str, n: int = 10) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def aggregate_scaled_results(path: Path, pool_name: str) -> pd.DataFrame:
    raw = read_csv(path).copy()
    ok = raw[raw["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError(f"No ok scaled rows in {path}")
    idx = ok.groupby("target_id")["ptm"].idxmax()
    best = ok.loc[idx].copy()
    best["aggregation"] = "best_by_ptm"
    best["evaluation_pool"] = pool_name
    # Keep mean summaries for diagnostics without changing the main metric.
    means = ok.groupby("target_id").agg(
        scaled_mean_ca_lddt=("ca_lddt", "mean"),
        scaled_sd_ca_lddt=("ca_lddt", "std"),
        scaled_mean_gdt_ts=("gdt_ts", "mean"),
        scaled_mean_ptm=("ptm", "mean"),
        scaled_n_samples=("sample_index", "nunique"),
    )
    best = best.merge(means, left_on="target_id", right_index=True, how="left")
    return best


def aa_fractions(seq: str) -> dict:
    seq = str(seq or "").upper()
    n = len(seq)
    if n == 0:
        return {}
    counts = {aa: seq.count(aa) for aa in AA}
    charged = counts["D"] + counts["E"] + counts["K"] + counts["R"] + counts["H"]
    return {
        "aa20_fraction": sum(counts.values()) / n,
        "basic_fraction": sum(counts[a] for a in BASIC) / n,
        "acidic_fraction": sum(counts[a] for a in ACIDIC) / n,
        "net_charge_per_residue": (counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]) / n,
        "abs_charge_per_residue": abs(counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]) / n,
        "fraction_charged_residues_fcr": charged / n,
        "charged_fraction": charged / n,
        "hydrophobic_fraction": sum(counts[a] for a in HYDROPHOBIC) / n,
        "aromatic_fraction": sum(counts[a] for a in AROMATIC) / n,
        "polar_fraction": sum(counts[a] for a in POLAR) / n,
        "proline_fraction": counts["P"] / n,
        "cys_fraction": counts["C"] / n,
        "seq_cys_fraction": counts["C"] / n,
        "gly_fraction": counts["G"] / n,
        "gly_pro_fraction": (counts["G"] + counts["P"]) / n,
        "tiny_fraction": sum(counts[a] for a in TINY) / n,
        "disorder_promoting_fraction": sum(counts[a] for a in DISORDER_PROMOTING) / n,
        "order_promoting_fraction": sum(counts[a] for a in ORDER_PROMOTING) / n,
    }


def longest_run(seq: str, chars: set[str] | None = None) -> int:
    best = cur = 0
    prev = None
    for ch in str(seq or "").upper():
        hit = ch in chars if chars is not None else ch == prev
        if chars is None:
            cur = cur + 1 if ch == prev else 1
            prev = ch
        else:
            cur = cur + 1 if hit else 0
        best = max(best, cur)
    return best


def sequence_extra_features(seq: str) -> dict:
    seq = str(seq or "").upper()
    n = len(seq)
    out = aa_fractions(seq)
    if n == 0:
        return out
    cysteine_positions = [i for i, ch in enumerate(seq) if ch == "C"]
    motifs = {
        "cys_count": len(cysteine_positions),
        "cys_pair_capacity": len(cysteine_positions) // 2,
        "cys_even_pairable": int(len(cysteine_positions) >= 2 and len(cysteine_positions) % 2 == 0),
        "cxxc_motif_count": sum(1 for i in range(n - 3) if seq[i] == "C" and seq[i + 3] == "C"),
        "cxxxh_motif_count": sum(1 for i in range(n - 4) if seq[i] == "C" and seq[i + 4] == "H"),
        "histidine_fraction": seq.count("H") / n,
        "cys_his_fraction": (seq.count("C") + seq.count("H")) / n,
        "max_cys_spacing_norm": 0.0,
        "min_cys_spacing_norm": 0.0,
        "longest_low_complexity_run_norm": max(
            longest_run(seq, set("GP")),
            longest_run(seq, set("ST")),
            longest_run(seq, set("KR")),
            longest_run(seq, set("DE")),
        ) / n,
        "longest_homopolymer_run_norm": longest_run(seq) / n,
        "low_length_extreme_score": 1.0 / math.sqrt(n),
        "log_seq_len": math.log10(n),
    }
    if len(cysteine_positions) >= 2:
        spacings = np.diff(cysteine_positions)
        motifs["max_cys_spacing_norm"] = float(np.max(spacings) / n)
        motifs["min_cys_spacing_norm"] = float(np.min(spacings) / n)
    out.update(motifs)
    return out


def ensure_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [sequence_extra_features(seq) for seq in df.get("sequence", pd.Series([""] * len(df))).fillna("")]
    extra = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in extra.columns:
        if col not in out.columns or out[col].isna().all():
            out[col] = extra[col]
        else:
            out[col] = out[col].fillna(extra[col])
    if "seq_len" not in out.columns:
        out["seq_len"] = out["sequence"].fillna("").str.len()
    out["length_extreme_from_cameo_median"] = np.nan
    return out


def parse_ca_coords(path: Path) -> np.ndarray:
    coords = []
    if not path.exists():
        return np.empty((0, 3), dtype=float)
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    return np.asarray(coords, dtype=float)


def topology_from_coords(coords: np.ndarray) -> dict:
    n = int(coords.shape[0])
    out = {
        "pdb_ca_count": n,
        "pdb_ca_parse_status": "ok" if n >= 3 else "too_few_ca",
        "rg": np.nan,
        "rg_len13_norm": np.nan,
        "end_to_end_distance": np.nan,
        "end_to_end_over_rg": np.nan,
        "shape_anisotropy": np.nan,
        "contact_count_ca8_seqsep3": np.nan,
        "contacts_per_residue": np.nan,
        "contact_density_ca8_seqsep3": np.nan,
        "relative_contact_order_ca8": np.nan,
        "long_contacts_per_residue": np.nan,
        "long_contact_fraction": np.nan,
        "medium_contact_fraction": np.nan,
        "local_contact_fraction": np.nan,
    }
    if n < 3:
        return out
    centered = coords - coords.mean(axis=0, keepdims=True)
    rg = float(np.sqrt((centered * centered).sum(axis=1).mean()))
    cov = np.cov(centered.T)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    out["rg"] = rg
    out["rg_len13_norm"] = rg / (n ** (1.0 / 3.0))
    out["end_to_end_distance"] = float(np.linalg.norm(coords[-1] - coords[0]))
    out["end_to_end_over_rg"] = out["end_to_end_distance"] / rg if rg > 0 else np.nan
    out["shape_anisotropy"] = float(eig[-1] / eig.sum()) if eig.sum() > 0 else np.nan
    seps = []
    for i in range(n - 3):
        d = np.linalg.norm(coords[i + 3 :] - coords[i], axis=1)
        hits = np.where(d <= 8.0)[0]
        if hits.size:
            seps.extend((hits + 3).tolist())
    if seps:
        seps_arr = np.asarray(seps, dtype=float)
        c = len(seps)
        out["contact_count_ca8_seqsep3"] = c
        out["contacts_per_residue"] = c / n
        out["contact_density_ca8_seqsep3"] = c / max(1, (n * (n - 1) / 2))
        out["relative_contact_order_ca8"] = float(seps_arr.mean() / n)
        long_mask = seps_arr >= 24
        med_mask = (seps_arr >= 12) & (seps_arr < 24)
        local_mask = seps_arr < 12
        out["long_contacts_per_residue"] = float(long_mask.sum() / n)
        out["long_contact_fraction"] = float(long_mask.mean())
        out["medium_contact_fraction"] = float(med_mask.mean())
        out["local_contact_fraction"] = float(local_mask.mean())
    else:
        out.update(
            {
                "contact_count_ca8_seqsep3": 0,
                "contacts_per_residue": 0.0,
                "contact_density_ca8_seqsep3": 0.0,
                "relative_contact_order_ca8": 0.0,
                "long_contacts_per_residue": 0.0,
                "long_contact_fraction": 0.0,
                "medium_contact_fraction": 0.0,
                "local_contact_fraction": 0.0,
            }
        )
    return out


def add_external_topology(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = df.copy()
    topo_cols = [
        "pdb_ca_count",
        "pdb_ca_parse_status",
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
    ]
    if all(c in out.columns and out[c].notna().any() for c in topo_cols):
        return out
    rows = []
    for _, row in out.iterrows():
        p = row.get("pdb_path")
        if not isinstance(p, str) or not p:
            rows.append(topology_from_coords(np.empty((0, 3))))
            continue
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        rows.append(topology_from_coords(parse_ca_coords(path)))
    topo = pd.DataFrame(rows, index=out.index)
    for col in topo.columns:
        if col not in out.columns or out[col].isna().all():
            out[col] = topo[col]
        else:
            out[col] = out[col].fillna(topo[col])
    return out


def add_common_derived(df: pd.DataFrame, cameo_len_median: float) -> pd.DataFrame:
    out = df.copy()
    out["length_extreme_from_cameo_median"] = (np.log10(out["seq_len"].clip(lower=1)) - math.log10(cameo_len_median)).abs()
    if "disorder_minus_order_fraction" not in out.columns:
        out["disorder_minus_order_fraction"] = out["disorder_promoting_fraction"] - out["order_promoting_fraction"]
    if "cys_fraction" not in out.columns and "seq_cys_fraction" in out.columns:
        out["cys_fraction"] = out["seq_cys_fraction"]
    out["cys_disulfide_proxy"] = out.get("cys_fraction", 0).fillna(0) * np.log1p(out.get("cys_pair_capacity", 0).fillna(0))
    out["metal_cys_his_proxy"] = out.get("cys_his_fraction", 0).fillna(0) + 0.05 * out.get("cxxc_motif_count", 0).fillna(0)
    out["charge_hydropathy_extreme"] = (
        zscore_series(out.get("abs_charge_per_residue", pd.Series(np.nan, index=out.index)))
        - zscore_series(out.get("hydropathy_norm_uversky", pd.Series(np.nan, index=out.index)))
    )
    out["compact_high_contact_proxy"] = zscore_series(out.get("contacts_per_residue", pd.Series(np.nan, index=out.index))) - zscore_series(
        out.get("rg_len13_norm", pd.Series(np.nan, index=out.index))
    )
    return out


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def merge_features_results(features: pd.DataFrame, results: pd.DataFrame, pool: str, dataset: str) -> pd.DataFrame:
    res = results.copy()
    res["evaluation_pool"] = pool
    res["dataset"] = dataset
    merged = features.merge(res, on="target_id", how="inner", suffixes=("", "_result"))
    merged["analysis_id"] = dataset + ":" + merged["target_id"].astype(str)
    merged["hash_fold10"] = merged["target_id"].map(stable_fold)
    return merged


def make_quantile_levels(s: pd.Series, q: int = 5, higher_is_harder: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    valid = x.dropna()
    if valid.nunique() < 3 or len(valid) < 12:
        return out
    bins = min(q, valid.nunique())
    try:
        labels = pd.qcut(valid.rank(method="first"), bins, labels=False, duplicates="drop")
    except ValueError:
        return out
    if labels.nunique() < 3:
        return out
    vals = labels.astype(float)
    if not higher_is_harder:
        vals = vals.max() - vals
    out.loc[valid.index] = vals
    return out


def residualize(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    xdf = controls.copy()
    xdf = xdf.apply(pd.to_numeric, errors="coerce")
    mask = y.notna()
    for col in xdf.columns:
        mask &= xdf[col].notna()
    resid = pd.Series(np.nan, index=y.index)
    if mask.sum() < max(12, xdf.shape[1] + 4):
        return resid
    X = np.column_stack([np.ones(mask.sum()), xdf.loc[mask].to_numpy(dtype=float)])
    yy = y.loc[mask].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    except np.linalg.LinAlgError:
        return resid
    resid.loc[mask] = yy - X @ beta
    return resid


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if mask.sum() < 5 or aa.loc[mask].nunique() < 2 or bb.loc[mask].nunique() < 2:
        return np.nan
    return float(aa.loc[mask].rank().corr(bb.loc[mask].rank()))


def score_axis(
    df: pd.DataFrame,
    axis_name: str,
    feature: str,
    direction: str,
    subset_name: str,
    metric: str = "ca_lddt",
    q: int = 5,
) -> tuple[dict, pd.DataFrame]:
    if feature not in df.columns:
        return {}, pd.DataFrame()
    higher = direction != "low"
    levels = make_quantile_levels(df[feature], q=q, higher_is_harder=higher)
    work = df.copy()
    work["axis_level"] = levels
    work = work[work["axis_level"].notna() & work[metric].notna()].copy()
    if work["axis_level"].nunique() < 3:
        return {}, pd.DataFrame()
    work["axis_level"] = work["axis_level"].astype(int)
    low_level = int(work["axis_level"].min())
    high_level = int(work["axis_level"].max())
    ctrl_cols = [c for c in KNOWN_CONTROL_FEATURES if c in work.columns and c != feature]
    resid = residualize(work[metric], work[ctrl_cols]) if ctrl_cols else pd.Series(np.nan, index=work.index)
    work["_metric_resid"] = resid
    group = (
        work.groupby("axis_level")
        .agg(
            n=("target_id", "nunique"),
            feature_mean=(feature, "mean"),
            feature_min=(feature, "min"),
            feature_max=(feature, "max"),
            ca_lddt_mean=("ca_lddt", "mean"),
            ca_lddt_median=("ca_lddt", "median"),
            gdt_ts_mean=("gdt_ts", "mean"),
            baseline_ca_lddt_mean=("baseline_ca_lddt", "mean"),
            seq_len_mean=("seq_len", "mean"),
            msa_log10_nonquery_rows_mean=("msa_log10_nonquery_rows", "mean"),
            msa_top_nonquery_identity_mean=("msa_top_nonquery_identity", "mean"),
            difficulty_mean=("difficulty", "mean"),
            flexible_nonhelical_composite_rank_mean=("flexible_nonhelical_composite_rank", "mean"),
            metric_resid_mean=("_metric_resid", "mean"),
        )
        .reset_index()
    )
    low = group[group["axis_level"].eq(low_level)].iloc[0]
    high = group[group["axis_level"].eq(high_level)].iloc[0]
    lddt_drop = float(low["ca_lddt_mean"] - high["ca_lddt_mean"])
    baseline_drop = float(low["baseline_ca_lddt_mean"] - high["baseline_ca_lddt_mean"])
    resid_drop = float(low["metric_resid_mean"] - high["metric_resid_mean"]) if np.isfinite(low["metric_resid_mean"]) and np.isfinite(high["metric_resid_mean"]) else np.nan
    monotone_corr = spearman_corr(group["axis_level"], group["ca_lddt_mean"])
    feature_order_corr = spearman_corr(group["axis_level"], group["feature_mean"])
    control_corrs = {
        f"level_corr_{c}": spearman_corr(group["axis_level"], group[f"{c}_mean"])
        for c in ["seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty", "flexible_nonhelical_composite_rank"]
        if f"{c}_mean" in group.columns
    }
    summary = {
        "axis_name": axis_name,
        "feature": feature,
        "direction": direction,
        "subset": subset_name,
        "q": q,
        "n": int(work["target_id"].nunique()),
        "levels": int(work["axis_level"].nunique()),
        "min_level_n": int(group["n"].min()),
        "feature_order_corr": feature_order_corr,
        "ca_lddt_drop_easy_minus_hard": lddt_drop,
        "baseline_ca_lddt_drop_easy_minus_hard": baseline_drop,
        "target_minus_baseline_drop": lddt_drop - baseline_drop,
        "residual_ca_lddt_drop_easy_minus_hard": resid_drop,
        "level_vs_ca_lddt_spearman": monotone_corr,
        "easy_level": low_level,
        "hard_level": high_level,
        "easy_feature_mean": float(low["feature_mean"]),
        "hard_feature_mean": float(high["feature_mean"]),
        "easy_ca_lddt_mean": float(low["ca_lddt_mean"]),
        "hard_ca_lddt_mean": float(high["ca_lddt_mean"]),
    }
    summary.update(control_corrs)
    group.insert(0, "axis_name", axis_name)
    group.insert(1, "feature", feature)
    group.insert(2, "direction", direction)
    group.insert(3, "subset", subset_name)
    return summary, group


def build_tables(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single = Path(args.single_root)
    beyond = Path(args.beyond_root)
    out_root = Path(args.out_root)
    cameo_features = read_csv(beyond / "spectra_ready_dataset/prospective_features.followup.csv")
    cameo_features = ensure_sequence_features(cameo_features)
    cameo_len_median = float(cameo_features["seq_len"].median())
    cameo_features = add_common_derived(cameo_features, cameo_len_median)
    cameo_results = read_csv(single / "target_model_results.csv")
    cameo_default = merge_features_results(cameo_features, cameo_results, "cameo_default_numloops3_steps14_one_sample", "cameo")

    scaled_features = read_csv(beyond / "inference_time_scaling_10loops68steps_5samples_full_top_tail_matched/spectra_ready_dataset/candidate_features.csv")
    scaled_features = scaled_features.drop(columns=[c for c in ["ca_lddt", "gdt_ts", "ca_rmsd", "baseline_ca_lddt"] if c in scaled_features.columns])
    scaled_features = scaled_features.merge(
        cameo_features.drop(columns=[c for c in scaled_features.columns if c != "target_id"], errors="ignore"),
        on="target_id",
        how="left",
    )
    scaled_features = ensure_sequence_features(scaled_features)
    scaled_features = add_common_derived(scaled_features, cameo_len_median)
    scaled_results = aggregate_scaled_results(
        beyond / "inference_time_scaling_10loops68steps_5samples_full_top_tail_matched/target_model_results.scaled_samples.csv",
        "cameo_scaled_10loops68steps_5samples_best_ptm",
    )
    cameo_scaled = merge_features_results(scaled_features, scaled_results, "cameo_scaled_10loops68steps_5samples_best_ptm", "cameo_scaled")

    external_root = beyond / "external_rcsb_same_bucket_expansion_20260531"
    external_features = read_csv(external_root / "spectra_ready_dataset/candidate_features.csv")
    external_features = ensure_sequence_features(external_features)
    external_features = add_external_topology(external_features, external_root)
    external_features = add_common_derived(external_features, cameo_len_median)
    external_results = aggregate_scaled_results(
        external_root / "target_model_results.scaled_samples.csv",
        "external_rcsb_scaled_10loops68steps_5samples_best_ptm",
    )
    external_scaled = merge_features_results(
        external_features,
        external_results,
        "external_rcsb_scaled_10loops68steps_5samples_best_ptm",
        "external_rcsb",
    )

    all_rows = pd.concat([cameo_default, cameo_scaled, external_scaled], ignore_index=True, sort=False)
    target_cols = [
        "analysis_id",
        "dataset",
        "evaluation_pool",
        "target_id",
        "sample_index",
        "status",
        "aggregation",
        "ptm",
        "plddt_mean",
        "plddt_ca_mean",
        "ca_lddt",
        "gdt_ts",
        "ca_rmsd",
        "baseline_ca_lddt",
        "baseline_gdt_ts",
        "baseline_ca_rmsd",
        "scaled_mean_ca_lddt",
        "scaled_sd_ca_lddt",
        "scaled_n_samples",
    ]
    target_results = all_rows[[c for c in target_cols if c in all_rows.columns]].copy()
    feature_cols = [
        c
        for c in all_rows.columns
        if c
        not in {
            "status",
            "error",
            "wall_sec",
            "wall_sec_target_all_samples",
            "seed",
            "num_loops",
            "num_sampling_steps",
            "num_diffusion_samples",
            "ptm",
            "plddt_mean",
            "plddt_ca_mean",
            "alignment_identity",
            "alignment_coverage",
            "aligned_ca",
            "ca_lddt",
            "ca_rmsd",
            "gdt_ts",
            "ca_median_error",
            "ca_p90_error",
            "baseline_ca_lddt",
            "baseline_gdt_ts",
            "baseline_ca_rmsd",
            "sample_index",
            "scaled_mean_ca_lddt",
            "scaled_sd_ca_lddt",
            "scaled_mean_gdt_ts",
            "scaled_mean_ptm",
            "scaled_n_samples",
        }
    ]
    prospective = all_rows[feature_cols].copy()
    ready = out_root / "spectra_ready_dataset"
    ready.mkdir(parents=True, exist_ok=True)
    prospective.to_csv(ready / "combined_prospective_features.csv", index=False)
    all_rows.to_csv(ready / "combined_feature_outcome_table.csv", index=False)
    target_results.to_csv(out_root / "target_model_results.csv", index=False)
    return all_rows, prospective, target_results


def axis_catalog() -> list[dict]:
    return [
        {"axis_name": "length_extreme_from_cameo_median", "feature": "length_extreme_from_cameo_median", "direction": "high", "family": "sequence_length_shape"},
        {"axis_name": "short_sequence_low_length_extreme", "feature": "low_length_extreme_score", "direction": "high", "family": "sequence_length_shape"},
        {"axis_name": "long_sequence_length", "feature": "seq_len", "direction": "high", "family": "sequence_length_shape"},
        {"axis_name": "absolute_charge", "feature": "abs_charge_per_residue", "direction": "high", "family": "charge_hydropathy"},
        {"axis_name": "fraction_charged_residues", "feature": "fraction_charged_residues_fcr", "direction": "high", "family": "charge_hydropathy"},
        {"axis_name": "charge_hydropathy_extreme", "feature": "charge_hydropathy_extreme", "direction": "high", "family": "charge_hydropathy"},
        {"axis_name": "hydrophobic_TM_proxy", "feature": "tm_helix_proxy", "direction": "high", "family": "charge_hydropathy"},
        {"axis_name": "hydrophobic_gravy", "feature": "kd_gravy", "direction": "high", "family": "charge_hydropathy"},
        {"axis_name": "cysteine_fraction", "feature": "cys_fraction", "direction": "high", "family": "cys_disulfide_metal"},
        {"axis_name": "cys_disulfide_proxy", "feature": "cys_disulfide_proxy", "direction": "high", "family": "cys_disulfide_metal"},
        {"axis_name": "cys_his_metal_proxy", "feature": "metal_cys_his_proxy", "direction": "high", "family": "cys_disulfide_metal"},
        {"axis_name": "cxxc_motif_count", "feature": "cxxc_motif_count", "direction": "high", "family": "cys_disulfide_metal"},
        {"axis_name": "low_complexity_window", "feature": "low_complexity_window30_max_fraction", "direction": "high", "family": "motif_repeat_low_complexity"},
        {"axis_name": "dipeptide_repeat_fraction", "feature": "dipeptide_repeat_fraction", "direction": "high", "family": "motif_repeat_low_complexity"},
        {"axis_name": "homopolymer_run", "feature": "longest_homopolymer_run_norm", "direction": "high", "family": "motif_repeat_low_complexity"},
        {"axis_name": "heptad_hydrophobic_periodicity", "feature": "heptad_hydrophobic_periodicity", "direction": "high", "family": "motif_repeat_low_complexity"},
        {"axis_name": "coiled_coil_sequence_proxy", "feature": "coiled_coil_sequence_proxy", "direction": "high", "family": "motif_repeat_low_complexity"},
        {"axis_name": "quaternary_state", "feature": "quaternary_state", "direction": "high", "family": "oligomer_metadata"},
        {"axis_name": "resolution", "feature": "resolution", "direction": "high", "family": "experiment_provenance"},
        {"axis_name": "submission_ordinal", "feature": "submission_ordinal", "direction": "high", "family": "experiment_provenance"},
        {"axis_name": "relative_contact_order", "feature": "relative_contact_order_ca8", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "long_contacts_per_residue", "feature": "long_contacts_per_residue", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "long_contact_fraction", "feature": "long_contact_fraction", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "contacts_per_residue", "feature": "contacts_per_residue", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "rg_len13_norm", "feature": "rg_len13_norm", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "shape_anisotropy", "feature": "shape_anisotropy", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "end_to_end_over_rg", "feature": "end_to_end_over_rg", "direction": "high", "family": "reference_topology_benchmark_only"},
        {"axis_name": "compact_high_contact_proxy", "feature": "compact_high_contact_proxy", "direction": "high", "family": "reference_topology_benchmark_only"},
    ]


def discovery_confirmation_score(all_rows: pd.DataFrame, out_root: Path) -> dict:
    axes = axis_catalog()
    discovery = all_rows[(all_rows["dataset"].eq("cameo")) & (all_rows["hash_fold10"] <= 5)].copy()
    confirmation = all_rows[(all_rows["dataset"].eq("cameo")) & (all_rows["hash_fold10"] >= 6)].copy()
    cameo_all = all_rows[all_rows["dataset"].eq("cameo")].copy()
    scaled = all_rows[all_rows["dataset"].eq("cameo_scaled")].copy()
    external = all_rows[all_rows["dataset"].eq("external_rcsb")].copy()
    subsets = {
        "cameo_discovery_hash0_5": discovery,
        "cameo_confirmation_hash6_9": confirmation,
        "cameo_full_default": cameo_all,
        "cameo_scaled_top_tail_best_ptm": scaled,
        "external_rcsb_same_bucket_scaled_best_ptm": external,
    }
    summaries = []
    curves = []
    for axis in axes:
        for subset_name, subset in subsets.items():
            q = 7 if subset_name == "cameo_full_default" and len(subset) >= 70 else 5
            summary, group = score_axis(
                subset,
                axis["axis_name"],
                axis["feature"],
                axis["direction"],
                subset_name,
                q=q,
            )
            if summary:
                summary["family"] = axis["family"]
                summaries.append(summary)
                group["family"] = axis["family"]
                curves.append(group)
    summary_df = pd.DataFrame(summaries)
    curve_df = pd.concat(curves, ignore_index=True, sort=False) if curves else pd.DataFrame()
    summary_df.to_csv(out_root / "candidate_axis_summary.csv", index=False)
    curve_df.to_csv(out_root / "candidate_axis_curves.csv", index=False)

    # Discovery ranking: target degradation in discovery, little baseline-only explanation,
    # and not dominated by the known flexible bucket or MSA controls.
    ranked = summary_df[summary_df["subset"].eq("cameo_discovery_hash0_5")].copy()
    if not ranked.empty:
        ranked["known_confound_corr_max_abs"] = ranked[
            [c for c in ranked.columns if c.startswith("level_corr_") and c not in {"level_corr_seq_len"}]
        ].abs().max(axis=1)
        ranked["selection_score"] = (
            ranked["ca_lddt_drop_easy_minus_hard"].fillna(0)
            + 0.5 * ranked["target_minus_baseline_drop"].fillna(0)
            + 0.5 * ranked["residual_ca_lddt_drop_easy_minus_hard"].fillna(0)
            - 0.05 * ranked["known_confound_corr_max_abs"].fillna(0)
        )
        ranked = ranked.sort_values("selection_score", ascending=False)
    ranked.to_csv(out_root / "candidate_axis_discovery_ranking.csv", index=False)

    return {
        "n_rows": int(len(all_rows)),
        "n_cameo_default": int(len(cameo_all)),
        "n_cameo_scaled": int(len(scaled)),
        "n_external_scaled": int(len(external)),
        "n_candidate_axis_scores": int(len(summary_df)),
        "top_discovery_axes": ranked.head(12).replace({np.nan: None}).to_dict(orient="records") if not ranked.empty else [],
    }


def write_state(out_root: Path, summary: dict, command_log: list[dict]) -> None:
    state = {
        "session": "esmfold2_new_axes_after_scaling_external_20260531T193531Z",
        "mode": "broad_beyond_paper_generalizability_discovery",
        "model": "Biohub ESMFold2",
        "phase": "candidate_axis_discovery_and_saved_output_confirmation",
        "known_nonterminal_controls": [
            "MSA depth/evolutionary support",
            "top homolog identity/CAMEO difficulty/evolutionary support",
            "flexible_nonhelical_composite_rank/proline+beta-over-helix+turn-over-helix bucket",
        ],
        "saved_evaluation_pools_loaded": {
            "cameo_default_targets": summary["n_cameo_default"],
            "cameo_scaled_top_tail_targets": summary["n_cameo_scaled"],
            "external_rcsb_scaled_targets": summary["n_external_scaled"],
        },
        "current_status": "candidate axes scored; closure gate pending primary-axis confirmation/densification",
        "top_discovery_axes": summary["top_discovery_axes"],
    }
    ledger = {
        "axes": [],
        "negative_or_weak_axes": [],
        "live_axis": None,
        "closure_gate": {
            "passes": False,
            "reason": "primary axis not yet selected and audited after discovery/confirmation scoring",
        },
    }
    (out_root / "spectra_loop_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (out_root / "axis_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    (out_root / "commands_run.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-root", default="/ewsc/yektefai/spectra_runs/esmfold2_single_controller_final_run")
    parser.add_argument("--beyond-root", default="/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run")
    parser.add_argument("--out-root", default=".")
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    command_log = [
        {
            "cmd": "python scripts/build_combined_and_score_axes.py --out-root .",
            "purpose": "load saved ESMFold2 target outputs, build combined feature/outcome table, score candidate non-evolutionary axes",
        }
    ]
    all_rows, _, _ = build_tables(args)
    summary = discovery_confirmation_score(all_rows, out_root)
    write_state(out_root, summary, command_log)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
