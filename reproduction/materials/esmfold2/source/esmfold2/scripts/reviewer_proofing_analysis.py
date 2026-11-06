#!/usr/bin/env python3
"""Reviewer-proofing analyses for the ESMFold2 disulfide/cysteine axis.

This script consumes the global all-scored evidence table and performs:
- approximate sequence-cluster-aware validation,
- matched high-vs-low disulfide controls,
- pTM/pLDDT confidence checks,
- per-source figures,
- example target selection for structure visualization.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_reviewer_proofing_20260602")
GLOBAL = Path("/ewsc/yektefai/spectra_runs/esmfold2_global_hypothesis_audit_20260602")
DATA = GLOBAL / "spectra_ready_dataset" / "primary_exact_sequence_dedup.csv"
OUT = ROOT / "analysis_outputs"
FIG = ROOT / "figures"
EX = ROOT / "example_structures"


def numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def bool_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float(np.nanmean(values)) if len(values) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_two_group_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float(np.nanmean(a) - np.nanmean(b)), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return float(a.mean() - b.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def kmer_set(seq: str, k: int = 5) -> set[str]:
    seq = str(seq)
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def make_clusters(df: pd.DataFrame, threshold: float = 0.35, k: int = 5) -> list[int]:
    """Single-linkage approximate sequence clusters from k-mer Jaccard."""
    seqs = df["sequence_clean"].fillna("").astype(str).tolist()
    sets = [kmer_set(s, k) for s in seqs]
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        si = sets[i]
        li = len(seqs[i])
        for j in range(i + 1, n):
            lj = len(seqs[j])
            if min(li, lj) / max(li, lj) < 0.55:
                continue
            sj = sets[j]
            denom = len(si | sj)
            if denom == 0:
                continue
            jac = len(si & sj) / denom
            if jac >= threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_id: dict[int, int] = {}
    cluster_ids = []
    for r in roots:
        if r not in root_to_id:
            root_to_id[r] = len(root_to_id)
        cluster_ids.append(root_to_id[r])
    return cluster_ids


def cluster_effects(df: pd.DataFrame, high: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cluster_tables = []
    for threshold in [0.50, 0.35, 0.25]:
        cid = make_clusters(df, threshold=threshold, k=5)
        d = df.copy()
        d["cluster_id"] = cid
        d["high_disulfide"] = high.to_numpy()
        n_clusters = d["cluster_id"].nunique()
        sizes = d.groupby("cluster_id").size()

        # One deterministic representative per cluster.
        reps = d.sort_values(["cluster_id", "target_id"]).drop_duplicates("cluster_id", keep="first")
        rep_top = reps[reps["high_disulfide"]]["ca_lddt"].to_numpy()
        rep_low = reps[~reps["high_disulfide"]]["ca_lddt"].to_numpy()
        rep_eff, rep_lo, rep_hi = bootstrap_two_group_ci(rep_top, rep_low, seed=int(threshold * 1000))

        # Cluster-mean effect: one vote per cluster, classified by mean stress.
        cg = d.groupby("cluster_id").agg(
            ca_lddt=("ca_lddt", "mean"),
            high_share=("high_disulfide", "mean"),
            disulfide=("disulfide_capacity_per100_calc", "mean"),
            n=("target_id", "size"),
        )
        cluster_high = cg["high_share"] >= 0.5
        c_eff, c_lo, c_hi = bootstrap_two_group_ci(cg.loc[cluster_high, "ca_lddt"], cg.loc[~cluster_high, "ca_lddt"], seed=int(threshold * 1000) + 1)

        # Cluster bootstrap over rows, resampling clusters.
        rng = np.random.default_rng(int(threshold * 1000) + 2)
        cluster_ids = cg.index.to_numpy()
        vals = []
        grouped = {k: v for k, v in d.groupby("cluster_id").groups.items()}
        for _ in range(3000):
            sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            idx = np.concatenate([np.asarray(list(grouped[c])) for c in sampled_clusters])
            sub = d.loc[idx]
            vals.append(sub.loc[sub["high_disulfide"], "ca_lddt"].mean() - sub.loc[~sub["high_disulfide"], "ca_lddt"].mean())
        vals = np.asarray(vals, dtype=float)

        rows.append(
            {
                "cluster_jaccard_threshold": threshold,
                "n_rows": len(d),
                "n_clusters": int(n_clusters),
                "largest_cluster": int(sizes.max()),
                "clusters_size_gt1": int((sizes > 1).sum()),
                "one_rep_top_n": int(reps["high_disulfide"].sum()),
                "one_rep_lower_n": int((~reps["high_disulfide"]).sum()),
                "one_rep_effect": rep_eff,
                "one_rep_ci_low": rep_lo,
                "one_rep_ci_high": rep_hi,
                "cluster_mean_top_clusters": int(cluster_high.sum()),
                "cluster_mean_lower_clusters": int((~cluster_high).sum()),
                "cluster_mean_effect": c_eff,
                "cluster_mean_ci_low": c_lo,
                "cluster_mean_ci_high": c_hi,
                "cluster_bootstrap_row_effect_mean": float(np.nanmean(vals)),
                "cluster_bootstrap_row_ci_low": float(np.nanquantile(vals, 0.025)),
                "cluster_bootstrap_row_ci_high": float(np.nanquantile(vals, 0.975)),
            }
        )
        cluster_tables.append(d[["target_id", "sequence_hash_calc", "cluster_id", "high_disulfide", "ca_lddt", "data_family"]].assign(cluster_threshold=threshold))
    return pd.DataFrame(rows), pd.concat(cluster_tables, ignore_index=True)


def matched_controls(df: pd.DataFrame, high: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["high_disulfide"] = high.to_numpy()
    d["viral_status"] = bool_col(d.get("tax_virus_or_phage_calc", pd.Series(False, index=d.index))).astype(int)
    d["log_len"] = np.log(d["calc_seq_len"].clip(lower=1))
    features = ["log_len", "tm_helix_proxy_calc2", "hydrophobic_window21_max_frac_calc2"]
    for f in features:
        mu = d[f].mean()
        sd = d[f].std()
        d[f + "_z"] = (d[f] - mu) / (sd if sd and np.isfinite(sd) else 1.0)
    used_controls: set[int] = set()
    pairs = []
    high_rows = d[d["high_disulfide"]].sort_values(["data_family", "viral_status", "disulfide_capacity_per100_calc"], ascending=[True, True, False])
    for hi_idx, hi in high_rows.iterrows():
        pool = d[
            (~d["high_disulfide"])
            & (d["data_family"] == hi["data_family"])
            & (d["viral_status"] == hi["viral_status"])
            & (~d.index.isin(used_controls))
        ].copy()
        relax_stage = "exact_source_viral"
        if pool.empty:
            pool = d[(~d["high_disulfide"]) & (d["data_family"] == hi["data_family"]) & (~d.index.isin(used_controls))].copy()
            relax_stage = "source_only"
        if pool.empty:
            continue
        dist = np.zeros(len(pool))
        for f in features:
            dist += (pool[f + "_z"].to_numpy() - float(hi[f + "_z"])) ** 2
        pool = pool.assign(match_distance=np.sqrt(dist))
        ctrl_idx = int(pool.sort_values("match_distance").index[0])
        used_controls.add(ctrl_idx)
        ctrl = d.loc[ctrl_idx]
        pairs.append(
            {
                "pair_id": len(pairs),
                "relax_stage": relax_stage,
                "data_family": hi["data_family"],
                "viral_status": int(hi["viral_status"]),
                "high_target_id": hi["target_id"],
                "control_target_id": ctrl["target_id"],
                "high_ca_lddt": hi["ca_lddt"],
                "control_ca_lddt": ctrl["ca_lddt"],
                "delta_high_minus_control": hi["ca_lddt"] - ctrl["ca_lddt"],
                "high_disulfide_capacity": hi["disulfide_capacity_per100_calc"],
                "control_disulfide_capacity": ctrl["disulfide_capacity_per100_calc"],
                "high_len": hi["calc_seq_len"],
                "control_len": ctrl["calc_seq_len"],
                "high_tm_proxy": hi["tm_helix_proxy_calc2"],
                "control_tm_proxy": ctrl["tm_helix_proxy_calc2"],
                "match_distance": float(pool.loc[ctrl_idx, "match_distance"]),
            }
        )
    pairs_df = pd.DataFrame(pairs)
    rows = []
    if not pairs_df.empty:
        mean, lo, hi = bootstrap_mean_ci(pairs_df["delta_high_minus_control"].to_numpy(), seed=22)
        rows.append(
            {
                "slice": "all_matched",
                "n_pairs": len(pairs_df),
                "mean_delta_high_minus_control": mean,
                "ci_low": lo,
                "ci_high": hi,
                "median_delta": float(pairs_df["delta_high_minus_control"].median()),
                "high_worse_fraction": float((pairs_df["delta_high_minus_control"] < 0).mean()),
                "exact_source_viral_pairs": int((pairs_df["relax_stage"] == "exact_source_viral").sum()),
            }
        )
        for fam, g in pairs_df.groupby("data_family"):
            mean, lo, hi = bootstrap_mean_ci(g["delta_high_minus_control"].to_numpy(), seed=23)
            rows.append(
                {
                    "slice": str(fam),
                    "n_pairs": len(g),
                    "mean_delta_high_minus_control": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "median_delta": float(g["delta_high_minus_control"].median()),
                    "high_worse_fraction": float((g["delta_high_minus_control"] < 0).mean()),
                    "exact_source_viral_pairs": int((g["relax_stage"] == "exact_source_viral").sum()),
                }
            )
    return pd.DataFrame(rows), pairs_df


def confidence_analysis(df: pd.DataFrame, high: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["high_disulfide"] = high.to_numpy()
    d["classii"] = bool_col(d.get("classii_e_e2_envelopeE_calc", pd.Series(False, index=d.index)))
    d["failure_ca_lddt_lt_0_6"] = d["ca_lddt"] < 0.6
    rows = []
    for axis, mask in [
        ("high_disulfide", d["high_disulfide"]),
        ("classII_E_E2_envelopeE", d["classii"]),
    ]:
        eligible = d["ca_lddt"].notna() & mask.notna()
        if axis.startswith("classII"):
            eligible &= bool_col(d.get("metadata_text_available", pd.Series(False, index=d.index)))
        sub = d[eligible].copy()
        top = mask.loc[sub.index].astype(bool)
        lower = ~top
        row = {
            "axis": axis,
            "n": len(sub),
            "n_top": int(top.sum()),
            "n_lower": int(lower.sum()),
            "top_ca_lddt": float(sub.loc[top, "ca_lddt"].mean()) if top.sum() else np.nan,
            "lower_ca_lddt": float(sub.loc[lower, "ca_lddt"].mean()) if lower.sum() else np.nan,
            "top_failure_rate_ca_lddt_lt_0_6": float(sub.loc[top, "failure_ca_lddt_lt_0_6"].mean()) if top.sum() else np.nan,
            "lower_failure_rate_ca_lddt_lt_0_6": float(sub.loc[lower, "failure_ca_lddt_lt_0_6"].mean()) if lower.sum() else np.nan,
        }
        for conf in ["ptm", "plddt_mean", "plddt_ca_mean"]:
            if conf not in sub.columns:
                continue
            c = pd.to_numeric(sub[conf], errors="coerce")
            row[f"top_{conf}"] = float(c.loc[top].mean()) if top.sum() else np.nan
            row[f"lower_{conf}"] = float(c.loc[lower].mean()) if lower.sum() else np.nan
            row[f"{conf}_diff_top_minus_lower"] = row[f"top_{conf}"] - row[f"lower_{conf}"]
            ok = c.notna() & sub["ca_lddt"].notna()
            row[f"{conf}_spearman_with_ca_lddt"] = float(c.loc[ok].rank().corr(sub.loc[ok, "ca_lddt"].rank())) if ok.sum() > 5 else np.nan
            row[f"{conf}_auc_predict_good_ca_lddt_ge_0_6"] = auc_score(sub.loc[ok, "ca_lddt"].to_numpy() >= 0.6, c.loc[ok].to_numpy())
            median_conf = c.median()
            row[f"overconfident_failure_rate_{conf}_ge_median"] = float(((sub["ca_lddt"] < 0.6) & (c >= median_conf)).mean())
            row[f"top_overconfident_failure_rate_{conf}_ge_median"] = float(((sub.loc[top, "ca_lddt"] < 0.6) & (c.loc[top] >= median_conf)).mean()) if top.sum() else np.nan
        rows.append(row)

    bins = []
    if "ptm" in d.columns:
        ok = d["ptm"].notna() & d["ca_lddt"].notna()
        try:
            d.loc[ok, "ptm_bin"] = pd.qcut(d.loc[ok, "ptm"], 5, labels=False, duplicates="drop")
            for (bin_id, hi), g in d.loc[ok].groupby(["ptm_bin", "high_disulfide"]):
                bins.append(
                    {
                        "ptm_bin": int(bin_id),
                        "high_disulfide": bool(hi),
                        "n": len(g),
                        "ptm_mean": float(g["ptm"].mean()),
                        "ca_lddt_mean": float(g["ca_lddt"].mean()),
                        "failure_rate_ca_lddt_lt_0_6": float((g["ca_lddt"] < 0.6).mean()),
                    }
                )
        except ValueError:
            pass
    return pd.DataFrame(rows), pd.DataFrame(bins)


def source_summary(df: pd.DataFrame, high: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d["high_disulfide"] = high.to_numpy()
    rows = []
    for fam, g in d.groupby("data_family"):
        for hi, sub in g.groupby("high_disulfide"):
            mean, lo, ci_hi = bootstrap_mean_ci(sub["ca_lddt"].to_numpy(), seed=11)
            rows.append(
                {
                    "data_family": fam,
                    "high_disulfide": bool(hi),
                    "n": len(sub),
                    "ca_lddt_mean": mean,
                    "ci_low": lo,
                    "ci_high": ci_hi,
                    "ca_lddt_median": float(sub["ca_lddt"].median()),
                }
            )
    return pd.DataFrame(rows)


def plot_source_summary(src: pd.DataFrame) -> None:
    families = list(src["data_family"].drop_duplicates())
    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, hi in enumerate([False, True]):
        sub = src[src["high_disulfide"] == hi].set_index("data_family").reindex(families)
        means = sub["ca_lddt_mean"].to_numpy()
        err_low = means - sub["ci_low"].to_numpy()
        err_hi = sub["ci_high"].to_numpy() - means
        ax.bar(x + (i - 0.5) * width, means, width, label="high disulfide" if hi else "lower disulfide")
        ax.errorbar(x + (i - 0.5) * width, means, yerr=[err_low, err_hi], fmt="none", color="black", lw=1, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=9)
    ax.set_ylabel("mean CA-lDDT")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("High-disulfide effect by source slice")
    fig.tight_layout()
    fig.savefig(FIG / "high_disulfide_by_source.png", dpi=220)
    fig.savefig(FIG / "high_disulfide_by_source.pdf")
    plt.close(fig)


def plot_level_curve(df: pd.DataFrame) -> None:
    d = df.copy()
    d["disulfide_level"] = pd.qcut(d["disulfide_capacity_per100_calc"], 4, labels=False, duplicates="drop")
    rows = []
    for level, g in d.groupby("disulfide_level"):
        mean, lo, hi = bootstrap_mean_ci(g["ca_lddt"].to_numpy(), seed=12)
        rows.append({"level": int(level), "n": len(g), "mean": mean, "lo": lo, "hi": hi})
    cur = pd.DataFrame(rows)
    cur.to_csv(OUT / "high_disulfide_quartile_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(cur["level"], cur["mean"], marker="o", color="#1f77b4")
    ax.fill_between(cur["level"], cur["lo"], cur["hi"], color="#1f77b4", alpha=0.2)
    ax.set_xlabel("disulfide-capacity quartile")
    ax.set_ylabel("mean CA-lDDT")
    ax.set_ylim(0, 1.02)
    ax.set_title("Disulfide-capacity threshold curve")
    ax.set_xticks(cur["level"])
    fig.tight_layout()
    fig.savefig(FIG / "high_disulfide_quartile_curve.png", dpi=220)
    fig.savefig(FIG / "high_disulfide_quartile_curve.pdf")
    plt.close(fig)


def plot_confidence(df: pd.DataFrame, high: pd.Series) -> None:
    d = df[df["ptm"].notna() & df["ca_lddt"].notna()].copy()
    d["high_disulfide"] = high.reindex(d.index).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    colors = np.where(d["high_disulfide"], "#d62728", "#7f7f7f")
    ax.scatter(d["ptm"], d["ca_lddt"], c=colors, s=18, alpha=0.65, linewidth=0)
    ax.axhline(0.6, color="black", lw=1, ls="--")
    ax.set_xlabel("ESMFold2 pTM")
    ax.set_ylabel("CA-lDDT")
    ax.set_title("Confidence versus realized accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "ptm_vs_ca_lddt_high_disulfide.png", dpi=220)
    fig.savefig(FIG / "ptm_vs_ca_lddt_high_disulfide.pdf")
    plt.close(fig)


def example_manifest(df: pd.DataFrame, high: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d["high_disulfide"] = high.to_numpy()
    d["classii"] = bool_col(d.get("classii_e_e2_envelopeE_calc", pd.Series(False, index=d.index)))
    candidates = d[(d["classii"] | d["high_disulfide"]) & d["pdb_path"].notna()].sort_values("ca_lddt").head(12)
    examples = []
    seen = set()
    for _, r in candidates.iterrows():
        org = str(r.get("organism", ""))
        desc = str(r.get("entity_description", ""))
        key = (org.split()[0] if org else "", desc[:20])
        if key in seen and len(examples) >= 3:
            continue
        seen.add(key)
        examples.append(r)
        if len(examples) >= 5:
            break
    ex = pd.DataFrame(examples)
    cols = [
        "target_id",
        "data_family",
        "ca_lddt",
        "gdt_ts",
        "ca_rmsd",
        "ptm",
        "plddt_mean",
        "calc_seq_len",
        "disulfide_capacity_per100_calc",
        "cys_fraction_calc",
        "classii",
        "high_disulfide",
        "organism",
        "entity_description",
        "pdb_path",
        "actual_ref_chain",
        "ref_chain",
        "sequence",
    ]
    cols = [c for c in cols if c in ex.columns]
    ex = ex[cols].copy()
    ex["example_status"] = "reference structure available; predicted coordinates not saved in final scaled run"
    ex.to_csv(EX / "example_structure_targets.csv", index=False)

    # Write simple ChimeraX templates for once predicted PDBs are generated.
    for _, r in ex.iterrows():
        target = str(r["target_id"])
        ref_path = str(r["pdb_path"])
        script = "\n".join(
            [
                f"# ChimeraX template for {target}",
                f"open {ref_path}",
                f"open predicted_best_by_ptm/{target}.pdb",
                "matchmaker #2 to #1",
                "color #1 dodger blue",
                "color #2 crimson",
                "cartoon #1 #2",
                "view",
                f"save {target}_overlay.png width 1600 height 1200",
                "",
            ]
        )
        (EX / f"{target}_chimerax_template.cxc").write_text(script, encoding="utf-8")
    return ex


def write_report(
    threshold: float,
    n_total: int,
    n_high: int,
    cluster_df: pd.DataFrame,
    match_summary: pd.DataFrame,
    conf: pd.DataFrame,
    src: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# ESMFold2 Disulfide-Axis Reviewer-Proofing")
    lines.append("")
    lines.append("No new ESMFold2 inference was run in this pass; analyses use saved target-level outputs.")
    lines.append("")
    lines.append("## Axis")
    lines.append("")
    lines.append(f"High disulfide capacity is the top 20% of exact-sequence primary rows: `disulfide_capacity_per100 >= {threshold:.4f}`.")
    lines.append(f"Total rows: {n_total}; high-disulfide rows: {n_high}.")
    lines.append("")
    lines.append("## 1. Cluster-Aware Validation")
    lines.append("")
    lines.append("| k-mer Jaccard cluster threshold | clusters | largest cluster | one-rep effect | one-rep CI | cluster-mean effect | cluster-mean CI | cluster-bootstrap row CI |")
    lines.append("|---:|---:|---:|---:|---|---:|---|---|")
    for _, r in cluster_df.iterrows():
        lines.append(
            f"| {r.cluster_jaccard_threshold:.2f} | {int(r.n_clusters)} | {int(r.largest_cluster)} | "
            f"{r.one_rep_effect:.3f} | [{r.one_rep_ci_low:.3f}, {r.one_rep_ci_high:.3f}] | "
            f"{r.cluster_mean_effect:.3f} | [{r.cluster_mean_ci_low:.3f}, {r.cluster_mean_ci_high:.3f}] | "
            f"[{r.cluster_bootstrap_row_ci_low:.3f}, {r.cluster_bootstrap_row_ci_high:.3f}] |"
        )
    lines.append("")
    lines.append("This is approximate k-mer clustering, not a replacement for final MMseqs2/CD-HIT clustering, but it directly tests the family-overrepresentation objection.")
    lines.append("")
    lines.append("## 2. Matched Controls")
    lines.append("")
    lines.append("High-disulfide rows were greedily matched 1:1 to lower-disulfide controls on source family and viral/nonviral status, then nearest log length, TM/hydrophobic proxy.")
    lines.append("")
    lines.append("| slice | pairs | mean high-control CA-lDDT | CI | median | high worse fraction | exact source+viral pairs |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for _, r in match_summary.iterrows():
        lines.append(
            f"| {r['slice']} | {int(r.n_pairs)} | {r.mean_delta_high_minus_control:.3f} | "
            f"[{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.median_delta:.3f} | {r.high_worse_fraction:.3f} | {int(r.exact_source_viral_pairs)} |"
        )
    lines.append("")
    lines.append("## 3. Confidence/Calibration")
    lines.append("")
    lines.append("| axis | n | top n | top CA-lDDT | lower CA-lDDT | top failure rate | lower failure rate | top pTM | lower pTM | pTM AUC good | top overconfident failure rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in conf.iterrows():
        lines.append(
            f"| {r.axis} | {int(r.n)} | {int(r.n_top)} | {r.top_ca_lddt:.3f} | {r.lower_ca_lddt:.3f} | "
            f"{r.top_failure_rate_ca_lddt_lt_0_6:.3f} | {r.lower_failure_rate_ca_lddt_lt_0_6:.3f} | "
            f"{r.get('top_ptm', np.nan):.3f} | {r.get('lower_ptm', np.nan):.3f} | "
            f"{r.get('ptm_auc_predict_good_ca_lddt_ge_0_6', np.nan):.3f} | "
            f"{r.get('top_overconfident_failure_rate_ptm_ge_median', np.nan):.3f} |"
        )
    lines.append("")
    lines.append("## 4. Per-Source Visualization")
    lines.append("")
    lines.append("Figures:")
    lines.append("")
    lines.append("- `figures/high_disulfide_by_source.png`")
    lines.append("- `figures/high_disulfide_quartile_curve.png`")
    lines.append("- `figures/ptm_vs_ca_lddt_high_disulfide.png`")
    lines.append("")
    lines.append("## 5. Structure Examples")
    lines.append("")
    lines.append("The final scaled evaluations did not save predicted coordinates, so this pass selected examples and wrote ChimeraX templates only. Actual overlays require rerunning these few targets with coordinate saving.")
    lines.append("")
    lines.append("| target | CA-lDDT | pTM | class-II | high disulfide | organism | description |")
    lines.append("|---|---:|---:|---|---|---|---|")
    for _, r in examples.iterrows():
        lines.append(
            f"| {r.get('target_id')} | {float(r.get('ca_lddt', np.nan)):.3f} | {float(r.get('ptm', np.nan)):.3f} | "
            f"{bool(r.get('classii', False))} | {bool(r.get('high_disulfide', False))} | "
            f"{str(r.get('organism', ''))[:60]} | {str(r.get('entity_description', ''))[:80]} |"
        )
    lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    lines.append("- `analysis_outputs/cluster_aware_disulfide_effects.csv`")
    lines.append("- `analysis_outputs/cluster_assignments.csv`")
    lines.append("- `analysis_outputs/matched_control_summary.csv`")
    lines.append("- `analysis_outputs/matched_control_pairs.csv`")
    lines.append("- `analysis_outputs/confidence_calibration_summary.csv`")
    lines.append("- `analysis_outputs/ptm_calibration_bins.csv`")
    lines.append("- `analysis_outputs/per_source_high_disulfide_summary.csv`")
    lines.append("- `example_structures/example_structure_targets.csv`")
    (ROOT / "reviewer_proofing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    EX.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA, low_memory=False)
    numeric(
        df,
        [
            "ca_lddt",
            "gdt_ts",
            "ca_rmsd",
            "ptm",
            "plddt_mean",
            "plddt_ca_mean",
            "calc_seq_len",
            "disulfide_capacity_per100_calc",
            "cys_fraction_calc",
            "tm_helix_proxy_calc2",
            "hydrophobic_window21_max_frac_calc2",
            "baseline_ca_lddt",
        ],
    )
    df = df[df["ca_lddt"].notna() & df["disulfide_capacity_per100_calc"].notna()].copy()
    threshold = float(df["disulfide_capacity_per100_calc"].quantile(0.8))
    high = df["disulfide_capacity_per100_calc"] >= threshold
    df["high_disulfide"] = high.to_numpy()

    cluster_df, cluster_assignments = cluster_effects(df, high)
    cluster_df.to_csv(OUT / "cluster_aware_disulfide_effects.csv", index=False)
    cluster_assignments.to_csv(OUT / "cluster_assignments.csv", index=False)

    match_summary, pairs = matched_controls(df, high)
    match_summary.to_csv(OUT / "matched_control_summary.csv", index=False)
    pairs.to_csv(OUT / "matched_control_pairs.csv", index=False)

    conf, bins = confidence_analysis(df, high)
    conf.to_csv(OUT / "confidence_calibration_summary.csv", index=False)
    bins.to_csv(OUT / "ptm_calibration_bins.csv", index=False)

    src = source_summary(df, high)
    src.to_csv(OUT / "per_source_high_disulfide_summary.csv", index=False)
    plot_source_summary(src)
    plot_level_curve(df)
    plot_confidence(df, high)

    examples = example_manifest(df, high)
    write_report(threshold, len(df), int(high.sum()), cluster_df, match_summary, conf, src, examples)

    summary = {
        "n": int(len(df)),
        "high_disulfide_threshold": threshold,
        "n_high": int(high.sum()),
        "cluster_thresholds": cluster_df.to_dict(orient="records"),
        "matched_all": match_summary[match_summary["slice"] == "all_matched"].to_dict(orient="records"),
        "confidence": conf.to_dict(orient="records"),
        "figures": [
            str(FIG / "high_disulfide_by_source.png"),
            str(FIG / "high_disulfide_quartile_curve.png"),
            str(FIG / "ptm_vs_ca_lddt_high_disulfide.png"),
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
