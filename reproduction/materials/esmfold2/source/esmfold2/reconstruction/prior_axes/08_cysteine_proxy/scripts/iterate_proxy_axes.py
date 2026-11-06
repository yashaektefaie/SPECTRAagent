#!/usr/bin/env python3
"""Explore prospective proxy axes behind the ESMFold2 cysteine result.

This is exploratory outcome mining. Axes promoted here are hypotheses that need
fresh frozen confirmation before claim closure.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_all_scored_evidence_loop_20260601T230100Z")
OUT = Path("/ewsc/yektefai/spectra_runs/esmfold2_cys_proxy_axis_iteration_20260602")
COMBINED = ROOT / "spectra_ready_dataset" / "combined_evidence_with_cys_confirmation.csv"
PRIOR = ROOT / "spectra_ready_dataset" / "expanded_all_scored_evidence_table.csv"
FRESH = ROOT / "target_model_results.cys_confirmation.target_level.csv"
MANIFEST = ROOT / "cys_confirmation_panel_manifest.json"


def ffloat(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return None
        out = float(text)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def median(values):
    values = list(values)
    return statistics.median(values) if values else float("nan")


def cys_fraction(row):
    value = ffloat(row.get("cys_fraction"))
    if value is not None:
        return value
    return ffloat(row.get("seq_cys_fraction"))


def source_name(row):
    return row.get("evidence_source") or row.get("source") or "UNKNOWN"


def organism_flags(organism):
    org = (organism or "").lower()
    virus = int(("virus" in org) or ("phage" in org))
    eeev = int("eastern equine encephalitis" in org)
    human = int(("homo sapiens" in org) or ("human" in org))
    bacteria = int(any(x in org for x in ["escherichia", "klebsiella", "pseudomonas", "staphylococcus", "geobacter"]))
    parasite = int(any(x in org for x in ["heligmosomoides", "plasmodium", "toxoplasma", "schistosoma", "nematode"]))
    nonhuman_nonmodel = int((not human) and (org != ""))
    return {
        "org_virus_or_phage": virus,
        "org_eeev": eeev,
        "org_human": human,
        "org_bacteria_common": bacteria,
        "org_parasite_or_helminth": parasite,
        "org_nonhuman_nonmodel": nonhuman_nonmodel,
    }


def sequence_features(sequence):
    seq = (sequence or "").upper()
    length = len(seq)
    if not seq:
        return {}
    cys_pos = [i for i, aa in enumerate(seq) if aa == "C"]
    cys_n = len(cys_pos)
    gaps = [cys_pos[i + 1] - cys_pos[i] for i in range(len(cys_pos) - 1)]

    def max_window_frac(width):
        if length <= width:
            return seq.count("C") / length
        best = 0
        for start in range(length - width + 1):
            best = max(best, seq[start : start + width].count("C"))
        return best / width

    def overlap(pattern):
        return sum(1 for _ in re.finditer("(?=" + pattern + ")", seq))

    pair_counts = {}
    for limit in [2, 3, 4, 6, 8, 12, 20]:
        count = 0
        for a, b in combinations(cys_pos, 2):
            if b - a <= limit:
                count += 1
        pair_counts[f"cys_pair_le{limit}_per_cys"] = count / max(cys_n, 1)
        pair_counts[f"cys_pair_le{limit}_per100aa"] = count / (length / 100.0)

    hydrophobic = set("AILMFWYV")
    tiny = set("GAS")
    charged = set("DEKRH")

    return {
        "seq_len_calc": length,
        "cys_count": cys_n,
        "cys_fraction_calc": cys_n / length,
        "cys_odd_count": int(cys_n % 2 == 1),
        "cys_disulfide_capacity_per100": (cys_n // 2) / (length / 100.0),
        "cys_min_gap_inv": 0.0 if not gaps else 1.0 / (min(gaps) + 1.0),
        "cys_mean_gap_inv": 0.0 if not gaps else 1.0 / (mean(gaps) + 1.0),
        "cys_gap_cv": 0.0 if len(gaps) < 2 or mean(gaps) == 0 else statistics.stdev(gaps) / mean(gaps),
        "cys_window10_max_frac": max_window_frac(10),
        "cys_window20_max_frac": max_window_frac(20),
        "cys_window30_max_frac": max_window_frac(30),
        "cxc_per100aa": overlap(r"C.C") / (length / 100.0),
        "cxxc_per100aa": overlap(r"C..C") / (length / 100.0),
        "cys_close_pair_motif_per100aa": overlap(r"C.{0,4}C") / (length / 100.0),
        "nterm_hydrophobic_frac30_calc": sum(aa in hydrophobic for aa in seq[:30]) / max(1, min(30, length)),
        "hydrophobic_fraction_calc": sum(aa in hydrophobic for aa in seq) / length,
        "tiny_fraction_calc": sum(aa in tiny for aa in seq) / length,
        "charged_fraction_calc": sum(aa in charged for aa in seq) / length,
        **pair_counts,
    }


def load_rows(path, panel):
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            ca = ffloat(raw.get("ca_lddt"))
            cys = cys_fraction(raw)
            seq = raw.get("sequence") or raw.get("sequence_upper") or ""
            if ca is None or cys is None or not seq:
                continue
            row = dict(raw)
            row["panel"] = panel
            row["source_group"] = source_name(raw)
            row["ca_lddt_num"] = ca
            row["cys_fraction_num"] = cys
            row["target_key"] = row.get("target_id") or row.get("sequence_hash") or f"{panel}_{len(rows)}"
            row["sequence_upper_norm"] = seq.upper()
            row["entry_id_norm"] = row.get("entry_id") or ""
            row["method_em"] = int("electron" in (row.get("method") or "").lower())
            row["method_xray"] = int("x-ray" in (row.get("method") or "").lower())
            row.update(organism_flags(row.get("organism")))
            row.update(sequence_features(seq))
            for col in [
                "seq_len",
                "resolution",
                "reference_quality_stress",
                "alignment_coverage",
                "alignment_identity",
                "baseline_ca_lddt",
                "gdt_ts",
                "ca_rmsd",
                "tm_helix_proxy",
                "order_promoting_fraction",
                "disorder_promoting_fraction",
                "disorder_minus_order_fraction",
                "proline_fraction",
                "acidic_fraction",
                "basic_fraction",
                "fraction_charged_residues_fcr",
                "seq_kmer4_isolation_stress_to_scored",
                "seq_kmer4_nearest_jaccard_to_scored",
                "seq_nterm_hydrophobic_frac30",
                "seq_hydrophobic_window21_max_frac",
            ]:
                value = ffloat(row.get(col))
                if value is not None:
                    row[col] = value
            rows.append(row)
    return rows


def kmer_set(seq, k=5):
    seq = seq.upper()
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def add_near_duplicate_clusters(rows):
    dsu = DSU(len(rows))
    by_hash = defaultdict(list)
    by_entry = defaultdict(list)
    kmers = []
    for i, row in enumerate(rows):
        by_hash[row.get("sequence_hash") or row["sequence_upper_norm"]].append(i)
        if row.get("entry_id_norm"):
            by_entry[row["entry_id_norm"]].append(i)
        kmers.append(kmer_set(row["sequence_upper_norm"], 5))
    for bucket in [by_hash, by_entry]:
        for inds in bucket.values():
            for idx in inds[1:]:
                dsu.union(inds[0], idx)
    # K-mer near-duplicate pass. This is intentionally strict and only meant to
    # collapse near-identical family pairs, not broad homologs.
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = kmers[i], kmers[j]
            jac = len(a & b) / max(1, len(a | b))
            if jac >= 0.85:
                dsu.union(i, j)
    cluster_ids = {}
    for i, row in enumerate(rows):
        root = dsu.find(i)
        if root not in cluster_ids:
            cluster_ids[root] = f"cluster_{len(cluster_ids):04d}"
        row["near_dup_cluster"] = cluster_ids[root]
    return rows


def add_candidate_axes(rows):
    for row in rows:
        cys = row["cys_fraction_num"]
        high_cys = int(cys >= 0.0338600451467268)
        close_cys = int(row.get("cys_pair_le8_per_cys", 0) >= 0.20)
        local_cys = int(row.get("cys_window30_max_frac", 0) >= 0.10)
        disulfide_dense = int(row.get("cys_disulfide_capacity_per100", 0) >= 1.2)
        viral_or_parasite = int(row.get("org_virus_or_phage", 0) or row.get("org_parasite_or_helminth", 0))
        nonhuman = int(row.get("org_nonhuman_nonmodel", 0))
        em_or_poor_ref = int(row.get("method_em", 0) or (ffloat(row.get("resolution")) or 0) >= 3.0)
        row["axis_high_cys_frozen"] = high_cys
        row["axis_close_cys_pairs"] = close_cys
        row["axis_local_cys_cluster"] = local_cys
        row["axis_disulfide_dense"] = disulfide_dense
        row["axis_high_cys_close_pairs"] = int(high_cys and close_cys)
        row["axis_high_cys_local_cluster"] = int(high_cys and local_cys)
        row["axis_high_cys_disulfide_dense"] = int(high_cys and disulfide_dense)
        row["axis_high_cys_nonhuman"] = int(high_cys and nonhuman)
        row["axis_high_cys_viral_or_parasite"] = int(high_cys and viral_or_parasite)
        row["axis_high_cys_em_or_poor_ref"] = int(high_cys and em_or_poor_ref)
        row["axis_high_cys_viral_parasite_or_nonhuman_surface"] = int(
            high_cys
            and nonhuman
            and (viral_or_parasite or close_cys or local_cys or row.get("method_em", 0))
        )
    return rows


def aggregate_clusters(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["near_dup_cluster"]].append(row)
    out = []
    numeric_cols = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_cols.add(key)
    for cluster, members in groups.items():
        base = dict(members[0])
        base["cluster_member_count"] = len(members)
        base["target_key"] = cluster
        for col in numeric_cols:
            vals = [m[col] for m in members if isinstance(m.get(col), (int, float, np.integer, np.floating)) and math.isfinite(float(m[col]))]
            if vals:
                # Binary axes should be cluster-positive if any member is positive.
                if col.startswith("axis_") or col.startswith("org_") or col.startswith("method_"):
                    base[col] = int(max(vals) > 0)
                else:
                    base[col] = mean(vals)
        base["ca_lddt_num"] = mean(m["ca_lddt_num"] for m in members)
        srcs = sorted({m["source_group"] for m in members})
        base["source_group"] = "+".join(srcs) if len(srcs) <= 2 else "MIXED"
        out.append(base)
    return out


def axis_effect(rows, feature, mode="continuous_high", min_high=5):
    vals = []
    for row in rows:
        value = row.get(feature)
        if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
            continue
        vals.append((float(value), row))
    if len(vals) < max(20, min_high * 3):
        return None
    if mode == "binary":
        high = [r for v, r in vals if v >= 0.5]
        low = [r for v, r in vals if v < 0.5]
        high_feature_mean = 1.0
        low_feature_mean = 0.0
    else:
        reverse = mode == "continuous_high"
        vals.sort(key=lambda x: x[0], reverse=reverse)
        high_n = max(min_high, len(vals) // 5)
        high = [r for _, r in vals[:high_n]]
        low = [r for _, r in vals[high_n:]]
        high_feature_mean = mean(v for v, _ in vals[:high_n])
        low_feature_mean = mean(v for v, _ in vals[high_n:])
    if len(high) < min_high or len(low) < min_high:
        return None
    high_ca = mean(r["ca_lddt_num"] for r in high)
    low_ca = mean(r["ca_lddt_num"] for r in low)
    high_gdt = mean(ffloat(r.get("gdt_ts")) for r in high if ffloat(r.get("gdt_ts")) is not None)
    low_gdt = mean(ffloat(r.get("gdt_ts")) for r in low if ffloat(r.get("gdt_ts")) is not None)
    high_rmsd = median(ffloat(r.get("ca_rmsd")) for r in high if ffloat(r.get("ca_rmsd")) is not None)
    low_rmsd = median(ffloat(r.get("ca_rmsd")) for r in low if ffloat(r.get("ca_rmsd")) is not None)
    try:
        rho, pvalue = spearmanr([v for v, _ in vals], [r["ca_lddt_num"] for _, r in vals])
    except Exception:
        rho, pvalue = float("nan"), float("nan")
    return {
        "feature": feature,
        "mode": mode,
        "n": len(vals),
        "n_high": len(high),
        "n_low": len(low),
        "high_feature_mean": high_feature_mean,
        "low_feature_mean": low_feature_mean,
        "high_ca_lddt": high_ca,
        "low_ca_lddt": low_ca,
        "effect_ca_lddt": high_ca - low_ca,
        "effect_gdt_ts": high_gdt - low_gdt if math.isfinite(high_gdt) and math.isfinite(low_gdt) else float("nan"),
        "effect_ca_rmsd_median": high_rmsd - low_rmsd if math.isfinite(high_rmsd) and math.isfinite(low_rmsd) else float("nan"),
        "spearman_rho": rho,
        "spearman_p": pvalue,
    }


def screen_axes(row_sets, features, binary_features):
    records = []
    for table_name, rows in row_sets.items():
        for feature in features:
            for mode in ["continuous_high", "continuous_low"]:
                result = axis_effect(rows, feature, mode)
                if result:
                    result["table"] = table_name
                    records.append(result)
        for feature in binary_features:
            result = axis_effect(rows, feature, "binary")
            if result:
                result["table"] = table_name
                records.append(result)
    return records


def source_consistency(rows, feature, mode):
    out = []
    for src in sorted({r["source_group"] for r in rows}):
        src_rows = [r for r in rows if r["source_group"] == src]
        result = axis_effect(src_rows, feature, mode, min_high=max(3, len(src_rows) // 5))
        if result:
            result["source_group"] = src
            out.append(result)
    return out


def ols_high_axis(rows, feature, controls):
    y = []
    x = []
    for row in rows:
        value = row.get(feature)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        line = [1.0, float(value)]
        ok = True
        for control in controls:
            if control == "log_len":
                cv = ffloat(row.get("seq_len")) or ffloat(row.get("seq_len_calc"))
                cv = math.log(max(cv or 1.0, 1.0))
            else:
                cv = ffloat(row.get(control))
            if cv is None:
                ok = False
                break
            line.append(float(cv))
        if ok:
            y.append(row["ca_lddt_num"])
            x.append(line)
    if len(y) <= len(controls) + 3:
        return None
    y = np.array(y)
    x = np.array(x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    try:
        cov = (resid @ resid / dof) * np.linalg.inv(x.T @ x)
        se = np.sqrt(np.diag(cov))[1]
        return {"n": len(y), "coef": beta[1], "se": se, "t": beta[1] / se}
    except Exception:
        return {"n": len(y), "coef": beta[1], "se": float("nan"), "t": float("nan")}


def write_csv(path, rows, preferred=None):
    if not rows:
        return
    keys = []
    if preferred:
        keys.extend([k for k in preferred if k in rows[0]])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = add_candidate_axes(add_near_duplicate_clusters(load_rows(PRIOR, "prior")))
    fresh = add_candidate_axes(add_near_duplicate_clusters(load_rows(FRESH, "fresh_cys_confirmation")))
    combined = add_candidate_axes(add_near_duplicate_clusters(load_rows(COMBINED, "combined")))
    prior_clusters = add_candidate_axes(aggregate_clusters(prior))
    fresh_clusters = add_candidate_axes(aggregate_clusters(fresh))
    combined_clusters = add_candidate_axes(aggregate_clusters(combined))

    continuous = [
        "cys_fraction_num",
        "cys_count",
        "cys_disulfide_capacity_per100",
        "cys_min_gap_inv",
        "cys_mean_gap_inv",
        "cys_gap_cv",
        "cys_window10_max_frac",
        "cys_window20_max_frac",
        "cys_window30_max_frac",
        "cxc_per100aa",
        "cxxc_per100aa",
        "cys_close_pair_motif_per100aa",
        "cys_pair_le4_per_cys",
        "cys_pair_le8_per_cys",
        "cys_pair_le12_per_cys",
        "seq_len",
        "resolution",
        "reference_quality_stress",
        "alignment_coverage",
        "tm_helix_proxy",
        "order_promoting_fraction",
        "disorder_promoting_fraction",
        "disorder_minus_order_fraction",
        "proline_fraction",
        "seq_nterm_hydrophobic_frac30",
        "seq_hydrophobic_window21_max_frac",
        "charged_fraction_calc",
        "hydrophobic_fraction_calc",
    ]
    binary = [
        "axis_high_cys_frozen",
        "axis_close_cys_pairs",
        "axis_local_cys_cluster",
        "axis_disulfide_dense",
        "axis_high_cys_close_pairs",
        "axis_high_cys_local_cluster",
        "axis_high_cys_disulfide_dense",
        "axis_high_cys_nonhuman",
        "axis_high_cys_viral_or_parasite",
        "axis_high_cys_em_or_poor_ref",
        "axis_high_cys_viral_parasite_or_nonhuman_surface",
        "org_virus_or_phage",
        "org_parasite_or_helminth",
        "org_nonhuman_nonmodel",
        "method_em",
        "method_xray",
    ]

    row_sets = {
        "prior_rows": prior,
        "fresh_rows": fresh,
        "combined_rows": combined,
        "prior_clusters": prior_clusters,
        "fresh_clusters": fresh_clusters,
        "combined_clusters": combined_clusters,
    }
    screen = screen_axes(row_sets, continuous, binary)
    screen.sort(key=lambda r: (r["table"], r["effect_ca_lddt"]))
    write_csv(
        OUT / "candidate_axis_screen.csv",
        screen,
        [
            "table",
            "feature",
            "mode",
            "n",
            "n_high",
            "n_low",
            "effect_ca_lddt",
            "effect_gdt_ts",
            "effect_ca_rmsd_median",
            "high_ca_lddt",
            "low_ca_lddt",
            "high_feature_mean",
            "low_feature_mean",
            "spearman_rho",
            "spearman_p",
        ],
    )

    # Pick candidates that matter in combined clusters or fresh clusters.
    candidate_pairs = []
    for row in screen:
        if row["table"] in {"combined_clusters", "fresh_clusters"} and row["effect_ca_lddt"] <= -0.04:
            candidate_pairs.append((row["feature"], row["mode"]))
    candidate_pairs = list(dict.fromkeys(candidate_pairs))

    consistency_records = []
    for feature, mode in candidate_pairs:
        for table, rows in {
            "combined_rows": combined,
            "combined_clusters": combined_clusters,
            "fresh_rows": fresh,
            "fresh_clusters": fresh_clusters,
        }.items():
            for result in source_consistency(rows, feature, mode):
                result["table"] = table
                consistency_records.append(result)
    write_csv(
        OUT / "candidate_source_consistency.csv",
        consistency_records,
        ["table", "source_group", "feature", "mode", "n", "n_high", "n_low", "effect_ca_lddt", "high_ca_lddt", "low_ca_lddt"],
    )

    controls = ["log_len", "resolution", "method_em", "org_virus_or_phage", "org_nonhuman_nonmodel", "cys_fraction_num"]
    ols_records = []
    for feature, mode in candidate_pairs:
        if mode != "binary":
            continue
        for table, rows in row_sets.items():
            result = ols_high_axis(rows, feature, [c for c in controls if c != feature])
            if result:
                result.update({"table": table, "feature": feature, "controls": ",".join(controls)})
                ols_records.append(result)
    write_csv(OUT / "candidate_control_models.csv", ols_records, ["table", "feature", "n", "coef", "se", "t", "controls"])

    # Save compact feature tables for manual inspection.
    compact_keys = [
        "panel",
        "source_group",
        "target_id",
        "entry_id",
        "near_dup_cluster",
        "cluster_member_count",
        "organism",
        "method",
        "resolution",
        "seq_len",
        "ca_lddt_num",
        "gdt_ts",
        "ca_rmsd",
        "baseline_ca_lddt",
        "cys_fraction_num",
        "cys_count",
        "cys_disulfide_capacity_per100",
        "cys_min_gap_inv",
        "cys_window30_max_frac",
        "cys_pair_le8_per_cys",
        "axis_high_cys_frozen",
        "axis_high_cys_nonhuman",
        "axis_high_cys_viral_or_parasite",
        "axis_high_cys_viral_parasite_or_nonhuman_surface",
        "org_virus_or_phage",
        "org_parasite_or_helminth",
        "org_nonhuman_nonmodel",
        "method_em",
        "method_xray",
    ]
    write_csv(OUT / "combined_proxy_feature_table.csv", combined, compact_keys)
    write_csv(OUT / "combined_proxy_cluster_table.csv", combined_clusters, compact_keys)

    # Build a report from the most relevant candidates.
    def lookup(table, feature, mode):
        hits = [r for r in screen if r["table"] == table and r["feature"] == feature and r["mode"] == mode]
        return hits[0] if hits else None

    promoted = [
        ("axis_high_cys_frozen", "binary"),
        ("axis_high_cys_nonhuman", "binary"),
        ("axis_high_cys_viral_or_parasite", "binary"),
        ("axis_high_cys_viral_parasite_or_nonhuman_surface", "binary"),
        ("cys_close_pair_motif_per100aa", "continuous_high"),
        ("cys_pair_le8_per_cys", "continuous_high"),
        ("cys_fraction_num", "continuous_high"),
        ("org_virus_or_phage", "binary"),
        ("method_em", "binary"),
        ("reference_quality_stress", "continuous_high"),
    ]

    lines = []
    lines.append("# ESMFold2 Cysteine-Proxy Axis Iteration")
    lines.append("")
    lines.append("This is exploratory outcome-informed mining. Any better axis here must be frozen and confirmed on new evidence before replacing the cysteine claim.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Prior scored rows loaded: {len(prior)}; near-duplicate clusters: {len(prior_clusters)}")
    lines.append(f"- Fresh cysteine confirmation rows loaded: {len(fresh)}; near-duplicate clusters: {len(fresh_clusters)}")
    lines.append(f"- Combined scored rows loaded: {len(combined)}; near-duplicate clusters: {len(combined_clusters)}")
    lines.append("")
    lines.append("## Candidate Effects")
    lines.append("")
    lines.append("| Axis | Table | n high | CA-lDDT effect | high mean | low mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for feature, mode in promoted:
        for table in ["fresh_rows", "fresh_clusters", "combined_rows", "combined_clusters"]:
            hit = lookup(table, feature, mode)
            if hit:
                lines.append(
                    f"| {feature} | {table} | {hit['n_high']} | {hit['effect_ca_lddt']:+.4f} | {hit['high_ca_lddt']:.4f} | {hit['low_ca_lddt']:.4f} |"
                )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Raw high cysteine remains a real risk-region marker, but it is not the best mechanism-level explanation.")
    lines.append("- The strongest sharper proxy in the fresh panel is high cysteine combined with nonhuman/viral/parasitic provenance. That axis explains the severe EEEV failures but is too sparse and family-concentrated to be claim-valid yet.")
    lines.append("- Cysteine local-clustering features are more biologically specific than raw fraction and survive better as prospective sequence-only hypotheses, but they are not uniformly stronger across all evidence slices.")
    lines.append("- EM/provenance/reference-quality stress explains some degradation but is a data-quality/provenance confound, not a satisfying model-generalization mechanism by itself.")
    lines.append("")
    lines.append("## Next Frozen Hypothesis")
    lines.append("")
    lines.append("Best current follow-up axis: disulfide-rich nonhuman/viral surface-protein proxy, operationalized prospectively as high cysteine fraction plus either viral/parasitic/nonhuman provenance or local Cys clustering. Confirmation must use homology-clustered sampling with at most one close homolog per viral family/entry cluster.")
    lines.append("")
    lines.append("Recommended confirmation axes:")
    lines.append("")
    lines.append("1. `axis_high_cys_viral_parasite_or_nonhuman_surface`: high cysteine and nonhuman, plus viral/parasitic provenance or Cys-local-cluster/EM surface-protein proxy.")
    lines.append("2. Sequence-only fallback: high local cysteine close-pair motif density (`cys_close_pair_motif_per100aa` or `cys_pair_le8_per_cys`) with homology clustering.")
    lines.append("")
    lines.append("Do not promote Eastern equine encephalitis as an axis; it is a cluster stress test showing why broader homology controls are needed.")
    (OUT / "proxy_axis_iteration_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
