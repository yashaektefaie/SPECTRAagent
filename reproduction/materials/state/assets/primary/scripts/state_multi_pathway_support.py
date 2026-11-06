#!/usr/bin/env python3
"""Cross-ontology SPECTRA support screen for STATE/Replogle.

The Reactome result suggested a broader train-coverage boundary. This script
tests that directly by swapping in independent prospective gene-set systems:
GO BP/MF/CC, WikiPathways, KEGG, HGNC gene groups, and Reactome.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import state_spectra_loop as base  # noqa: E402


ROOT = Path("./spectra_runs/state_generalizability_continued_20260603")
ART = ROOT / "artifacts"
READY = ROOT / "spectra_ready_dataset"
EXT = ROOT / "external"
TARGET_METRIC = "overlap_at_N"

GOA_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
WIKIPATHWAYS_URL = "https://data.wikipathways.org/current/gmt/wikipathways-20260510-gmt-Homo_sapiens.gmt"
KEGG_LIST_HSA = "https://rest.kegg.jp/list/hsa"
KEGG_LIST_PATHWAY = "https://rest.kegg.jp/list/pathway/hsa"
KEGG_LINK_PATHWAY = "https://rest.kegg.jp/link/pathway/hsa"

LEVEL_ORDER = {
    "L0_no_annotation": 0,
    "L1_low": 1,
    "L2_mid": 2,
    "L3_high": 3,
    "L4_very_high": 4,
}


def download(url: str, path: Path, timeout: int = 90) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, path.open("wb") as out:
            out.write(response.read())
        return True
    except Exception as exc:
        print(f"WARN urllib failed download {url}: {exc}; trying curl", file=sys.stderr)
    try:
        subprocess.run(
            ["curl", "-L", "--fail", "--max-time", str(timeout), "-A", "Mozilla/5.0", "-o", str(path), url],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return path.exists() and path.stat().st_size > 0
    except Exception as exc:
        print(f"WARN failed download {url}: {exc}", file=sys.stderr)
        return False


def fetch_text(url: str, path: Path, timeout: int = 60) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    try:
        data = urllib.request.urlopen(url, timeout=timeout).read()
        path.write_bytes(data)
        return True
    except Exception as exc:
        print(f"WARN failed fetch {url}: {exc}", file=sys.stderr)
        return False


def safe_mean(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def canonical(sym: object, alias: dict[str, str]) -> str:
    s = base.norm_symbol(sym)
    return alias.get(s, s)


def parse_go_names() -> dict[str, str]:
    path = EXT / "go-basic.obo"
    names: dict[str, str] = {}
    if not download(GO_OBO_URL, path):
        return names
    current_id = None
    current_name = None
    in_term = False
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current_id and current_name:
                    names[current_id] = current_name
                current_id, current_name, in_term = None, None, True
                continue
            if line.startswith("["):
                if current_id and current_name:
                    names[current_id] = current_name
                current_id, current_name, in_term = None, None, False
                continue
            if in_term and line.startswith("id: "):
                current_id = line[4:]
            elif in_term and line.startswith("name: "):
                current_name = line[6:]
        if current_id and current_name:
            names[current_id] = current_name
    return names


def parse_go_collections(alias: dict[str, str]) -> dict[str, dict[str, set[str]]]:
    path = EXT / "goa_human.gaf.gz"
    if not download(GOA_URL, path):
        return {}
    names = parse_go_names()
    aspect_name = {"P": "GO_BP", "F": "GO_MF", "C": "GO_CC"}
    out: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in aspect_name.values()}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            qualifier = parts[3]
            if "NOT" in qualifier.split("|"):
                continue
            symbol = canonical(parts[2], alias)
            go_id = parts[4]
            aspect = parts[8]
            coll = aspect_name.get(aspect)
            if not coll or not symbol:
                continue
            term = names.get(go_id, go_id)
            out[coll][symbol].add(f"{go_id}:{term}")
    return {name: dict(mapping) for name, mapping in out.items()}


def parse_wikipathways(alias: dict[str, str]) -> dict[str, set[str]]:
    path = EXT / "wikipathways-20260510-gmt-Homo_sapiens.gmt"
    if not download(WIKIPATHWAYS_URL, path):
        return {}
    hgnc = pd.read_csv(EXT / "hgnc_complete_set.txt", sep="\t", dtype=str, keep_default_na=False)
    entrez_to_symbol: dict[str, str] = {}
    for _, row in hgnc.iterrows():
        entrez = str(row.get("entrez_id", "")).strip()
        sym = canonical(row.get("symbol", ""), alias)
        if entrez and sym:
            entrez_to_symbol[entrez] = sym
    out: dict[str, set[str]] = defaultdict(set)
    with path.open(errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathway = parts[0]
            for gene in parts[2:]:
                sym = entrez_to_symbol.get(str(gene).strip()) if str(gene).strip().isdigit() else canonical(gene, alias)
                if sym:
                    out[sym].add(pathway)
    return dict(out)


def parse_kegg(alias: dict[str, str]) -> dict[str, set[str]]:
    list_path = EXT / "kegg_list_hsa.txt"
    pathway_path = EXT / "kegg_list_pathway_hsa.txt"
    link_path = EXT / "kegg_link_pathway_hsa.txt"
    ok = (
        fetch_text(KEGG_LIST_HSA, list_path)
        and fetch_text(KEGG_LIST_PATHWAY, pathway_path)
        and fetch_text(KEGG_LINK_PATHWAY, link_path)
    )
    if not ok:
        return {}
    gene_to_symbol: dict[str, set[str]] = defaultdict(set)
    with list_path.open(errors="replace") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            kid, desc = line.rstrip("\n").split("\t", 1)
            symbols = desc.split(";", 1)[0]
            for sym in re.split(r",\s*", symbols):
                cs = canonical(sym, alias)
                if cs:
                    gene_to_symbol[kid].add(cs)
    pathway_names: dict[str, str] = {}
    with pathway_path.open(errors="replace") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            pid, name = line.rstrip("\n").split("\t", 1)
            pathway_names[f"path:{pid}"] = name
    out: dict[str, set[str]] = defaultdict(set)
    with link_path.open(errors="replace") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            kid, pid = line.rstrip("\n").split("\t", 1)
            pname = pathway_names.get(pid, pid)
            for sym in gene_to_symbol.get(kid, set()):
                out[sym].add(pname)
    return dict(out)


def hgnc_group_collection(hgnc_info: dict[str, dict[str, object]]) -> dict[str, set[str]]:
    return {sym: set(info.get("groups", set())) for sym, info in hgnc_info.items() if info.get("groups")}


def reactome_collection(hgnc_info: dict[str, dict[str, object]], alias: dict[str, str]) -> dict[str, set[str]]:
    return base.load_reactome(hgnc_info, alias)


def support_stat(feature_set: set[str], index: dict[str, set[str]], train_n: int) -> tuple[float, float, float, int]:
    if not feature_set or train_n <= 0:
        return 0.0, 0.0, 0.0, 0
    counts = np.array([len(index.get(feature, set())) for feature in feature_set], dtype=float)
    raw = float(np.mean(np.log1p(counts)))
    norm = raw / float(np.log1p(train_n))
    frac = float(np.mean(counts > 0))
    return raw, norm, frac, int(len(feature_set))


def feature_index(feature_map: dict[str, set[str]], train_genes: set[str]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for gene in train_genes:
        for feature in feature_map.get(gene, set()):
            idx[feature].add(gene)
    return idx


def assign_bins(values: pd.Series, annot_count: pd.Series, cutpoints: list[float]) -> pd.Series:
    labels = ["L1_low", "L2_mid", "L3_high", "L4_very_high"]

    def label(v: object) -> str:
        if pd.isna(v):
            return "L0_no_annotation"
        fv = float(v)
        for cp, lab in zip(cutpoints, labels):
            if fv <= cp:
                return lab
        return "L4_very_high"

    out = values.map(label).astype(object)
    out[pd.to_numeric(annot_count, errors="coerce").fillna(0) <= 0] = "L0_no_annotation"
    return out


def infer_fewshot_train_sets(df: pd.DataFrame, alias: dict[str, str]) -> dict[str, set[str]]:
    specs = base.load_replogle_tomls()
    z = df[(df["dataset_family"] == "Replogle-Nadig") & (df["protocol"] == "zeroshot")]
    all_by_context = {
        str(ctx): {canonical(g, alias) for g in sub["gene_symbol"].dropna() if canonical(g, alias) != "NON-TARGETING"}
        for ctx, sub in z.groupby("context")
    }
    out = {}
    for ctx, genes in all_by_context.items():
        spec = specs.get(ctx, {"test": set(), "val": set()})
        test = {canonical(g, alias) for g in spec.get("test", set())}
        val = {canonical(g, alias) for g in spec.get("val", set())}
        out[ctx] = genes - test - val
    return out


def build_fewshot_examples(df: pd.DataFrame, alias: dict[str, str], feature_map: dict[str, set[str]], collection: str) -> pd.DataFrame:
    sub = df[(df["dataset_family"] == "Replogle-Nadig") & (df["protocol"] == "fewshot")].copy()
    sub["gene_symbol_canonical"] = sub["gene_symbol"].map(lambda x: canonical(x, alias))
    sub = sub[sub["gene_symbol_canonical"] != "NON-TARGETING"].copy()
    train_by_context = infer_fewshot_train_sets(df, alias)
    rows = []
    for ctx, ctx_df in sub.groupby("context"):
        train = train_by_context.get(str(ctx), set())
        idx = feature_index(feature_map, train)
        for row_index, row in ctx_df.iterrows():
            gene = row["gene_symbol_canonical"]
            feats = feature_map.get(gene, set())
            raw, norm, frac, n_features = support_stat(feats, idx, len(train))
            rows.append(
                {
                    "row_index": row_index,
                    "collection": collection,
                    "train_gene_count": len(train),
                    "feature_count": n_features,
                    "support_log_mean": raw,
                    "support_norm_log_mean": norm,
                    "supported_fraction": frac,
                }
            )
    feat = pd.DataFrame(rows).set_index("row_index")
    out = sub.join(feat)
    out["spectra_subset"] = np.where(out["context"].isin({"hepg2", "k562"}), "discovery_hepg2_k562", "confirmation_jurkat_rpe1")
    return out


def build_zeroshot_examples(df: pd.DataFrame, alias: dict[str, str], feature_map: dict[str, set[str]], collection: str) -> pd.DataFrame:
    sub = df[(df["dataset_family"] == "Replogle-Nadig") & (df["protocol"] == "zeroshot")].copy()
    sub["gene_symbol_canonical"] = sub["gene_symbol"].map(lambda x: canonical(x, alias))
    sub = sub[sub["gene_symbol_canonical"] != "NON-TARGETING"].copy()
    all_by_context = {
        str(ctx): {canonical(g, alias) for g in ctx_df["gene_symbol"].dropna() if canonical(g, alias) != "NON-TARGETING"}
        for ctx, ctx_df in sub.groupby("context")
    }
    contexts = set(all_by_context)
    rows = []
    for ctx, ctx_df in sub.groupby("context"):
        train = set().union(*(all_by_context[c] for c in contexts - {str(ctx)}))
        idx = feature_index(feature_map, train)
        for row_index, row in ctx_df.iterrows():
            gene = row["gene_symbol_canonical"]
            feats = feature_map.get(gene, set())
            raw, norm, frac, n_features = support_stat(feats, idx, len(train))
            rows.append(
                {
                    "row_index": row_index,
                    "collection": collection,
                    "zeroshot_train_gene_count": len(train),
                    "feature_count": n_features,
                    "support_log_mean": raw,
                    "support_norm_log_mean": norm,
                    "supported_fraction": frac,
                }
            )
    feat = pd.DataFrame(rows).set_index("row_index")
    out = sub.join(feat)
    out["spectra_subset"] = "zeroshot_cross_protocol_confirmation"
    return out


def summarize(examples: pd.DataFrame, level_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (subset, rep, level), sub in examples.groupby(["spectra_subset", "representation", level_col]):
        rows.append(
            {
                "spectra_subset": subset,
                "representation": rep,
                "level": level,
                "level_order": LEVEL_ORDER.get(str(level), -1),
                "n_rows": int(len(sub)),
                "n_unique_contexts": int(sub["context"].nunique()),
                "n_unique_perturbations": int(sub["gene_symbol_canonical"].nunique()),
                "similarity_mean": safe_mean(sub["support_norm_log_mean"]),
                "feature_count_mean": safe_mean(sub["feature_count"]),
                "overlap_at_N_mean": safe_mean(sub["overlap_at_N"]),
                "de_sig_genes_recall_mean": safe_mean(sub["de_sig_genes_recall"]),
                "pearson_delta_mean": safe_mean(sub["pearson_delta"]),
            }
        )
    levels = pd.DataFrame(rows).sort_values(["spectra_subset", "representation", "level_order"])
    trend_rows = []
    for (subset, rep), sub in levels.groupby(["spectra_subset", "representation"]):
        valid = sub[sub["level_order"] >= 0].sort_values("level_order")
        if len(valid) < 2:
            continue
        low = valid.iloc[0]
        high = valid.iloc[-1]
        trend_rows.append(
            {
                "spectra_subset": subset,
                "representation": rep,
                "n_levels": int(len(valid)),
                "n_rows": int(valid["n_rows"].sum()),
                "similarity_monotone_increasing": bool(valid["similarity_mean"].is_monotonic_increasing),
                "spearman_level_vs_overlap_at_N": float(valid["level_order"].corr(valid["overlap_at_N_mean"], method="spearman")),
                "low_level": low["level"],
                "high_level": high["level"],
                "low_overlap_at_N_mean": float(low["overlap_at_N_mean"]),
                "high_overlap_at_N_mean": float(high["overlap_at_N_mean"]),
                "high_minus_low_overlap_at_N": float(high["overlap_at_N_mean"] - low["overlap_at_N_mean"]),
            }
        )
    return levels, pd.DataFrame(trend_rows)


def ols_support_coef(examples: pd.DataFrame) -> dict[str, float]:
    work = examples[(examples["feature_count"] > 0)].dropna(subset=["support_norm_log_mean", TARGET_METRIC]).copy()
    if len(work) < 50:
        return {"n_rows": int(len(work)), "coef": float("nan"), "t": float("nan"), "r2": float("nan")}
    y = pd.to_numeric(work[TARGET_METRIC], errors="coerce").to_numpy(float)
    cols = []
    names = []

    def add(name: str, vals: np.ndarray, standardize: bool = True) -> None:
        vals = vals.astype(float)
        if standardize:
            sd = float(np.std(vals))
            vals = (vals - float(np.mean(vals))) / sd if sd > 0 else vals * 0
        cols.append(vals)
        names.append(name)

    add("support_norm_log_mean_z", pd.to_numeric(work["support_norm_log_mean"], errors="coerce").to_numpy(float))
    add("feature_count_z", pd.to_numeric(work["feature_count"], errors="coerce").to_numpy(float))
    add("log1p_de_nsig_counts_real_z", np.log1p(pd.to_numeric(work["de_nsig_counts_real"], errors="coerce").fillna(0).to_numpy(float)))
    for col in ["context", "representation", "spectra_subset"]:
        dummies = pd.get_dummies(work[col], prefix=col, drop_first=True, dtype=float)
        for dcol in dummies.columns:
            cols.append(dummies[dcol].to_numpy(float))
            names.append(dcol)
    x = np.column_stack([np.ones(len(work)), *cols])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    sigma2 = ss_res / max(len(work) - x.shape[1], 1)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    idx = names.index("support_norm_log_mean_z") + 1
    return {
        "n_rows": int(len(work)),
        "coef": float(beta[idx]),
        "se": float(se[idx]),
        "t": float(beta[idx] / se[idx]) if se[idx] else float("nan"),
        "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def test_collection(df: pd.DataFrame, alias: dict[str, str], name: str, feature_map: dict[str, set[str]]) -> dict[str, object]:
    if not feature_map:
        return {"collection": name, "status": "not_available"}
    few = build_fewshot_examples(df, alias, feature_map, name)
    discovery = few[(few["spectra_subset"] == "discovery_hepg2_k562") & (few["feature_count"] > 0)]
    positive = discovery["support_norm_log_mean"].dropna()
    positive = positive[positive > 0]
    if positive.nunique() < 4:
        return {"collection": name, "status": "not_evaluable", "reason": "insufficient nonzero support diversity"}
    cutpoints = [float(x) for x in positive.quantile([0.25, 0.50, 0.75]).tolist()]
    few["support_level"] = assign_bins(few["support_norm_log_mean"], few["feature_count"], cutpoints)
    zero = build_zeroshot_examples(df, alias, feature_map, name)
    zero["support_level"] = assign_bins(zero["support_norm_log_mean"], zero["feature_count"], cutpoints)

    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    few.to_csv(READY / f"multi_pathway_{safe}_fewshot_examples.csv", index=False)
    zero.to_csv(READY / f"multi_pathway_{safe}_zeroshot_examples.csv", index=False)
    few_levels, few_trends = summarize(few, "support_level")
    zero_levels, zero_trends = summarize(zero, "support_level")
    levels = pd.concat([few_levels, zero_levels], ignore_index=True)
    trends = pd.concat([few_trends, zero_trends], ignore_index=True)
    levels.to_csv(ART / f"spc_multi_pathway_{safe}.csv", index=False)
    trends.to_csv(ART / f"spc_multi_pathway_{safe}_trends.csv", index=False)

    conf = trends[trends["spectra_subset"] == "confirmation_jurkat_rpe1"]
    zconf = trends[trends["spectra_subset"] == "zeroshot_cross_protocol_confirmation"]
    decision = (
        len(conf) == 2
        and len(zconf) == 2
        and conf["similarity_monotone_increasing"].all()
        and zconf["similarity_monotone_increasing"].all()
        and (conf["n_levels"] >= 3).all()
        and (zconf["n_levels"] >= 3).all()
        and (conf["high_minus_low_overlap_at_N"] > 0).all()
        and (zconf["high_minus_low_overlap_at_N"] > 0).all()
    )
    ols = ols_support_coef(few)
    result = {
        "collection": name,
        "status": "supports_broader_claim" if decision else "does_not_support_broader_claim",
        "n_genes_with_features": int(sum(1 for feats in feature_map.values() if feats)),
        "n_unique_features": int(len(set().union(*feature_map.values()))) if feature_map else 0,
        "cutpoints_from_fewshot_discovery_norm_support": cutpoints,
        "fewshot_confirmation_hvg_gap": float(conf[conf["representation"] == "HVG"]["high_minus_low_overlap_at_N"].iloc[0]) if not conf[conf["representation"] == "HVG"].empty else float("nan"),
        "fewshot_confirmation_se_gap": float(conf[conf["representation"] == "SE"]["high_minus_low_overlap_at_N"].iloc[0]) if not conf[conf["representation"] == "SE"].empty else float("nan"),
        "zeroshot_hvg_gap": float(zconf[zconf["representation"] == "HVG"]["high_minus_low_overlap_at_N"].iloc[0]) if not zconf[zconf["representation"] == "HVG"].empty else float("nan"),
        "zeroshot_se_gap": float(zconf[zconf["representation"] == "SE"]["high_minus_low_overlap_at_N"].iloc[0]) if not zconf[zconf["representation"] == "SE"].empty else float("nan"),
        "fewshot_support_coef_controlled": ols,
        "levels_artifact": str(ART / f"spc_multi_pathway_{safe}.csv"),
        "trends_artifact": str(ART / f"spc_multi_pathway_{safe}_trends.csv"),
    }
    return result


def top_pathway_examples(collections: dict[str, dict[str, set[str]]], examples_path: Path) -> dict[str, object]:
    ex = pd.read_csv(examples_path, low_memory=False)
    out = {}
    for coll_name, fmap in collections.items():
        counter = Counter()
        high_genes = ex[ex["support_level"] == "L4_very_high"]["gene_symbol_canonical"].drop_duplicates()
        for gene in high_genes:
            for feature in fmap.get(str(gene), set()):
                counter[feature] += 1
        out[coll_name] = counter.most_common(20)
    return out


def main() -> None:
    ART.mkdir(exist_ok=True)
    READY.mkdir(exist_ok=True)
    df = base.load_results()
    hgnc_info, alias = base.load_hgnc()
    collections: dict[str, dict[str, set[str]]] = {}
    collections["Reactome"] = reactome_collection(hgnc_info, alias)
    collections["HGNC_gene_group"] = hgnc_group_collection(hgnc_info)
    collections.update(parse_go_collections(alias))
    collections["WikiPathways"] = parse_wikipathways(alias)
    collections["KEGG"] = parse_kegg(alias)

    manifest = {
        "sources": {
            "GOA": GOA_URL,
            "GO_basic": GO_OBO_URL,
            "WikiPathways": WIKIPATHWAYS_URL,
            "KEGG_list_hsa": KEGG_LIST_HSA,
            "KEGG_list_pathway": KEGG_LIST_PATHWAY,
            "KEGG_link_pathway": KEGG_LINK_PATHWAY,
        },
        "collections": {name: {"n_genes": len(fmap), "n_features": len(set().union(*fmap.values())) if fmap else 0} for name, fmap in collections.items()},
    }
    (ART / "multi_pathway_source_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = []
    for name, fmap in collections.items():
        print(f"testing {name} genes={len(fmap)}")
        results.append(test_collection(df, alias, name, fmap))
    screen = pd.DataFrame(results)
    screen.to_csv(ART / "multi_pathway_support_screen.csv", index=False)
    summary = {
        "status": "completed",
        "interpretation": "Independent gene-set systems were tested using the same frozen train-support SPECTRA protocol.",
        "screen": results,
        "supporting_collections": [r["collection"] for r in results if r.get("status") == "supports_broader_claim"],
        "artifacts": [
            str(ART / "multi_pathway_source_manifest.json"),
            str(ART / "multi_pathway_support_screen.csv"),
        ],
    }
    (ART / "multi_pathway_support_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
