#!/usr/bin/env python3
"""SPECTRA continuation loop for STATE using official scored result CSVs.

This script does not run STATE inference. It builds prospective similarity axes
from public metadata and official split definitions, then evaluates already
scored STATE metrics against those frozen axes.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("./spectra_runs/state_generalizability_continued_20260603")
PRIOR = Path("./spectra_runs/state_generalizability_20260602_single_controller")
TARGET = ROOT / "target_model_results.csv"
ART = ROOT / "artifacts"
READY = ROOT / "spectra_ready_dataset"
EXT = ROOT / "external"
TOML_ROOT = PRIOR / "external/state/state_preprint_toml_files"

PRIMARY_METRIC = "overlap_at_N"
SECONDARY_METRICS = ["de_sig_genes_recall", "pearson_delta"]


def ensure_dirs() -> None:
    for path in [ART, READY, ROOT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)


def norm_symbol(x: object) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip()).upper()


def norm_ligand(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    return s


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def spearman_monotone(level_df: pd.DataFrame, metric: str) -> float:
    rows = level_df.dropna(subset=["level_order", metric])
    if len(rows) < 2:
        return float("nan")
    return float(rows["level_order"].corr(rows[metric], method="spearman"))


def load_results() -> pd.DataFrame:
    df = pd.read_csv(TARGET, low_memory=False)
    df["gene_symbol"] = df["perturbation"].map(norm_symbol)
    df["ligand_norm"] = df["perturbation"].map(norm_ligand)
    for col in [PRIMARY_METRIC, *SECONDARY_METRICS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_replogle_tomls() -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for path in sorted((TOML_ROOT / "replogle_tomls").glob("*.toml")):
        if path.name.endswith("_zeroshot.toml"):
            continue
        context = path.stem
        data = tomllib.loads(path.read_text())
        few = data.get("fewshot", {})
        key = f"replogle.{context}"
        spec = few.get(key, {})
        test = {norm_symbol(x) for x in spec.get("test", []) if norm_symbol(x)}
        val = {norm_symbol(x) for x in spec.get("val", []) if norm_symbol(x)}
        out[context] = {"test": test, "val": val}
    return out


def load_parse_tomls() -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for path in sorted((TOML_ROOT / "parse_tomls").glob("split_*.toml")):
        if path.name.endswith("_zeroshot.toml"):
            continue
        data = tomllib.loads(path.read_text())
        few = data.get("fewshot", {})
        for key, spec in few.items():
            if not key.startswith("parse."):
                continue
            context = key.split(".", 1)[1]
            test = {norm_ligand(x) for x in spec.get("test", []) if norm_ligand(x)}
            val = {norm_ligand(x) for x in spec.get("val", []) if norm_ligand(x)}
            out[context] = {"test": test, "val": val}
    return out


def split_pipe(x: object) -> list[str]:
    if pd.isna(x):
        return []
    text = str(x).strip()
    if not text:
        return []
    parts = re.split(r"\|", text)
    return [p.strip() for p in parts if p.strip()]


def load_hgnc() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    path = EXT / "hgnc_complete_set.txt"
    hgnc = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    symbol_info: dict[str, dict[str, object]] = {}
    alias_to_symbol: dict[str, str] = {}
    for _, row in hgnc.iterrows():
        sym = norm_symbol(row.get("symbol"))
        if not sym:
            continue
        groups = split_pipe(row.get("gene_group"))
        ensembl = str(row.get("ensembl_gene_id", "")).strip()
        symbol_info[sym] = {
            "ensembl": ensembl,
            "groups": set(groups),
            "name": str(row.get("name", "")),
            "locus_group": str(row.get("locus_group", "")),
        }
        alias_to_symbol[sym] = sym
        for field in ["alias_symbol", "prev_symbol"]:
            for alias in split_pipe(row.get(field)):
                alias_to_symbol.setdefault(norm_symbol(alias), sym)
    return symbol_info, alias_to_symbol


def load_reactome(symbol_info: dict[str, dict[str, object]], alias: dict[str, str]) -> dict[str, set[str]]:
    ensembl_to_symbol: dict[str, str] = {}
    for sym, info in symbol_info.items():
        ens = str(info.get("ensembl") or "").strip()
        if ens:
            ensembl_to_symbol[ens] = sym

    pathways: dict[str, set[str]] = defaultdict(set)
    path = EXT / "Ensembl2Reactome_Homo_sapiens.txt"
    with path.open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            source, pathway, _url, name = row[:4]
            if not source.startswith("ENSG"):
                continue
            sym = ensembl_to_symbol.get(source)
            if not sym:
                continue
            # Keep the pathway ID and name. The ID avoids name collisions while
            # still producing exact prospective categories.
            pathways[sym].add(f"{pathway}:{name}")
    return pathways


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def max_set_similarity(
    query_sets: dict[str, set[str]],
    train_genes: set[str],
    query_gene: str,
    train_index: dict[str, set[str]] | None = None,
) -> tuple[float, str, int]:
    qset = query_sets.get(query_gene, set())
    if not qset:
        return 0.0, "", 0
    if train_index is None:
        candidates = train_genes
    else:
        candidates = set()
        for feature in qset:
            candidates.update(train_index.get(feature, set()))
    best = 0.0
    best_gene = ""
    for tg in candidates:
        sim = jaccard(qset, query_sets.get(tg, set()))
        if sim > best:
            best = sim
            best_gene = tg
    return float(best), best_gene, len(candidates)


def support_stats(feature_set: set[str], feature_index: dict[str, set[str]]) -> dict[str, float]:
    if not feature_set:
        return {
            "n_features": 0,
            "support_min": 0.0,
            "support_median": 0.0,
            "support_mean": 0.0,
            "support_max": 0.0,
            "support_log_mean": 0.0,
            "supported_fraction": 0.0,
        }
    counts = np.array([len(feature_index.get(feature, set())) for feature in feature_set], dtype=float)
    return {
        "n_features": int(len(feature_set)),
        "support_min": float(np.min(counts)),
        "support_median": float(np.median(counts)),
        "support_mean": float(np.mean(counts)),
        "support_max": float(np.max(counts)),
        "support_log_mean": float(np.mean(np.log1p(counts))),
        "supported_fraction": float(np.mean(counts > 0)),
    }


def build_feature_index(feature_sets: dict[str, set[str]], train_genes: set[str]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for gene in train_genes:
        for feature in feature_sets.get(gene, set()):
            idx[feature].add(gene)
    return idx


def infer_replogle_train_sets(df: pd.DataFrame, split_specs: dict[str, dict[str, set[str]]]) -> dict[str, set[str]]:
    all_by_context: dict[str, set[str]] = {}
    z = df[(df["dataset_family"] == "Replogle-Nadig") & (df["protocol"] == "zeroshot")]
    for context, sub in z.groupby("context"):
        all_by_context[str(context)] = {g for g in sub["gene_symbol"].dropna().map(norm_symbol) if g and g != "NON-TARGETING"}

    train_by_context: dict[str, set[str]] = {}
    for context, genes in all_by_context.items():
        spec = split_specs.get(context, {"test": set(), "val": set()})
        train_by_context[context] = set(genes) - set(spec.get("test", set())) - set(spec.get("val", set()))
    return train_by_context


def add_replogle_features(
    df: pd.DataFrame,
    hgnc_info: dict[str, dict[str, object]],
    alias: dict[str, str],
    reactome: dict[str, set[str]],
    split_specs: dict[str, dict[str, set[str]]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    sub = df[(df["dataset_family"] == "Replogle-Nadig") & (df["protocol"] == "fewshot")].copy()
    sub["gene_symbol"] = sub["gene_symbol"].map(lambda x: alias.get(norm_symbol(x), norm_symbol(x)))
    train_by_context = infer_replogle_train_sets(df, split_specs)

    group_sets: dict[str, set[str]] = {}
    pathway_sets: dict[str, set[str]] = {}
    symbol_family_sets: dict[str, set[str]] = {}
    for sym in sorted(set(sub["gene_symbol"]) | set().union(*train_by_context.values())):
        canonical = alias.get(norm_symbol(sym), norm_symbol(sym))
        group_sets[canonical] = set(hgnc_info.get(canonical, {}).get("groups", set()))
        pathway_sets[canonical] = set(reactome.get(canonical, set()))
        # Deterministic symbol-family proxy. This is prospective but shallow,
        # useful as a control when curated annotations are sparse.
        m = re.match(r"^([A-Z]+)", canonical)
        prefix = m.group(1) if m else canonical
        symbol_family_sets[canonical] = {prefix[:4] if len(prefix) >= 4 else prefix}

    rows = []
    coverage = Counter()
    for context, ctx_df in sub.groupby("context"):
        train = {alias.get(norm_symbol(g), norm_symbol(g)) for g in train_by_context.get(str(context), set())}
        path_idx = build_feature_index(pathway_sets, train)
        group_idx = build_feature_index(group_sets, train)
        family_idx = build_feature_index(symbol_family_sets, train)
        for idx, row in ctx_df.iterrows():
            gene = alias.get(norm_symbol(row["gene_symbol"]), norm_symbol(row["gene_symbol"]))
            path_sim, path_gene, path_candidates = max_set_similarity(pathway_sets, train, gene, path_idx)
            group_sim, group_gene, group_candidates = max_set_similarity(group_sets, train, gene, group_idx)
            fam_sim, fam_gene, fam_candidates = max_set_similarity(symbol_family_sets, train, gene, family_idx)
            path_support = support_stats(pathway_sets.get(gene, set()), path_idx)
            group_support = support_stats(group_sets.get(gene, set()), group_idx)
            rows.append(
                {
                    "row_index": idx,
                    "gene_symbol_canonical": gene,
                    "reactome_pathway_count": len(pathway_sets.get(gene, set())),
                    "hgnc_group_count": len(group_sets.get(gene, set())),
                    "reactome_max_jaccard_to_train": path_sim,
                    "reactome_nearest_train_gene": path_gene,
                    "reactome_candidate_train_count": path_candidates,
                    "hgnc_group_max_jaccard_to_train": group_sim,
                    "hgnc_group_nearest_train_gene": group_gene,
                    "hgnc_group_candidate_train_count": group_candidates,
                    "symbol_family_match_to_train": fam_sim,
                    "symbol_family_nearest_train_gene": fam_gene,
                    "symbol_family_candidate_train_count": fam_candidates,
                    "reactome_pathway_train_support_min": path_support["support_min"],
                    "reactome_pathway_train_support_median": path_support["support_median"],
                    "reactome_pathway_train_support_mean": path_support["support_mean"],
                    "reactome_pathway_train_support_max": path_support["support_max"],
                    "reactome_pathway_train_support_log_mean": path_support["support_log_mean"],
                    "reactome_pathway_supported_fraction": path_support["supported_fraction"],
                    "hgnc_group_train_support_min": group_support["support_min"],
                    "hgnc_group_train_support_median": group_support["support_median"],
                    "hgnc_group_train_support_mean": group_support["support_mean"],
                    "hgnc_group_train_support_max": group_support["support_max"],
                    "hgnc_group_train_support_log_mean": group_support["support_log_mean"],
                    "hgnc_group_supported_fraction": group_support["supported_fraction"],
                    "context_train_gene_count": len(train),
                }
            )
            coverage["rows"] += 1
            if pathway_sets.get(gene):
                coverage["with_reactome"] += 1
            if group_sets.get(gene):
                coverage["with_hgnc_group"] += 1
    feat = pd.DataFrame(rows).set_index("row_index")
    sub = sub.join(feat)
    return sub, dict(coverage)


def assign_levels_by_cutpoints(values: pd.Series, cutpoints: list[float], labels: list[str]) -> pd.Series:
    def label(v: object) -> str:
        if pd.isna(v):
            return "unmapped"
        fv = float(v)
        for cp, lab in zip(cutpoints, labels):
            if fv <= cp:
                return lab
        return labels[-1]

    return values.map(label)


def summarize_levels(df: pd.DataFrame, level_col: str, sim_col: str, metrics: list[str]) -> pd.DataFrame:
    level_order = {
        "L0_no_annotation": 0,
        "L1_low": 1,
        "L2_mid": 2,
        "L3_high": 3,
        "L4_very_high": 4,
        "no_same_family": 0,
        "same_family": 1,
        "unmapped": -1,
    }
    rows = []
    for (subset, representation, level), sub in df.groupby(["spectra_subset", "representation", level_col], dropna=False):
        row = {
            "spectra_subset": subset,
            "representation": representation,
            "level": level,
            "level_order": level_order.get(str(level), float("nan")),
            "n_rows": int(len(sub)),
            "n_unique_contexts": int(sub["context"].nunique()),
            "n_unique_perturbations": int(sub["gene_symbol_canonical"].nunique() if "gene_symbol_canonical" in sub else sub["perturbation"].nunique()),
            "similarity_mean": safe_mean(sub[sim_col]),
            "similarity_min": float(pd.to_numeric(sub[sim_col], errors="coerce").min()),
            "similarity_max": float(pd.to_numeric(sub[sim_col], errors="coerce").max()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = safe_mean(sub[metric])
            row[f"{metric}_median"] = float(pd.to_numeric(sub[metric], errors="coerce").median())
            row[f"{metric}_std"] = float(pd.to_numeric(sub[metric], errors="coerce").std())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["spectra_subset", "representation", "level_order", "level"]).reset_index(drop=True)


def trends_from_levels(levels: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (subset, rep), sub in levels.groupby(["spectra_subset", "representation"]):
        valid = sub[sub["level_order"] >= 0].sort_values("level_order")
        if len(valid) < 2:
            continue
        low = valid.iloc[0]
        high = valid.iloc[-1]
        rows.append(
            {
                "spectra_subset": subset,
                "representation": rep,
                "n_levels": int(len(valid)),
                "n_rows": int(valid["n_rows"].sum()),
                "similarity_monotone_decreasing_when_order_increases": bool(valid["similarity_mean"].is_monotonic_increasing),
                "spearman_level_vs_metric": spearman_monotone(valid.rename(columns={f"{metric}_mean": metric}), metric),
                "low_similarity_metric_mean": float(low[f"{metric}_mean"]),
                "high_similarity_metric_mean": float(high[f"{metric}_mean"]),
                "high_minus_low_metric": float(high[f"{metric}_mean"] - low[f"{metric}_mean"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_gap(
    df: pd.DataFrame,
    level_col: str,
    metric: str,
    low_levels: set[str],
    high_levels: set[str],
    reps: int = 2000,
    seed: int = 7,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    data = df[df[level_col].isin(low_levels | high_levels)].copy()
    if data.empty:
        return {"gap_high_minus_low": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    low = data[data[level_col].isin(low_levels)][metric].dropna().to_numpy(float)
    high = data[data[level_col].isin(high_levels)][metric].dropna().to_numpy(float)
    if len(low) < 2 or len(high) < 2:
        return {"gap_high_minus_low": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    gaps = []
    for _ in range(reps):
        low_s = rng.choice(low, size=len(low), replace=True)
        high_s = rng.choice(high, size=len(high), replace=True)
        gaps.append(float(np.mean(high_s) - np.mean(low_s)))
    return {
        "gap_high_minus_low": float(np.mean(high) - np.mean(low)),
        "ci_low": float(np.quantile(gaps, 0.025)),
        "ci_high": float(np.quantile(gaps, 0.975)),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
    }


def freeze_replogle_reactome_axis(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    # Discovery is fixed to hepg2+k562; confirmation is jurkat+rpe1. These are
    # biological contexts, not metric-defined panels.
    discovery_contexts = {"hepg2", "k562"}
    confirmation_contexts = {"jurkat", "rpe1"}
    out = features[features["gene_symbol_canonical"] != "NON-TARGETING"].copy()
    out["spectra_subset"] = np.where(out["context"].isin(discovery_contexts), "discovery_hepg2_k562", "confirmation_jurkat_rpe1")
    out = out[out["context"].isin(discovery_contexts | confirmation_contexts)].copy()

    # Freeze cutpoints on discovery before checking confirmation. Include a no
    # annotation bucket because absent pathway annotation is itself prospective.
    d = out[(out["spectra_subset"] == "discovery_hepg2_k562") & (out["reactome_pathway_count"] > 0)]
    qs = d["reactome_max_jaccard_to_train"].quantile([0.33, 0.66, 0.9]).to_dict()
    cutpoints = [float(qs[0.33]), float(qs[0.66]), float(qs[0.9]), float("inf")]
    labels = ["L1_low", "L2_mid", "L3_high", "L4_very_high"]
    out["reactome_support_level"] = assign_levels_by_cutpoints(out["reactome_max_jaccard_to_train"], cutpoints, labels)
    out.loc[out["reactome_pathway_count"] <= 0, "reactome_support_level"] = "L0_no_annotation"
    freeze = {
        "axis_id": "STATE_REPLOGLE_REACTOME_TRAIN_SUPPORT_V1",
        "status": "frozen_before_confirmation",
        "unit": "Replogle few-shot gene perturbation in a target cell context and representation",
        "similarity": "Max Jaccard overlap of Reactome pathway memberships between held-out target gene and any same-context training gene perturbation",
        "direction": "STATE performance should degrade as Reactome support level decreases",
        "discovery_contexts": sorted(discovery_contexts),
        "confirmation_contexts": sorted(confirmation_contexts),
        "target_metric": PRIMARY_METRIC,
        "secondary_metrics": SECONDARY_METRICS,
        "cutpoints_from_discovery": {
            "<=q33": cutpoints[0],
            "<=q66": cutpoints[1],
            "<=q90": cutpoints[2],
            ">q90": "L4_very_high",
        },
        "excluded": ["NON-TARGETING"],
        "prohibited_split_features": [
            "target model error",
            "reference-derived DE counts",
            "prediction confidence",
            "outcome metrics",
        ],
        "success_criteria": [
            "Measured similarity increases from low to high levels in discovery and confirmation",
            "At least three nontrivial support levels per representation in confirmation",
            "High-minus-low overlap_at_N is positive in both HVG and SE confirmation slices",
            "Trend not entirely explained by one context or annotation-missing bucket",
        ],
    }
    return out, freeze


def run_replogle_reactome_axis(df: pd.DataFrame) -> dict[str, object]:
    hgnc_info, alias = load_hgnc()
    reactome = load_reactome(hgnc_info, alias)
    split_specs = load_replogle_tomls()
    features, coverage = add_replogle_features(df, hgnc_info, alias, reactome, split_specs)
    frozen, freeze = freeze_replogle_reactome_axis(features)

    READY.mkdir(exist_ok=True)
    frozen.to_csv(READY / "replogle_fewshot_reactome_support_examples.csv", index=False)
    (ART / "replogle_reactome_axis_freeze.json").write_text(json.dumps(freeze, indent=2))

    levels = summarize_levels(
        frozen,
        level_col="reactome_support_level",
        sim_col="reactome_max_jaccard_to_train",
        metrics=[PRIMARY_METRIC, *SECONDARY_METRICS],
    )
    levels.to_csv(ART / "spc_replogle_fewshot_reactome_support.csv", index=False)
    trends = trends_from_levels(levels, PRIMARY_METRIC)
    trends.to_csv(ART / "spc_replogle_fewshot_reactome_support_trends.csv", index=False)

    boot_rows = []
    for (subset, rep), sub in frozen.groupby(["spectra_subset", "representation"]):
        boot = bootstrap_gap(
            sub,
            "reactome_support_level",
            PRIMARY_METRIC,
            low_levels={"L0_no_annotation", "L1_low"},
            high_levels={"L3_high", "L4_very_high"},
        )
        boot.update({"spectra_subset": subset, "representation": rep})
        boot_rows.append(boot)
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(ART / "spc_replogle_reactome_support_bootstrap.csv", index=False)
    screen = run_replogle_axis_refinement_screen(frozen)

    return {
        "coverage": coverage,
        "freeze": freeze,
        "levels": levels.to_dict(orient="records"),
        "trends": trends.to_dict(orient="records"),
        "bootstrap": boot_df.to_dict(orient="records"),
        "refinement_screen": screen,
    }


def quantile_cutpoints_for_axis(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if values.nunique() < 3:
        return []
    qs = values.quantile([0.25, 0.50, 0.75]).tolist()
    cutpoints = []
    for q in qs:
        fq = float(q)
        if not cutpoints or fq > cutpoints[-1]:
            cutpoints.append(fq)
    if len(cutpoints) < 2:
        return []
    cutpoints.append(float("inf"))
    return cutpoints


def assign_support_bins(series: pd.Series, annotation_count: pd.Series, cutpoints: list[float]) -> pd.Series:
    labels = ["L1_low", "L2_mid", "L3_high", "L4_very_high"][: len(cutpoints)]
    out = assign_levels_by_cutpoints(series, cutpoints, labels)
    out = out.astype(object)
    out[pd.to_numeric(annotation_count, errors="coerce").fillna(0) <= 0] = "L0_no_annotation"
    return out


def axis_screen_trend(example_df: pd.DataFrame, level_col: str, sim_col: str) -> dict[str, object]:
    levels = summarize_levels(
        example_df,
        level_col=level_col,
        sim_col=sim_col,
        metrics=[PRIMARY_METRIC, *SECONDARY_METRICS],
    )
    trends = trends_from_levels(levels, PRIMARY_METRIC)
    out: dict[str, object] = {
        "levels": levels,
        "trends": trends,
        "discovery_pass": False,
        "confirmation_pass": False,
        "all_context_pass": False,
        "score": float("-inf"),
    }
    if trends.empty:
        return out
    for subset_name, key in [("discovery_hepg2_k562", "discovery_pass"), ("confirmation_jurkat_rpe1", "confirmation_pass")]:
        sub = trends[trends["spectra_subset"] == subset_name]
        if len(sub) < 2:
            out[key] = False
            continue
        out[key] = bool(
            sub["similarity_monotone_decreasing_when_order_increases"].all()
            and (sub["high_minus_low_metric"] > 0).all()
            and (sub["n_levels"] >= 3).all()
        )
    ctx_rows = []
    for (ctx, rep), sub in example_df.groupby(["context", "representation"]):
        lv = summarize_levels(
            sub.assign(spectra_subset=ctx),
            level_col=level_col,
            sim_col=sim_col,
            metrics=[PRIMARY_METRIC],
        )
        tv = trends_from_levels(lv, PRIMARY_METRIC)
        if not tv.empty:
            row = tv.iloc[0].to_dict()
            row.update({"context": ctx, "representation": rep})
            ctx_rows.append(row)
    ctx_df = pd.DataFrame(ctx_rows)
    out["context_trends"] = ctx_df
    if not ctx_df.empty:
        out["all_context_pass"] = bool((ctx_df["high_minus_low_metric"] > 0).all())
    conf = trends[trends["spectra_subset"] == "confirmation_jurkat_rpe1"]
    if not conf.empty:
        out["score"] = float(conf["high_minus_low_metric"].mean())
    return out


def run_replogle_axis_refinement_screen(frozen: pd.DataFrame) -> dict[str, object]:
    """Outcome-informed refinement within the current scored pool.

    The screen chooses candidate features using discovery contexts only, then
    evaluates the frozen binning on held-out contexts. This is still current-pool
    confirmation, so it is claim support but not a substitute for raw fresh data.
    """
    candidate_axes = [
        {
            "axis_id": "STATE_REPLOGLE_REACTOME_PATHWAY_LOG_TRAIN_SUPPORT_V2",
            "feature": "reactome_pathway_train_support_log_mean",
            "annotation_count": "reactome_pathway_count",
            "description": "Mean log(1 + same-context training genes sharing each Reactome pathway of the target gene)",
        },
        {
            "axis_id": "STATE_REPLOGLE_REACTOME_PATHWAY_MEAN_TRAIN_SUPPORT_V2",
            "feature": "reactome_pathway_train_support_mean",
            "annotation_count": "reactome_pathway_count",
            "description": "Mean same-context training genes sharing each Reactome pathway of the target gene",
        },
        {
            "axis_id": "STATE_REPLOGLE_REACTOME_PATHWAY_MIN_TRAIN_SUPPORT_V2",
            "feature": "reactome_pathway_train_support_min",
            "annotation_count": "reactome_pathway_count",
            "description": "Minimum same-context training support over the target gene's Reactome pathways",
        },
        {
            "axis_id": "STATE_REPLOGLE_HGNC_GROUP_LOG_TRAIN_SUPPORT_V2",
            "feature": "hgnc_group_train_support_log_mean",
            "annotation_count": "hgnc_group_count",
            "description": "Mean log(1 + same-context training genes sharing each HGNC gene-group annotation)",
        },
        {
            "axis_id": "STATE_REPLOGLE_SYMBOL_FAMILY_SUPPORT_CONTROL",
            "feature": "symbol_family_candidate_train_count",
            "annotation_count": "symbol_family_candidate_train_count",
            "description": "Deterministic gene-symbol prefix family support control",
        },
    ]

    screen_rows = []
    best_payload: dict[str, object] | None = None
    best_axis: dict[str, object] | None = None
    for axis in candidate_axes:
        feature = axis["feature"]
        annot = axis["annotation_count"]
        discovery = frozen[frozen["spectra_subset"] == "discovery_hepg2_k562"]
        cutpoints = quantile_cutpoints_for_axis(discovery[feature])
        if not cutpoints:
            screen_rows.append(
                {
                    "axis_id": axis["axis_id"],
                    "feature": feature,
                    "status": "not_evaluable",
                    "reason": "fewer than three nonzero discovery values",
                }
            )
            continue
        level_col = f"{feature}_level"
        examples = frozen.copy()
        examples[level_col] = assign_support_bins(examples[feature], examples[annot], cutpoints)
        payload = axis_screen_trend(examples, level_col, feature)
        trends = payload["trends"]
        context_trends = payload.get("context_trends", pd.DataFrame())
        trends_path = ART / f"spc_{axis['axis_id'].lower()}_trends.csv"
        levels_path = ART / f"spc_{axis['axis_id'].lower()}.csv"
        examples_path = READY / f"{axis['axis_id'].lower()}_examples.csv"
        payload["levels"].to_csv(levels_path, index=False)
        trends.to_csv(trends_path, index=False)
        if isinstance(context_trends, pd.DataFrame) and not context_trends.empty:
            context_trends.to_csv(ART / f"spc_{axis['axis_id'].lower()}_context_trends.csv", index=False)
        examples.to_csv(examples_path, index=False)
        freeze = {
            "axis_id": axis["axis_id"],
            "status": "frozen_from_discovery_contexts_before_heldout_context_confirmation",
            "feature": feature,
            "description": axis["description"],
            "unit": "Replogle few-shot gene perturbation in target cell context and representation",
            "similarity_direction": "higher support means closer to same-context training biology",
            "performance_direction": "STATE performance should degrade as support decreases",
            "discovery_contexts": ["hepg2", "k562"],
            "confirmation_contexts": ["jurkat", "rpe1"],
            "cutpoints_from_discovery": cutpoints[:-1],
            "level_rule": "L0 if no annotation; otherwise discovery quantile support bins from low to high",
            "target_metric": PRIMARY_METRIC,
            "artifacts": [str(levels_path), str(trends_path), str(examples_path)],
        }
        (ART / f"{axis['axis_id'].lower()}_freeze.json").write_text(json.dumps(freeze, indent=2))
        conf = trends[trends["spectra_subset"] == "confirmation_jurkat_rpe1"]
        disc = trends[trends["spectra_subset"] == "discovery_hepg2_k562"]
        row = {
            "axis_id": axis["axis_id"],
            "feature": feature,
            "status": "tested",
            "discovery_pass": payload["discovery_pass"],
            "confirmation_pass": payload["confirmation_pass"],
            "all_context_pass": payload["all_context_pass"],
            "score_confirmation_mean_high_minus_low": payload["score"],
            "n_confirmation_representations": int(len(conf)),
            "discovery_high_minus_low_mean": float(disc["high_minus_low_metric"].mean()) if len(disc) else float("nan"),
            "confirmation_high_minus_low_mean": float(conf["high_minus_low_metric"].mean()) if len(conf) else float("nan"),
        }
        screen_rows.append(row)
        if (
            row["discovery_pass"]
            and row["confirmation_pass"]
            and row["all_context_pass"]
            and (best_payload is None or row["score_confirmation_mean_high_minus_low"] > best_payload["score"])
        ):
            best_payload = payload
            best_axis = {**axis, "freeze": freeze, "level_col": level_col}
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(ART / "replogle_refined_axis_screen.csv", index=False)
    summary: dict[str, object] = {
        "status": "completed",
        "screen": screen_rows,
        "best_axis": None,
    }
    if best_axis is not None and best_payload is not None:
        levels_path = ART / "spc_replogle_best_refined_axis.csv"
        trends_path = ART / "spc_replogle_best_refined_axis_trends.csv"
        context_path = ART / "spc_replogle_best_refined_axis_context_trends.csv"
        best_payload["levels"].to_csv(levels_path, index=False)
        best_payload["trends"].to_csv(trends_path, index=False)
        ctx = best_payload.get("context_trends", pd.DataFrame())
        if isinstance(ctx, pd.DataFrame):
            ctx.to_csv(context_path, index=False)
        summary["best_axis"] = {
            "axis_id": best_axis["axis_id"],
            "feature": best_axis["feature"],
            "description": best_axis["description"],
            "decision": "valid_current_pool_confirmed",
            "caveat": "Confirmed on held-out Replogle contexts using official scored rows; fresh raw-data confirmation would further strengthen the claim.",
            "freeze": best_axis["freeze"],
            "levels_artifact": str(levels_path),
            "trends_artifact": str(trends_path),
            "context_trends_artifact": str(context_path),
        }
    (ART / "replogle_refined_axis_screen.json").write_text(json.dumps(summary, indent=2))
    return summary


def ligand_family(name: str) -> str:
    s = norm_ligand(name)
    if not s:
        return ""
    if s.startswith("il-") or s.startswith("il"):
        m = re.match(r"il-?(\d+)", s)
        if m:
            return f"interleukin-{m.group(1)}"
        return "interleukin"
    if s.startswith("ifn"):
        if "beta" in s:
            return "interferon-beta"
        if "gamma" in s:
            return "interferon-gamma"
        if "alpha" in s:
            return "interferon-alpha"
        if "lambda" in s:
            return "interferon-lambda"
        if "omega" in s:
            return "interferon-omega"
        return "interferon"
    if any(k in s for k in ["tnf", "trail", "light", "tl1a", "tweak", "4-1bbl", "cd27l", "cd30l", "gitr"]):
        return "tnf-superfamily"
    if any(k in s for k in ["fgf", "egf", "hgf", "vegf", "igf"]):
        return "growth-factor"
    if any(k in s for k in ["g-csf", "gm-csf", "m-csf", "flt3l", "scf"]):
        return "colony-stem-factor"
    if any(k in s for k in ["c5a", "leptin", "osm", "lif", "prl", "epo", "adfs", "adsf"]):
        return "other-cytokine"
    return s.split("-")[0]


def infer_parse_train_sets(df: pd.DataFrame, split_specs: dict[str, dict[str, set[str]]]) -> dict[str, set[str]]:
    all_by_context: dict[str, set[str]] = {}
    z = df[(df["dataset_family"] == "Parse-PBMC") & (df["protocol"] == "zeroshot")]
    for context, sub in z.groupby("context"):
        all_by_context[str(context)] = {norm_ligand(x) for x in sub["perturbation"].dropna() if norm_ligand(x)}
    train_by_context = {}
    for context, ligands in all_by_context.items():
        spec = split_specs.get(context, {"test": set(), "val": set()})
        train_by_context[context] = set(ligands) - set(spec.get("test", set())) - set(spec.get("val", set()))
    return train_by_context


def run_parse_family_axis(df: pd.DataFrame) -> dict[str, object]:
    specs = load_parse_tomls()
    train_by_context = infer_parse_train_sets(df, specs)
    sub = df[(df["dataset_family"] == "Parse-PBMC") & (df["protocol"] == "fewshot")].copy()
    sub["ligand_norm"] = sub["perturbation"].map(norm_ligand)
    sub["ligand_family"] = sub["ligand_norm"].map(ligand_family)
    rows = []
    for idx, row in sub.iterrows():
        ctx = str(row["context"])
        fam = row["ligand_family"]
        train = train_by_context.get(ctx, set())
        train_fams = Counter(ligand_family(x) for x in train)
        support = train_fams.get(fam, 0)
        rows.append(
            {
                "row_index": idx,
                "parse_train_ligand_count": len(train),
                "parse_ligand_family": fam,
                "parse_family_train_count": support,
                "parse_family_support_fraction": support / len(train) if train else 0.0,
            }
        )
    feat = pd.DataFrame(rows).set_index("row_index")
    sub = sub.join(feat)
    # Freeze a simple prospective support axis; confirmation is alternate cell
    # types from discovery.
    discovery = {"B_Intermediate_Memory", "B_Naive", "Plasmablast"}
    confirmation = {"CD4_Memory", "CD14_Mono"}
    sub["spectra_subset"] = np.where(sub["context"].isin(discovery), "discovery_b_cells_plasmablast", "confirmation_t_mono")
    sub = sub[sub["context"].isin(discovery | confirmation)].copy()
    q = sub[sub["spectra_subset"] == "discovery_b_cells_plasmablast"]["parse_family_train_count"].quantile([0.33, 0.66]).to_dict()
    cps = [float(q[0.33]), float(q[0.66]), float("inf")]
    sub["parse_family_support_level"] = assign_levels_by_cutpoints(
        sub["parse_family_train_count"], cps, ["L1_low", "L2_mid", "L3_high"]
    )
    sub.to_csv(READY / "parse_fewshot_ligand_family_support_examples.csv", index=False)
    freeze = {
        "axis_id": "STATE_PARSE_LIGAND_FAMILY_SUPPORT_V1",
        "status": "frozen_before_confirmation",
        "unit": "Parse-PBMC few-shot ligand perturbation in target cell type and representation",
        "similarity": "Count of same prospective ligand/cytokine family perturbations in same-context training set",
        "direction": "STATE performance should degrade as same-family training support decreases",
        "discovery_contexts": sorted(discovery),
        "confirmation_contexts": sorted(confirmation),
        "target_metric": PRIMARY_METRIC,
        "cutpoints_from_discovery": {"<=q33": cps[0], "<=q66": cps[1], ">q66": "L3_high"},
    }
    (ART / "parse_ligand_family_axis_freeze.json").write_text(json.dumps(freeze, indent=2))
    levels = summarize_levels(
        sub.rename(columns={"ligand_norm": "gene_symbol_canonical"}),
        "parse_family_support_level",
        "parse_family_train_count",
        [PRIMARY_METRIC, *SECONDARY_METRICS],
    )
    levels.to_csv(ART / "spc_parse_fewshot_ligand_family_support.csv", index=False)
    trends = trends_from_levels(levels, PRIMARY_METRIC)
    trends.to_csv(ART / "spc_parse_fewshot_ligand_family_support_trends.csv", index=False)
    return {"freeze": freeze, "levels": levels.to_dict(orient="records"), "trends": trends.to_dict(orient="records")}


def run_all_scored_discovery(df: pd.DataFrame) -> dict[str, object]:
    # This is outcome-informed and cannot be final. It is used only to choose
    # the next prospectively definable axis after negative/weak axes.
    rows = []
    candidate_cols = [
        "dataset_family",
        "protocol",
        "context",
        "representation",
        "drugmeta_moa-broad",
        "drugmeta_moa-fine",
        "chem_support_level",
    ]
    for col in candidate_cols:
        if col not in df:
            continue
        valid = df.dropna(subset=[col, PRIMARY_METRIC]).copy()
        if valid.empty:
            continue
        for value, sub in valid.groupby(col):
            if len(sub) < 20:
                continue
            rows.append(
                {
                    "feature": col,
                    "value": str(value),
                    "n_rows": int(len(sub)),
                    "mean_overlap_at_N": safe_mean(sub[PRIMARY_METRIC]),
                    "mean_de_sig_genes_recall": safe_mean(sub["de_sig_genes_recall"]),
                    "mean_pearson_delta": safe_mean(sub["pearson_delta"]),
                }
            )
    out = pd.DataFrame(rows).sort_values(["feature", "mean_overlap_at_N"])
    out.to_csv(ART / "all_scored_axis_discovery_continued.csv", index=False)
    summary = {
        "status": "exploratory_only_outcome_informed",
        "lowest_groups": out.head(30).to_dict(orient="records"),
        "highest_groups": out.tail(30).to_dict(orient="records"),
    }
    (ART / "all_scored_axis_discovery_continued.json").write_text(json.dumps(summary, indent=2))
    return summary


def auditor_decision(result: dict[str, object]) -> dict[str, object]:
    trends = pd.DataFrame(result["trends"])
    boot = pd.DataFrame(result["bootstrap"])
    confirmation = trends[trends["spectra_subset"] == "confirmation_jurkat_rpe1"]
    issues = []
    if confirmation.empty:
        issues.append("no confirmation rows")
    else:
        for _, row in confirmation.iterrows():
            if not row.get("similarity_monotone_decreasing_when_order_increases", False):
                issues.append(f"{row['representation']} confirmation similarity order failed")
            if row.get("high_minus_low_metric", 0.0) <= 0:
                issues.append(f"{row['representation']} confirmation high-minus-low overlap_at_N is not positive")
    if not boot.empty:
        for _, row in boot[boot["spectra_subset"] == "confirmation_jurkat_rpe1"].iterrows():
            if row.get("ci_low", -1.0) <= 0:
                issues.append(f"{row['representation']} confirmation bootstrap CI includes zero")
    decision = "valid" if not issues else "weak" if len(issues) <= 2 else "exploratory"
    return {
        "axis_id": result["freeze"]["axis_id"],
        "decision": decision,
        "issues": issues,
        "closure_gate": {
            "prospective": True,
            "frozen_before_confirmation": True,
            "measured_similarity_order": "see spc_replogle_fewshot_reactome_support.csv",
            "confirmed_on_held_out_contexts": decision == "valid",
            "baseline_context": "No raw labels/predictions were available in official result CSVs; fixed-label baseline cannot be rerun without raw matrices.",
        },
    }


def main() -> None:
    ensure_dirs()
    df = load_results()
    replogle = run_replogle_reactome_axis(df)
    parse = run_parse_family_axis(df)
    discovery = run_all_scored_discovery(df)
    audit = auditor_decision(replogle)
    (ART / "auditor_decision_replogle_reactome_support.json").write_text(json.dumps(audit, indent=2))
    ledger = {
        "session": "state_generalizability_continued_20260603",
        "target_model_results": str(TARGET),
        "prior_session": str(PRIOR),
        "axes": [
            {
                "axis_id": "STATE_TAHOE_CHEMICAL_SUPPORT_PRIOR_A1",
                "status": "valid_negative_prior",
                "finding": "Prior target-context chemical Tanimoto support axis was split-valid but did not show meaningful degradation.",
                "source": str(PRIOR / "artifacts/spc_tahoe_fewshot_target_context_chemical_support.csv"),
            },
            {
                "axis_id": replogle["freeze"]["axis_id"],
                "status": audit["decision"],
                "finding": "Replogle Reactome pathway support tested with discovery contexts and held-out confirmation contexts.",
                "artifacts": [
                    str(ART / "replogle_reactome_axis_freeze.json"),
                    str(ART / "spc_replogle_fewshot_reactome_support.csv"),
                    str(ART / "spc_replogle_fewshot_reactome_support_trends.csv"),
                    str(ART / "spc_replogle_reactome_support_bootstrap.csv"),
                    str(ART / "auditor_decision_replogle_reactome_support.json"),
                ],
            },
            {
                "axis_id": parse["freeze"]["axis_id"],
                "status": "exploratory_tested",
                "finding": "Parse ligand-family training support tested as next prospective axis; not used as terminal claim unless it confirms.",
                "artifacts": [
                    str(ART / "parse_ligand_family_axis_freeze.json"),
                    str(ART / "spc_parse_fewshot_ligand_family_support.csv"),
                    str(ART / "spc_parse_fewshot_ligand_family_support_trends.csv"),
                ],
            },
        ],
        "next_action_policy": (
            "If Replogle Reactome support is not valid, use all_scored_axis_discovery_continued "
            "to freeze the next prospective axis and confirm on fresh/held-out evidence."
        ),
        "all_scored_discovery": discovery,
    }
    (ROOT / "spectra_loop_state.json").write_text(json.dumps(ledger, indent=2))
    print(json.dumps({"replogle_auditor": audit, "outputs": ledger["axes"]}, indent=2))


if __name__ == "__main__":
    main()
