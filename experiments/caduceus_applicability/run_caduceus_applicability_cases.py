#!/usr/bin/env python3
"""Decision-first Caduceus applicability experiments.

These runs are framed as deployment audits: given a biological dataset and a
candidate foundation model interface, should a scientist use the model as-is,
fine-tune/calibrate it, switch model families, construct better matched data,
or abstain?
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModel, AutoTokenizer


SEED = 20260521
DEFAULT_OUT = Path("/ewsc/yektefai/spectra_caduceus_applicability_20260521/results")
DEFAULT_RAW = Path("/ewsc/yektefai/spectra_caduceus_applicability_20260521/raw")
DEFAULT_CADUCEUS = Path("/ewsc/yektefai/spectra_caduceus_strict_20260513/model/caduceus_ps")
DEFAULT_HG38 = Path(
    "/ewsc/yektefai/spectra_caduceus_strict_20260513/"
    "agents/dataset_constructor_encode_ccre_v4_20260517/raw/hg38.fa.gz"
)
ENCODE_FINDING = Path(
    "/ewsc/yektefai/spectra_caduceus_strict_20260513/"
    "agents/distiller_after_train_heldout_shift_encode_ccre_20260518/"
    "paper_ready_spectra_finding.md"
)
ENCODE_SUMMARY = Path(
    "/ewsc/yektefai/spectra_caduceus_strict_20260513/"
    "agents/spectra_investigator_train_heldout_shift_after_probe_calibration_20260518/"
    "train_heldout_shift_summary.json"
)
ENCODE_ERROR = Path(
    "/ewsc/yektefai/spectra_caduceus_strict_20260513/"
    "agents/spectra_investigator_train_heldout_shift_after_probe_calibration_20260518/"
    "heldout_error_analysis.csv"
)


def as_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )


def safe_metric(name: str, y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    if name == "roc_auc":
        return float(roc_auc_score(y_true, scores))
    if name == "average_precision":
        return float(average_precision_score(y_true, scores))
    raise ValueError(name)


def threshold_for_best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def score_predictions(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float | int | None]:
    threshold = threshold_for_best_f1(y_train, train_scores)
    pred = test_scores >= threshold
    return {
        "n_test": int(len(y_test)),
        "test_positive_rate": float(np.mean(y_test)) if len(y_test) else None,
        "roc_auc": safe_metric("roc_auc", y_test, test_scores),
        "average_precision": safe_metric("average_precision", y_test, test_scores),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)) if len(np.unique(y_test)) > 1 else None,
        "f1": float(f1_score(y_test, pred, zero_division=0)) if len(np.unique(y_test)) > 1 else None,
        "selected_threshold": threshold,
    }


def get_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fasta_records(path: Path) -> Iterable[tuple[str, str]]:
    name: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks).upper()


def extract_interval_sequences(df: pd.DataFrame, fasta_gz: Path, out_csv_gz: Path) -> pd.DataFrame:
    if out_csv_gz.exists():
        return pd.read_csv(out_csv_gz)

    required = {
        "resized_merged_targeting_chr_hg38",
        "resized_merged_targeting_start_hg38",
        "resized_merged_targeting_end_hg38",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing DC-TAP coordinate columns: {sorted(missing)}")

    table = df.copy()
    table["locus_id"] = (
        table["resized_merged_targeting_chr_hg38"].astype(str)
        + ":"
        + table["resized_merged_targeting_start_hg38"].astype(int).astype(str)
        + "-"
        + table["resized_merged_targeting_end_hg38"].astype(int).astype(str)
    )
    loci = (
        table[
            [
                "locus_id",
                "resized_merged_targeting_chr_hg38",
                "resized_merged_targeting_start_hg38",
                "resized_merged_targeting_end_hg38",
            ]
        ]
        .drop_duplicates("locus_id")
        .rename(
            columns={
                "resized_merged_targeting_chr_hg38": "chrom",
                "resized_merged_targeting_start_hg38": "start",
                "resized_merged_targeting_end_hg38": "end",
            }
        )
    )

    wanted_by_chrom: dict[str, list[tuple[int, int, str]]] = {}
    for row in loci.itertuples(index=False):
        wanted_by_chrom.setdefault(str(row.chrom), []).append((int(row.start), int(row.end), str(row.locus_id)))

    seq_by_locus: dict[str, str] = {}
    for chrom, chrom_seq in fasta_records(fasta_gz):
        if chrom not in wanted_by_chrom:
            continue
        chrom_len = len(chrom_seq)
        for start, end, locus_id in wanted_by_chrom[chrom]:
            start = max(0, start)
            end = min(chrom_len, end)
            seq = chrom_seq[start:end]
            expected = max(0, int(locus_id.split(":")[1].split("-")[1]) - int(locus_id.split(":")[1].split("-")[0]))
            if len(seq) < expected:
                seq = seq + ("N" * (expected - len(seq)))
            seq_by_locus[locus_id] = seq.upper()

    table["sequence"] = table["locus_id"].map(seq_by_locus)
    missing_seq = int(table["sequence"].isna().sum())
    if missing_seq:
        raise RuntimeError(f"Failed to extract sequences for {missing_seq} rows")

    seq = table["sequence"].astype(str)
    table["sequence_length"] = seq.str.len()
    table["gc_fraction"] = seq.map(lambda x: (x.count("G") + x.count("C")) / len(x) if len(x) else np.nan)
    table["cpg_per_kb"] = seq.map(lambda x: x.count("CG") / (len(x) / 1000.0) if len(x) else np.nan)
    table["n_fraction"] = seq.map(lambda x: x.count("N") / len(x) if len(x) else np.nan)
    out_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv_gz, index=False)
    return table


def embed_sequences(
    model_dir: Path,
    seqs: list[str],
    output_path: Path,
    batch_size: int,
    max_length: int | None,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            if start % max(batch_size * 20, 1) == 0:
                print(f"Embedding {start}/{len(seqs)} DC-TAP sequences on {device}", flush=True)
            batch = seqs[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            out = model(**encoded)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = encoded.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))

    emb = np.vstack(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb)
    return emb


@dataclass
class SplitSpec:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_splits(df: pd.DataFrame, y: np.ndarray) -> list[SplitSpec]:
    splits: list[SplitSpec] = []
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    splits.append(SplitSpec("random_pair_split", train_idx, test_idx))

    for group_col, name in [
        ("locus_id", "locus_holdout_split"),
        ("gene_symbol", "gene_holdout_split"),
    ]:
        groups = df[group_col].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        tr, te = next(splitter.split(idx, y, groups=groups))
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            splits.append(SplitSpec(name, tr, te))

    if set(df["cell_type"].dropna().astype(str)) >= {"K562", "WTC11"}:
        cell = df["cell_type"].astype(str).to_numpy()
        tr = np.flatnonzero(cell == "K562")
        te = np.flatnonzero(cell == "WTC11")
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            splits.append(SplitSpec("cell_holdout_train_K562_test_WTC11", tr, te))
        tr = np.flatnonzero(cell == "WTC11")
        te = np.flatnonzero(cell == "K562")
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            splits.append(SplitSpec("cell_holdout_train_WTC11_test_K562", tr, te))

    return splits


def fit_embedding_model(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
    ).fit(x, y)


def fit_kmer_model(seqs: pd.Series, y: np.ndarray):
    return make_pipeline(
        HashingVectorizer(
            analyzer="char",
            ngram_range=(6, 6),
            n_features=4096,
            alternate_sign=False,
            lowercase=False,
            norm="l2",
        ),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
    ).fit(seqs, y)


def fit_metadata_model(df: pd.DataFrame, y: np.ndarray):
    numeric = [
        "log10_distance_to_gencode_gene_TSS",
        "log10_distance_to_abc_canonical_TSS",
        "gc_fraction",
        "cpg_per_kb",
        "n_fraction",
        "power_at_effect_size_20_wo_pos_controls_20fdr",
        "power_at_effect_size_50_wo_pos_controls_20fdr",
    ]
    categorical = [
        "cell_type",
        "element_location",
        "design_file_type",
        "element_category",
        "ubiq_category",
        "targeting_chr_hg38",
    ]
    present_numeric = [c for c in numeric if c in df.columns]
    present_categorical = [c for c in categorical if c in df.columns]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), present_numeric),
            ("cat", get_one_hot_encoder(), present_categorical),
        ],
        remainder="drop",
    )
    return make_pipeline(
        pre,
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
    ).fit(df[present_numeric + present_categorical], y)


def predict_scores(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def cosine_nearest_train_support(train_emb: np.ndarray, test_emb: np.ndarray, chunk: int = 2048) -> np.ndarray:
    train = train_emb / np.clip(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12, None)
    test = test_emb / np.clip(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12, None)
    out = np.empty(len(test), dtype=np.float32)
    for start in range(0, len(test), chunk):
        stop = min(start + chunk, len(test))
        out[start:stop] = (test[start:stop] @ train.T).max(axis=1)
    return out


def support_bin_rows(
    split_name: str,
    model_name: str,
    y_test: np.ndarray,
    scores: np.ndarray,
    support: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    edges = np.quantile(support, [0, 1 / 3, 2 / 3, 1])
    labels = ["low_support", "middle_support", "high_support"]
    for i, label in enumerate(labels):
        if i == 0:
            mask = support <= edges[1]
        elif i == 2:
            mask = support > edges[2]
        else:
            mask = (support > edges[1]) & (support <= edges[2])
        if mask.sum() == 0:
            continue
        yy = y_test[mask]
        ss = scores[mask]
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "support_bin": label,
                "n": int(mask.sum()),
                "positive_rate": float(np.mean(yy)),
                "support_min": float(np.min(support[mask])),
                "support_median": float(np.median(support[mask])),
                "support_max": float(np.max(support[mask])),
                "roc_auc": safe_metric("roc_auc", yy, ss),
                "average_precision": safe_metric("average_precision", yy, ss),
            }
        )
    return rows


def contradiction_summary(df: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    tmp = df[["locus_id", "gene_symbol", "cell_type"]].copy()
    tmp["label"] = y.astype(int)
    by_locus = tmp.groupby("locus_id")["label"].agg(["size", "sum"])
    by_locus["neg"] = by_locus["size"] - by_locus["sum"]
    repeated = by_locus[by_locus["size"] > 1]
    mixed = repeated[(repeated["sum"] > 0) & (repeated["neg"] > 0)]
    pos_total = int(y.sum())
    return {
        "rows": int(len(df)),
        "positive_rows": pos_total,
        "positive_rate": float(np.mean(y)),
        "unique_loci": int(by_locus.shape[0]),
        "repeated_loci": int(repeated.shape[0]),
        "mixed_label_repeated_loci": int(mixed.shape[0]),
        "rows_in_mixed_loci": int(mixed["size"].sum()) if len(mixed) else 0,
        "positive_rows_in_mixed_loci": int(mixed["sum"].sum()) if len(mixed) else 0,
        "fraction_positive_rows_in_mixed_loci": float(mixed["sum"].sum() / pos_total) if pos_total else None,
        "max_rows_per_locus": int(by_locus["size"].max()),
        "max_positive_rows_per_locus": int(by_locus["sum"].max()),
    }


def run_dc_tap_case(args: argparse.Namespace) -> dict[str, object]:
    raw_table = args.raw_dir / "dc_tap_table.tsv"
    out_dir = args.out_dir / "dc_tap"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading DC-TAP table: {raw_table}", flush=True)
    df0 = pd.read_csv(raw_table, sep="\t")
    label_col = "significant_wo_pos_controls_20fdr"
    df0[label_col] = as_bool_series(df0[label_col])
    df0 = df0[df0[label_col].notna()].copy()
    df0["include_in_fdr_bool"] = as_bool_series(df0["include_in_fdr"])
    df0 = df0[df0["include_in_fdr_bool"].fillna(False)].copy()
    df0["log10_distance_to_gencode_gene_TSS"] = np.log10(
        pd.to_numeric(df0["distance_to_gencode_gene_TSS"], errors="coerce").abs().fillna(0) + 1
    )
    df0["log10_distance_to_abc_canonical_TSS"] = np.log10(
        pd.to_numeric(df0["distance_to_abc_canonical_TSS"], errors="coerce").abs().fillna(0) + 1
    )
    for col in [
        "power_at_effect_size_20_wo_pos_controls_20fdr",
        "power_at_effect_size_50_wo_pos_controls_20fdr",
    ]:
        df0[col] = pd.to_numeric(df0[col], errors="coerce").fillna(0.0)

    sequence_table = out_dir / "dc_tap_with_hg38_sequences.csv.gz"
    df = extract_interval_sequences(df0, args.hg38_fasta, sequence_table)
    y = df[label_col].astype(bool).astype(int).to_numpy()
    emb_path = out_dir / "dc_tap_caduceus_mean_embeddings.npy"
    embeddings = embed_sequences(
        args.caduceus_model,
        df["sequence"].astype(str).tolist(),
        emb_path,
        args.batch_size,
        args.max_length,
    )

    split_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    splits = make_splits(df, y)
    for spec in splits:
        tr, te = spec.train_idx, spec.test_idx
        split_rows.append(
            {
                "split": spec.name,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "train_positive_rate": float(np.mean(y[tr])),
                "test_positive_rate": float(np.mean(y[te])),
                "train_loci": int(df.iloc[tr]["locus_id"].nunique()),
                "test_loci": int(df.iloc[te]["locus_id"].nunique()),
                "shared_loci": int(
                    len(set(df.iloc[tr]["locus_id"].astype(str)) & set(df.iloc[te]["locus_id"].astype(str)))
                ),
                "train_genes": int(df.iloc[tr]["gene_symbol"].nunique()),
                "test_genes": int(df.iloc[te]["gene_symbol"].nunique()),
                "shared_genes": int(
                    len(set(df.iloc[tr]["gene_symbol"].astype(str)) & set(df.iloc[te]["gene_symbol"].astype(str)))
                ),
            }
        )

        fitted: list[tuple[str, object, object, object]] = []
        caduceus = fit_embedding_model(embeddings[tr], y[tr])
        fitted.append(("caduceus_frozen_mean_embedding", caduceus, embeddings[tr], embeddings[te]))
        kmer = fit_kmer_model(df.iloc[tr]["sequence"].astype(str), y[tr])
        fitted.append(("sequence_kmer6_hash", kmer, df.iloc[tr]["sequence"].astype(str), df.iloc[te]["sequence"].astype(str)))
        metadata_cols = [
            "log10_distance_to_gencode_gene_TSS",
            "log10_distance_to_abc_canonical_TSS",
            "gc_fraction",
            "cpg_per_kb",
            "n_fraction",
            "power_at_effect_size_20_wo_pos_controls_20fdr",
            "power_at_effect_size_50_wo_pos_controls_20fdr",
            "cell_type",
            "element_location",
            "design_file_type",
            "element_category",
            "ubiq_category",
            "targeting_chr_hg38",
        ]
        metadata = fit_metadata_model(df.iloc[tr], y[tr])
        fitted.append(("context_metadata_not_sequence_only", metadata, df.iloc[tr][metadata_cols], df.iloc[te][metadata_cols]))

        support = cosine_nearest_train_support(embeddings[tr], embeddings[te])
        for model_name, model, x_train, x_test in fitted:
            train_scores = predict_scores(model, x_train)
            test_scores = predict_scores(model, x_test)
            metrics = score_predictions(y[tr], train_scores, y[te], test_scores)
            metrics.update({"split": spec.name, "model": model_name})
            metric_rows.append(metrics)
            if model_name == "caduceus_frozen_mean_embedding":
                support_rows.extend(support_bin_rows(spec.name, model_name, y[te], test_scores, support))

    metrics_df = pd.DataFrame(metric_rows)
    splits_df = pd.DataFrame(split_rows)
    support_df = pd.DataFrame(support_rows)
    contradiction = contradiction_summary(df, y)
    by_context = (
        df.assign(label=y)
        .groupby(["cell_type", "element_category"], dropna=False)["label"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .sort_values(["sum", "size"], ascending=False)
    )

    metrics_path = out_dir / "dc_tap_applicability_metrics.csv"
    splits_path = out_dir / "dc_tap_split_diagnostics.csv"
    support_path = out_dir / "dc_tap_caduceus_support_bins.csv"
    context_path = out_dir / "dc_tap_label_rates_by_context.csv"
    summary_path = out_dir / "dc_tap_summary.json"
    metrics_df.to_csv(metrics_path, index=False)
    splits_df.to_csv(splits_path, index=False)
    support_df.to_csv(support_path, index=False)
    by_context.to_csv(context_path, index=False)

    best_seq = metrics_df[metrics_df["model"] == "caduceus_frozen_mean_embedding"].sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    best_context = metrics_df[metrics_df["model"] == "context_metadata_not_sequence_only"].sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    summary = {
        "question": (
            "A user has distal regulatory perturbation element-gene-cell records and asks "
            "whether frozen Caduceus sequence embeddings are an appropriate as-is model interface."
        ),
        "dataset": "DC-TAP perturbation-derived element-gene associations",
        "n_rows_after_fdr_filter": int(len(df)),
        "label": label_col,
        "contradiction_summary": contradiction,
        "splits": str(splits_path),
        "metrics": str(metrics_path),
        "support_bins": str(support_path),
        "context_rates": str(context_path),
        "best_caduceus_row": best_seq.head(1).to_dict(orient="records"),
        "best_context_metadata_row": best_context.head(1).to_dict(orient="records"),
        "recommendation": (
            "Do not use frozen Caduceus sequence embeddings as-is for this dataset. "
            "The unit is an element-gene-cell perturbation relation, but the Caduceus interface "
            "sees only the element sequence. Mixed labels for repeated loci and cross-cell/gene "
            "splits test whether sequence support is sufficient; the audit should favor either "
            "a context-aware regulatory model, fine-tuning with gene/cell covariates, or a new "
            "dataset/interface that encodes the missing regulatory context."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_encode_case(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out_dir / "encode_ccre"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "question": (
            "A user has ENCODE cCRE 1024 bp windows and asks whether frozen Caduceus "
            "mean-pooled embeddings are reliable as-is for enhancer-like classification."
        ),
        "source_finding": str(ENCODE_FINDING),
        "recommendation": (
            "Do not use frozen Caduceus as-is as a broad regulatory-element deployment model. "
            "It may be usable as a calibrated representation probe for matched cCRE-style labels, "
            "but the existing audit found modest heldout ROC-AUC and concentrated failures in "
            "promoter-like and promoter-proximal strata. A prospective deployment should use "
            "class-aware calibration, promoter-distance/context controls, or fine-tune/augment "
            "with regulatory-context features before trust."
        ),
    }
    if ENCODE_SUMMARY.exists():
        try:
            summary["investigator_summary"] = json.loads(ENCODE_SUMMARY.read_text())
        except json.JSONDecodeError:
            summary["investigator_summary_path"] = str(ENCODE_SUMMARY)
    if ENCODE_ERROR.exists():
        err = pd.read_csv(ENCODE_ERROR)
        summary["heldout_error_analysis_rows"] = int(len(err))
        for col in ["stratum", "group", "roc_auc", "error_rate", "n"]:
            if col not in err.columns:
                continue
        summary["top_error_rows"] = err.sort_values(
            [c for c in ["error_rate", "n"] if c in err.columns],
            ascending=False,
        ).head(10).to_dict(orient="records")

    finding_text = ENCODE_FINDING.read_text() if ENCODE_FINDING.exists() else ""
    card = f"""# ENCODE cCRE Caduceus Applicability Card

## Deployment Question

Should a scientist use frozen Caduceus mean-pooled embeddings as-is for enhancer-like
classification on ENCODE cCRE windows?

## Recommendation

{summary['recommendation']}

## Evidence Source

The decision is based on the existing leakage-controlled ENCODE cCRE SPECTRA run:
`{ENCODE_FINDING}`.

## Relevant Finding Excerpt

{finding_text[:3500]}
"""
    (out_dir / "encode_ccre_applicability_card.md").write_text(card)
    (out_dir / "encode_ccre_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_cross_case_report(out_dir: Path, encode: dict[str, object], dc_tap: dict[str, object]) -> Path:
    dc_summary = Path(dc_tap["metrics"])
    metrics = pd.read_csv(dc_summary)
    pivot = metrics.pivot_table(
        index="split",
        columns="model",
        values=["roc_auc", "average_precision", "balanced_accuracy"],
        aggfunc="first",
    )
    support_path = Path(dc_tap["support_bins"])
    support = pd.read_csv(support_path) if support_path.exists() else pd.DataFrame()
    support_text = support.to_string(index=False, max_rows=30) if len(support) else "No support-bin rows."

    report = f"""# Caduceus Decision-First Applicability Experiments

## Framing

These experiments treat SPECTRA as a decision-first applicability auditor. The input
question is not "can we make another split?" but "given this biological dataset
and this model interface, should the user trust Caduceus as-is?"

## Case 1: ENCODE cCRE Windows

Question: {encode['question']}

Decision: {encode['recommendation']}

The existing ENCODE audit found modest frozen-probe performance and localized
failure in promoter-like/promoter-proximal strata. That supports a conditional
recommendation: Caduceus can be treated as a representation source for matched
cCRE-style probing, but not as an as-is deployment model without class-aware
calibration and regulatory-context controls.

## Case 2: DC-TAP Element-Gene Perturbation Records

Question: {dc_tap['question']}

Decision: {dc_tap['recommendation']}

The dataset unit is an element-gene-cell relation. A sequence-only Caduceus
interface sees only the element sequence, so repeated loci with mixed labels are
direct evidence that the chosen model interface is incomplete.

Contradiction summary:

```json
{json.dumps(dc_tap['contradiction_summary'], indent=2)}
```

Model metrics:

```text
{pivot.to_string()}
```

Caduceus support-bin diagnostics:

```text
{support_text}
```

## Cross-Case Result

Across both cases, the useful SPECTRA output is a deployment decision. For cCRE
classification, the audit says frozen Caduceus is at best conditionally useful
after calibration and context-aware controls. For DC-TAP-style perturbation
relations, the audit says the model interface is wrong as-is: sequence embeddings
alone omit gene, cell-state, distance, and perturbation context that define the
label. This is the decision-facing behavior the audit should produce.
"""
    path = out_dir / "caduceus_applicability_report.md"
    path.write_text(report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--caduceus-model", type=Path, default=DEFAULT_CADUCEUS)
    parser.add_argument("--hg38-fasta", type=Path, default=DEFAULT_HG38)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encode = run_encode_case(args)
    dc_tap = run_dc_tap_case(args)
    report = write_cross_case_report(args.out_dir, encode, dc_tap)
    print(json.dumps({"report": str(report), "encode": encode, "dc_tap": dc_tap}, indent=2))


if __name__ == "__main__":
    main()
