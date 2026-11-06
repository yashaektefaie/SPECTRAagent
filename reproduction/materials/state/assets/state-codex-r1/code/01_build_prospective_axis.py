#!/usr/bin/env python3
"""Build the prospective context-distance axis without reading STATE outcomes."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


HELD_OUT = {"C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"}
MANUAL_DEPMAP_NAMES = {
    "HepG2/C3A": "C3A",
    "Panc 03.27": "PANC0327",
}
MANUAL_DEPMAP_IDS = {"J82": "ACH-000396"}


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def load_torch_keyed_pickle(path: Path) -> dict:
    """Load a dict with torch tensor values without importing torch."""
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch._utils")
    torch_storage = types.ModuleType("torch.storage")
    torch_utils._rebuild_tensor_v2 = lambda *args: None
    torch_storage._load_from_bytes = lambda value: None
    torch._utils = torch_utils
    torch.storage = torch_storage
    sys.modules.update(
        {
            "torch": torch,
            "torch._utils": torch_utils,
            "torch.storage": torch_storage,
        }
    )
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_alias_index(sample_info: pd.DataFrame) -> dict[str, set[str]]:
    fields = ["stripped_cell_line_name", "CCLE_Name", "Alias"]
    index: dict[str, set[str]] = {}
    for _, row in sample_info.iterrows():
        depmap_id = row["DepMap_ID"]
        if pd.isna(depmap_id):
            continue
        values: list[str] = []
        for field in fields:
            value = row.get(field)
            if pd.isna(value):
                continue
            values.extend(re.split(r"[,;/|]", str(value)))
            if field == "CCLE_Name":
                values.append(str(value).split("_")[0])
        for value in values:
            key = normalize_name(value)
            if key:
                index.setdefault(key, set()).add(str(depmap_id))
    return index


def map_cell_lines(cell_lines: list[str], sample_info: pd.DataFrame) -> pd.DataFrame:
    alias_index = build_alias_index(sample_info)
    rows = []
    for cell in cell_lines:
        lookup = MANUAL_DEPMAP_NAMES.get(cell, cell)
        candidates = (
            [MANUAL_DEPMAP_IDS[cell]]
            if cell in MANUAL_DEPMAP_IDS
            else sorted(alias_index.get(normalize_name(lookup), set()))
        )
        status = "exact" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched"
        depmap_id = candidates[0] if len(candidates) == 1 else None
        rows.append(
            {
                "cell_line": cell,
                "held_out": cell in HELD_OUT,
                "lookup_name": lookup,
                "depmap_id": depmap_id,
                "match_status": status,
                "candidate_ids": "|".join(candidates),
            }
        )
    mapping = pd.DataFrame(rows)
    details = sample_info[
        [
            "DepMap_ID",
            "stripped_cell_line_name",
            "CCLE_Name",
            "Sanger_Model_ID",
            "lineage",
            "lineage_subtype",
        ]
    ].drop_duplicates("DepMap_ID")
    return mapping.merge(details, left_on="depmap_id", right_on="DepMap_ID", how="left")


def choose_primary_genes(
    expression: pd.DataFrame, train_ids: list[str], n_genes: int
) -> tuple[pd.DataFrame, list[str]]:
    renamed = {}
    for column in expression.columns:
        match = re.match(r"^(.+?)\s*\(\d+\)$", str(column))
        renamed[column] = match.group(1) if match else str(column)
    expression = expression.rename(columns=renamed)
    expression = expression.loc[:, ~expression.columns.duplicated()].astype(float)
    available_train = [value for value in train_ids if value in expression.index]
    variance = expression.loc[available_train].var(axis=0, ddof=1)
    genes = variance.sort_values(ascending=False).head(n_genes).index.tolist()
    return expression.loc[:, genes], genes


def panel_distance(
    expression: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = expression.loc[train_ids]
    test = expression.loc[test_ids]
    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=1).replace(0, np.nan)
    valid = scales.notna() & np.isfinite(scales)
    train_z = (train.loc[:, valid] - means[valid]) / scales[valid]
    test_z = (test.loc[:, valid] - means[valid]) / scales[valid]
    pairwise = np.sqrt(
        ((test_z.to_numpy()[:, None, :] - train_z.to_numpy()[None, :, :]) ** 2).mean(axis=2)
    )
    pairwise_df = pd.DataFrame(pairwise, index=test_ids, columns=train_ids)
    k_eff = min(k, len(train_ids))
    summary = pd.DataFrame(
        {
            "sample_id": test_ids,
            "distance": np.median(np.sort(pairwise, axis=1)[:, :k_eff], axis=1),
            "nearest_distance": pairwise.min(axis=1),
            "nearest_training_id": pairwise_df.idxmin(axis=1).to_numpy(),
            "n_genes": int(valid.sum()),
            "n_training_contexts": len(train_ids),
            "k_neighbors": k_eff,
        }
    )
    return summary, pairwise_df


def load_cmp_subset(path: Path, sanger_ids: list[str]) -> pd.DataFrame:
    usecols = ["model_id", "Unnamed: 1", *sanger_ids]
    data = pd.read_csv(path, usecols=usecols, low_memory=False)
    data = data.iloc[4:].copy()
    genes = data.pop("Unnamed: 1").astype(str)
    data = data.drop(columns="model_id")
    data.index = genes
    data = data[~data.index.duplicated(keep="first")]
    data = data.apply(pd.to_numeric, errors="coerce").T
    return np.log2(data + 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-map", type=Path, required=True)
    parser.add_argument("--sample-info", type=Path, required=True)
    parser.add_argument("--ccle-expression", type=Path, required=True)
    parser.add_argument("--cmp-expression", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--k-neighbors", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cell_lines = sorted(map(str, load_torch_keyed_pickle(args.cell_map).keys()))
    sample_info = pd.read_csv(args.sample_info, low_memory=False)
    sample_info = sample_info.rename(
        columns={
            "ModelID": "DepMap_ID",
            "StrippedCellLineName": "stripped_cell_line_name",
            "CCLEName": "CCLE_Name",
            "CellLineName": "Alias",
            "SangerModelID": "Sanger_Model_ID",
            "OncotreeLineage": "lineage",
            "OncotreeSubtype": "lineage_subtype",
        }
    )
    mapping = map_cell_lines(cell_lines, sample_info)
    mapping.to_csv(args.output_dir / "cell_line_mapping.csv", index=False)
    unresolved = mapping[mapping["held_out"] & (mapping["match_status"] != "exact")]
    if not unresolved.empty:
        raise RuntimeError(
            "Unresolved Tahoe-to-DepMap mappings:\n" + unresolved.to_string(index=False)
        )

    train_map = mapping[(~mapping["held_out"]) & mapping["depmap_id"].notna()].copy()
    test_map = mapping[mapping["held_out"]].copy()

    # Primary atlas: actual models in Sanger Cell Model Passports RNA-seq.
    sanger_all = mapping["Sanger_Model_ID"].dropna().astype(str).tolist()
    cmp = load_cmp_subset(args.cmp_expression, sanger_all)
    train_sanger = train_map[train_map["Sanger_Model_ID"].isin(cmp.index)]["Sanger_Model_ID"].astype(str).tolist()
    test_sanger = test_map[test_map["Sanger_Model_ID"].isin(cmp.index)]["Sanger_Model_ID"].astype(str).tolist()
    if len(test_sanger) != len(HELD_OUT):
        raise RuntimeError(f"Primary CMP atlas lacks held-out contexts: {sorted(HELD_OUT - set(test_map[test_map['Sanger_Model_ID'].isin(test_sanger)]['cell_line']))}")
    cmp, genes = choose_primary_genes(cmp, train_sanger, args.n_genes)
    primary, primary_pairs = panel_distance(cmp, train_sanger, test_sanger, args.k_neighbors)
    primary = primary.merge(
        test_map[["cell_line", "depmap_id", "Sanger_Model_ID", "lineage", "lineage_subtype"]],
        left_on="sample_id",
        right_on="Sanger_Model_ID",
        how="left",
    ).drop(columns="sample_id")
    primary["rank_near_to_far"] = primary["distance"].rank(method="first").astype(int)
    primary = primary.sort_values("distance")
    primary.to_csv(args.output_dir / "primary_cmp_distances.csv", index=False)
    primary_pairs.to_csv(args.output_dir / "primary_cmp_pairwise_distances.csv")
    (args.output_dir / "primary_gene_set.txt").write_text("\n".join(genes) + "\n")

    # Independent replication: DepMap/CCLE. C3A lacks a CCLE RNA-seq row, so
    # the biologically pre-specified parental HepG2 line is used as its proxy.
    ccle = pd.read_csv(args.ccle_expression, index_col=0, low_memory=False)
    renamed = {}
    for column in ccle.columns:
        match = re.match(r"^(.+?)\s*\(\d+\)$", str(column))
        renamed[column] = match.group(1) if match else str(column)
    ccle = ccle.rename(columns=renamed)
    ccle = ccle.loc[:, ~ccle.columns.duplicated()].astype(float)
    ccle_genes = [gene for gene in genes if gene in ccle.columns]
    train_ccle = [value for value in train_map["depmap_id"].astype(str) if value in ccle.index]
    test_ccle_map = test_map[["cell_line", "depmap_id"]].copy()
    test_ccle_map["proxy_used"] = False
    test_ccle_map.loc[test_ccle_map["cell_line"] == "HepG2/C3A", "depmap_id"] = "ACH-000739"
    test_ccle_map.loc[test_ccle_map["cell_line"] == "HepG2/C3A", "proxy_used"] = True
    test_ccle = test_ccle_map["depmap_id"].tolist()
    missing_ccle_test = sorted(set(test_ccle) - set(ccle.index))
    if missing_ccle_test:
        raise RuntimeError(f"Independent CCLE atlas lacks held-out contexts/proxies: {missing_ccle_test}")
    independent, independent_pairs = panel_distance(
        ccle.loc[:, ccle_genes], train_ccle, test_ccle, args.k_neighbors
    )
    independent = independent.merge(
        test_ccle_map,
        left_on="sample_id",
        right_on="depmap_id",
        how="left",
    ).drop(columns="sample_id")
    independent["rank_near_to_far"] = independent["distance"].rank(method="first").astype(int)
    independent = independent.sort_values("distance")
    independent.to_csv(args.output_dir / "independent_ccle_distances.csv", index=False)
    independent_pairs.to_csv(args.output_dir / "independent_ccle_pairwise_distances.csv")

    merged = primary.merge(
        independent[["cell_line", "distance", "rank_near_to_far"]],
        on="cell_line",
        suffixes=("_cmp", "_ccle"),
    )
    replication = {
        "n_tahoe_contexts": len(mapping),
        "n_training_contexts_cmp": len(train_sanger),
        "n_held_out_contexts_cmp": len(test_sanger),
        "n_training_contexts_ccle": len(train_ccle),
        "n_held_out_contexts_ccle_including_parental_proxy": len(test_ccle),
        "n_primary_genes": len(genes),
        "n_replication_genes": len(ccle_genes),
        "distance_rank_spearman": float(
            merged["distance_cmp"].corr(merged["distance_ccle"], method="spearman")
        ),
        "primary_order": primary["cell_line"].tolist(),
        "independent_order": independent["cell_line"].tolist(),
    }
    (args.output_dir / "axis_replication_summary.json").write_text(
        json.dumps(replication, indent=2) + "\n"
    )
    print(json.dumps(replication, indent=2))
    print(primary[["cell_line", "distance", "rank_near_to_far"]].to_string(index=False))
    print(independent[["cell_line", "distance", "rank_near_to_far"]].to_string(index=False))


if __name__ == "__main__":
    main()
