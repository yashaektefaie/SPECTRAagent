#!/usr/bin/env python3
"""Analyze the frozen STATE Tahoe distance axis and fixed identity reference."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.api as sm
from scipy.stats import spearmanr


CONTEXT_FILES = {
    "C32": "C32",
    "HOP62": "HOP62",
    "HepG2/C3A": "HepG2-C3A",
    "Hs 766T": "Hs 766T",
    "PANC-1": "PANC-1",
}


def parse_target(value: str) -> tuple[str, float, str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 1 or len(parsed[0]) != 3:
        raise ValueError(f"Unexpected Tahoe target: {value}")
    name, dose, unit = parsed[0]
    return str(name), float(dose), str(unit)


def normalize_compound(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def partition_for(compound: str) -> str:
    bucket = int(hashlib.sha256(compound.encode("utf-8")).hexdigest()[:16], 16) % 10
    return "discovery" if bucket <= 2 else "confirmation"


def extract_cell_counts(path: Path, output: Path) -> pd.DataFrame:
    if output.exists():
        return pd.read_csv(output)
    dataset = ds.dataset(path, format="parquet")
    held_out = list(CONTEXT_FILES)
    scanner = dataset.scanner(
        columns=["cell_name", "drugname_drugconc", "pass_filter"],
        filter=ds.field("cell_name").isin(held_out),
        batch_size=500_000,
    )
    all_counts: Counter[tuple[str, str]] = Counter()
    full_counts: Counter[tuple[str, str]] = Counter()
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        keys = list(zip(frame["cell_name"], frame["drugname_drugconc"]))
        all_counts.update(keys)
        full = frame[frame["pass_filter"].astype(str).str.lower() == "full"]
        full_counts.update(zip(full["cell_name"], full["drugname_drugconc"]))
    rows = [
        {
            "cell_line": key[0],
            "target": key[1],
            "n_cells_all": value,
            "n_cells_full": full_counts.get(key, 0),
        }
        for key, value in all_counts.items()
    ]
    result = pd.DataFrame(rows).sort_values(["cell_line", "target"])
    result.to_csv(output, index=False)
    return result


def compute_lfc_scores(real_path: Path, pred_path: Path, chunksize: int = 250_000) -> pd.DataFrame:
    use_real = ["target", "feature", "fold_change", "fdr"]
    use_pred = ["target", "feature", "fold_change"]
    real_iter = pd.read_csv(real_path, usecols=use_real, chunksize=chunksize)
    pred_iter = pd.read_csv(pred_path, usecols=use_pred, chunksize=chunksize)
    accum: dict[str, list[float]] = {}
    total_rows = 0
    for real, pred in zip(real_iter, pred_iter, strict=True):
        if len(real) != len(pred):
            raise RuntimeError("real/pred DE chunk length mismatch")
        if not np.array_equal(real["target"].to_numpy(), pred["target"].to_numpy()):
            raise RuntimeError("real/pred DE target order mismatch")
        if not np.array_equal(real["feature"].to_numpy(), pred["feature"].to_numpy()):
            raise RuntimeError("real/pred DE feature order mismatch")
        real_lfc = np.log2(real["fold_change"].to_numpy(dtype=float))
        pred_lfc = np.log2(pred["fold_change"].to_numpy(dtype=float))
        if not (np.isfinite(real_lfc).all() and np.isfinite(pred_lfc).all()):
            raise RuntimeError("non-finite released fold change encountered")
        temp = pd.DataFrame(
            {
                "target": real["target"].to_numpy(),
                "state_sq": (pred_lfc - real_lfc) ** 2,
                "identity_sq": real_lfc**2,
                "is_de": real["fdr"].to_numpy(dtype=float) < 0.05,
                "n": 1,
            }
        )
        grouped = temp.groupby("target", sort=False).agg(
            state_sse=("state_sq", "sum"),
            identity_sse=("identity_sq", "sum"),
            n_de=("is_de", "sum"),
            n_genes=("n", "sum"),
        )
        for target, row in grouped.iterrows():
            current = accum.setdefault(target, [0.0, 0.0, 0.0, 0.0])
            current[0] += float(row["state_sse"])
            current[1] += float(row["identity_sse"])
            current[2] += float(row["n_de"])
            current[3] += float(row["n_genes"])
        total_rows += len(real)
    result = pd.DataFrame(
        [
            {
                "target": target,
                "state_lfc_rmse": math.sqrt(values[0] / values[3]),
                "identity_lfc_rmse": math.sqrt(values[1] / values[3]),
                "n_de": int(values[2]),
                "n_genes_lfc": int(values[3]),
            }
            for target, values in accum.items()
        ]
    )
    if int(result["n_genes_lfc"].sum()) != total_rows:
        raise RuntimeError("DE row accounting mismatch")
    return result


def within_design(frame: pd.DataFrame, outcome: str, controls: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = ["distance_z", *controls]
    work = frame[["target", outcome, *names]].copy()
    for column in names:
        sd = work[column].std(ddof=1)
        if column != "distance_z" and sd > 0:
            work[column] = (work[column] - work[column].mean()) / sd
    centered = work.groupby("target", sort=False)[[outcome, *names]].transform(lambda x: x - x.mean())
    return centered[outcome].to_numpy(float), centered[names].to_numpy(float), names


def fit_within(frame: pd.DataFrame, outcome: str, controls: list[str]) -> dict:
    y, x, names = within_design(frame, outcome, controls)
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": frame["compound"].to_numpy()})
    idx = names.index("distance_z")
    estimate = float(fit.params[idx])
    p_two = float(fit.pvalues[idx])
    p_one = p_two / 2 if estimate >= 0 else 1 - p_two / 2
    return {
        "estimate": estimate,
        "se_cluster": float(fit.bse[idx]),
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "n_rows": len(frame),
        "n_targets": frame["target"].nunique(),
        "n_compounds": frame["compound"].nunique(),
        "controls": controls,
    }


def bootstrap_slope(
    frame: pd.DataFrame,
    outcome: str,
    controls: list[str],
    reps: int,
    seed: int,
) -> np.ndarray:
    y, x, names = within_design(frame, outcome, controls)
    compounds = frame["compound"].to_numpy()
    unique = np.unique(compounds)
    xtx = []
    xty = []
    for compound in unique:
        mask = compounds == compound
        xc = x[mask]
        yc = y[mask]
        xtx.append(xc.T @ xc)
        xty.append(xc.T @ yc)
    xtx = np.stack(xtx)
    xty = np.stack(xty)
    rng = np.random.default_rng(seed)
    output = np.empty(reps)
    idx = names.index("distance_z")
    for i in range(reps):
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        lhs = np.tensordot(weights, xtx, axes=(0, 0))
        rhs = np.tensordot(weights, xty, axes=(0, 0))
        output[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0][idx]
    return output


def context_spc(frame: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    compound_context = frame.groupby(["compound", "cell_line"], as_index=False).agg(
        state_lfc_rmse=("state_lfc_rmse", "mean"),
        identity_lfc_rmse=("identity_lfc_rmse", "mean"),
        distance=("distance", "first"),
        ccle_distance=("ccle_distance", "first"),
        n_conditions=("target", "nunique"),
    )
    compounds = sorted(compound_context["compound"].unique())
    rng = np.random.default_rng(seed)
    rows = []
    for cell, group in compound_context.groupby("cell_line"):
        indexed = group.set_index("compound")
        state_values = indexed.loc[compounds, "state_lfc_rmse"].to_numpy()
        reference_values = indexed.loc[compounds, "identity_lfc_rmse"].to_numpy()
        state_boot = np.empty(reps)
        reference_boot = np.empty(reps)
        for i in range(reps):
            sample = rng.integers(0, len(compounds), len(compounds))
            state_boot[i] = state_values[sample].mean()
            reference_boot[i] = reference_values[sample].mean()
        rows.append(
            {
                "cell_line": cell,
                "distance": group["distance"].iloc[0],
                "ccle_distance": group["ccle_distance"].iloc[0],
                "n_compounds": len(compounds),
                "n_conditions": int(group["n_conditions"].sum()),
                "state_rmse_mean": state_values.mean(),
                "state_rmse_ci_low": np.quantile(state_boot, 0.025),
                "state_rmse_ci_high": np.quantile(state_boot, 0.975),
                "identity_rmse_mean": reference_values.mean(),
                "identity_rmse_ci_low": np.quantile(reference_boot, 0.025),
                "identity_rmse_ci_high": np.quantile(reference_boot, 0.975),
            }
        )
    return pd.DataFrame(rows).sort_values("distance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--primary-distances", type=Path, required=True)
    parser.add_argument("--ccle-distances", type=Path, required=True)
    parser.add_argument("--obs-metadata", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    cell_counts = extract_cell_counts(args.obs_metadata, args.results_dir / "cell_counts.csv")
    primary = pd.read_csv(args.primary_distances)[["cell_line", "distance", "rank_near_to_far"]]
    ccle = pd.read_csv(args.ccle_distances)[["cell_line", "distance", "proxy_used"]].rename(
        columns={"distance": "ccle_distance"}
    )
    frames = []
    released_frames = []
    for cell_line, stem in CONTEXT_FILES.items():
        custom = compute_lfc_scores(
            args.scores_dir / f"{stem}_real_de.csv",
            args.scores_dir / f"{stem}_pred_de.csv",
        )
        released = pd.read_csv(args.scores_dir / f"{stem}_results.csv").rename(columns={"perturbation": "target"})
        merged = released.merge(custom, on="target", how="inner", validate="one_to_one")
        if len(merged) != len(released):
            raise RuntimeError(f"Custom/released target mismatch for {cell_line}")
        merged["cell_line"] = cell_line
        frames.append(merged)
        released_frames.append(released.assign(cell_line=cell_line))
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.merge(primary, on="cell_line", validate="many_to_one")
    scores = scores.merge(ccle, on="cell_line", validate="many_to_one")
    scores = scores.merge(cell_counts, on=["cell_line", "target"], how="left", validate="one_to_one")
    parsed = scores["target"].map(parse_target)
    scores["compound_name"] = parsed.map(lambda x: x[0])
    scores["dose"] = parsed.map(lambda x: x[1])
    scores["dose_unit"] = parsed.map(lambda x: x[2])
    scores["compound"] = scores["compound_name"].map(normalize_compound)
    scores["partition"] = scores["compound"].map(partition_for)
    scores["distance_z"] = (scores["distance"] - scores["distance"].mean()) / scores["distance"].std(ddof=1)
    scores["ccle_distance_z"] = (scores["ccle_distance"] - scores["ccle_distance"].mean()) / scores["ccle_distance"].std(ddof=1)
    scores["log_n_de"] = np.log1p(scores["n_de"])
    scores["n_cells"] = scores["n_cells_full"].where(scores["n_cells_full"] > 0, scores["n_cells_all"])
    scores["log_n_cells"] = np.log1p(scores["n_cells"])

    coverage = scores.groupby("target")["cell_line"].nunique()
    common_targets = coverage[coverage == len(CONTEXT_FILES)].index
    common = scores[scores["target"].isin(common_targets)].copy()
    eligibility = common.groupby("compound").agg(
        n_contexts=("cell_line", "nunique"),
        n_doses=("dose", "nunique"),
        n_targets=("target", "nunique"),
        partition=("partition", "first"),
    )
    eligible_compounds = eligibility[
        (eligibility["n_contexts"] == len(CONTEXT_FILES)) & (eligibility["n_doses"] >= 2)
    ].index
    common = common[common["compound"].isin(eligible_compounds)].copy()
    if common["n_cells"].isna().any():
        raise RuntimeError("Missing official per-condition cell count for eligible targets")
    common.to_csv(args.results_dir / "per_condition_scores.csv", index=False)
    pd.concat(released_frames, ignore_index=True).to_csv(
        args.results_dir / "released_metrics_combined.csv", index=False
    )
    eligibility.reset_index().to_csv(args.results_dir / "compound_eligibility.csv", index=False)

    controls = ["identity_lfc_rmse", "log_n_de", "log_n_cells"]
    tests: dict[str, dict] = {}
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    for partition in ["discovery", "confirmation", "all"]:
        subset = common if partition == "all" else common[common["partition"] == partition]
        tests[partition] = {
            "state_raw": fit_within(subset, "state_lfc_rmse", []),
            "state_adjusted": fit_within(subset, "state_lfc_rmse", controls),
            "identity_raw": fit_within(subset, "identity_lfc_rmse", []),
        }
        bootstrap[partition] = {
            "state_raw": bootstrap_slope(subset, "state_lfc_rmse", [], args.bootstrap_reps, args.seed + 1),
            "state_adjusted": bootstrap_slope(subset, "state_lfc_rmse", controls, args.bootstrap_reps, args.seed + 2),
            "identity_raw": bootstrap_slope(subset, "identity_lfc_rmse", [], args.bootstrap_reps, args.seed + 3),
        }
        for key, values in bootstrap[partition].items():
            tests[partition][key]["bootstrap_ci_low"] = float(np.quantile(values, 0.025))
            tests[partition][key]["bootstrap_ci_high"] = float(np.quantile(values, 0.975))

    confirm = common[common["partition"] == "confirmation"].copy()
    # Independent-atlas sensitivity: substitute CCLE distance, retain all other frozen choices.
    confirm_ccle = confirm.copy()
    confirm_ccle["distance_z"] = confirm_ccle["ccle_distance_z"]
    tests["confirmation"]["state_adjusted_ccle_distance"] = fit_within(
        confirm_ccle, "state_lfc_rmse", controls
    )
    ccle_boot = bootstrap_slope(
        confirm_ccle, "state_lfc_rmse", controls, args.bootstrap_reps, args.seed + 4
    )
    tests["confirmation"]["state_adjusted_ccle_distance"].update(
        bootstrap_ci_low=float(np.quantile(ccle_boot, 0.025)),
        bootstrap_ci_high=float(np.quantile(ccle_boot, 0.975)),
    )

    diff_boot = bootstrap["confirmation"]["state_raw"] - bootstrap["confirmation"]["identity_raw"]
    diff_ci = np.quantile(diff_boot, [0.025, 0.975])
    if diff_ci[0] > 0:
        gate = "STATE degradation larger than identity reference"
    elif diff_ci[1] < 0:
        gate = "STATE degradation smaller than identity reference"
    else:
        gate = "STATE and identity-reference degradations comparable"
    reference = pd.DataFrame(
        [
            {
                "partition": "confirmation",
                "state_raw_slope": tests["confirmation"]["state_raw"]["estimate"],
                "identity_raw_slope": tests["confirmation"]["identity_raw"]["estimate"],
                "state_minus_identity_slope": tests["confirmation"]["state_raw"]["estimate"]
                - tests["confirmation"]["identity_raw"]["estimate"],
                "difference_ci_low": diff_ci[0],
                "difference_ci_high": diff_ci[1],
                "gate_call": gate,
            }
        ]
    )
    reference.to_csv(args.results_dir / "reference_comparison.csv", index=False)

    spc = context_spc(confirm, args.bootstrap_reps, args.seed + 5)
    state_rho = spearmanr(spc["distance"], spc["state_rmse_mean"]).statistic
    identity_rho = spearmanr(spc["distance"], spc["identity_rmse_mean"]).statistic
    spc["state_spearman_rho"] = state_rho
    spc["identity_spearman_rho"] = identity_rho
    spc.to_csv(args.results_dir / "spc_table.csv", index=False)

    tests["frozen_counts"] = {
        "released_rows": len(scores),
        "common_eligible_rows": len(common),
        "common_eligible_targets": common["target"].nunique(),
        "eligible_compounds_total": common["compound"].nunique(),
        "eligible_compounds_discovery": common.loc[common["partition"] == "discovery", "compound"].nunique(),
        "eligible_compounds_confirmation": confirm["compound"].nunique(),
        "confirmation_contexts": confirm["cell_line"].nunique(),
        "confirmation_state_context_rho": float(state_rho),
        "confirmation_identity_context_rho": float(identity_rho),
        "cell_count_min": int(common["n_cells"].min()),
        "cell_count_median": float(common["n_cells"].median()),
        "cell_count_max": int(common["n_cells"].max()),
    }
    tests["reference_gate"] = {
        "call": gate,
        "difference_ci_low": float(diff_ci[0]),
        "difference_ci_high": float(diff_ci[1]),
    }
    (args.results_dir / "model_tests.json").write_text(json.dumps(tests, indent=2) + "\n")

    # Confound summaries and matching/residualized sensitivity.
    confounds = common.groupby(["partition", "cell_line"], as_index=False).agg(
        distance=("distance", "first"),
        state_rmse=("state_lfc_rmse", "mean"),
        identity_rmse=("identity_lfc_rmse", "mean"),
        n_de=("n_de", "mean"),
        n_cells=("n_cells", "mean"),
        n_conditions=("target", "nunique"),
    )
    confounds.to_csv(args.results_dir / "confound_controls.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for metric, label, color in [
        ("state", "STATE", "#8B1E3F"),
        ("identity", "Identity reference", "#367588"),
    ]:
        ax.errorbar(
            spc["distance"],
            spc[f"{metric}_rmse_mean"],
            yerr=[
                spc[f"{metric}_rmse_mean"] - spc[f"{metric}_rmse_ci_low"],
                spc[f"{metric}_rmse_ci_high"] - spc[f"{metric}_rmse_mean"],
            ],
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
    for _, row in spc.iterrows():
        ax.annotate(row["cell_line"], (row["distance"], row["state_rmse_mean"]), xytext=(3, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Basal transcriptomic distance from training panel")
    ax.set_ylabel("LFC RMSE (lower is better)")
    ax.set_title("STATE spectral performance curve — confirmation compounds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.plots_dir / "spc_primary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.scatter(primary["distance"], ccle["ccle_distance"], s=55, color="#5B5F97")
    for cell in primary["cell_line"]:
        x = primary.loc[primary["cell_line"] == cell, "distance"].iloc[0]
        y = ccle.loc[ccle["cell_line"] == cell, "ccle_distance"].iloc[0]
        ax.annotate(cell, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Cell Model Passports distance")
    ax.set_ylabel("CCLE distance")
    ax.set_title("Prospective axis replication")
    fig.tight_layout()
    fig.savefig(args.plots_dir / "atlas_replication.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(confirm["identity_lfc_rmse"], confirm["state_lfc_rmse"], s=7, alpha=0.25, c=confirm["distance"], cmap="viridis")
    ax.set_xlabel("Identity-reference RMSE (effect magnitude)")
    ax.set_ylabel("STATE LFC RMSE")
    ax.set_title("Effect-size control — confirmation")
    fig.tight_layout()
    fig.savefig(args.plots_dir / "effect_magnitude_control.png", dpi=180)
    plt.close(fig)

    print(json.dumps(tests["frozen_counts"], indent=2))
    print(json.dumps(tests["confirmation"], indent=2))
    print(reference.to_string(index=False))
    print(spc.to_string(index=False))


if __name__ == "__main__":
    main()
