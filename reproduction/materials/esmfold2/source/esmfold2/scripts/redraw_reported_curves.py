#!/usr/bin/env python3
"""Redraw every packaged SPECTRA curve table without upstream run paths.

The script deliberately consumes the already-published aggregate curve tables.
Per-example/scored inputs and the original analysis scripts remain beside those
tables so individual aggregates can also be independently recomputed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUNDLE_ROOT = Path(__file__).resolve().parents[1]

CURVE_NAME_TOKENS = ("curve", "spc", "spectral")
EXTRA_CURVE_TABLES = {
    "primary_axis_threshold_sweep.csv",
    "ptm_calibration_bins.csv",
    "per_source_high_disulfide_summary.csv",
}
X_CANDIDATES = (
    "level_order",
    "level",
    "stress_rank",
    "axis_level",
    "charge_level",
    "cys_level",
    "external_axis_level",
    "viral_disulfide_score",
    "envelope_fusion_score",
    "surface_lowhelical_score",
    "tail_quantile",
    "ptm_bin",
    "data_family",
)
Y_CANDIDATES = (
    "target_ca_lddt_mean",
    "ca_lddt_mean",
    "ca_lddt",
    "mean",
    "overlap_at_N_mean",
    "de_sig_genes_recall_mean",
    "pearson_delta_mean",
    "similarity_mean",
    "target_gdt_ts_mean",
    "gdt_ts_mean",
    "gdt_ts",
    "target_ca_rmsd_median",
    "ca_rmsd_median",
    "ca_rmsd",
    "baseline_ca_lddt_mean",
    "baseline_ca_lddt",
    "target_minus_baseline_mean",
    "ca_lddt_top_minus_lower",
    "gdt_ts_top_minus_lower",
    "ca_rmsd_top_minus_lower",
    "broad_adjusted_top_tail_coef",
    "failure_rate_ca_lddt_lt_0_6",
)
FACET_CANDIDATES = ("axis", "axis_id", "axis_name", "feature")
GROUP_CANDIDATES = (
    "spectra_subset",
    "subset",
    "representation",
    "selection",
    "level_schema",
    "source",
    "inference_setting",
    "high_disulfide",
)


def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text[:140] or "all"


def curve_tables() -> list[Path]:
    tables = []
    for path in BUNDLE_ROOT.rglob("*.csv"):
        lower = path.name.lower()
        if any(token in lower for token in CURVE_NAME_TOKENS) or path.name in EXTRA_CURVE_TABLES:
            if lower.endswith("_trends.csv") or "bootstrap" in lower:
                continue
            tables.append(path)
    return sorted(set(tables))


def numeric_x(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, list[str] | None]:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float), None
    categories = list(dict.fromkeys(frame[column].astype(str)))
    mapping = {value: idx for idx, value in enumerate(categories)}
    return frame[column].astype(str).map(mapping).to_numpy(dtype=float), categories


def render_table(path: Path, output_root: Path) -> dict:
    relative = path.relative_to(BUNDLE_ROOT)
    frame = pd.read_csv(path, low_memory=False)
    x_col = next((column for column in X_CANDIDATES if column in frame.columns), None)
    y_cols = [column for column in Y_CANDIDATES if column in frame.columns]
    if x_col is None or not y_cols:
        return {"input": relative.as_posix(), "status": "skipped", "reason": "no recognized x/y schema"}

    facet_col = next(
        (column for column in FACET_CANDIDATES if column in frame.columns and 1 < frame[column].nunique(dropna=False) <= 100),
        None,
    )
    facets = [("all", frame)] if facet_col is None else list(frame.groupby(facet_col, dropna=False, sort=False))
    plots: list[str] = []

    for facet_value, facet in facets:
        group_cols = [
            column
            for column in GROUP_CANDIDATES
            if column in facet.columns and 1 < facet[column].nunique(dropna=False) <= 24
        ]
        if group_cols:
            grouped = list(facet.groupby(group_cols, dropna=False, sort=False))
        else:
            grouped = [("all", facet)]

        for y_col in y_cols:
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            drawn = 0
            category_labels: list[str] | None = None
            for group_value, group in grouped:
                work = group[[x_col, y_col]].copy()
                work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
                work = work.dropna(subset=[y_col])
                if work.empty:
                    continue
                x_values, labels = numeric_x(work, x_col)
                order = np.argsort(x_values)
                if labels is not None:
                    category_labels = labels
                if isinstance(group_value, tuple):
                    legend = " | ".join(f"{col}={val}" for col, val in zip(group_cols, group_value))
                elif group_cols:
                    legend = f"{group_cols[0]}={group_value}"
                else:
                    legend = "curve"
                ax.plot(x_values[order], work[y_col].to_numpy()[order], marker="o", linewidth=1.5, label=legend)
                drawn += 1
            if not drawn:
                plt.close(fig)
                continue
            if category_labels is not None:
                ax.set_xticks(range(len(category_labels)))
                ax.set_xticklabels(category_labels, rotation=30, ha="right")
            title = relative.as_posix()
            if facet_col is not None:
                title += f" | {facet_col}={facet_value}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            if drawn > 1:
                ax.legend(frameon=False, fontsize=7)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            out_dir = output_root / relative.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"__{safe_name(facet_value)}" if facet_col is not None else ""
            out_path = out_dir / f"{path.stem}{suffix}__{safe_name(y_col)}.png"
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            plots.append(out_path.relative_to(output_root).as_posix())

    return {
        "input": relative.as_posix(),
        "status": "rendered" if plots else "skipped",
        "rows": int(len(frame)),
        "x": x_col,
        "metrics": y_cols,
        "plots": plots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=BUNDLE_ROOT / "reconstructed_figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = [render_table(path, args.out) for path in curve_tables()]
    manifest = {
        "bundle_root": str(BUNDLE_ROOT),
        "curve_tables": len(records),
        "rendered_tables": sum(record["status"] == "rendered" for record in records),
        "plot_files": sum(len(record.get("plots", [])) for record in records),
        "records": records,
    }
    (args.out / "redraw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("curve_tables", "rendered_tables", "plot_files")}, indent=2))


if __name__ == "__main__":
    main()
