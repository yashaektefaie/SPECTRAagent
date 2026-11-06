#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
FEATURES = ROOT / "spectra_ready_dataset" / "candidate_features.csv"
RESULTS = ROOT / "target_model_results.scaled_samples.csv"
MANIFEST = ROOT / "charge_confirmation_panel_manifest.json"


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def best_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ok = df[df["status"].fillna("ok").eq("ok") & (pd.to_numeric(df["sample_index"], errors="coerce") >= 0)].copy()
    ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok = ok.sort_values(["target_id", col, "sample_index"], ascending=[True, False, True])
    return ok.groupby("target_id", as_index=False).head(1)


def first_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].fillna("ok").eq("ok") & (pd.to_numeric(df["sample_index"], errors="coerce") == 0)].copy()


def spearman(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 4 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return {"rho": None, "p": None}
    if stats is not None:
        rho, p = stats.spearmanr(a[mask], b[mask])
        return {"rho": finite_float(rho), "p": finite_float(p)}
    return {"rho": finite_float(a[mask].rank().corr(b[mask].rank())), "p": None}


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not math.isfinite(float(sd)) or float(sd) == 0:
        return x * 0
    return (x - x.mean()) / sd


def regression_axis_coef(df: pd.DataFrame, y_col: str = "ca_lddt") -> dict[str, Any]:
    cols = [
        y_col,
        "calc_seq_abs_charge_per_residue",
        "seq_len",
        "resolution",
        "uversky_disorder_score",
        "fraction_charged_residues_fcr",
    ]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(18, len(cols) + 6):
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None, "controls": [c for c in cols if c != y_col]}
    y = sub[y_col].to_numpy(float)
    x_cols = [zscore(sub["calc_seq_abs_charge_per_residue"]).to_numpy(float)]
    control_names = []
    for c in ["seq_len", "resolution", "uversky_disorder_score", "fraction_charged_residues_fcr"]:
        if c in sub.columns:
            x_cols.append(zscore(sub[c]).to_numpy(float))
            control_names.append(c)
    X = np.column_stack([np.ones(len(sub))] + x_cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return {"n": int(len(sub)), "coef": finite_float(beta[1]), "se": None, "t": None, "controls": control_names}
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    coef = float(beta[1])
    return {
        "n": int(len(sub)),
        "coef": coef,
        "se": se,
        "t": coef / se if se else None,
        "controls": control_names,
    }


def bootstrap_diff(high: np.ndarray, low: np.ndarray, seed: int = 20260601, reps: int = 5000) -> dict[str, Any]:
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if len(high) < 5 or len(low) < 10:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_high": int(len(high)), "n_low": int(len(low))}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        vals.append(float(rng.choice(high, size=len(high), replace=True).mean() - rng.choice(low, size=len(low), replace=True).mean()))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
    }


def summarize_selection(name: str, selected: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = features.merge(selected, on="target_id", how="left", suffixes=("", "_metric"))
    df = df[df["status"].fillna("").eq("ok") & df["ca_lddt"].notna()].copy()
    df["charge_level"] = pd.to_numeric(df["charge_level"], errors="coerce").astype("Int64")
    curve = (
        df.groupby(["charge_level", "charge_level_name"])
        .agg(
            n=("target_id", "count"),
            entries=("entry_id", "nunique"),
            charge_min=("calc_seq_abs_charge_per_residue", "min"),
            charge_mean=("calc_seq_abs_charge_per_residue", "mean"),
            charge_max=("calc_seq_abs_charge_per_residue", "max"),
            seq_len_median=("seq_len", "median"),
            resolution_median=("resolution", "median"),
            ca_lddt_mean=("ca_lddt", "mean"),
            ca_lddt_median=("ca_lddt", "median"),
            gdt_ts_mean=("gdt_ts", "mean"),
            ca_rmsd_median=("ca_rmsd", "median"),
            baseline_ca_lddt_mean=("baseline_ca_lddt", "mean"),
            ptm_mean=("ptm", "mean"),
            plddt_mean=("plddt_mean", "mean"),
        )
        .reset_index()
        .sort_values("charge_level")
    )
    levels = sorted([int(x) for x in df["charge_level"].dropna().unique()])
    top_level = max(levels) if levels else None
    high = df[df["charge_level"].eq(top_level)] if top_level is not None else pd.DataFrame()
    lower = df[~df["charge_level"].eq(top_level)] if top_level is not None else pd.DataFrame()
    low = df[df["charge_level"].eq(min(levels))] if levels else pd.DataFrame()
    high2 = df[df["charge_level"].isin(levels[-2:])] if len(levels) >= 4 else pd.DataFrame()
    low2 = df[df["charge_level"].isin(levels[:2])] if len(levels) >= 4 else pd.DataFrame()
    effect = {
        "selection": name,
        "n": int(len(df)),
        "n_levels": int(len(levels)),
        "min_level_n": int(curve["n"].min()) if len(curve) else 0,
        "top_level": top_level,
        "top_level_n": int(len(high)),
        "lower_n": int(len(lower)),
        "measured_charge_order_ok": bool(curve["charge_mean"].is_monotonic_increasing) if len(curve) else False,
        "top_minus_lower_ca_lddt": finite_float(high["ca_lddt"].mean() - lower["ca_lddt"].mean()) if len(high) and len(lower) else None,
        "top_minus_lowest_ca_lddt": finite_float(high["ca_lddt"].mean() - low["ca_lddt"].mean()) if len(high) and len(low) else None,
        "high2_minus_low2_ca_lddt": finite_float(high2["ca_lddt"].mean() - low2["ca_lddt"].mean()) if len(high2) and len(low2) else None,
        "top_minus_lower_gdt_ts": finite_float(high["gdt_ts"].mean() - lower["gdt_ts"].mean()) if len(high) and len(lower) else None,
        "top_minus_lower_baseline_ca_lddt": finite_float(high["baseline_ca_lddt"].mean() - lower["baseline_ca_lddt"].mean()) if len(high) and len(lower) else None,
        "top_minus_lower_ca_rmsd_median": finite_float(high["ca_rmsd"].median() - lower["ca_rmsd"].median()) if len(high) and len(lower) else None,
        "spearman_level_vs_ca_lddt": spearman(df["charge_level"], df["ca_lddt"]),
        "spearman_charge_vs_ca_lddt": spearman(df["calc_seq_abs_charge_per_residue"], df["ca_lddt"]),
        "bootstrap_top_minus_lower_ca_lddt": bootstrap_diff(high["ca_lddt"].to_numpy(float), lower["ca_lddt"].to_numpy(float)) if len(high) and len(lower) else {},
        "length_resolution_adjusted": regression_axis_coef(df, "ca_lddt"),
    }
    return curve, effect


def closure_decision(primary: dict[str, Any]) -> dict[str, Any]:
    ok_power = primary.get("n_levels") == 5 and primary.get("min_level_n", 0) >= 15
    order_ok = bool(primary.get("measured_charge_order_ok"))
    effect = primary.get("top_minus_lower_ca_lddt")
    adjusted = primary.get("length_resolution_adjusted") or {}
    ci = primary.get("bootstrap_top_minus_lower_ca_lddt") or {}
    baseline = primary.get("top_minus_lower_baseline_ca_lddt")
    supports = (
        ok_power
        and order_ok
        and effect is not None
        and effect < -0.05
        and adjusted.get("coef") is not None
        and adjusted.get("coef") < 0
        and (ci.get("ci_high") is None or ci.get("ci_high") < 0)
    )
    if supports:
        status = "valid_fresh_confirmation"
        finding = "Fresh heldout RCSB charge-band evaluation confirms the SPECTRA high-charge degradation axis."
    elif effect is not None and effect < -0.05 and adjusted.get("coef") is not None and adjusted.get("coef") < 0:
        status = "weak_fresh_confirmation"
        finding = "Fresh heldout RCSB evaluation supports the high-charge axis, but uncertainty or power gates are not fully satisfied."
    else:
        status = "not_confirmed"
        finding = "Fresh heldout RCSB evaluation does not confirm the high-charge axis under the frozen charge-band panel."
    return {
        "status": status,
        "finding": finding,
        "power_gate_ok": bool(ok_power),
        "order_gate_ok": bool(order_ok),
        "baseline_explanation_note": "baseline moves in same direction" if baseline is not None and baseline < 0 else "baseline does not explain target degradation",
    }


def write_report(manifest: dict[str, Any], primary_curve: pd.DataFrame, primary: dict[str, Any], effects: list[dict[str, Any]], closure: dict[str, Any]) -> None:
    lines = [
        "# Fresh RCSB Charge-Axis Confirmation for ESMFold2",
        "",
        f"Status: `{closure['status']}`",
        "",
        "## Finding",
        "",
        closure["finding"],
        "",
        "The axis was frozen before target-model scoring: `calc_seq_abs_charge_per_residue`, with higher charge as the predicted harder tail. The panel excludes prior exact sequences and target IDs used in the CAMEO/RCSB discovery evidence.",
        "",
        "## Data",
        "",
        f"- Eligible unscored RCSB candidates: {manifest['eligible_unscored_rows']}",
        f"- Final scored panel requested: {manifest['panel_rows']} targets across 5 charge bands",
        f"- Unique PDB entries in panel: {manifest['panel_entries']}",
        "- ESMFold2 setting: `num_loops=10`, `num_sampling_steps=68`, `num_diffusion_samples=5`; target summaries use best-by-pTM unless noted.",
        "",
        "## Best-by-pTM Curve",
        "",
        primary_curve[
            [
                "charge_level",
                "n",
                "entries",
                "charge_mean",
                "seq_len_median",
                "resolution_median",
                "ca_lddt_mean",
                "gdt_ts_mean",
                "ca_rmsd_median",
                "baseline_ca_lddt_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Primary Effect",
        "",
        f"Top charge band minus lower bands CA-lDDT: {primary.get('top_minus_lower_ca_lddt'):.3f}.",
        f"Top charge band minus lowest band CA-lDDT: {primary.get('top_minus_lowest_ca_lddt'):.3f}.",
        f"Top two bands minus lowest two bands CA-lDDT: {primary.get('high2_minus_low2_ca_lddt'):.3f}.",
        f"Top charge band minus lower bands GDT-TS: {primary.get('top_minus_lower_gdt_ts'):.3f}.",
        f"Top charge band minus lower bands baseline CA-lDDT: {primary.get('top_minus_lower_baseline_ca_lddt'):.3f}.",
        f"Spearman charge level vs CA-lDDT: rho={primary['spearman_level_vs_ca_lddt']['rho']:.3f}, p={primary['spearman_level_vs_ca_lddt']['p']:.3g}.",
        f"Bootstrap top-minus-lower CA-lDDT mean={primary['bootstrap_top_minus_lower_ca_lddt']['mean']:.3f}, 95% CI [{primary['bootstrap_top_minus_lower_ca_lddt']['ci_low']:.3f}, {primary['bootstrap_top_minus_lower_ca_lddt']['ci_high']:.3f}].",
        f"Adjusted charge coefficient={primary['length_resolution_adjusted']['coef']:.3f}, t={primary['length_resolution_adjusted']['t']:.2f}; controls={primary['length_resolution_adjusted']['controls']}.",
        "",
        "## Selection Sensitivity",
        "",
        pd.DataFrame(effects)[
            [
                "selection",
                "n",
                "min_level_n",
                "top_minus_lower_ca_lddt",
                "top_minus_lower_gdt_ts",
                "top_minus_lower_baseline_ca_lddt",
                "high2_minus_low2_ca_lddt",
            ]
        ].to_markdown(index=False),
        "",
        "## Artifacts",
        "",
        "- `charge_confirmation_panel_manifest.json`",
        "- `spectra_ready_dataset/eligible_unscored_charge_pool.csv`",
        "- `spectra_ready_dataset/candidate_features.csv`",
        "- `target_model_results.scaled_samples.csv`",
        "- `analysis_outputs/charge_confirmation_level_curves.csv`",
        "- `analysis_outputs/charge_confirmation_effects.json`",
        "- `spectra_loop_state.json`",
    ]
    (ROOT / "final_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    features = pd.read_csv(FEATURES)
    results = pd.read_csv(RESULTS)
    manifest = json.loads(MANIFEST.read_text())
    selections = {
        "best_by_ptm": best_by(results, "ptm"),
        "best_by_plddt": best_by(results, "plddt_mean"),
        "oracle_best_ca_lddt": best_by(results, "ca_lddt"),
        "first_sample": first_sample(results),
    }
    curves = []
    effects = []
    for name, selected in selections.items():
        curve, effect = summarize_selection(name, selected, features)
        curve.insert(0, "selection", name)
        curves.append(curve)
        effects.append(effect)
    curve_df = pd.concat(curves, ignore_index=True, sort=False)
    primary_curve = curve_df[curve_df["selection"].eq("best_by_ptm")].copy()
    primary = next(e for e in effects if e["selection"] == "best_by_ptm")
    closure = closure_decision(primary)

    (ROOT / "analysis_outputs").mkdir(exist_ok=True)
    curve_df.to_csv(ROOT / "analysis_outputs" / "charge_confirmation_level_curves.csv", index=False)
    (ROOT / "analysis_outputs" / "charge_confirmation_effects.json").write_text(json.dumps(effects, indent=2, sort_keys=True) + "\n")
    state = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Fresh heldout confirmation of the SPECTRA ESMFold2 high absolute charge per residue axis.",
        "closure": closure,
        "primary_effect": primary,
        "manifest": manifest,
        "artifacts": {
            "final_report": "final_report.md",
            "candidate_features": "spectra_ready_dataset/candidate_features.csv",
            "target_model_results": "target_model_results.scaled_samples.csv",
            "curves": "analysis_outputs/charge_confirmation_level_curves.csv",
            "effects": "analysis_outputs/charge_confirmation_effects.json",
        },
    }
    (ROOT / "spectra_loop_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_report(manifest, primary_curve, primary, effects, closure)
    print(json.dumps({"closure": closure, "primary_effect": primary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
