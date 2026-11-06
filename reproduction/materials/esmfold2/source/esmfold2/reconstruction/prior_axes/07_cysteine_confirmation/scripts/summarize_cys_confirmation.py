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
PANEL = ROOT / "spectra_ready_dataset" / "cys_confirmation_candidate_features.csv"
SAMPLES = ROOT / "target_model_results.cys_confirmation.scaled_samples.csv"
OUT_TARGET = ROOT / "target_model_results.cys_confirmation.target_level.csv"
OUT_CURVES = ROOT / "analysis_outputs" / "cys_confirmation_level_curves.csv"
OUT_EFFECTS = ROOT / "analysis_outputs" / "cys_confirmation_effects.json"
OUT_COMBINED = ROOT / "spectra_ready_dataset" / "combined_evidence_with_cys_confirmation.csv"
OUT_STATE = ROOT / "spectra_loop_state.json"
OUT_LEDGER = ROOT / "axis_ledger.json"
OUT_COMMANDS = ROOT / "commands_run.json"
EXPANDED = ROOT / "spectra_ready_dataset" / "expanded_all_scored_evidence_table.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.load(open(path))
    return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def select_samples(samples: pd.DataFrame, policy: str) -> pd.DataFrame:
    df = samples[samples["status"].fillna("ok").eq("ok")].copy()
    if policy == "best_by_ptm":
        df["_sort"] = pd.to_numeric(df["ptm"], errors="coerce").fillna(-np.inf)
        return df.sort_values(["target_id", "_sort", "sample_index"], ascending=[True, False, True]).groupby("target_id", as_index=False).head(1).drop(columns=["_sort"])
    if policy == "best_by_plddt":
        df["_sort"] = pd.to_numeric(df["plddt_ca_mean"], errors="coerce").fillna(-np.inf)
        return df.sort_values(["target_id", "_sort", "sample_index"], ascending=[True, False, True]).groupby("target_id", as_index=False).head(1).drop(columns=["_sort"])
    if policy == "first_sample":
        return df.sort_values(["target_id", "sample_index"]).groupby("target_id", as_index=False).head(1)
    if policy == "oracle_best_ca_lddt":
        df["_sort"] = pd.to_numeric(df["ca_lddt"], errors="coerce").fillna(-np.inf)
        return df.sort_values(["target_id", "_sort", "sample_index"], ascending=[True, False, True]).groupby("target_id", as_index=False).head(1).drop(columns=["_sort"])
    raise ValueError(policy)


def spearman(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 5 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return {"rho": None, "p": None}
    if stats is not None:
        rho, p = stats.spearmanr(a[mask], b[mask])
        return {"rho": finite_float(rho), "p": finite_float(p)}
    return {"rho": finite_float(a[mask].rank().corr(b[mask].rank())), "p": None}


def bootstrap_diff(high: np.ndarray, low: np.ndarray, seed: int = 88, n_boot: int = 5000) -> dict[str, Any]:
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if len(high) < 8 or len(low) < 20:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_high": int(len(high)), "n_low": int(len(low))}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(float(rng.choice(high, len(high), replace=True).mean() - rng.choice(low, len(low), replace=True).mean()))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
    }


def ols_axis_coef(df: pd.DataFrame, controls: list[str]) -> dict[str, Any]:
    cols = ["cys_fraction", "ca_lddt"] + [c for c in controls if c in df.columns and c != "cys_fraction"]
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(20, len(cols) + 5) or sub["cys_fraction"].nunique() < 3:
        return {"n": int(len(sub)), "coef": None, "se": None, "t": None, "controls": controls}
    x_cols = []
    for col in ["cys_fraction"] + [c for c in controls if c in sub.columns and c != "cys_fraction"]:
        arr = sub[col].astype(float)
        sd = float(np.nanstd(arr, ddof=0))
        if not math.isfinite(sd) or sd == 0:
            continue
        x_cols.append(((arr - float(np.nanmean(arr))) / sd).to_numpy())
    X = np.column_stack([np.ones(len(sub))] + x_cols)
    y = sub["ca_lddt"].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return {"n": int(len(sub)), "coef": finite_float(beta[1]), "se": None, "t": None, "controls": controls}
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    coef = float(beta[1])
    return {"n": int(len(sub)), "coef": coef, "se": se, "t": coef / se if se else None, "controls": controls}


def summarize_policy(panel: pd.DataFrame, samples: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    chosen = select_samples(samples, policy)
    merged = panel.merge(chosen, on="target_id", how="left", suffixes=("", "_perf"))
    finite = merged[pd.to_numeric(merged["ca_lddt"], errors="coerce").notna()].copy()
    level_rows = []
    for level, part in finite.groupby("cys_level"):
        level_rows.append(
            {
                "selection": policy,
                "cys_level": int(level),
                "n": int(len(part)),
                "cys_mean": finite_float(part["cys_fraction"].mean()),
                "seq_len_median": finite_float(part["seq_len"].median()),
                "resolution_median": finite_float(part["resolution"].median()),
                "em_n": int(part["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).sum()),
                "xray_n": int(part["method"].astype(str).str.contains("X-RAY", case=False, regex=True).sum()),
                "ca_lddt_mean": finite_float(part["ca_lddt"].mean()),
                "gdt_ts_mean": finite_float(part["gdt_ts"].mean()),
                "ca_rmsd_median": finite_float(part["ca_rmsd"].median()),
                "baseline_ca_lddt_mean": finite_float(part["baseline_ca_lddt"].mean()),
                "ptm_mean": finite_float(part["ptm"].mean()),
            }
        )
    curve = pd.DataFrame(level_rows).sort_values("cys_level")
    levels = sorted(curve["cys_level"].tolist()) if len(curve) else []
    high = finite[finite["cys_level"].eq(max(levels))] if levels else finite.iloc[0:0]
    lower = finite[~finite["cys_level"].eq(max(levels))] if levels else finite.iloc[0:0]
    low = finite[finite["cys_level"].eq(min(levels))] if levels else finite.iloc[0:0]
    high2 = finite[finite["cys_level"].isin([3, 4])]
    low2 = finite[finite["cys_level"].isin([0, 1])]
    lr = ["seq_len", "resolution"]
    full = [
        "seq_len",
        "resolution",
        "prov_method_em",
        "tax_nonhuman_nonmodel_stress",
        "order_promoting_fraction",
        "calc_seq_abs_charge_per_residue",
        "fraction_charged_residues_fcr",
    ]
    effect = {
        "selection": policy,
        "n_panel": int(len(panel)),
        "n_scored_targets": int(chosen["target_id"].nunique()),
        "n_finite": int(len(finite)),
        "n_levels": int(len(levels)),
        "min_level_n": int(curve["n"].min()) if len(curve) else 0,
        "level_counts": ",".join(str(int(x)) for x in curve["n"]) if len(curve) else "",
        "measured_order_ok": bool(curve["cys_mean"].is_monotonic_increasing) if len(curve) else False,
        "top_level": int(max(levels)) if levels else None,
        "top_level_n": int(len(high)),
        "top_minus_lower_ca_lddt": finite_float(high["ca_lddt"].mean() - lower["ca_lddt"].mean()) if len(high) and len(lower) else None,
        "top_minus_lowest_ca_lddt": finite_float(high["ca_lddt"].mean() - low["ca_lddt"].mean()) if len(high) and len(low) else None,
        "high2_minus_low2_ca_lddt": finite_float(high2["ca_lddt"].mean() - low2["ca_lddt"].mean()) if len(high2) and len(low2) else None,
        "top_minus_lower_gdt_ts": finite_float(high["gdt_ts"].mean() - lower["gdt_ts"].mean()) if len(high) and len(lower) else None,
        "top_minus_lower_ca_rmsd_median": finite_float(high["ca_rmsd"].median() - lower["ca_rmsd"].median()) if len(high) and len(lower) else None,
        "top_minus_lower_baseline_ca_lddt": finite_float(high["baseline_ca_lddt"].mean() - lower["baseline_ca_lddt"].mean()) if len(high) and len(lower) else None,
        "spearman_level_vs_ca_lddt": spearman(finite["cys_level"], finite["ca_lddt"]),
        "spearman_cys_vs_ca_lddt": spearman(finite["cys_fraction"], finite["ca_lddt"]),
        "bootstrap_top_minus_lower_ca_lddt": bootstrap_diff(high["ca_lddt"].to_numpy(dtype=float), lower["ca_lddt"].to_numpy(dtype=float)),
        "adjusted_len_res": ols_axis_coef(finite, lr),
        "adjusted_full_controls": ols_axis_coef(finite, full),
    }
    return merged, curve, effect


def decision(primary: dict[str, Any]) -> dict[str, Any]:
    top = primary.get("top_minus_lower_ca_lddt")
    high2 = primary.get("high2_minus_low2_ca_lddt")
    gdt = primary.get("top_minus_lower_gdt_ts")
    baseline = primary.get("top_minus_lower_baseline_ca_lddt")
    adj = primary.get("adjusted_len_res", {})
    full = primary.get("adjusted_full_controls", {})
    passed = (
        primary.get("measured_order_ok") is True
        and primary.get("n_levels", 0) >= 5
        and primary.get("min_level_n", 0) >= 15
        and top is not None
        and top <= -0.04
        and high2 is not None
        and high2 <= -0.04
        and adj.get("coef") is not None
        and adj.get("coef") < 0
        and full.get("coef") is not None
        and full.get("coef") < 0
        and baseline is not None
        and abs(baseline) <= 0.02
        and gdt is not None
        and gdt <= -0.05
    )
    return {
        "terminal_passed": bool(passed),
        "status": "claim_valid_confirmed" if passed else "failed_or_not_claim_valid",
        "failed_criteria": [] if passed else [
            "requires measured order, min level n>=15, top-vs-lower CA-lDDT<=-0.04, high2-vs-low2<=-0.04, negative adjusted coefficients, small baseline movement, and GDT agreement"
        ],
    }


def update_artifacts(primary: dict[str, Any], effects: list[dict[str, Any]], decision_obj: dict[str, Any]) -> None:
    state = load_json(OUT_STATE, {})
    state.update(
        {
            "created_at_utc": now(),
            "phase": "cys_confirmation_summarized",
            "cys_confirmation": {
                "status": decision_obj["status"],
                "primary_best_by_ptm": primary,
                "selection_sensitivity": effects,
            },
            "artifacts": {
                **state.get("artifacts", {}),
                "cys_confirmation_samples": str(SAMPLES),
                "cys_confirmation_target_level": str(OUT_TARGET),
                "cys_confirmation_effects": str(OUT_EFFECTS),
                "cys_confirmation_level_curves": str(OUT_CURVES),
                "combined_evidence_with_cys_confirmation": str(OUT_COMBINED),
            },
            "closure_gate": {
                "terminal_passed": bool(decision_obj["terminal_passed"]),
                "final_report_written": False,
                "reason": "High-cysteine confirmation passed the predeclared claim gate." if decision_obj["terminal_passed"] else "High-cysteine confirmation failed or did not meet all predeclared claim-valid criteria; all-scored-evidence discovery must continue.",
            },
        }
    )
    write_json(OUT_STATE, state)

    ledger = load_json(OUT_LEDGER, {"entries": []})
    ledger["entries"] = [e for e in ledger.get("entries", []) if e.get("axis_id") != "frozen_high_cys_confirmation_result"]
    ledger["entries"].append(
        {
            "axis_id": "frozen_high_cys_confirmation_result",
            "category": "fresh_confirmation",
            "status": decision_obj["status"],
            "prospective": True,
            "outcome_informed": True,
            "frozen_before_target_scoring": True,
            "feature": "cys_fraction",
            "direction": "high",
            "primary_effect": primary,
            "selection_sensitivity": effects,
            "artifacts": {
                "samples": str(SAMPLES),
                "target_level": str(OUT_TARGET),
                "effects": str(OUT_EFFECTS),
                "curves": str(OUT_CURVES),
            },
        }
    )
    ledger["created_at_utc"] = now()
    write_json(OUT_LEDGER, ledger)

    commands = load_json(OUT_COMMANDS, [])
    commands.extend(
        [
            {
                "command": "/ewsc/yektefai/envs/envs/pgt/bin/python scripts/evaluate_esmfold2_scaling_multisample.py --features spectra_ready_dataset/cys_confirmation_candidate_features.csv --out target_model_results.cys_confirmation.scaled_samples.csv --max-len 500 --num-loops 10 --num-sampling-steps 68 --num-diffusion-samples 5 --seed 0 > evaluate_esmfold2_cys_confirmation.log 2>&1",
                "cwd": str(ROOT),
                "purpose": "Run ESMFold2 target-model scoring on the frozen high-cysteine confirmation panel.",
                "status": "completed",
            },
            {
                "command": "/ewsc/yektefai/envs/envs/pgt/bin/python scripts/summarize_cys_confirmation.py",
                "cwd": str(ROOT),
                "purpose": "Summarize the frozen high-cysteine confirmation SPC and update state/ledger.",
                "status": "completed",
            },
        ]
    )
    write_json(OUT_COMMANDS, commands)


def main() -> None:
    panel = pd.read_csv(PANEL)
    samples = pd.read_csv(SAMPLES)
    policies = ["best_by_ptm", "best_by_plddt", "oracle_best_ca_lddt", "first_sample"]
    curves = []
    effects = []
    target_primary = None
    for policy in policies:
        merged, curve, effect = summarize_policy(panel, samples, policy)
        curves.append(curve)
        effects.append(effect)
        if policy == "best_by_ptm":
            target_primary = merged
    assert target_primary is not None
    target_primary.to_csv(OUT_TARGET, index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT_CURVES, index=False)
    write_json(OUT_EFFECTS, effects)

    expanded = pd.read_csv(EXPANDED)
    cys_rows = target_primary.copy()
    cys_rows["evidence_source"] = "RCSB_cys_confirmation"
    cys_rows["broad_source"] = "RCSB"
    cys_rows["target_result_origin"] = "cys_confirmation_target_level_best_by_ptm"
    cys_rows["selection_policy"] = "best_by_ptm"
    cys_rows["inference_setting"] = "scaled_10loops68steps_5samples_best_by_ptm_cys_confirmation"
    combined = pd.concat([expanded, cys_rows], ignore_index=True, sort=False)
    combined.to_csv(OUT_COMBINED, index=False)

    primary = next(e for e in effects if e["selection"] == "best_by_ptm")
    decision_obj = decision(primary)
    update_artifacts(primary, effects, decision_obj)
    print(json.dumps({"decision": decision_obj, "primary": primary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
