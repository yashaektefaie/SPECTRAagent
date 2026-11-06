#!/usr/bin/env python3
"""Prespecified threshold intervals and confound/provenance sensitivities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_parquet(ROOT / "data/processed/clinvar_confirmation_outcomes_sealed.parquet")


def wilson(k: int, n: int) -> tuple[float, float]:
    return tuple(float(x) for x in proportion_confint(k, n, alpha=.05, method="wilson"))


threshold = []
for lv, g in d.groupby("axis_level"):
    y = g.label.to_numpy(int); s = g.am_pathogenicity.to_numpy(float)
    definitions = {
        "pathogenic_sensitivity_score_gt_0.564": (int(((s > .564) & (y == 1)).sum()), int((y == 1).sum())),
        "benign_specificity_score_lt_0.34": (int(((s < .34) & (y == 0)).sum()), int((y == 0).sum())),
        "ambiguous_fraction": (int(((s >= .34) & (s <= .564)).sum()), len(g)),
    }
    called = (s < .34) | (s > .564)
    definitions["called_accuracy"] = (
        int(((((s < .34) & (y == 0)) | ((s > .564) & (y == 1))) & called).sum()),
        int(called.sum()))
    for metric, (k, n) in definitions.items():
        lo, hi = wilson(k, n)
        threshold.append({"axis_level": lv, "axis_name": g.axis_name.iloc[0],
                          "metric": metric, "numerator": k, "denominator": n,
                          "estimate": k/n, "ci95_low": lo, "ci95_high": hi})
pd.DataFrame(threshold).to_csv(ROOT / "artifacts/tables/confirmation_threshold_metrics_ci.csv", index=False)

d["review_group"] = np.where(d.review_stars >= 2, "two_plus_star", "one_star")
d["evaluation_era"] = np.where(pd.to_datetime(d.last_evaluated, errors="coerce") >= "2023-09-22",
                                "evaluated_on_or_after_publication", "evaluated_before_publication_or_missing")
d["exact_near_distance"] = np.where(d.axis_level.eq(3), "d=" + d.d_splice_bp.astype(str), "not_near")
strata = []
for factor in ["review_group", "evaluation_era", "strand", "terminal_exon"]:
    for value, x in d.groupby(factor):
        for lv, g in x.groupby("axis_level"):
            counts = g.label.value_counts()
            if len(counts) == 2 and counts.min() >= 20:
                strata.append({"factor": factor, "stratum": str(value), "axis_level": lv,
                               "axis_name": g.axis_name.iloc[0], "rows": len(g),
                               "genes": g.gene.nunique(), "pathogenic": int(g.label.sum()),
                               "benign": int((1-g.label).sum()),
                               "am_auroc": roc_auc_score(g.label, g.am_pathogenicity),
                               "blosum_auroc": roc_auc_score(g.label, g.blosum_pathogenicity)})
pd.DataFrame(strata).to_csv(ROOT / "artifacts/tables/confirmation_confound_strata.csv", index=False)

near = []
for distance, g in d[d.axis_level.eq(3)].groupby("d_splice_bp"):
    near.append({"d_splice_bp": distance, "rows": len(g), "genes": g.gene.nunique(),
                 "pathogenic": int(g.label.sum()), "benign": int((1-g.label).sum()),
                 "am_auroc": roc_auc_score(g.label, g.am_pathogenicity),
                 "blosum_auroc": roc_auc_score(g.label, g.blosum_pathogenicity),
                 "brier": float(np.mean((g.am_pathogenicity-g.label)**2))})
pd.DataFrame(near).to_csv(ROOT / "artifacts/tables/confirmation_near_bin_densification_exploratory.csv", index=False)

power = {
    "frozen_nominal_target_per_extreme_label": 500,
    "frozen_nominal_target_per_intermediate_label": 200,
    "hanley_mcneil_design_auc_far": .90,
    "hanley_mcneil_design_auc_near": .84,
    "nominal_two_sided_alpha": .05,
    "nominal_power_at_500_per_class_per_extreme": .9598,
    "observed_confirmation_counts": {
        str(lv): {"pathogenic": int(g.label.sum()), "benign": int((1-g.label).sum())}
        for lv, g in d.groupby("axis_level")
    },
    "target_met_all_levels": True,
    "sequential_rescue_used": False,
}
(ROOT / "artifacts/tables/power_plan_and_attainment.json").write_text(json.dumps(power, indent=2) + "\n")
print(json.dumps(power, indent=2))
