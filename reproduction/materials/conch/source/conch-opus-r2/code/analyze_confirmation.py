"""Apply the pre-declared decision rules to the frozen confirmation outputs.

Reads confirmation/confirmation_per_tile.csv (written by run_confirmation.py) and
computes:
  Test A  (real degradation): CONCH accuracy drop sigma0 -> sigma_max, paired
          bootstrap 95% CI. Real if CI excludes 0.
  Test B  (gate / not-robustness): difference-in-differences
          DiD = CONCH_drop - baseline_drop, paired bootstrap 95% CI.
          CONCH is "more robust" (=> negative) only if DiD CI lies entirely < 0.
          model-specific if DiD CI entirely > 0; shared if CI spans 0.
  Trend: monotonic decrease of CONCH accuracy across levels (secondary).
  Axis-order validation: mean varlap and proximity-to-reference decrease.
  Confound (stain vs blur): mean brightness / saturation across levels.
  Per-class SPC (descriptive).

Writes confirmation/analysis_results.json and confirmation/spc_table.csv.
"""
import csv, json
import numpy as np
from collections import defaultdict

PT = "confirmation/confirmation_per_tile.csv"
rows = list(csv.DictReader(open(PT)))
sigmas = sorted({float(r["sigma"]) for r in rows})
smin, smax = min(sigmas), max(sigmas)

# index by (sample_id, sigma)
tiles = sorted({r["sample_id"] for r in rows})
idx = {(r["sample_id"], float(r["sigma"])): r for r in rows}


def arr(field, sigma, cast=float):
    return np.array([cast(idx[(t, sigma)][field]) for t in tiles])


def paired_boot(delta_vec, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(delta_vec)
    idxb = rng.integers(0, n, size=(n_boot, n))
    means = delta_vec[idxb].mean(1)
    return float(delta_vec.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# per-level accuracy
levels = []
for s in sigmas:
    cc = arr("conch_correct", s, int)
    bc = arr("baseline_correct", s, int)
    def ci(x):
        rng = np.random.default_rng(1)
        n = len(x); b = rng.integers(0, n, (4000, n))
        m = x[b].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    levels.append({
        "sigma": s,
        "conch_acc": float(cc.mean()), "conch_ci": ci(cc),
        "baseline_acc": float(bc.mean()), "baseline_ci": ci(bc),
        "mean_varlap": float(arr("varlap", s).mean()),
        "mean_conch_prox": float(arr("conch_prox", s).mean()),
        "mean_resnet_prox": float(arr("resnet_prox", s).mean()),
        "mean_brightness": float(arr("brightness", s).mean()),
        "mean_saturation": float(arr("saturation_mean", s).mean()),
    })

# Test A: CONCH drop sigma0 -> smax (paired)
c0 = arr("conch_correct", smin, int); cm = arr("conch_correct", smax, int)
conch_delta = (c0 - cm).astype(float)            # per-tile drop
drop_conch, ci_conch = paired_boot(conch_delta)

# Test B: DiD
b0 = arr("baseline_correct", smin, int); bm = arr("baseline_correct", smax, int)
base_delta = (b0 - bm).astype(float)
did_vec = conch_delta - base_delta
did, ci_did = paired_boot(did_vec)
drop_base = float(base_delta.mean())

# monotonic?
accs = [l["conch_acc"] for l in levels]
mono = all(accs[i] >= accs[i + 1] - 1e-9 for i in range(len(accs) - 1))

# axis order checks
varlaps = [l["mean_varlap"] for l in levels]
prox = [l["mean_resnet_prox"] for l in levels]
varlap_dec = all(varlaps[i] >= varlaps[i + 1] for i in range(len(varlaps) - 1))
prox_dec = all(prox[i] >= prox[i + 1] - 1e-6 for i in range(len(prox) - 1))

# per-class SPC (conch)
per_class = {}
cls_of = {t: idx[(t, smin)]["label_name"] for t in tiles}
for s in sigmas:
    for t in tiles:
        per_class.setdefault(cls_of[t], {}).setdefault(str(s), []).append(int(idx[(t, s)]["conch_correct"]))
per_class_acc = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in per_class.items()}

# verdict logic
real_degradation = ci_conch[0] > 0
conch_more_robust = ci_did[1] < 0          # DiD entirely negative => CONCH drops less
conch_worse = ci_did[0] > 0                # DiD entirely positive => CONCH drops more
if conch_more_robust:
    gate = "NEGATIVE_ROBUST"               # CONCH robust where baseline collapses
    label = "model_robust"
elif conch_worse:
    gate = "PASS_MODEL_SPECIFIC"
    label = "model_specific"
else:
    gate = "PASS_SHARED"
    label = "shared"
scientific = (real_degradation and gate.startswith("PASS"))

out = {
    "n_tiles_per_level": len(tiles), "sigmas": sigmas,
    "levels": levels,
    "test_A_real_degradation": {"conch_drop": drop_conch, "ci95": ci_conch, "real": bool(real_degradation)},
    "test_B_did": {"conch_drop": drop_conch, "baseline_drop": drop_base,
                   "did_conch_minus_baseline": did, "ci95": ci_did,
                   "conch_more_robust_than_baseline": bool(conch_more_robust),
                   "conch_worse_than_baseline": bool(conch_worse)},
    "monotonic_conch": bool(mono),
    "axis_order": {"varlap_decreasing": bool(varlap_dec), "resnet_prox_decreasing": bool(prox_dec),
                   "varlaps": varlaps, "resnet_prox": prox},
    "stain_confound_check": {"brightness_by_level": [l["mean_brightness"] for l in levels],
                             "saturation_by_level": [l["mean_saturation"] for l in levels]},
    "per_class_conch_acc": per_class_acc,
    "gate_verdict": gate, "degradation_label": label,
    "scientific_pass": bool(scientific),
}
json.dump(out, open("confirmation/analysis_results.json", "w"), indent=2)

with open("confirmation/spc_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sigma", "conch_acc", "conch_ci_lo", "conch_ci_hi", "baseline_acc",
                "baseline_ci_lo", "baseline_ci_hi", "mean_varlap", "mean_resnet_prox",
                "mean_brightness", "mean_saturation"])
    for l in levels:
        w.writerow([l["sigma"], l["conch_acc"], l["conch_ci"][0], l["conch_ci"][1],
                    l["baseline_acc"], l["baseline_ci"][0], l["baseline_ci"][1],
                    l["mean_varlap"], l["mean_resnet_prox"], l["mean_brightness"], l["mean_saturation"]])

print(json.dumps({k: out[k] for k in ["test_A_real_degradation", "test_B_did",
      "monotonic_conch", "axis_order", "gate_verdict", "degradation_label", "scientific_pass"]}, indent=2))
print("WROTE confirmation/analysis_results.json and spc_table.csv")
