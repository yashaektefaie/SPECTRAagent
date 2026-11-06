"""Render the confirmation SPC figures from confirmation/spc_table.csv and
confirmation/analysis_results.json. Saves PNGs to results/figures/.
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open("confirmation/spc_table.csv")))
res = json.load(open("confirmation/analysis_results.json"))
sig = [float(r["sigma"]) for r in rows]
cacc = [float(r["conch_acc"]) for r in rows]
clo = [float(r["conch_ci_lo"]) for r in rows]
chi = [float(r["conch_ci_hi"]) for r in rows]
bacc = [float(r["baseline_acc"]) for r in rows]
blo = [float(r["baseline_ci_lo"]) for r in rows]
bhi = [float(r["baseline_ci_hi"]) for r in rows]
varlap = [float(r["mean_varlap"]) for r in rows]
prox = [float(r["mean_resnet_prox"]) for r in rows]

C_CONCH = "#D1495B"   # target
C_BASE = "#30638E"    # reference
CHANCE = 1.0 / 9

# Figure 1: SPC — CONCH vs baseline
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(sig, cacc, yerr=[np.array(cacc) - np.array(clo), np.array(chi) - np.array(cacc)],
            marker="o", color=C_CONCH, capsize=4, lw=2, label="CONCH (zero-shot, target)")
ax.errorbar(sig, bacc, yerr=[np.array(bacc) - np.array(blo), np.array(bhi) - np.array(bacc)],
            marker="s", color=C_BASE, capsize=4, lw=2, label="ResNet50+logreg (fixed baseline)")
ax.axhline(CHANCE, ls="--", color="#888", lw=1, label=f"chance ({CHANCE:.2f})")
ax.set_xlabel("Gaussian blur sigma (focus degradation →)")
ax.set_ylabel("9-class top-1 accuracy (balanced)")
ax.set_title("SPECTRA SPC: CONCH focus-blur fragility vs fixed baseline\n(NCT-CRC-HE-100K confirmation, n=%d/level)" % res["n_tiles_per_level"])
ax.set_ylim(0, 1); ax.set_xticks(sig); ax.legend(frameon=False); ax.grid(alpha=0.25)
did = res["test_B_did"]
ax.text(0.02, 0.03, f"CONCH drop={did['conch_drop']:.3f}  baseline drop={did['baseline_drop']:.3f}\n"
        f"DiD={did['did_conch_minus_baseline']:.3f} CI{tuple(round(x,3) for x in did['ci95'])}  "
        f"verdict: {res['gate_verdict']} ({res['degradation_label']})",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
fig.tight_layout(); fig.savefig(OUT / "fig1_spc_conch_vs_baseline.png", dpi=150)

# Figure 2: axis-order validation (sharpness + proximity decrease)
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(sig, varlap, marker="o", color="#2E7D32", lw=2)
ax1.set_xlabel("Gaussian blur sigma"); ax1.set_ylabel("mean variance-of-Laplacian (sharpness)", color="#2E7D32")
ax1.tick_params(axis="y", labelcolor="#2E7D32")
ax2 = ax1.twinx()
ax2.plot(sig, prox, marker="s", color="#6A1B9A", lw=2)
ax2.set_ylabel("mean proximity to sharp reference manifold\n(ResNet, model-independent)", color="#6A1B9A")
ax2.tick_params(axis="y", labelcolor="#6A1B9A")
ax1.set_title("Axis-order validation: measured sharpness and\nreference-manifold proximity both decrease with blur")
ax1.set_xticks(sig); ax1.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "fig2_axis_order_validation.png", dpi=150)

# Figure 3: per-class CONCH SPC
pc = res["per_class_conch_acc"]
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.tab10(np.linspace(0, 1, len(pc)))
for (cls, d), col in zip(sorted(pc.items()), cmap):
    xs = sorted(float(k) for k in d)
    ys = [d[str(x) if str(x) in d else ("%g" % x)] for x in xs]
    ax.plot(xs, ys, marker="o", label=cls, color=col, lw=1.5)
ax.axhline(CHANCE, ls="--", color="#888", lw=1)
ax.set_xlabel("Gaussian blur sigma"); ax.set_ylabel("CONCH per-class accuracy")
ax.set_title("CONCH per-class SPC under focus blur (confirmation)")
ax.set_xticks(sorted(float(k) for k in next(iter(pc.values())))); ax.set_ylim(0, 1.02)
ax.legend(ncol=3, frameon=False, fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "fig3_per_class_spc.png", dpi=150)
print("WROTE figures to", OUT)
