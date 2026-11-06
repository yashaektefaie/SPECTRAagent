#!/usr/bin/env python
"""SPC figure: ESMFold TM-score and baseline TM vs frozen axis level."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = json.load(open("results/spc_summary.json"))
LEVELS = ["A1", "A2", "A3", "A4"]
labels = ["A1\n>=80%", "A2\n50-80%", "A3\n30-50%", "A4\n<30%"]
x = np.arange(4)

tm = [s["levels"][l]["tm_mean"] for l in LEVELS]
sd = [s["levels"][l]["tm_sd"] for l in LEVELS]
n = [s["levels"][l]["n"] for l in LEVELS]
ci = [1.96 * sd[i] / np.sqrt(n[i]) for i in range(4)]
btm = [s["levels"][l]["baseline_tm_mean"] for l in LEVELS]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.errorbar(x, tm, yerr=ci, marker="o", lw=2, label="ESMFold TM-score (n=80/level)")
bx = [i for i, b in enumerate(btm) if b is not None]
by = [b for b in btm if b is not None]
ax.plot(bx, by, marker="s", ls="--", color="gray",
        label="Naive template baseline (covered only)")
pl = [s["levels"][l]["plddt_mean"] / 100 for l in LEVELS]
ax.plot(x, pl, marker="^", ls=":", color="orange", label="ESMFold mean pLDDT/100")
ax.axhline(0.5, color="red", lw=0.8, alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Frozen Axis A level: max seq identity to PDB <= 2020-05-01 training pool")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)
ax.set_title("SPECTRA SPC: ESMFold on post-cutoff PDB (2023-2024H1, X-ray <= 3A)")
fig.tight_layout()
fig.savefig("results/spc_figure.png", dpi=150)
print("saved results/spc_figure.png")
