"""Confirmatory statistics + gate call for Axis A (NCT confirmation set)."""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "axisA_confirm")
LEVELS = [1, 2, 3, 4, 6, 8]

pi = pd.read_csv(os.path.join(OUT, "per_image_eval.csv"))
summ = pd.read_csv(os.path.join(OUT, "axisA_summary.csv"))
y = pi["label"].values

def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p

rows = []
for model, col in [("CONCH", "pred_CONCH_s{}"), ("R1-OpenCLIP-B32-LAION", "pred_R1-OpenCLIP-B32-LAION_s{}"),
                   ("R2-DINOv2-logreg", "pred_R2-DINOv2-logreg_s{}")]:
    c1 = (pi[col.format(1)].values == y)
    for s in LEVELS[1:]:
        cs = (pi[col.format(s)].values == y)
        b, c, p = mcnemar(c1, cs)
        rows.append({"model": model, "compare": f"s1_vs_s{s}", "n_discordant_s1_better": b,
                     "n_discordant_sS_better": c, "mcnemar_p": p})
stats = pd.DataFrame(rows)
stats.to_csv(os.path.join(OUT, "confirmatory_stats.csv"), index=False)

piv = summ.pivot_table(index="level_s", columns="model", values="balanced_acc")
drops = {m: float(piv.loc[1, m] - piv.loc[8, m]) for m in piv.columns}
rel = {m: float((piv.loc[1, m] - piv.loc[8, m]) / piv.loc[1, m]) for m in piv.columns}
print("=== Balanced accuracy by level ===")
print(piv.round(4))
print("\n=== Absolute drop s=1 -> s=8 ===", {k: round(v, 4) for k, v in drops.items()})
print("=== Relative drop ===", {k: round(v, 3) for k, v in rel.items()})
print("\nGATE: CONCH drop >= R1 drop:", drops["CONCH"] >= drops["R1-OpenCLIP-B32-LAION"])
print("GATE: CONCH drop >= R2 drop:", drops["CONCH"] >= drops["R2-DINOv2-logreg"])
print("\n", stats.to_string())

pc = summ[summ.model == "CONCH"][["level_s"] + [c for c in summ.columns if c.startswith("recall_")]]
print("\nCONCH per-class recall by level:")
print(pc.round(3).to_string())
with open(os.path.join(OUT, "gate_call.txt"), "w") as f:
    f.write(f"pivot:\n{piv.round(4).to_string()}\n\ndrops: {drops}\nrelative: {rel}\n")
    f.write(f"GATE CONCH>=R1: {drops['CONCH'] >= drops['R1-OpenCLIP-B32-LAION']}\n")
    f.write(f"GATE CONCH>=R2: {drops['CONCH'] >= drops['R2-DINOv2-logreg']}\n")
