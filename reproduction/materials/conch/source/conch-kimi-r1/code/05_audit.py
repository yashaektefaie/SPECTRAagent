"""Audit: stats, CIs, gate call, SPC table, plot. Reads per-image CSVs only."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ART, LEVELS, RES, SEED

rng = np.random.default_rng(SEED)
MPP = {1: 0.5, 2: 1.0, 4: 2.0, 8: 4.0, 16: 8.0}


def boot_ci(x, n=10000):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_model(csv):
    df = pd.read_csv(csv)
    out = {}
    for s in LEVELS:
        d = df[df["level"] == s].sort_values("path")
        out[s] = d["correct"].values.astype(bool)
    assert all((df[df["level"] == s].sort_values("path")["path"].values
                == df[df["level"] == 1].sort_values("path")["path"].values).all() for s in LEVELS)
    return out


def mcnemar_p(b, c):
    # exact binomial
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    conch = per_model(os.path.join(RES, "conch_per_image.csv"))
    ref = per_model(os.path.join(RES, "reference_per_image.csv"))  # fit_native condition
    rows = []
    for s in LEVELS:
        for name, d in [("CONCH", conch), ("RN50kNN_fitNative", ref)]:
            acc = d[s].mean()
            lo, hi = boot_ci(d[s])
            rows.append({"model": name, "s": s, "mpp": MPP[s], "acc": acc,
                         "ci_lo": lo, "ci_hi": hi, "n": len(d[s])})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "spc_table.csv"), index=False)

    # paired comparison s=1 vs s=8 and s=1 vs s=16
    stats = {}
    for name, d in [("CONCH", conch), ("RN50kNN_fitNative", ref)]:
        for s2 in [8, 16]:
            c1, c2 = d[1], d[s2]
            b = int((c1 & ~c2).sum())  # correct@1, wrong@s2
            c = int((~c1 & c2).sum())
            diff = c1.mean() - c2.mean()
            lo, hi = boot_ci(c1.astype(float) - c2.astype(float))
            stats[f"{name}_s1_vs_s{s2}"] = {
                "acc_s1": float(c1.mean()), f"acc_s{s2}": float(c2.mean()),
                "abs_drop": float(diff), "ci95": [lo, hi],
                "rel_drop": float(diff / c1.mean()),
                "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mcnemar_p(b, c)}
        # monotonic trend
        accs = [d[s].mean() for s in LEVELS]
        from scipy.stats import spearmanr
        rho, p = spearmanr([np.log2(s) for s in LEVELS], accs)
        stats[f"{name}_trend"] = {"spearman_rho_vs_log2s": float(rho), "p": float(p)}

    # gate call
    g_conch = stats["CONCH_s1_vs_s8"]["rel_drop"]
    g_ref = stats["RN50kNN_fitNative_s1_vs_s8"]["rel_drop"]
    stats["gate_call"] = {
        "question": "Does CONCH degrade at least as much as the fixed model-independent reference along the frozen axis (same task, same examples)?",
        "conch_rel_drop_s1_s8": g_conch, "reference_rel_drop_s1_s8": g_ref,
        "answer": bool(g_conch >= g_ref),
        "label": "model-specific (CONCH degrades strictly more than the reference; reference also degrades, so a shared component exists, but the dominant failure is CONCH-specific)" if g_conch >= g_ref else "shared"}

    json.dump(stats, open(os.path.join(RES, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, mk in [("CONCH", "o-"), ("RN50kNN_fitNative", "s--")]:
        t = tab[tab["model"] == name]
        ax.errorbar(t["mpp"], t["acc"], yerr=[t["acc"] - t["ci_lo"], t["ci_hi"] - t["acc"]],
                    fmt=mk, label=name, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("effective microns/pixel (downsampling = lower magnification)")
    ax.set_ylabel("zero-shot / reference accuracy (CRC-VAL-HE-7K, 9 classes)")
    ax.set_title("SPC: CONCH vs fixed reference along magnification axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "spc_plot.png"), dpi=150)


if __name__ == "__main__":
    main()
