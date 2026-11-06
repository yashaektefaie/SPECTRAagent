"""Audit analyses on frozen axis: cross-check, stats, confounds, gate call."""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/per_unit_scores.csv")
off = pd.read_csv("models/zeroshot_k562/official_eval/k562_results.csv")

# ---- 1. cross-check vs official release ----
m = df.merge(off[["perturbation", "pearson_delta"]], left_on="pert", right_on="perturbation", how="inner")
r_xcheck = stats.pearsonr(m.state_pearson_delta.fillna(0), m.pearson_delta.fillna(0))[0]
sp_xcheck = stats.spearmanr(m.state_pearson_delta.fillna(0), m.pearson_delta.fillna(0))[0]
print(f"[cross-check] n={len(m)} pearson(our,official)={r_xcheck:.3f} spearman={sp_xcheck:.3f}")
print(f"[cross-check] mean ours={m.state_pearson_delta.mean():.3f} official={m.pearson_delta.mean():.3f}")

# ---- 2. frozen primary tests ----
S = df.support.values
pd_state = df.state_pearson_delta.values
rho, p_rho = stats.spearmanr(S, pd_state, alternative="less" )  # placeholder; recompute properly below
rho, p_rho = stats.spearmanr(S, pd_state)
print(f"\n[primary] Spearman rho(S, STATE pearson_delta) = {rho:.3f}, two-sided p = {p_rho:.3e}, one-sided p = {p_rho/2:.3e}")

lvl = df.groupby("level", observed=True)["state_pearson_delta"]
print("\n[levels] STATE pearson_delta:")
print(df.groupby("level", observed=True).agg(
    n=("pert", "size"),
    state_mean=("state_pearson_delta", "mean"),
    state_med=("state_pearson_delta", "median"),
    r1_mean=("r1_pearson_delta", "mean"),
    r2_mean=("r2_pearson_delta", "mean"),
    state_mse=("state_mse_delta", "mean"),
    r1_mse=("r1_mse_delta", "mean"),
    r2_mse=("r2_mse_delta", "mean"),
    true_l2=("true_l2", "mean"),
    ncells=("n_k562_cells", "mean"),
    nlines=("n_train_lines", "mean"),
))

rng = np.random.default_rng(0)
def boot_diff(a, b, n=10000):
    a, b = np.asarray(a), np.asarray(b)
    d = np.empty(n)
    for i in range(n):
        d[i] = a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
    return np.percentile(d, [2.5, 50, 97.5])

g = {l: df.loc[df.level == l, "state_pearson_delta"].values for l in ["L0","L1","L2","L3"]}
print("\n[boot] mean(L3)-mean(L0) CI:", boot_diff(g["L3"], g["L0"]))
print("[boot] mean(L2)-mean(L1) CI:", boot_diff(g["L2"], g["L1"]))
print("[boot] mean(L3)-mean(L1) CI:", boot_diff(g["L3"], g["L1"]))
rest = df.loc[df.level != "L0", "state_pearson_delta"].values
print("[boot] mean(L1..L3)-mean(L0) CI:", boot_diff(rest, g["L0"]))

# S>0 only: does support magnitude matter once seen?
dfp = df[df.support > 0]
rho_p, p_p = stats.spearmanr(dfp.support, dfp.state_pearson_delta)
print(f"\n[S>0 only] Spearman rho = {rho_p:.3f}, p = {p_p:.3e} (n={len(dfp)})")

# ---- 3. confound controls ----
df["l2_terc"] = pd.qcut(df.true_l2, 3, labels=["weak", "mid", "strong"])
print("\n[C1 effect strength] mean STATE pearson_delta by level within true_l2 terciles:")
print(df.pivot_table(index="level", columns="l2_terc", values="state_pearson_delta", observed=True))
print("\ncounts:")
print(df.pivot_table(index="level", columns="l2_terc", values="pert", aggfunc="size", observed=True))
print("\ntrue_l2 by level:", df.groupby("level", observed=True).true_l2.mean().round(3).to_dict())

df["nc_terc"] = pd.qcut(df.n_k562_cells, 3, labels=["few", "mid", "many"])
print("\n[C2 cell count] STATE pearson_delta by level within n_cells terciles:")
print(df.pivot_table(index="level", columns="nc_terc", values="state_pearson_delta", observed=True))

# partial spearman: rho(S, perf) controlling true_l2
def partial_spearman(x, y, z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    sx = np.polyfit(rz, rx, 1); sy = np.polyfit(rz, ry, 1)
    ex = rx - np.polyval(sx, rz); ey = ry - np.polyval(sy, rz)
    r = stats.pearsonr(ex, ey)
    return r
r_part = partial_spearman(df.support, df.state_pearson_delta.fillna(0), df.true_l2)
print(f"\n[partial] Spearman(S, perf | true_l2) approx r={r_part[0]:.3f}, p={r_part[1]:.3e}")

# strong-effect subset (top two terciles) level means
sub = df[df.l2_terc != "weak"]
print("\n[strong/mid effects only] level means:")
print(sub.groupby("level", observed=True)[["state_pearson_delta","r1_pearson_delta"]].mean())
rho_s, p_s = stats.spearmanr(sub.support, sub.state_pearson_delta)
print(f"[strong/mid only] Spearman rho={rho_s:.3f} p={p_s:.3e} n={len(sub)}")

# ---- 4. gate call ----
print("\n[GATE] level-wise degradation L3->L0:")
st = df.groupby("level", observed=True)["state_pearson_delta"].mean()
r1 = df.groupby("level", observed=True)["r1_pearson_delta"].mean()
r2 = df.groupby("level", observed=True)["r2_pearson_delta"].mean()
print(f"STATE: {st['L3']:.3f} -> {st['L0']:.3f}  drop={st['L3']-st['L0']:.3f}")
print(f"R1   : {r1['L3']:.3f} -> {r1['L0']:.3f}  drop={r1['L3']-r1['L0']:.3f}")
print(f"R2   : {r2['L3']:.3f} -> {r2['L0']:.3f}  drop={r2['L3']-r2['L0']:.3f}")

# MSE direction
stm = df.groupby("level", observed=True)["state_mse_delta"].mean()
r1m = df.groupby("level", observed=True)["r1_mse_delta"].mean()
print(f"[GATE mse] STATE {stm['L3']:.5f}->{stm['L0']:.5f} (rise={stm['L0']-stm['L3']:.5f}); R1 {r1m['L3']:.5f}->{r1m['L0']:.5f} (rise={r1m['L0']-r1m['L3']:.5f})")

# STATE below trivial reference at L0?
l0 = df[df.level=="L0"]
print(f"\n[L0 detail] STATE mean={l0.state_pearson_delta.mean():.3f} vs R2(global-mean)={l0.r2_pearson_delta.mean():.3f}; "
      f"STATE<R2 in {(l0.state_pearson_delta<l0.r2_pearson_delta).mean()*100:.0f}% of L0 perts")
print(f"[L0 detail] median STATE={l0.state_pearson_delta.median():.3f}; frac STATE pearson<0: {(l0.state_pearson_delta<0).mean()*100:.0f}%")
