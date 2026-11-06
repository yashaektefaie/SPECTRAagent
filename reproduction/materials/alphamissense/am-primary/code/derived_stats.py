#!/usr/bin/env python3
"""Derive easy->hard drops (absolute + proportional) for AM/PSSM/BLOSUM62 and the DMS dynamic-range
shrink, strictly from axis3_baselines_binstats.tsv and axis3_dms_signal_perbin.tsv, so every number
in the .tex traces to a file. READ-ONLY; writes a derived-stats artifact."""
import csv
STEPB="/path/to/audit-home/spectra_alphamissense/stepB"
B={}
with open(f"{STEPB}/axis3_baselines_binstats.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        if r["wmean_absrho"]=="NA": continue
        B[(r["predictor"],r["bin"])]=float(r["wmean_absrho"])
S={}
with open(f"{STEPB}/axis3_dms_signal_perbin.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        S[r["bin"]]=float(r["mean_relative_spread"])

out=[]
out.append("# Derived stats (from axis3_baselines_binstats.tsv + axis3_dms_signal_perbin.tsv)")
out.append("# easy->hard drops, |rho| to 3dp, percentages to 1dp")
out.append("")
def drop(pred,hi,lo):
    a=B.get((pred,hi)); b=B.get((pred,lo))
    if a is None or b is None: return
    out.append(f"{pred}: {hi}={a:.3f} {lo}={b:.3f}  abs_drop={a-b:.3f}  rel_drop={100*(a-b)/a:.1f}%")
out.append("=== q4 -> q0 (all three predictors defined) ===")
for p in ["AM","PSSM","BLOSUM62"]: drop(p,"q4","q0")
out.append("")
out.append("=== q4 -> insert (PSSM undefined at insert) ===")
for p in ["AM","BLOSUM62"]: drop(p,"q4","insert")
out.append("")
out.append("=== AM advantage (|rho| gap) over baselines per bin ===")
for b in ["q4","q0","insert"]:
    if ("AM",b) in B and ("BLOSUM62",b) in B:
        out.append(f"AM-BLOSUM62 @ {b}: {B[('AM',b)]-B[('BLOSUM62',b)]:.3f}")
for b in ["q4","q0"]:
    if ("AM",b) in B and ("PSSM",b) in B:
        out.append(f"AM-PSSM @ {b}: {B[('AM',b)]-B[('PSSM',b)]:.3f}")
out.append("")
out.append("=== intrinsic-difficulty counterfactual ===")
# if AM declined by BLOSUM's proportional q4->q0 rate, its q0 would be:
bl_rel=(B[("BLOSUM62","q4")]-B[("BLOSUM62","q0")])/B[("BLOSUM62","q4")]
cf=B[("AM","q4")]*(1-bl_rel)
out.append(f"BLOSUM62 proportional q4->q0 decline = {100*bl_rel:.1f}%")
out.append(f"AM q0 if it tracked that decline = {cf:.3f}; ACTUAL AM q0 = {B[('AM','q0')]:.3f} "
           f"(actual {'>' if B[('AM','q0')]>cf else '<='} counterfactual => AM {'sub-intrinsic/more robust' if B[('AM','q0')]>cf else 'excess'})")
out.append("")
out.append("=== DMS dynamic-range shrink (mean relative spread) ===")
out.append(f"q4={S['q4']:.3f} q0={S['q0']:.3f} insert={S['insert']:.3f}")
out.append(f"shrink q4->q0  = {100*(1-S['q0']/S['q4']):.1f}%")
out.append(f"shrink q4->insert = {100*(1-S['insert']/S['q4']):.1f}%")
txt="\n".join(out)+"\n"
open(f"{STEPB}/derived_stats_for_tex.txt","w").write(txt)
print(txt)
