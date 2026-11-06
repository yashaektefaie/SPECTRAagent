#!/usr/bin/env python
"""r2 step 01: parse ClinVar VCF -> clean binary-labeled missense SNVs.

Labels: CLNSIG Pathogenic/Likely_pathogenic -> 1; Benign/Likely_benign -> 0.
Excludes: conflicts, VUS, non-SNV, non-missense (MC must contain missense_variant),
and any compound/ambiguous significance strings.
Output: tables/r2_clinvar_missense.tsv
"""
import gzip, re, csv, sys

IN = "data/clinvar.vcf.gz"
OUT = "tables/r2_clinvar_missense.tsv"

PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BEN = {"Benign", "Likely_benign", "Benign/Likely_benign"}

n_total = n_snv = n_missense = n_labeled = 0
sig_counts = {}

with gzip.open(IN, "rt") as f, open(OUT, "w", newline="") as o:
    w = csv.writer(o, delimiter="\t")
    w.writerow(["chrom","pos","ref","alt","label","clnsig","clnrevstat","geneinfo","mc","rsid"])
    for line in f:
        if line.startswith("#"):
            continue
        n_total += 1
        c = line.rstrip("\n").split("\t")
        chrom, pos, rsid, ref, alt, info = c[0], int(c[1]), c[2], c[3], c[4], c[7]
        if len(ref) != 1 or len(alt) != 1 or "," in alt:
            continue
        n_snv += 1
        d = dict(kv.split("=",1) for kv in info.split(";") if "=" in kv)
        mc = d.get("MC","")
        if "missense_variant" not in mc:
            continue
        n_missense += 1
        raw_sig = d.get("CLNSIG","")
        # take the primary significance before any '|' (e.g. 'Pathogenic|_other')
        sig = raw_sig.split("|")[0]
        sig_counts[sig] = sig_counts.get(sig,0)+1
        if sig in PATH:
            label = 1
        elif sig in BEN:
            label = 0
        else:
            continue
        n_labeled += 1
        w.writerow([chrom, pos, ref, alt, label, sig, d.get("CLNREVSTAT",""),
                    d.get("GENEINFO",""), mc.split("|")[1] if "|" in mc else mc,
                    rsid])

print(f"total_lines={n_total} snv={n_snv} missense={n_missense} clean_labeled={n_labeled}")
print("significance distribution (missense SNVs):")
for k,v in sorted(sig_counts.items(), key=lambda x:-x[1]):
    print(f"  {v:8d}  {k}")
