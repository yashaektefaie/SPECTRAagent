#!/usr/bin/env python3
"""01_filter_clinvar.py — extract labeled missense SNVs from ClinVar VCF (GRCh38).

Filters per docs/AXIS_SPEC.md:
  CLNVC == single_nucleotide_variant; MC contains missense_variant;
  CLNSIG exactly one of Pathogenic / Likely_pathogenic / Benign / Likely_benign;
  CLNREVSTAT >= 1 star (exclude no_assertion_* and no_interpretation_*).
Output: data/processed/clinvar_labeled.tsv
"""
import gzip, csv, sys

IN = "data/raw/clinvar.vcf.gz"
OUT = "data/processed/clinvar_labeled.tsv"

KEEP_SIG = {"Pathogenic": 1, "Likely_pathogenic": 1, "Benign": 0, "Likely_benign": 0}
BAD_REV = ("no_assertion", "no_interpretation")

n_total = n_snv = n_miss = n_sig = n_rev = 0
with gzip.open(IN, "rt") as f, open(OUT, "w", newline="") as o:
    w = csv.writer(o, delimiter="\t")
    w.writerow(["variation_id", "chrom", "pos", "ref", "alt", "clnsig", "label",
                "revstat", "gene"])
    for line in f:
        if line.startswith("#"):
            continue
        n_total += 1
        chrom, pos, vid, ref, alt, qual, filt, info = line.rstrip("\n").split("\t")[:8]
        if len(ref) != 1 or len(alt) != 1:
            continue
        d = dict(kv.split("=", 1) for kv in info.split(";") if "=" in kv)
        if d.get("CLNVC") != "single_nucleotide_variant":
            continue
        n_snv += 1
        if "missense_variant" not in d.get("MC", ""):
            continue
        n_miss += 1
        sig = d.get("CLNSIG", "")
        if sig not in KEEP_SIG:
            continue
        n_sig += 1
        rev = d.get("CLNREVSTAT", "")
        if any(rev.startswith(b) for b in BAD_REV):
            continue
        n_rev += 1
        gene = d.get("GENEINFO", "").split(":")[0].split("|")[0]
        w.writerow([vid, chrom, pos, ref, alt, sig, KEEP_SIG[sig], rev, gene])

print(f"total_rows={n_total} snv={n_snv} missense={n_miss} unambiguous_PLP_BLB={n_sig} ge1star={n_rev}")
