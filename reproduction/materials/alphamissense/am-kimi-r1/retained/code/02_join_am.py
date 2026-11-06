#!/usr/bin/env python3
"""02_join_am.py — join labeled ClinVar variants to official AlphaMissense hg38
precomputed scores by (CHROM,POS,REF,ALT). Streams the 614MB gz file.

Dedup per AXIS_SPEC: unique (uniprot_id, protein_variant), first occurrence kept.
Output: data/processed/am_matched.tsv
"""
import gzip, csv, sys

CLIN = "data/processed/clinvar_labeled.tsv"
AM = "data/raw/AlphaMissense_hg38.tsv.gz"
OUT = "data/processed/am_matched.tsv"

keys = {}
with open(CLIN) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        keys[("chr" + row["chrom"], int(row["pos"]), row["ref"], row["alt"])] = row
print(f"clinvar_keys={len(keys)}")

n_rows = n_hit = n_dup = 0
seen_prot = set()
with gzip.open(AM, "rt") as f, open(OUT, "w", newline="") as o:
    w = csv.writer(o, delimiter="\t")
    w.writerow(["variation_id", "chrom", "pos", "ref", "alt", "clnsig", "label",
                "revstat", "gene", "uniprot_id", "transcript_id", "protein_variant",
                "am_pathogenicity", "am_class"])
    for line in f:
        if line.startswith("#"):
            continue
        n_rows += 1
        p = line.rstrip("\n").split("\t")
        if len(p) < 10:
            continue
        key = (p[0], int(p[1]), p[2], p[3])
        cv = keys.get(key)
        if cv is None:
            continue
        n_hit += 1
        pk = (p[5], p[7])
        if pk in seen_prot:
            n_dup += 1
            continue
        seen_prot.add(pk)
        w.writerow([cv["variation_id"], cv["chrom"], cv["pos"], cv["ref"], cv["alt"],
                    cv["clnsig"], cv["label"], cv["revstat"], cv["gene"],
                    p[5], p[6], p[7], p[8], p[9]])

print(f"am_rows_scanned={n_rows} matched={n_hit} dup_protein_variant_dropped={n_dup}")
