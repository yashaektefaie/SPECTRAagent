#!/usr/bin/env python
"""r2 step 11: mutational-context (CpG) feature from hg38 FASTA.
Levels: CTX0 = not at CpG site; CTX1 = CpG-site transversion/other;
CTX2 = CpG-site transition (C>T or G>A). QC: REF allele must match FASTA base.
Output: tables/r2_eval_table3.tsv
"""
import gzip
import pandas as pd
import numpy as np

df = pd.read_csv("tables/r2_eval_table2.tsv", sep="\t", dtype={"chrom":str})
need = df.groupby("chrom").pos.apply(lambda s: sorted(s.unique())).to_dict()

# stream FASTA, process one chromosome at a time
ctx = {}   # (chrom, pos) -> level
qc_match = qc_total = 0
cur, seq_parts = None, []
def process(chrom, seq):
    global qc_match, qc_total
    if chrom not in need: return
    sub = df[df.chrom == chrom]
    for pos, ref, alt, key in zip(sub.pos, sub.ref, sub.alt, sub.key):
        i = pos - 1
        if i < 0 or i >= len(seq): continue
        base = seq[i]
        qc_total += 1
        if base.upper() != ref.upper():
            continue
        qc_match += 1
        prevb = seq[i-1].upper() if i > 0 else "N"
        nextb = seq[i+1].upper() if i+1 < len(seq) else "N"
        at_cpg = (ref == "C" and nextb == "G") or (ref == "G" and prevb == "C")
        if not at_cpg:
            ctx[key] = 0
        else:
            transition = (ref == "C" and alt == "T") or (ref == "G" and alt == "A")
            ctx[key] = 2 if transition else 1

with gzip.open("data/hg38.fa.gz", "rt") as f:
    for line in f:
        if line.startswith(">"):
            if cur is not None:
                process(cur, "".join(seq_parts))
            cur = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if cur is not None:
        process(cur, "".join(seq_parts))

print(f"QC ref-match: {qc_match}/{qc_total} ({qc_match/max(qc_total,1):.4f})")
df["cpg_ctx"] = df.key.map(ctx)
df.to_csv("tables/r2_eval_table3.tsv", sep="\t", index=False)
print(df.groupby(["cpg_ctx","label"]).size())
print("missing ctx:", df.cpg_ctx.isna().sum())
