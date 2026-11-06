#!/usr/bin/env python
"""r2 step 03: gene-level evolutionary depth feature.
For each protein_coding gene in GENCODE v45 basic, take the longest-CDS transcript,
compute mean phyloP100way over its CDS exons (mean over defined values) and CDS coverage.
Output: tables/r2_gene_phyloP.tsv
"""
import gzip, re
import pyBigWig
import pandas as pd

# parse GTF: collect CDS exons per transcript, keep protein_coding genes
cds = {}   # tx -> list of (chrom, start, end)  (0-based half-open)
tx2gene = {}
tx_gene_name = {}
with gzip.open("data/gencode.v45.basic.annotation.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        if c[2] != "CDS":
            continue
        attrs = c[8]
        if 'gene_type "protein_coding"' not in attrs and 'gene_biotype "protein_coding"' not in attrs:
            continue
        mtx = re.search(r'transcript_id "([^"]+)"', attrs)
        mgn = re.search(r'gene_name "([^"]+)"', attrs)
        if not mtx or not mgn:
            continue
        tx = mtx.group(1).split(".")[0]
        cds.setdefault(tx, []).append((c[0], int(c[3])-1, int(c[4])))
        tx_gene_name[tx] = mgn.group(1)

# longest-CDS transcript per gene
best = {}
for tx, exons in cds.items():
    L = sum(e-s for _, s, e in exons)
    g = tx_gene_name[tx]
    if g not in best or L > best[g][1]:
        best[g] = (tx, L)
print(f"genes with CDS: {len(best)}")

bw = pyBigWig.open("data/hg38.phyloP100way.bw")
rows = []
for g, (tx, L) in best.items():
    tot_w = 0.0; tot_cov = 0; tot_len = 0
    for chrom, s, e in cds[tx]:
        n = e - s
        tot_len += n
        try:
            m = bw.stats(chrom, s, e, type="mean", exact=True)
            cov = bw.stats(chrom, s, e, type="coverage", exact=True)
        except RuntimeError:
            continue
        if m and m[0] is not None and cov and cov[0] is not None:
            w = cov[0]*n
            tot_w += m[0]*w
            tot_cov += w
    if tot_cov > 0:
        rows.append((g, tx, tot_len, tot_w/tot_cov, tot_cov/tot_len))
bw.close()

df = pd.DataFrame(rows, columns=["gene","tx","cds_len","gene_mean_phyloP","cds_phylop_cov"])
df.to_csv("tables/r2_gene_phyloP.tsv", sep="\t", index=False)
print(df.describe().to_string())
