#!/usr/bin/env python
"""r2 step 04: build eval table = ClinVar labels + AM scores + prospective features.
Features: position phyloP100way (axis A1), Grantham (reference R1),
gene-level mean CDS phyloP (axis A2 / reference R2).
Output: tables/r2_eval_table.tsv
"""
import re
import pandas as pd
import pyBigWig

GRANTHAM = {
 'S': {'R':110,'L':145,'T':58,'K':74,'N':46,'G':56,'A':99,'V':124,'I':142,'P':74,'C':112,'M':135,'F':155,'W':177,'Y':144,'H':99,'D':92,'E':80,'Q':98},
 'R': {'L':102,'T':71,'K':26,'N':86,'G':125,'A':112,'V':96,'I':97,'P':103,'C':180,'M':91,'F':97,'W':101,'Y':96,'H':29,'D':96,'E':98,'Q':43},
 'L': {'T':92,'K':107,'N':113,'G':115,'A':138,'V':32,'I':5,'P':98,'C':198,'M':15,'F':22,'W':61,'Y':36,'H':99,'D':164,'E':181,'Q':113},
 'T': {'K':78,'N':65,'G':59,'A':58,'V':69,'I':89,'P':38,'C':149,'M':81,'F':103,'W':128,'Y':92,'H':47,'D':85,'E':65,'Q':29},
 'K': {'N':94,'G':127,'A':111,'V':97,'I':102,'P':103,'C':202,'M':95,'F':102,'W':110,'Y':85,'H':24,'D':101,'E':56,'Q':53},
 'N': {'G':80,'A':111,'V':133,'I':149,'P':91,'C':139,'M':142,'F':158,'W':174,'Y':143,'H':68,'D':23,'E':42,'Q':46},
 'G': {'A':60,'V':109,'I':94,'P':42,'C':159,'M':127,'F':153,'W':184,'Y':147,'H':98,'D':94,'E':98,'Q':87},
 'A': {'V':64,'I':44,'P':27,'C':195,'M':84,'F':113,'W':148,'Y':112,'H':86,'D':126,'E':107,'Q':98},
 'V': {'I':10,'P':32,'C':169,'M':21,'F':50,'W':88,'Y':55,'H':84,'D':152,'E':121,'Q':109},
 'I': {'P':95,'C':198,'M':10,'F':21,'W':61,'Y':33,'H':94,'D':168,'E':168,'Q':109},
 'P': {'C':169,'M':87,'F':114,'W':147,'Y':110,'H':77,'D':91,'E':93,'Q':76},
 'C': {'M':33,'F':205,'W':215,'Y':194,'H':174,'D':154,'E':170,'Q':154},
 'M': {'F':28,'W':67,'Y':36,'H':87,'D':160,'E':172,'Q':101},
 'F': {'W':40,'Y':7,'H':100,'D':175,'E':173,'Q':158},
 'W': {'Y':37,'H':115,'D':181,'E':184,'Q':165},
 'Y': {'H':83,'D':160,'E':122,'Q':99},
 'H': {'D':81,'E':68,'Q':24},
 'D': {'E':45,'Q':61},
 'E': {'Q':29},
}
def grantham(a, b):
    if a == b: return 0.0
    try:
        return float(GRANTHAM[a][b] if b in GRANTHAM[a] else GRANTHAM[b][a])
    except KeyError:
        return None

AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
def pv_grantham(pv):
    m = re.match(r"^([A-Z][a-z]{2}|[A-Z])(\d+)([A-Z][a-z]{2}|[A-Z]|Ter)$", str(pv))
    if not m: return None
    a, b = m.group(1), m.group(3)
    a = AA3.get(a, a); b = AA3.get(b, b)
    if len(a)!=1 or len(b)!=1: return None
    return grantham(a, b)

cv = pd.read_csv("tables/r2_clinvar_missense.tsv", sep="\t", dtype={"chrom":str})
cv["chrom"] = cv.chrom.map(lambda c: "chrM" if c=="MT" else "chr"+c)
cv["key"] = cv.chrom + ":" + cv.pos.astype(str) + ":" + cv.ref + ":" + cv.alt
cv = cv.drop_duplicates(subset=["key"])

am = pd.read_csv("tables/r2_am_hits.tsv", sep="\t",
                 names=["chrom","pos","ref","alt","protein_variant","am_pathogenicity"],
                 dtype={"chrom":str})
am["key"] = am.chrom + ":" + am.pos.astype(str) + ":" + am.ref + ":" + am.alt
agg = am.groupby("key").agg(am_mean=("am_pathogenicity","mean"),
                            am_min=("am_pathogenicity","min"),
                            am_max=("am_pathogenicity","max"),
                            am_ntx=("am_pathogenicity","size"),
                            protein_variant=("protein_variant","first")).reset_index()

df = cv.merge(agg, on="key", how="inner")
print(f"clinvar={len(cv)} joined={len(df)}")

df["gene"] = df.geneinfo.astype(str).str.split("|").str[0].str.split(":").str[0]

bw = pyBigWig.open("data/hg38.phyloP100way.bw")
ph = []
for chrom, pos in zip(df.chrom, df.pos):
    try:
        v = bw.values(chrom, pos-1, pos)
        ph.append(v[0] if v else float("nan"))
    except Exception:
        ph.append(float("nan"))
bw.close()
df["phyloP"] = ph

df["grantham"] = df.protein_variant.map(pv_grantham)

gp = pd.read_csv("tables/r2_gene_phyloP.tsv", sep="\t")
df = df.merge(gp[["gene","gene_mean_phyloP","cds_len"]], on="gene", how="left")

df.to_csv("tables/r2_eval_table.tsv", sep="\t", index=False)
print(f"final: {len(df)} variants")
print(f"missing: phyloP={df.phyloP.isna().sum()} grantham={df.grantham.isna().sum()} gene_phyloP={df.gene_mean_phyloP.isna().sum()}")
print(df.groupby("label").size())
print(df.phyloP.describe())
