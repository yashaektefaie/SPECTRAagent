#!/usr/bin/env python3
"""Build ClinVar missense benchmark joined to AlphaMissense (genome-coordinate join)."""
import subprocess, gzip, sys, csv
from collections import Counter

DATA = "/path/to/audit-home/spectra_alphamissense/data"
OUT  = "/path/to/audit-home/spectra_alphamissense/work"

# review status -> star rating
STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,  # we drop conflicting by CLNSIG anyway
}
PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BEN  = {"Benign", "Likely_benign", "Benign/Likely_benign"}
BASES = set("ACGT")

print("[1] extracting ClinVar fields via bcftools...", flush=True)
cmd = ["bcftools", "query", "-f",
       "%CHROM\t%POS\t%REF\t%ALT\t%INFO/CLNSIG\t%INFO/CLNREVSTAT\t%INFO/MC\t%INFO/GENEINFO\n",
       f"{DATA}/clinvar.vcf.gz"]
p = subprocess.run(cmd, capture_output=True, text=True)
if p.returncode != 0:
    sys.exit("bcftools failed: " + p.stderr[:500])

clin = {}   # (chrom,pos,ref,alt) -> dict
cnt = Counter()
for line in p.stdout.splitlines():
    chrom, pos, ref, alt, clnsig, revstat, mc, gene = line.split("\t")
    cnt["total"] += 1
    if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES:
        continue
    if "missense_variant" not in mc:
        continue
    cnt["missense_snv"] += 1
    if clnsig in PATH:
        label = 1
    elif clnsig in BEN:
        label = 0
    else:
        continue
    stars = STARS.get(revstat, 0)
    if stars < 1:
        continue
    cnt["labeled_1star+"] += 1
    g = gene.split(":")[0] if gene and gene != "." else ""
    clin[("chr"+chrom, pos, ref, alt)] = (label, stars, g, clnsig)

print("  counts:", dict(cnt), flush=True)
print(f"  clinvar keyed variants: {len(clin)}", flush=True)

print("[2] streaming AlphaMissense hg38 and joining...", flush=True)
rows = []
seen = 0
with gzip.open(f"{DATA}/AlphaMissense_hg38.tsv.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        # CHROM POS REF ALT genome uniprot_id transcript_id protein_variant am_path am_class
        key = (f[0], f[1], f[2], f[3])
        hit = clin.get(key)
        if hit is None:
            continue
        seen += 1
        label, stars, gene, clnsig = hit
        rows.append([f[5], f[7], f[8], f[9], f[0], f[1], f[2], f[3],
                     label, stars, gene, clnsig])

print(f"  joined rows: {len(rows)} (of {len(clin)} clinvar keys)", flush=True)

outp = f"{OUT}/clinvar_am_benchmark.tsv"
with open(outp, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["uniprot_id","protein_variant","am_pathogenicity","am_class",
                "chrom","pos","ref","alt","label","stars","gene","clnsig"])
    w.writerows(rows)

# summary
lab = Counter(r[8] for r in rows)
prot = len(set(r[0] for r in rows))
st = Counter(r[9] for r in rows)
print(f"  wrote {outp}")
print(f"  label balance: pathogenic={lab[1]} benign={lab[0]}")
print(f"  unique uniprot proteins: {prot}")
print(f"  star distribution: {dict(st)}")
