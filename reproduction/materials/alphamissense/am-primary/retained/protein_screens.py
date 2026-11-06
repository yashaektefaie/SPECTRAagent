"""Reproduce the two rank tests in Supplementary Figure 4 from the archived graphic.
Usage: python protein_screens.py. Requires SciPy. Rank the original ten bins from
highest to lowest support. This verifies the plotted trend; it does not re-estimate
within-bin AUROC or bootstrap intervals from ClinVar variants.
"""
import csv
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
rows=defaultdict(list)
with (Path(__file__).parent/"protein_screens_figure_values.csv").open() as f:
    for r in csv.DictReader(f):rows[r["axis"]].append(r)
for name,rs in rows.items():
    print(name,spearmanr([-int(r["bin_from_high_support"]) for r in rs],
                         [float(r["auroc_from_vector"]) for r in rs]))
