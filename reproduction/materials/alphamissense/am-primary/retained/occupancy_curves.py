"""Print the Figure 5 series from archived published bin estimates.
If the original per-variant joined table has been recovered, the optional CSV
must contain assay, bin, predictor, score, and dms. It recomputes point estimates.
Bootstrap intervals in the released table are archived estimates.
"""
from pathlib import Path
import csv,sys
if len(sys.argv)==1:
    with (Path(__file__).parent/"axis3_baselines_binstats.tsv").open() as f:
        for row in csv.DictReader(f,delimiter="\t"):
            print(row["predictor"],row["bin"],row["wmean_absrho"],row["ci2.5"],row["ci97.5"],sep="\t")
else:
    import pandas as pd
    d=pd.read_csv(sys.argv[1]).dropna(subset=["score","dms"])
    out=[]
    for (predictor,bin_id,assay),v in d.groupby(["predictor","bin","assay"]):
        if len(v)>=20:
            rho=v["score"].corr(v["dms"],method="spearman")
            if pd.notna(rho):out.append((predictor,bin_id,len(v),abs(rho)*len(v)))
    t=pd.DataFrame(out,columns=["predictor","bin","n","weighted"]).groupby(["predictor","bin"]).sum()
    print((t.weighted/t.n).to_csv())
