"""Rebuild the CONCH discovery and confirmation curves from frozen tile assignments.
Run: python curves.py (no third-party dependencies). All rates are proportions.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import csv,sys
base=Path(__file__).resolve().parent
out=csv.writer(sys.stdout);out.writerow(["cohort","axis","level","n","raw_accuracy","balanced_accuracy","tumor_recall","tumor_n"])
for cohort in ("discovery","confirmation"):
    with (base/(cohort+".csv")).open() as f:
        rows=list(csv.DictReader(f))
    for axis in (k for k in rows[0] if k.startswith("level_nct_")):
        groups=defaultdict(list)
        for row in rows:groups[int(row[axis])].append(row)
        for level,group in sorted(groups.items()):
            classes=defaultdict(list)
            for row in group: classes[row["label_name"]].append(int(row["correct"]))
            tumor=classes.get("TUM",[])
            out.writerow([cohort,axis,level,len(group),mean(int(r["correct"]) for r in group),
                          mean(mean(values) for values in classes.values()),mean(tumor) if tumor else "",len(tumor)])
