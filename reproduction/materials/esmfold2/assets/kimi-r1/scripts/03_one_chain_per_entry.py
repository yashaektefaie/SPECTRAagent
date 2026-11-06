#!/usr/bin/env python
"""Select one chain per test entry (longest qualifying chain; deterministic,
blind to outcomes), write fasta for clustering."""
import csv

best = {}
with open("data/test_chains.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        e, c, l, s = row["entry"], row["chain"], int(row["length"]), row["seq"]
        if e not in best or l > best[e][0]:
            best[e] = (l, c, s)

with open("data/test_one_per_entry.fasta", "w") as f:
    for e in sorted(best):
        l, c, s = best[e]
        f.write(">%s_%s\n%s\n" % (e, c, s))
print("entries:", len(best))
