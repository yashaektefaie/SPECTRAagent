#!/usr/bin/env python
"""Parse pdb_seqres.txt; build:
  data/train_pool.fasta   sequences of chains from train entries (<=2020-05-01)
  data/test_chains.tsv    candidate test chains (one chosen per entry later)
pdb_seqres headers: >pdbid_chain mol:protein length:NNN ...
"""
import gzip

STD = set("ACDEFGHIKLMNPQRSTVWY")

test_ids = set(x.strip().upper() for x in open("data/test_entry_ids.txt"))
train_ids = set(x.strip().upper() for x in open("data/train_entry_ids.txt"))

n_test = n_train = 0
with open("data/train_pool.fasta", "w") as tr, open("data/test_chains.tsv", "w") as te:
    te.write("entry\tchain\tlength\tseq\n")
    pdbid = chain = None
    seq = []
    def flush():
        global n_test, n_train
        if pdbid is None:
            return
        s = "".join(seq).upper()
        if not s or not set(s) <= STD:
            return
        if pdbid in train_ids:
            tr.write(">%s_%s\n%s\n" % (pdbid, chain, s))
            n_train += 1
        if pdbid in test_ids and 40 <= len(s) <= 400:
            te.write("%s\t%s\t%d\t%s\n" % (pdbid, chain, len(s), s))
            n_test += 1
    for line in open("data/pdb_seqres.txt"):
        if line.startswith(">"):
            flush()
            head = line[1:].split()[0]
            pdbid, chain = head.rsplit("_", 1)
            pdbid = pdbid.upper()
            seq = []
        else:
            seq.append(line.strip())
    flush()

print("train chains:", n_train, "test chains:", n_test)
