#!/usr/bin/env python3
"""Build SPECTRA test pool + pre-cutoff corpus from pdb_seqres.txt.

Outputs:
  data/train_corpus.fasta   protein chains from entries released <= 2022-03-01
  data/test_pool.fasta      protein chains 40-350 aa from test-window entries
  data/test_pool_meta.tsv   entity_id, entry, length
"""
import re

post = set(open("data/post_cutoff_entries.txt").read().split())
test_entries = set(open("data/test_pool_entries_raw.txt").read().split())

corpus = open("data/train_corpus.fasta", "w")
pool = open("data/test_pool.fasta", "w")
meta = open("data/test_pool_meta.tsv", "w")
meta.write("entity\tentry\tlength\n")

n_corpus = n_pool = 0
hdr = None
seq_lines = []

def flush():
    global n_corpus, n_pool
    if hdr is None:
        return
    m = re.match(r">(\w{4})_(\S+)\s+mol:(\S+)\s+length:(\d+)", hdr)
    if not m:
        return
    entry, ent, mol, length = m.group(1).upper(), m.group(2), m.group(3), int(m.group(4))
    if mol != "protein":
        return
    seq = "".join(seq_lines).replace(" ", "")
    if not seq or set(seq) - set("ACDEFGHIKLMNPQRSTVWYX"):
        pass  # keep; filter unusual chars below
    clean = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
    frac_clean = len(clean) / max(len(seq), 1)
    if frac_clean < 0.8 or len(clean) == 0:
        return
    tag = f"{entry}_{ent}"
    if entry in test_entries and 40 <= len(clean) <= 350:
        pool.write(f">{tag}\n{clean}\n")
        meta.write(f"{tag}\t{entry}\t{len(clean)}\n")
        n_pool += 1
    if entry not in post:
        corpus.write(f">{tag}\n{clean}\n")
        n_corpus += 1

for line in open("data/pdb_seqres.txt"):
    line = line.rstrip("\n")
    if line.startswith(">"):
        flush()
        hdr = line
        seq_lines = []
    else:
        seq_lines.append(line.strip())
flush()
corpus.close(); pool.close(); meta.close()
print(f"corpus chains: {n_corpus}  test-pool chains: {n_pool}")
