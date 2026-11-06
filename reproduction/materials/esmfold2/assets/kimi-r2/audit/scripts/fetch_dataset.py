#!/usr/bin/env python
"""Fetch SPECTRA dataset pieces from RCSB + wwPDB.

Provenance:
- RCSB Search API v2 (https://search.rcsb.org/rcsbsearch/v2/query)
  Q1: entries with initial_release_date < 2021-10-01 (training-pool proxy; model
      card states PDB training data released prior to 2021-09-30)
  Q2: polymer entities: protein, X-RAY DIFFRACTION, resolution <= 3.0,
      length 40-400, initial_release_date >= 2023-01-14 (test pool; after the
      paper's own validation window end 2023-01-13)
- wwPDB derived data: https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
Outputs:
- audit/data/pool_seqs.fasta        (pre-cutoff PDB protein chain sequences)
- audit/data/test_candidates.fasta  (post-2023 test candidate entity sequences)
- audit/data/test_candidates.jsonl  (entity metadata: entry, entity, length)
"""
import requests, gzip, json, sys, time, os

OUT = "audit/data"
os.makedirs(OUT, exist_ok=True)

def search_all(query, return_type):
    ids, start = [], 0
    while True:
        body = {
            "query": query,
            "return_type": return_type,
            "request_options": {
                "paginate": {"start": start, "rows": 10000},
                "results_content_type": ["experimental"],
            },
        }
        for attempt in range(4):
            try:
                r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                                  json=body, timeout=180)
                r.raise_for_status()
                break
            except Exception as e:
                print("retry", attempt, e, flush=True)
                time.sleep(5 * (attempt + 1))
        d = r.json()
        batch = [x["identifier"] for x in d.get("result_set", [])]
        ids.extend(batch)
        total = d["total_count"]
        print(f"{return_type}: got {len(ids)}/{total}", flush=True)
        if len(ids) >= total or not batch:
            break
        start += len(batch)
    return ids

# Q1: pre-cutoff entries
q1 = {"type": "terminal", "service": "text", "parameters": {
    "attribute": "rcsb_accession_info.initial_release_date",
    "operator": "less", "value": "2021-10-01T00:00:00Z"}}
pool_entries = set(search_all(q1, "entry"))
json.dump(sorted(pool_entries), open(f"{OUT}/pool_entry_ids.json", "w"))

# Q2: test candidate polymer entities
q2 = {"type": "group", "logical_operator": "and", "nodes": [
    {"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_accession_info.initial_release_date",
        "operator": "greater_or_equal", "value": "2023-01-14T00:00:00Z"}},
    {"type": "terminal", "service": "text", "parameters": {
        "attribute": "entity_poly.rcsb_entity_polymer_type",
        "operator": "exact_match", "value": "Protein"}},
    {"type": "terminal", "service": "text", "parameters": {
        "attribute": "exptl.method", "operator": "exact_match",
        "value": "X-RAY DIFFRACTION"}},
    {"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_entry_info.resolution_combined",
        "operator": "less_or_equal", "value": 3.0}},
    {"type": "terminal", "service": "text", "parameters": {
        "attribute": "entity_poly.rcsb_sample_sequence_length",
        "operator": "greater_or_equal", "value": 40}},
    {"type": "terminal", "service": "text", "parameters": {
        "attribute": "entity_poly.rcsb_sample_sequence_length",
        "operator": "less_or_equal", "value": 400}},
]}
test_entities = search_all(q2, "polymer_entity")
json.dump(test_entities, open(f"{OUT}/test_entity_ids.json", "w"))

# sequences from pdb_seqres.txt.gz
seqres = f"{OUT}/pdb_seqres.txt.gz"
if not os.path.exists(seqres):
    r = requests.get("https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
                     stream=True, timeout=600)
    r.raise_for_status()
    with open(seqres, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print("downloaded pdb_seqres.txt.gz", flush=True)

test_set = set(t.upper() for t in test_entities)
pool_f = open(f"{OUT}/pool_seqs.fasta", "w")
test_f = open(f"{OUT}/test_candidates.fasta", "w")
meta_f = open(f"{OUT}/test_candidates.jsonl", "w")
n_pool = n_test = 0
header, seq = None, []
def flush():
    global n_pool, n_test
    if header is None:
        return
    hid = header.split()[0][1:]
    is_prot = "mol:protein" in header
    s = "".join(seq)
    entry = hid.split("_")[0].upper()
    if is_prot and entry in pool_entries and len(s) >= 20:
        pool_f.write(f">{hid}\n{s}\n")
        n_pool += 1
    if hid.upper() in test_set:
        test_f.write(f">{hid}\n{s}\n")
        meta_f.write(json.dumps({"entity": hid.upper(), "entry": entry,
                                 "length": len(s)}) + "\n")
        n_test += 1

with gzip.open(seqres, "rt") as fh:
    for line in fh:
        if line.startswith(">"):
            flush()
            header, seq = line.strip(), []
        else:
            seq.append(line.strip())
flush()
pool_f.close(); test_f.close(); meta_f.close()
print(f"pool chains: {n_pool}, test candidates: {n_test}", flush=True)
