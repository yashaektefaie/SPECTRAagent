#!/usr/bin/env python
"""Fetch sequences + auth chain ids for a random subsample of test candidate
polymer entities via RCSB REST data API.

Provenance: https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity}
Subsample: 3000 of 32788 entities, seed=20260722 (runtime budget; every unit in
the final confirmation set is drawn from this candidate table).
"""
import requests, json, random, time, os

OUT = "audit/data"
ids = json.load(open(f"{OUT}/test_entity_ids.json"))
random.seed(20260722)
random.shuffle(ids)
sub = ids[:3000]

S = requests.Session()
out_f = open(f"{OUT}/test_candidates.fasta", "w")
meta_f = open(f"{OUT}/test_candidates.jsonl", "w")
n_ok = n_err = 0
for i, eid in enumerate(sub):
    entry, ent = eid.split("_")
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{ent}"
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    if r.status_code != 200:
        n_err += 1
        continue
    d = r.json()
    seq = d.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
    seq = seq.replace("\n", "").replace(" ", "")
    chains = d.get("rcsb_polymer_entity_container_identifiers", {}) \
              .get("auth_asym_ids", [])
    nres_obs = d.get("rcsb_polymer_entity", {}).get(
        "rcsb_missing_residue_count", None)
    if not seq or not chains or len(seq) < 40 or len(seq) > 400:
        continue
    if set(seq) - set("ACDEFGHIKLMNPQRSTVWYX"):
        continue
    out_f.write(f">{eid}\n{seq}\n")
    meta_f.write(json.dumps({"entity": eid, "entry": entry,
                             "entity_id": ent, "length": len(seq),
                             "chains": chains}) + "\n")
    n_ok += 1
    if (i + 1) % 250 == 0:
        print(f"{i+1}/{len(sub)} ok={n_ok} err={n_err}", flush=True)
out_f.close(); meta_f.close()
print(f"DONE ok={n_ok} err={n_err}", flush=True)
