#!/usr/bin/env python
"""Fetch entry ID lists from RCSB Search API (identifiers only, paginated).

Writes:
  data/test_entry_ids.txt   (release 2023-01-01..2024-06-30, X-ray, res<=3.0)
  data/train_entry_ids.txt  (release <= 2020-05-01)
"""
import sys, time
import requests

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


def t(a, o, v):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": a, "operator": o, "value": v}}


def g(*nodes):
    return {"type": "group", "logical_operator": "and", "nodes": list(nodes)}


def fetch_ids(q, outpath, rows=10000):
    total, start = None, 0
    with open(outpath, "w") as f:
        while True:
            body = {"query": q, "return_type": "entry",
                    "request_options": {"paginate": {"start": start, "rows": rows}}}
            for attempt in range(6):
                try:
                    r = requests.post(SEARCH, json=body, timeout=120)
                    r.raise_for_status()
                    break
                except Exception as e:
                    print("retry", attempt, e, file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError("failed")
            js = r.json()
            if total is None:
                total = js["total_count"]
                print(outpath, "total:", total)
            ids = [x["identifier"] for x in js.get("result_set", [])]
            for i in ids:
                f.write(i + "\n")
            start += len(ids)
            print("  got", start, "/", total)
            if start >= total or not ids:
                break


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "test":
        q = g(t("rcsb_accession_info.initial_release_date", "greater_or_equal", "2023-01-01T00:00:00Z"),
              t("rcsb_accession_info.initial_release_date", "less_or_equal", "2024-06-30T00:00:00Z"),
              t("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
              t("rcsb_entry_info.resolution_combined", "less_or_equal", 3.0))
        fetch_ids(q, "data/test_entry_ids.txt")
    elif mode == "train":
        q = t("rcsb_accession_info.initial_release_date", "less_or_equal", "2020-05-01T00:00:00Z")
        fetch_ids(q, "data/train_entry_ids.txt")
