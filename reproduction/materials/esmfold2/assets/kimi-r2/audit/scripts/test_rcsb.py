import requests, json
q = {
  "query": {"type":"terminal","service":"date",
    "parameters":{"attribute":"rcsb_accession_info.initial_release_date","operator":"less","negation":False,"value":"2021-10-01"}},
  "return_type":"entry",
  "request_options":{"paginate":{"start":0,"rows":0},"results_content_type":["experimental"]}
}
r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
d = r.json()
print("pre-cutoff entries:", d.get("total_count"))

q2 = {
  "query": {"type":"group","logical_operator":"and","nodes":[
    {"type":"terminal","service":"date","parameters":{"attribute":"rcsb_accession_info.initial_release_date","operator":"greater_or_equal","negation":False,"value":"2023-01-14"}},
    {"type":"terminal","service":"text","parameters":{"attribute":"entity_poly.rcsb_entity_polymer_type","operator":"exact_match","value":"Protein"}},
    {"type":"terminal","service":"text","parameters":{"attribute":"exptl.method","operator":"exact_match","value":"X-RAY DIFFRACTION"}},
    {"type":"terminal","service":"text","parameters":{"attribute":"rcsb_entry_info.resolution_combined","operator":"less_or_equal","value":3.0}},
    {"type":"terminal","service":"text","parameters":{"attribute":"entity_poly.rcsb_sample_sequence_length","operator":"greater_or_equal","value":40}},
    {"type":"terminal","service":"text","parameters":{"attribute":"entity_poly.rcsb_sample_sequence_length","operator":"less_or_equal","value":400}}
  ]},
  "return_type":"polymer_entity",
  "request_options":{"paginate":{"start":0,"rows":0},"results_content_type":["experimental"]}
}
r2 = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q2, timeout=60)
print(r2.status_code)
d2 = r2.json()
print("post-2023 test entities:", d2.get("total_count"))
