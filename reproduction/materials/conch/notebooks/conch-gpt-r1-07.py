# Original notebook cell 1
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = Path.cwd()
if not (ROOT / 'source').is_dir(): ROOT = ROOT.parent
SOURCE = ROOT / 'source'
assert SOURCE.is_dir(), 'Run from the conch package or its notebooks directory.'
def read_csv(path): return pd.read_csv(SOURCE / path)
def read_json(path): return json.loads((SOURCE / path).read_text())
def paired_interval(values, seed=0, draws=5000):
    values = np.asarray(values, float); rng=np.random.default_rng(seed)
    # Chunk draws to keep memory bounded, preserving NumPy RNG order.
    means=np.concatenate([values[rng.integers(0,len(values),(min(100,draws-i),len(values)))].mean(1) for i in range(0,draws,100)])
    return float(values.mean()), np.quantile(means,[.025,.975])
def wilson(k,n):
    z=1.959963984540054; p=k/n; a=1+z*z/n
    center=(p+z*z/(2*n))/a; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return center-h, center+h
def show_curve(table, x='level', group='model', y='accuracy', xlabel='Level', ylabel='Accuracy'):
    fig,ax=plt.subplots(figsize=(7,3.6))
    for name,g in table.groupby(group,sort=False):
        g=g.sort_values(x); ax.plot(g[x],g[y],marker='o',label=name)
        if 'lo' in g and 'hi' in g: ax.fill_between(g[x],g.lo,g.hi,alpha=.13)
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);fig.tight_layout();plt.show()

# Original notebook cell 4
files=['conch-gpt-r1/matched_confound_summary.csv', 'conch-gpt-r1/report.md', 'conch-gpt-r1/candidate_axis_ledger.csv', 'conch-gpt-r1/spc_table.csv', 'conch-gpt-r1/extreme_contrasts.csv', 'conch-gpt-r1/code/finalize_population_gate.py', 'conch-gpt-r1/code/analyze_spc.py', 'conch-gpt-r1/code/derive_prospective_axis.py', 'conch-gpt-r1/code/summarize_levels.py', 'conch-gpt-r1/code/finalize_dedup.py', 'conch-gpt-r1/code/deduplicate.py', 'conch-gpt-r1/code/score_conch.py', 'conch-gpt-r1/code/extract_imagenet_embeddings.py', 'conch-gpt-r1/code/fit_fixed_references.py', 'conch-gpt-r1/code/blind_extract.py', 'conch-gpt-r1/code/extract_prospective_features.py', 'conch-gpt-r1/code/recheck_population_after_dedup.py', 'conch-gpt-r1/original_bundle/README.md', 'conch-gpt-r1/original_bundle/results/matched_confound_summary.csv', 'conch-gpt-r1/original_bundle/results/spc_table.csv', 'conch-gpt-r1/original_bundle/results/reference_only_within_class_spc_all_mapped.csv', 'conch-gpt-r1/original_bundle/results/integrity_validation.json', 'conch-gpt-r1/original_bundle/results/reference_only_within_class_spc.csv', 'conch-gpt-r1/original_bundle/results/extreme_contrasts.csv', 'conch-gpt-r1/original_bundle/results/closure_statistics.json', 'conch-gpt-r1/original_bundle/results/reference_only_spc_table.csv', 'conch-gpt-r1/original_bundle/results/within_class_spc_all_mapped.csv', 'conch-gpt-r1/original_bundle/results/within_class_spc.csv', 'conch-gpt-r1/original_bundle/results/reference_only_closure_statistics.json', 'conch-gpt-r1/original_bundle/results/deduplication_summary.json', 'conch-gpt-r1/original_bundle/provenance/sources.json', 'conch-gpt-r1/original_bundle/provenance/runtime.json', 'conch-gpt-r1/original_bundle/provenance/references.json', 'conch-gpt-r1/original_bundle/provenance/checkpoint_access.json', 'conch-gpt-r1/original_bundle/provenance/downloads.json', 'conch-gpt-r1/original_bundle/provenance/artifact_manifest.json', 'conch-gpt-r1/original_bundle/audit/candidate_axis_ledger.csv', 'conch-gpt-r1/original_bundle/audit/power_plan.json', 'conch-gpt-r1/original_bundle/audit/closure_checklist.md', 'conch-gpt-r1/original_bundle/audit/frozen_axis.json', 'conch-gpt-r1/original_bundle/reports/novelty_search.md', 'conch-gpt-r1/original_bundle/reports/preclosure_status.md', 'conch-gpt-r1/original_bundle/data/manifests/crc_val_7k.blind.csv', 'conch-gpt-r1/original_bundle/data/manifests/kather2016_5k.blind.csv']
display(pd.DataFrame([{'file':p,'bytes':(SOURCE/p).stat().st_size,'sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in files]))

# Original notebook cell 10
display(read_csv('conch-gpt-r1/candidate_axis_ledger.csv'))
