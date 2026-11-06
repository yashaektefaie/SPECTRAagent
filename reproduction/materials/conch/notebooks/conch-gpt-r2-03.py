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
files=['conch-gpt-r2/extreme_level_contrasts.csv', 'conch-gpt-r2/report.md', 'conch-gpt-r2/candidate_axis_ledger.csv', 'conch-gpt-r2/spc_table.csv', 'conch-gpt-r2/original_checksums.csv', 'conch-gpt-r2/original_bundle/audit/provenance.json', 'conch-gpt-r2/original_bundle/audit/README.md', 'conch-gpt-r2/original_bundle/audit/frozen_axis.json', 'conch-gpt-r2/original_bundle/audit/artifact_manifest.json', 'conch-gpt-r2/original_bundle/audit/novelty_search.md', 'conch-gpt-r2/original_bundle/audit/features/crc_val_he_7k.manifest.json', 'conch-gpt-r2/original_bundle/audit/features/stanford_crc_he_val_small.manifest.json', 'conch-gpt-r2/original_bundle/audit/features/nct_bounded_reference.manifest.json', 'conch-gpt-r2/original_bundle/audit/state/candidate_axis_ledger.csv', 'conch-gpt-r2/original_bundle/audit/results/extreme_level_contrasts.csv', 'conch-gpt-r2/original_bundle/audit/results/spc_table.csv', 'conch-gpt-r2/original_bundle/audit/results/power_plan.json', 'conch-gpt-r2/original_bundle/audit/manifests/nct_bounded_reference.json', 'conch-gpt-r2/original_bundle/audit/manifests/handcrafted_references.json', 'conch-gpt-r2/original_bundle/audit/manifests/resnet18_reference.json', 'conch-gpt-r2/original_bundle/audit/manifests/conch_weight_download.json', 'conch-gpt-r2/original_bundle/audit/manifests/conch_scoring.json']
display(pd.DataFrame([{'file':p,'bytes':(SOURCE/p).stat().st_size,'sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in files]))

# Original notebook cell 10
display(read_csv('conch-gpt-r2/candidate_axis_ledger.csv'))
