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
files=['conch-kimi-r2/report.md', 'conch-kimi-r2/native/REPORT.md', 'conch-kimi-r2/native/AXIS_SPEC.md', 'conch-kimi-r2/code/stats_axisA_kimi.py', 'conch-kimi-r2/code/zeroshot_common.py', 'conch-kimi-r2/code/pilot_resolution.py', 'conch-kimi-r2/code/build_nct_sample.py', 'conch-kimi-r2/code/run_axisA_confirm.py', 'conch-kimi-r2/code/pilot_axisB_stain.py', 'conch-kimi-r2/code/run_axisA_crcval.py', 'conch-kimi-r2/native/artifacts/provenance_manifest.json', 'conch-kimi-r2/native/artifacts/axisA_frozen_spec.md', 'conch-kimi-r2/native/notes/distillation_and_ledger.md', 'conch-kimi-r2/native/scripts/eval_resnet.py', 'conch-kimi-r2/native/scripts/stats_axisA_kimi.py', 'conch-kimi-r2/native/scripts/transforms_spec.py', 'conch-kimi-r2/native/scripts/zeroshot_common.py', 'conch-kimi-r2/native/scripts/pilot_resolution.py', 'conch-kimi-r2/native/scripts/build_nct_sample.py', 'conch-kimi-r2/native/scripts/stats_axisA.py', 'conch-kimi-r2/native/scripts/run_axisA_confirm.py', 'conch-kimi-r2/native/scripts/eval_conch.py', 'conch-kimi-r2/native/scripts/train_resnet.py', 'conch-kimi-r2/native/scripts/pilot_axisB_stain.py', 'conch-kimi-r2/native/scripts/run_axisA_crcval.py', 'conch-kimi-r2/native/data/NCT-sample/nct_sample_manifest.csv', 'conch-kimi-r2/native/artifacts/axisA_crcval/crcval_per_image.csv', 'conch-kimi-r2/native/artifacts/axisA_crcval/crcval_summary.csv', 'conch-kimi-r2/native/artifacts/pilot_axisB/axisB_pilot.csv', 'conch-kimi-r2/native/artifacts/pilot/pilot_results.csv', 'conch-kimi-r2/native/artifacts/pilot/pilot_sample.csv', 'conch-kimi-r2/native/artifacts/pilot/pilot_per_image.csv', 'conch-kimi-r2/native/artifacts/axisA_confirm/confirmatory_stats.csv', 'conch-kimi-r2/native/artifacts/axisA_confirm/per_image_eval.csv', 'conch-kimi-r2/native/artifacts/axisA_confirm/axis_order_verification.csv', 'conch-kimi-r2/native/artifacts/axisA_confirm/axisA_summary.csv']
display(pd.DataFrame([{'file':p,'bytes':(SOURCE/p).stat().st_size,'sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in files]))

# Original notebook cell 10
display(read_csv('conch-kimi-r2/native/artifacts/pilot_axisB/axisB_pilot.csv'))
