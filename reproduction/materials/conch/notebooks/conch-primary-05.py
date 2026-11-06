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

# Original notebook cell 3
dev=read_csv('conch-primary/discovery.csv'); confirm=read_csv('conch-primary/confirmation.csv')
pred=read_csv('shared/crcval_predictions.csv')
assert dev.sample_id.is_unique and confirm.sample_id.is_unique
dev=dev.merge(pred[['sample_id','image_path','pred','label','correct']].rename(columns={'label':'prediction_label','correct':'prediction_correct'}),on='sample_id',validate='one_to_one')
assert len(dev)==7180 and len(confirm)==1800
assert (dev.label==dev.prediction_label).all() and (dev.correct==dev.prediction_correct).all()
assert not set(dev.sample_id)&set(confirm.sample_id)
display(pd.DataFrame({'discovery':dev.label_name.value_counts(),'confirmation':confirm.label_name.value_counts()}).fillna(0).astype(int))
print('Original confirmation IDs; raw images and filename/hash mapping are not retained here:')
display(confirm[['sample_id','label','label_name']].head())

# Original notebook cell 5
level_col='level_nct_color_texture_proximity'; axis_col='axis_nct_color_texture_similarity'
assert dev[level_col].between(1,6).all()
display(dev.groupby(level_col)[axis_col].agg(['count','min','max','mean']))
print('These are retained prospective axis values and original level assignments. The reference feature matrix and original assignment implementation have not been recovered; this cell does not claim to recompute them.')

# Original notebook cell 7
classes=['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
pred_logits=pred[['logit_'+c for c in classes]].to_numpy().argmax(1)
assert np.array_equal(pred_logits,pred.pred)
assert np.array_equal(dev.pred==dev.label,dev.correct.astype(bool))
assert set(confirm.correct.unique()).issubset({0,1})
print('Confirmation target correctness is retained per image, but original predicted labels/logits are missing; it cannot be independently rescored.')
for frame in [dev,confirm]:
    for model in ['centroid','logistic']:
        frame[model+'_correct']=(frame['baseline_'+model+'_pred']==frame.label).astype(int)
print('Discovery correctness recalculated from saved predictions/logits for 7,180 images. Both confirmation baseline correctness arrays were recalculated from predicted labels.')

# Original notebook cell 9
display(pd.crosstab(dev[level_col],dev.label_name));display(pd.crosstab(dev[level_col],dev.label_name,normalize='index'))
print('These class counts use the same original distance bins as the pooled curve.')
