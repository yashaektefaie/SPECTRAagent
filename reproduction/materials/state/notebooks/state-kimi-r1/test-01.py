# Original notebook cell 2
from pathlib import Path
import sys,json,hashlib,io,os,tempfile,shutil,runpy,contextlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from IPython.display import display,Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'STATE_INPUTS.json').is_file()),None)
assert root is not None,'Extract the complete STATE bundle first.'
manifest=json.loads((root/'STATE_INPUTS.json').read_text())
P=root/'assets'/'state-kimi-r1'
for relative,meta in manifest.items():
    if relative.startswith('state-kimi-r1'+'/'):
        assert hashlib.sha256((root/'assets'/relative).read_bytes()).hexdigest()==meta['sha256'],relative
print('Verified all run input and code hashes.')

# Original notebook cell 3
def draw(data,x,y,group):
    rng=np.random.default_rng(7);rows=[]
    for keys,g in data.groupby([group,x],observed=True):
        v=pd.to_numeric(g[y],errors='coerce').dropna().values
        if not len(v):continue
        bs=np.array([rng.choice(v,len(v),True).mean() for _ in range(1000)])
        rows.append({group:keys[0],x:keys[1],'n':len(v),'mean':v.mean(),'lo':np.quantile(bs,.025),'hi':np.quantile(bs,.975)})
    t=pd.DataFrame(rows);display(t)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');levels=sorted(t[x].unique(),key=str)
    for name,g in t.groupby(group,observed=True):
        g=g.assign(xpos=g[x].map({v:i for i,v in enumerate(levels)})).sort_values('xpos')
        ax.errorbar(g.xpos,g['mean'],yerr=np.stack([g['mean']-g.lo,g.hi-g['mean']]),marker='o',capsize=2,label=str(name))
    ax.set_xticks(range(len(levels)),levels,rotation=25);ax.set(xlabel=x,ylabel=y);ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);display(Image(data=buf.getvalue()))

# Original notebook cell 5
sys.path.insert(0,str(P))
from replay_metrics import replay
data=replay(P)

# Original notebook cell 6
data['group']=data.context
data['exposure_label']=data.apply(lambda r:'seen_pair' if r.seen_pair else 'unseen_E'+str(r.exposure_E),axis=1)
draw(data,'seen_pair' if 1==1 else 'exposure_label','pearson_delta_state','group')
print('The accepted comparison is binary pair exposure; the E1/E2/E3 ordering is a separate graded test.')

# Original notebook cell 7
tmp=tempfile.TemporaryDirectory();run=Path(tmp.name);shutil.copytree(P/'results',run/'results');old=Path.cwd();os.chdir(run)
try:
    runpy.run_path(str(P/'scripts/10_confirm.py'),run_name='__main__')
finally:os.chdir(old)
actual=json.loads((run/'results/confirm_summary.json').read_text());expected=json.loads((P/'results/confirm_summary.json').read_text())
for ctx in ['rpe1','jurkat','hepg2']:
    np.testing.assert_allclose(actual['contexts'][ctx]['delta_state'],expected['contexts'][ctx]['delta_state'],atol=1e-10)
    np.testing.assert_allclose(actual['contexts'][ctx]['delta_boot_ci95'],expected['contexts'][ctx]['delta_boot_ci95'],atol=1e-10)
print('Original held-out-context differences and bootstrap intervals reproduced.')
tmp.cleanup()
