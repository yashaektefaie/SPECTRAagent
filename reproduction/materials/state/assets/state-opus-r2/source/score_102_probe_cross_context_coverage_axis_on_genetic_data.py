import pandas as pd, numpy as np
from scipy.stats import spearmanr
df = pd.read_csv('data/predictions.csv', low_memory=False)
r = df[df.dataset_family=='Replogle-Nadig'].copy()

# cross-context coverage: for each gene, how many of the 4 contexts it appears in (overall)
gene_ctx = r.groupby('perturbation_name')['context'].nunique()
r['gene_ncontexts']=r['perturbation_name'].map(gene_ctx)

# For a zeroshot row in context c, 'other-context coverage' = ncontexts - 1 (present elsewhere)
r['other_ctx_cov']=r['gene_ncontexts']-1

print('=== Replogle zeroshot: overlap by other_ctx_cov (0..3) ===')
z=r[r.protocol=='zeroshot']
for rep in ['HVG','SE']:
    zz=z[z.representation==rep]
    print(f'-- {rep} --')
    print(zz.groupby('other_ctx_cov')['overlap_at_N'].agg(['mean','median','count']).round(3))
    rho,p=spearmanr(zz['other_ctx_cov'],zz['overlap_at_N'])
    print(f'   spearman={rho:+.3f} p={p:.1e}')
