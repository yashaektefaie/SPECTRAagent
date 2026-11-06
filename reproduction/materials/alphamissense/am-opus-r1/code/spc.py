import pandas as pd, numpy as np
from scipy.stats import spearmanr

def assign_levels(sub, feat, nbin):
    """Within-assay: rank unique positions by feat, split into nbin equal-position levels (0=low..nbin-1=high)."""
    pp = sub[['position',feat]].drop_duplicates('position').sort_values(feat)
    pp = pp.reset_index(drop=True)
    # equal-count position bins by rank
    pp['lvl'] = pd.qcut(pp[feat].rank(method='first'), nbin, labels=False)
    return sub.merge(pp[['position','lvl']], on='position', how='left')

def cell_acc(sub):
    if len(sub)<50: return np.nan,np.nan,np.nan,len(sub),sub['position'].nunique()
    am=abs(spearmanr(sub.am_pathogenicity, sub.DMS_score).correlation)
    bl=abs(spearmanr(sub.blosum62, sub.DMS_score).correlation)
    return am,bl,sub.DMS_score.std(),len(sub),sub['position'].nunique()

def spc_table(v, feat='cov_frac', nbin=3, min_cell=50):
    rows=[]
    for a,sub in v.groupby('assay'):
        if sub[feat].nunique()<nbin: continue
        sub=assign_levels(sub, feat, nbin)
        cells={}
        ok=True
        for l in range(nbin):
            cs=sub[sub.lvl==l]
            am,bl,sd,n,npos=cell_acc(cs)
            cells[l]=(am,bl,sd,n,npos, cs[feat].mean(), cs['entropy'].mean())
            if n<min_cell or np.isnan(am): ok=False
        if not ok: continue
        for l in range(nbin):
            am,bl,sd,n,npos,mcf,ment=cells[l]
            rows.append(dict(assay=a, sel=sub.coarse_selection_type.iloc[0], split=sub.split.iloc[0],
                             lvl=l, am_acc=am, bl_acc=bl, dms_std=sd, n=n, npos=npos,
                             mean_covfrac=mcf, mean_entropy=ment))
    return pd.DataFrame(rows)

def summarize(tab, nbin, label, B=5000, seed=0):
    assays=sorted(tab.assay.unique())
    print(f"\n===== {label}: {len(assays)} assays x {nbin} levels (paired) =====")
    piv_am=tab.pivot(index='assay',columns='lvl',values='am_acc')
    piv_bl=tab.pivot(index='assay',columns='lvl',values='bl_acc')
    piv_sd=tab.pivot(index='assay',columns='lvl',values='dms_std')
    piv_cf=tab.pivot(index='assay',columns='lvl',values='mean_covfrac')
    print("level:  " + "  ".join([f"L{l}" for l in range(nbin)]) + "   (L0=lowest coverage .. L%d=highest)"%(nbin-1))
    print("mean cov_frac:   " + "  ".join(f"{piv_cf[l].mean():.3f}" for l in range(nbin)))
    print("AM   |Spearman|: " + "  ".join(f"{piv_am[l].mean():.3f}" for l in range(nbin)))
    print("BLOS |Spearman|: " + "  ".join(f"{piv_bl[l].mean():.3f}" for l in range(nbin)))
    print("AM-BLOS gap:     " + "  ".join(f"{(piv_am[l]-piv_bl[l]).mean():.3f}" for l in range(nbin)))
    print("DMS std:         " + "  ".join(f"{piv_sd[l].mean():.3f}" for l in range(nbin)))
    # effect: High - Low per assay
    hi, lo = nbin-1, 0
    d_am=(piv_am[hi]-piv_am[lo]).dropna()
    d_bl=(piv_bl[hi]-piv_bl[lo]).dropna()
    rng=np.random.default_rng(seed)
    def boot(x):
        x=x.values; idx=rng.integers(0,len(x),(B,len(x)))
        bs=x[idx].mean(1); return np.percentile(bs,[2.5,97.5])
    ci_am=boot(d_am); ci_bl=boot(d_bl)
    print(f"\nAM   High-Low (L{hi}-L0): mean={d_am.mean():+.3f}  95%CI[{ci_am[0]:+.3f},{ci_am[1]:+.3f}]  (>0 in {(d_am>0).sum()}/{len(d_am)} assays)")
    print(f"BLOS High-Low (L{hi}-L0): mean={d_bl.mean():+.3f}  95%CI[{ci_bl[0]:+.3f},{ci_bl[1]:+.3f}]")
    # AM-specific degradation: (AM_hi-AM_lo) - (BL_hi-BL_lo)
    both=pd.concat([d_am.rename('am'),d_bl.rename('bl')],axis=1).dropna()
    dd=(both.am-both.bl); idx=rng.integers(0,len(dd),(B,len(dd))); ci=np.percentile(dd.values[idx].mean(1),[2.5,97.5])
    print(f"AM-specific (AM drop - BLOS drop): mean={dd.mean():+.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
    # monotonicity (Spearman of mean acc vs level across the nbin means)
    mlevels=[piv_am[l].mean() for l in range(nbin)]
    print("monotone decreasing High->Low?", all(mlevels[l]>=mlevels[l-1]-1e-9 for l in range(1,nbin)),
          " level means:", [round(x,3) for x in mlevels])
    return dict(d_am_mean=float(d_am.mean()), ci_am=[float(ci_am[0]),float(ci_am[1])],
                d_bl_mean=float(d_bl.mean()), amspec=float(dd.mean()), amspec_ci=[float(ci[0]),float(ci[1])],
                level_am=[float(x) for x in mlevels], n_assays=len(assays))

if __name__=="__main__":
    import json
    v=pd.read_parquet('work/variants_feat.parquet')
    disc=v[v.split=='discovery']
    tab3=spc_table(disc, 'cov_frac', 3)
    tab3.to_csv('work/spc_discovery_t3.csv',index=False)
    s3=summarize(tab3,3,"DISCOVERY coverage terciles")
    tab5=spc_table(disc,'cov_frac',5)
    s5=summarize(tab5,5,"DISCOVERY coverage quintiles (densification)")
    json.dump({'t3':s3,'t5':s5}, open('work/discovery_summary.json','w'), indent=2)
