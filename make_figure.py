"""Figure for report.tex (matplotlib used for visualisation only).
Left: OOF F1 vs positive rate from artifacts/cv_oof.npz. Right: learning curve of gbB on
fold 0 (AP measured in the v7 ablation, see docs/v7_analysis.md)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
d=np.load('artifacts/cv_oof.npz'); y=d['y']
nn=np.mean([d[f'nn{s}'] for s in range(5)],0); gb=(d['gbA']+d['gbB'])/2; bl=0.6*gb+0.4*nn
plt.rcParams.update({'font.size':7}); fig,ax=plt.subplots(1,2,figsize=(3.5,1.8))
for s,l,c in [(bl,'blend','k'),(d['gbA'],'gbA','C0'),(nn,'NN (5 seeds)','C1')]:
    o=np.argsort(-s); tp=np.cumsum(y[o]); k=np.arange(1,len(y)+1); f=2*tp/(k+y.sum())
    q=k/len(y); m=(q>0.05)&(q<0.25); ax[0].plot(q[m]*100,f[m],c,label=l,lw=1.2)
ax[0].set_xlabel('predicted positive (%)'); ax[0].set_ylabel('OOF F1'); ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)
lc=[(65627,0.3898),(131254,0.4263),(262508,0.4353)]
ax[1].plot([a for a,_ in lc],[b for _,b in lc],'o-k'); ax[1].set_xscale('log',base=2)
ax[1].set_xticks([a for a,_ in lc]); ax[1].set_xticklabels([f'{int(a/1000)}k' for a,_ in lc])
ax[1].set_xlabel('training rows'); ax[1].set_ylabel('AP on fold 0'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('report_fig.pdf')
