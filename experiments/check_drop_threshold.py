"""Drop threshold mu_drop = lambda h^2 / (2 l) per channel, and the clipped law at 94 A.

Runs on the saved arrays of experiments/mu_theory.py; no solver or model. Writes
results/plots/24_paper_20261031/drop_threshold_check.json.
"""
import json

import numpy as np

d=np.load('output/meeting_followup/bp_mu_theory_val.npz')
mus=d['mus']; g=d['group']; obs=d['obs']; err=d['err']; lam=d['lam_lo']; st=d['lp_status']; viol=d['viol']; side=d['side']; rr=d['recon94_ratio']
CORE=g.max(); core=(g==CORE); feas=(st==0)
ch=[94,131,171,193,211,335]
theory=json.load(open('results/plots/23_meeting_followup_20261024/bp_mu_theory_val.json'))
bound=np.array([theory['overall']['single_channel_bound'][str(w)] for w in ch])
t=1.4
out={}
print('channel  bound  median sigma^2/o: flare cores (p10,p50,p90) | all bright  | mu_drop_inf = bound*t^2*g/2')
gs=[]
for c,w in enumerate(ch):
    m=core&(obs[:,c]>0.01)
    g_=err[m,c]**2/obs[m,c]
    br=(g>=4)&(obs[:,c]>10*err[:,c])
    g2=err[br,c]**2/obs[br,c]
    p=np.percentile(g_,[10,50,90])
    p2=np.percentile(g2,[10,50,90]) if br.sum() else [np.nan]*3
    md=bound[c]*t**2*p2[1]/2
    gs.append(p2[1])
    out[w]={'price_bound':bound[c],'sigma2_over_o_core_p10':p[0],'sigma2_over_o_core_p50':p[1],'sigma2_over_o_core_p90':p[2],'sigma2_over_o_bright_p50':p2[1],'sigma2_over_o_bright_p90':p2[2],'mu_drop_large_o':md}
    print(f"{w:4d}  {bound[c]:.4f}  core {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} | high-SNR Bright {p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f} | mu_drop {md:.3f}")
# clipped law at 94 A, feasible flare cores with the 94 A lower edge priced
c=0
h=t*err[:,c]; l=obs[:,c]-h; la=lam[:,c]
m=core&feas&np.isfinite(la)&(la>0)&(l>0)
print('n',m.sum())
rows=[]
for k,mu in enumerate(mus):
    pred=np.maximum(l[m]-la[m]*h[m]**2/(2*mu),0)
    meas=rr[k][m]*obs[m,c]
    rows.append((float(mu),float(pred.sum()/obs[m,c].sum()),float(meas.sum()/obs[m,c].sum())))
    print(f"mu {mu:6g}  predicted 94A/obs {pred.sum()/obs[m,c].sum():.3f}  measured {meas.sum()/obs[m,c].sum():.3f}")
# and medians per pixel
mud=la[m]*h[m]**2/(2*l[m])
out2={'n_pixels':int(m.sum()),'mu_drop_percentiles_p10_p25_p50_p75_p90':[float(x) for x in np.percentile(mud,[10,25,50,75,90])],
      'law_rows_mu_pred_meas':rows,'channels':out}
json.dump(out2,open('results/plots/24_paper_20261031/drop_threshold_check.json','w'),indent=1)
