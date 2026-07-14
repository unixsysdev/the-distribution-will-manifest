#!/usr/bin/env python3
"""loss_control.py — the 300s hold is the weakest live rule. Quantify replacements:
shorter time cap, hard stop, hazard death-cut, vs the current tp_50 + 300s.
Deduped net, OOS Jun10-11, lat1, fees. Also isolate the NON-WINNER subset
(tokens that never hit +50%) where loss-control actually acts.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path("/root/the-distribution-will-manifest"); JUN10=calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE=0.1,250.0,0.0015; RNG=np.random.default_rng(5)
P9=["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w","dts"]
def bt(vs,vt,d): return vt-(vs*vt)/(vs+d)
def sl(vs,vt,d): return vs-(vs*vt)/(vt+d)
def frr(p):
    op=gzip.open if p.endswith(".gz") else open
    with op(p,"rt") as f:
        for ln in f:
            try:
                r=json.loads(ln)
                if "slot" in r and "t" in r: return float(r["slot"]),float(r["t"])
            except: pass
def dedup(sc,v):
    by={}
    for s,x in zip(sc,v): by.setdefault(round(float(s),6),[]).append(x)
    pm=[np.mean(z) for z in by.values()]
    bs=[np.mean(np.array(pm)[RNG.integers(0,len(pm),len(pm))]) for _ in range(2000)] if len(pm)>1 else [np.mean(pm)]
    return float(np.mean(pm)),np.percentile(bs,5),np.percentile(bs,95),len(pm)

# hazard model
sn=pd.read_parquet(ROOT/"data/recovery_snaps_k3v03.parquet").sort_values(["mint","fwd_i"])
fmn=sn.groupby("mint")["ret"].transform(lambda s:s[::-1].cummin()[::-1].shift(-1))
sn=sn.assign(fut_min=fmn).dropna(subset=["fut_min"]); sn["collapse"]=(((1+sn.fut_min)/(1+sn.ret)-1)<=-0.40).astype(int)
hz=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,l2_regularization=2.0,random_state=0).fit(sn[sn.ready_ts<JUN10][P9].values,sn[sn.ready_ts<JUN10].collapse.values)
hzby={m:hz.predict_proba(g[P9].values)[:,1] for m,g in sn.groupby("mint")}

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); M=d["mints"]; mints=list(M)
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=frr(fs[0]),frr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
clf=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
X=np.array([M[m]["feats"] for m in mints]); S=clf.predict_proba(X)[:,1]; TT=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])

class Fire:
    def __init__(s,m):
        f=M[m]["fwd"]; j=min(1,len(f)-1); s.evs,s.evt=f[j][2],f[j][3]; s.tok=bt(s.evs,s.evt,Q*1e9); em=s.evs/s.evt
        rest=f[j+1:]; s.t=np.array([r[0]-f[j][0] for r in rest]); s.vs=np.array([r[2] for r in rest],float); s.vt=np.array([r[3] for r in rest],float)
        s.ret=(s.vs/s.vt)/em-1.0
        h=hzby.get(m); hh=h[j+1:] if (h is not None and len(h)>j) else np.array([])
        s.haz=np.concatenate([hh,np.repeat(hh[-1] if len(hh) else 0.0,max(0,len(s.ret)-len(hh)))])[:len(s.ret)] if len(s.ret) else np.array([])
    def out(s,i): return sl(s.vs[i],s.vt[i],s.tok)/(Q*1e9)-1-CB/1e4-2*FEE/Q

fires=[(m,S[i]) for i,m in enumerate(mints) if S[i]>=0.50 and M[m]["fwd"] and len(M[m]["fwd"])>2 and TT[i]>=JUN10]
F={m:Fire(m) for m,_ in fires}

def sim(m, mode):
    f=F[m]; n=len(f.ret)
    if n==0: return -CB/1e4-2*FEE/Q
    for i in range(n):
        if f.ret[i]>=0.5: return f.out(i)                      # tp_50 win (all modes)
        if mode=="cap120" and f.t[i]>=120: return f.out(i)
        if mode=="cap90"  and f.t[i]>=90:  return f.out(i)
        if mode=="stop30" and f.ret[i]<=-0.30: return f.out(i)
        if mode=="stop40" and f.ret[i]<=-0.40: return f.out(i)
        if mode=="hazard" and f.haz[i]>=0.60: return f.out(i)
        if mode=="stop30+cap120" and (f.ret[i]<=-0.30 or f.t[i]>=120): return f.out(i)
    return f.out(n-1)                                          # "stale300"/horizon

modes=["stale300 (CURRENT)","cap120","cap90","stop30","stop40","hazard","stop30+cap120"]
keys ={"stale300 (CURRENT)":"stale300","cap120":"cap120","cap90":"cap90","stop30":"stop30","stop40":"stop40","hazard":"hazard","stop30+cap120":"stop30+cap120"}
print(f"OOS Jun10-11 fires={len(fires)}  (tp_50 take-profit fixed; downside rule varies)\n")
print(f"  {'loss-control':22s} {'ALL fires deduped':>26s} | {'non-winner subset mean':>22s}")
nonwin=[m for m,_ in fires if F[m].ret.max()<0.5 if len(F[m].ret)]
for lab in modes:
    k=keys[lab]
    nets=[sim(m,k) for m,_ in fires]; scs=[sc for _,sc in fires]
    mu,lo,hi,npat=dedup(scs,nets)
    nw=[sim(m,k) for m in nonwin]
    print(f"  {lab:22s} {mu:+.3f} [{lo:+.3f},{hi:+.3f}] np={npat:2d} | n={len(nw):2d} mean={np.mean(nw):+.3f}")
print("\nread: a downside rule helps if ALL-deduped rises vs stale300 AND the non-winner mean rises")
print("(less bleed). tp_100_t120 already embeds cap120, so cap120 here ~ that exit's downside half.")
