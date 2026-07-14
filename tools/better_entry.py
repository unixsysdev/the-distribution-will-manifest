#!/usr/bin/env python3
"""better_entry.py — two experiments, judged on the metric that matters
(deduped tp_100_t120 net at matched top-3%, cross-day train Jun7-9 / test
Jun10-11), not AUC.

 A) BETTER ENTRY: does a different TARGET beat the current peak>=2x?
    - peak>=2x  (current deployed target)
    - peak>=1x  (matches the tp_100 exit)
    - exit-aligned: predict tp_100_t120 net >= +0.20 directly
    Train each on ALL ready mints (well-powered), 22 K+V feats.
 B) TIME-CONDITIONED PRICE LADDER: empirical P(reach next level | at level,
    time-bucket) from train; ladder holds the high tranches only while that
    conditional continuation prob stays high (path-based, needs no entry
    moon-detection). vs tp_100_t120. Deduped.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE = 0.1,250.0,0.0015
REG=dict(max_depth=3,max_iter=150,learning_rate=0.05,l2_regularization=5.0,random_state=42)

def bt(vs,vt,d): return vt-(vs*vt)/(vs+d)
def sl(vs,vt,d): return vs-(vs*vt)/(vt+d)
def fr(p):
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
    return float(np.mean([np.mean(z) for z in by.values()])),len(by)

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); names,M=d["names"],d["mints"]
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=fr(fs[0]),fr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
mints=list(M); X=np.array([M[m]["feats"] for m in mints]); TT=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])

def path(m, lat=1):
    f=M[m]["fwd"];
    if not f: return None
    j=min(lat,len(f)-1); evs,evt=f[j][2],f[j][3]
    rest=f[j+1:]
    if not rest: return None
    em=evs/evt; t0=f[j][0]
    ret=np.array([(r[2]/r[3])/em-1.0 for r in rest]); tt=np.array([r[0]-t0 for r in rest])
    vs=np.array([r[2] for r in rest],float); vt=np.array([r[3] for r in rest],float)
    return dict(evs=evs,evt=evt,em=em,ret=ret,t=tt,vs=vs,vt=vt,tok0=bt(evs,evt,Q*1e9))

def tp100t120(pp):
    if pp is None: return None
    xi=0
    for i in range(len(pp["ret"])):
        xi=i
        if pp["ret"][i]>=1.0 or pp["t"][i]>=120: break
    return sl(pp["vs"][xi],pp["vt"][xi],pp["tok0"])/(Q*1e9)-1.0-CB/1e4-2*FEE/Q

P={m:path(m) for m in mints}
NET={m:(tp100t120(P[m]) if P[m] else None) for m in mints}
PEAK=np.array([P[m]["ret"].max() if P[m] is not None and len(P[m]["ret"]) else -1.0 for m in mints])
valid=np.array([P[m] is not None for m in mints])
tr=valid&(TT<JUN10); te=valid&(TT>=JUN10)
netarr=np.array([NET[m] if NET[m] is not None else np.nan for m in mints])
print(f"ready mints with usable paths: train {tr.sum()} test {te.sum()}\n")

print("=== A) BETTER ENTRY: target choice, judged on deduped tp_100_t120 net @ top-3% ===")
targets={
  "peak>=2x (current)": (PEAK>=2.0).astype(int),
  "peak>=1x (exit-matched)": (PEAK>=1.0).astype(int),
  "exit-aligned net>=+0.2": (netarr>=0.20).astype(int),
}
for tname,y in targets.items():
    if y[tr].sum()<20 or y[te].sum()<10:
        print(f"  {tname:26s}: too few pos"); continue
    clf=HistGradientBoostingClassifier(**REG).fit(X[tr],y[tr])
    s=clf.predict_proba(X[te])[:,1]
    auc=roc_auc_score(y[te],s)
    thr=np.quantile(s,0.97)
    te_idx=np.where(te)[0]; fired=[(mints[i],s[k]) for k,i in enumerate(te_idx) if s[k]>=thr]
    nets=[NET[m] for m,_ in fired if NET[m] is not None]; scs=[sc for (m,sc) in fired if NET[m] is not None]
    dm,npat=dedup(scs,nets)
    print(f"  {tname:26s}: test_auc={auc:.3f}  fires={len(nets)}  DEDUP net={dm:+.3f} (n_pat={npat})")

print("\n=== B) TIME-CONDITIONED PRICE LADDER vs tp_100_t120 (deduped) ===")
LV=[0.5,1.0,2.0,4.0]; TB=[(0,30),(30,75),(75,150),(150,1e9)]
# empirical P(reach next level | first-crossed level L at time-bucket) from TRAIN
def tbucket(t):
    for k,(lo,hi) in enumerate(TB):
        if lo<=t<hi: return k
    return len(TB)-1
cont={}  # (level_idx, tbucket) -> [reached_next, total]
for m in [mm for mm,ok in zip(mints,tr) if ok]:
    pp=P[m]; r=pp["ret"]; tt=pp["t"]
    for li,L in enumerate(LV[:-1]):
        w=np.where(r>=L)[0]
        if not len(w): continue
        i=w[0]; tb=tbucket(tt[i]); key=(li,tb)
        reached_next=1 if (r[i:]>=LV[li+1]).any() else 0
        c=cont.setdefault(key,[0,0]); c[0]+=reached_next; c[1]+=1
Pc={k:(v[0]/v[1] if v[1] else 0.0) for k,v in cont.items()}
HOLD_THR=0.40  # hold the next tranche only if cond. continuation prob >= this

# need hazard for the runner cut
import pandas as pd
sn=pd.read_parquet(ROOT/"data/recovery_snaps_k3v03.parquet").sort_values(["mint","fwd_i"])
fmn=sn.groupby("mint")["ret"].transform(lambda s:s[::-1].cummin()[::-1].shift(-1))
sn=sn.assign(fut_min=fmn).dropna(subset=["fut_min"]); sn["collapse"]=(((1+sn.fut_min)/(1+sn.ret)-1)<=-0.40).astype(int)
P9=["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w","dts"]
hz=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,l2_regularization=2.0,random_state=0).fit(sn[sn.ready_ts<JUN10][P9].values, sn[sn.ready_ts<JUN10].collapse.values)
hzby={mm:hz.predict_proba(g[P9].values)[:,1] for mm,g in sn.groupby("mint")}

def ladder_net(m):
    pp=P[m]; r=pp["ret"]; tt=pp["t"]; n=len(r)
    if n==0: return None
    h=hzby.get(m); j=1
    haz=(h[j+1:j+1+n] if h is not None and len(h)>j+1 else np.zeros(n))
    haz=np.concatenate([haz,np.repeat(haz[-1] if len(haz) else 0.0,max(0,n-len(haz)))])[:n]
    ntr=len(LV); tok_each=pp["tok0"]/ntr; sold=[False]*ntr; proceeds=0.0; ntx=1
    for i in range(n):
        for k in range(ntr):
            if sold[k]: continue
            L=LV[k]
            if r[i]>=L:  # tranche k's target hit -> sell it
                proceeds+=sl(pp["vs"][i],pp["vt"][i],tok_each); sold[k]=True; ntx+=1; continue
            # not yet at this tranche's level: should we give up on it (sell now)?
            # give up if cond. continuation from the highest level reached is low, or hazard fires
            reached=[L2 for L2 in LV if r[i]>=L2]
            base_li = (LV.index(reached[-1]) if reached else None)
            if haz[i]>=0.60:
                proceeds+=sl(pp["vs"][i],pp["vt"][i],tok_each); sold[k]=True; ntx+=1; continue
            if base_li is not None and base_li<ntr-1:
                pc=Pc.get((base_li,tbucket(tt[i])),0.0)
                if pc<HOLD_THR and tt[i]>=30:  # unlikely to climb further -> take it
                    proceeds+=sl(pp["vs"][i],pp["vt"][i],tok_each); sold[k]=True; ntx+=1
        if all(sold): break
    for k in range(ntr):
        if not sold[k]: proceeds+=sl(pp["vs"][-1],pp["vt"][-1],tok_each); ntx+=1
    return proceeds/(Q*1e9)-1.0-CB/1e4-ntx*FEE/Q

tem=[mm for mm,ok in zip(mints,te) if ok and M[mm]["feats"] is not None]
# evaluate on the TEST fires of the deployed model (score>=0.5) for apples-to-apples
clf0=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
s0=clf0.predict_proba(X)[:,1]
fire_te=[mints[i] for i in np.where(te)[0] if s0[i]>=0.50]
for lab,fn in [("tp_100_t120", tp100t120_wrap:=lambda m: tp100t120(P[m])),("time-cond ladder", ladder_net)]:
    nets=[fn(m) for m in fire_te if P[m] is not None]; scs=[s0[mints.index(m)] for m in fire_te if P[m] is not None]
    nets2=[x for x in nets if x is not None]
    dm,npat=dedup(scs[:len(nets2)],nets2)
    print(f"  {lab:18s} DEDUP={dm:+.3f} (n_pat={npat}, n={len(nets2)})")
