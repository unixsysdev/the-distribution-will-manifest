#!/usr/bin/env python3
"""exit_followup.py — answer the follow-ups:
 A) hazard AUC: 9 path feats vs +11 K feats (headroom from more features).
 B) 50%->100% conditional transition time, and the "grab the 50% if +100%
    now unlikely" refinement vs tp_100_t120.
 C) does the hazard head SUBSUME the time cap? (hazard-gated exit vs time cap).
Design Jun7-9, one test look Jun10-11, deduped, lat1, fees.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE = 0.1,250.0,0.0015
THR=0.50
P9=["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w","dts"]
K11=["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol","single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]

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

# ---- A: hazard feature ablation ----
sn=pd.read_parquet(ROOT/"data/recovery_snaps_k3v03.parquet").sort_values(["mint","fwd_i"])
fmn=sn.groupby("mint")["ret"].transform(lambda s:s[::-1].cummin()[::-1].shift(-1))
sn=sn.assign(fut_min=fmn).dropna(subset=["fut_min"])
sn["collapse"]=(((1+sn.fut_min)/(1+sn.ret)-1)<=-0.40).astype(int)
tr=sn[sn.ready_ts<JUN10]; teh=sn[sn.ready_ts>=JUN10]
print("=== A) collapse-hazard AUC by feature set (test Jun10-11) ===")
for lab,feats in [("9 path only",P9),("9 path + 11 K",P9+K11)]:
    m=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,l2_regularization=2.0,random_state=0).fit(tr[feats].values,tr.collapse.values)
    auc=roc_auc_score(teh.collapse.values,m.predict_proba(teh[feats].values)[:,1])
    print(f"  {lab:16s} AUC={auc:.4f}")
# keep the path9 hazard model for part C
hz=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,l2_regularization=2.0,random_state=0).fit(tr[P9].values,tr.collapse.values)
hz_by={}
for m,g in sn.groupby("mint"):
    hz_by[m]=hz.predict_proba(g[P9].values)[:,1]

# ---- load fires ----
d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); names,M=d["names"],d["mints"]
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=fr(fs[0]),fr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
clf=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
mints=list(M); sc=clf.predict_proba(np.array([M[m]["feats"] for m in mints]))[:,1]
tt=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])
class Fire:
    def __init__(s,m,score,f,haz,lat=1):
        s.mint=m; s.score=score
        j=min(lat,len(f)-1); s.evs,s.evt=f[j][2],f[j][3]; s.tok0=bt(s.evs,s.evt,Q*1e9); em=s.evs/s.evt
        rest=f[j+1:]; s.t=np.array([r[0]-f[j][0] for r in rest])
        s.vs=np.array([r[2] for r in rest],float); s.vt=np.array([r[3] for r in rest],float)
        s.ret=(s.vs/s.vt)/em-1.0
        s.mnet=np.array([sl(s.vs[i],s.vt[i],s.tok0)/(Q*1e9)-1-CB/1e4-2*FEE/Q for i in range(len(rest))])
        # align hazard (decision-anchored, shift to lat1)
        h=haz.get(m);
        if h is not None and len(h)>j:
            hh=h[j+1:]; s.haz=np.concatenate([hh,np.repeat(hh[-1] if len(hh) else 0.0,max(0,len(s.ret)-len(hh)))])[:len(s.ret)]
        else: s.haz=np.zeros(len(s.ret))
mk=lambda m,score,f: Fire(m,score,f,hz_by)
TR=[mk(m,s,M[m]["fwd"]) for m,s,t in zip(mints,sc,tt) if s>=THR and M[m]["fwd"] and t<JUN10]
TE=[mk(m,s,M[m]["fwd"]) for m,s,t in zip(mints,sc,tt) if s>=THR and M[m]["fwd"] and t>=JUN10]

def tpi(f,lvl):
    w=np.where(f.ret>=lvl)[0]; return int(w[0]) if len(w) else None
def at(f,i):
    if len(f.mnet)==0: return -CB/1e4-2*FEE/Q
    if i is None or i>=len(f.mnet) or i<0: return f.mnet[-1]
    return f.mnet[i]

# ---- B: 50->100 transition time ----
print("\n=== B) 50%->100% transition timing (train fires) ===")
trans=[]
for f in TR:
    i50=tpi(f,0.5); i100=tpi(f,1.0)
    if i50 is not None and i100 is not None and i100>=i50:
        trans.append(f.t[i100]-f.t[i50])
hit50=[f for f in TR if tpi(f,0.5) is not None]
on=[f for f in hit50 if tpi(f,1.0) is not None]
print(f"  of fires hitting +50% (n={len(hit50)}): {len(on)} ({len(on)/max(len(hit50),1):.0%}) go on to +100%")
if trans: print(f"  conditional 50->100 time: p50={np.percentile(trans,50):.0f}s p75={np.percentile(trans,75):.0f}s p90={np.percentile(trans,90):.0f}s")
Tgive=np.percentile(trans,75) if trans else 60.0

# ---- policy comparison ----
def sim(f, kind):
    # returns net under a policy
    i100=tpi(f,1.0); i50=tpi(f,0.5)
    n=len(f.mnet)
    if n==0: return -CB/1e4-2*FEE/Q
    if kind=="tp100_t120":
        cap=np.where(f.t>=120)[0]; ci=int(cap[0]) if len(cap) else n-1
        if i100 is not None and i100<=ci: return at(f,i100)
        return at(f,ci)
    if kind=="tp100_grab50":
        # +100 sell; elif at >=50 and held past Tgive-from-50, grab; else 120 cap
        for i in range(n):
            if f.ret[i]>=1.0: return at(f,i)
            if i50 is not None and i>=i50 and (f.t[i]-f.t[i50])>=Tgive and f.ret[i]>=0.5: return at(f,i)
            if f.t[i]>=120: return at(f,i)
        return at(f,n-1)
    if kind=="tp100_hazard":
        for i in range(n):
            if f.ret[i]>=1.0: return at(f,i)
            if f.haz[i]>=0.60: return at(f,i)   # collapse predicted -> cut
        return at(f,n-1)
    if kind=="tp100_hazard_grab50":
        for i in range(n):
            if f.ret[i]>=1.0: return at(f,i)
            if f.haz[i]>=0.60: return at(f,i)
            if i50 is not None and i>=i50 and (f.t[i]-f.t[i50])>=Tgive and f.ret[i]>=0.5: return at(f,i)
        return at(f,n-1)
    return at(f, i100)
POL=["tp100_t120","tp100_grab50","tp100_hazard","tp100_hazard_grab50"]
print("\n=== C) policy comparison (TEST Jun10-11 deduped) ===")
for kind in POL:
    nets=[sim(f,kind) for f in TE]; dm,npat=dedup([f.score for f in TE],nets)
    print(f"  {kind:22s} DEDUP={dm:+.3f} raw={np.mean(nets):+.3f} (n_pat={npat})")
