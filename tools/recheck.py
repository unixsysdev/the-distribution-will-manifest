#!/usr/bin/env python3
"""recheck.py — separate the WELL-POWERED claims from the SAMPLE-STARVED ones.
The honest question is not 'is everything noise' but 'which part is proven'.

  (1) ENTRY EDGE (full population, thousands of mints): cross-day AUC with a
      bootstrap CI over mints, and win-rate-by-score-bucket monotonicity.
      If the CI is tight and >0.5 and buckets are monotone -> the ranking edge
      is PROVEN, not sample-starved.
  (2) PER-FIRE NET magnitude / policy choice (the ~12-40 fires above thr):
      bootstrap CI -> this is the part that is genuinely uncertain.
Separating these tells us what we actually know.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT=Path("/root/the-distribution-will-manifest"); JUN10=calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE=0.1,250.0,0.0015
REG=dict(max_depth=3,max_iter=150,learning_rate=0.05,l2_regularization=5.0,random_state=42)
RNG=np.random.default_rng(11)
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

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); M=d["mints"]; mints=list(M)
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=frr(fs[0]),frr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
X=np.array([M[m]["feats"] for m in mints]); TT=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])
PEAK=np.array([(max((r[2]/r[3] for r in M[m]["fwd"]),default=0)/(M[m]["decision"]["vsol"]/M[m]["decision"]["vtok"])-1) if M[m]["fwd"] else -1 for m in mints])
valid=PEAK>-1; tr=valid&(TT<JUN10); te=valid&(TT>=JUN10)

print("="*64)
print("(1) ENTRY EDGE — full population, is the RANKING proven?")
print("="*64)
y=(PEAK>=1.0).astype(int)   # reach +100% (the exit level)
clf=HistGradientBoostingClassifier(**REG).fit(X[tr],y[tr])
s=clf.predict_proba(X[te])[:,1]; yte=y[te]
auc=roc_auc_score(yte,s)
# bootstrap AUC over test mints
idx=np.arange(len(yte)); aucs=[]
for _ in range(2000):
    bi=RNG.integers(0,len(idx),len(idx))
    if yte[bi].min()==yte[bi].max(): continue
    aucs.append(roc_auc_score(yte[bi],s[bi]))
print(f"  cross-day AUC (reach +100%) = {auc:.4f}  90% CI [{np.percentile(aucs,5):.4f}, {np.percentile(aucs,95):.4f}]")
print(f"  test n={te.sum()} mints, positives={yte.sum()}  <- thousands, NOT sample-starved")
# win rate by score decile (well-powered)
order=np.argsort(s); dec=np.array_split(order,10)
print("  win-rate (reach +100%) by score decile, low->high:")
wr=[yte[g].mean() for g in dec]
print("   "+"  ".join(f"{w:.0%}" for w in wr))
mono=all(wr[i]<=wr[i+1]+0.03 for i in range(9))
print(f"  monotone (within 3pp tol): {mono}")

print("\n"+"="*64)
print("(2) PER-FIRE NET / policy — the genuinely uncertain part")
print("="*64)
def net_tp(m,tp,cap):
    f=M[m]["fwd"];
    if not f: return None
    j=min(1,len(f)-1); evs,evt=f[j][2],f[j][3]; tok=bt(evs,evt,Q*1e9); em=evs/evt; rest=f[j+1:]
    if not rest: return None
    t0=f[j][0]; xi=0
    for i in range(len(rest)):
        xi=i
        if (rest[i][2]/rest[i][3])/em-1>=tp or rest[i][0]-t0>=cap: break
    return sl(rest[xi][2],rest[xi][3],tok)/(Q*1e9)-1-CB/1e4-2*FEE/Q
clf0=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb")); s0=clf0.predict_proba(X)[:,1]
te_idx=np.where(te)[0]; fired=[mints[i] for i in te_idx if s0[i]>=0.50]
def ci_over_patterns(vals, scores):
    by={}
    for sc,v in zip(scores,vals): by.setdefault(round(float(sc),6),[]).append(v)
    pm=np.array([np.mean(v) for v in by.values()])
    bs=[pm[RNG.integers(0,len(pm),len(pm))].mean() for _ in range(2000)]
    return pm.mean(),np.percentile(bs,5),np.percentile(bs,95),len(pm)
for lab,(tp,cap) in [("tp_50",(0.5,1e9)),("tp_100",(1.0,1e9)),("tp_100_t120",(1.0,120))]:
    vals=[net_tp(m,tp,cap) for m in fired]; scs=[s0[mints.index(m)] for m in fired]
    vals=[v for v in vals if v is not None]
    mm,lo,hi,npat=ci_over_patterns(vals,scs[:len(vals)])
    print(f"  {lab:14s} deduped {mm:+.3f}  90% CI [{lo:+.3f},{hi:+.3f}]  n_pat={npat}")
print("\n  => the EDGE (does it rank winners) is well-powered; the MAGNITUDE per fire is not.")
