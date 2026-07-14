#!/usr/bin/env python3
"""farm_test.py — is there a strategy WITHOUT the farms?

A farm = a scripted actor replaying near-identical launches -> byte-identical
feature vectors -> identical model score (we saw 0.513838 x31). So:
  repeat-cluster (farm) = a round(score,6) value shared by >1 mint
  unique/organic        = a score seen exactly once

Decisive tests:
 P1 prevalence: what fraction of FIRES and of the ready universe are farm.
 P2 DETECTION on NOVEL patterns: train Jun7-9, score Jun10-11; split test into
    patterns SEEN in train vs NOVEL (new actors). AUC(reach+100%) on each.
    If novel-AUC stays high -> the edge GENERALIZES beyond known farms.
 P3 P&L on UNIQUE (non-farm) fires only: deduped == raw for singletons.
    Is the organic-only net positive, and is n even enough to tell?
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT=Path("/root/the-distribution-will-manifest"); JUN10=calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE=0.1,250.0,0.0015
REG=dict(max_depth=3,max_iter=150,learning_rate=0.05,l2_regularization=5.0,random_state=42)
RNG=np.random.default_rng(3)
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

def net_tp100t120(m):
    f=M[m]["fwd"]
    if not f: return None
    j=min(1,len(f)-1); evs,evt=f[j][2],f[j][3]; tok=bt(evs,evt,Q*1e9); em=evs/evt; rest=f[j+1:]
    if not rest: return None
    t0=f[j][0]; xi=0
    for i in range(len(rest)):
        xi=i
        if (rest[i][2]/rest[i][3])/em-1>=1.0 or rest[i][0]-t0>=120: break
    return sl(rest[xi][2],rest[xi][3],tok)/(Q*1e9)-1-CB/1e4-2*FEE/Q

# cluster by exact feature vector (the true farm signature, not just score)
def vkey(m): return tuple(np.round(M[m]["feats"],6))
vk=[vkey(m) for m in mints]
from collections import Counter
vc=Counter(vk)
is_repeat=np.array([vc[vkey(m)]>1 for m in mints])

# deployed scores
clf0=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb")); S0=clf0.predict_proba(X)[:,1]
fire=valid&(S0>=0.50)

print("=== P1: FARM PREVALENCE ===")
print(f"  ready universe: {valid.sum()} mints, {is_repeat[valid].mean():.0%} in a repeat-cluster (farm)")
print(f"  FIRES (score>=0.5): {fire.sum()}; farm-cluster {is_repeat[fire].mean():.0%}, unique {(~is_repeat[fire]).mean():.0%}")
print(f"  distinct vectors among fires: {len(set(vk[i] for i in range(len(mints)) if fire[i]))}")

print("\n=== P2: DETECTION on NOVEL patterns (train Jun7-9, test Jun10-11) ===")
y=(PEAK>=1.0).astype(int)
clf=HistGradientBoostingClassifier(**REG).fit(X[tr],y[tr])
s=clf.predict_proba(X[te])[:,1]
train_vecs=set(vk[i] for i in range(len(mints)) if tr[i])
te_i=np.where(te)[0]
novel=np.array([vk[i] not in train_vecs for i in te_i])
def auc_ci(yy,ss):
    if yy.min()==yy.max() or len(yy)<30: return None
    base=roc_auc_score(yy,ss); bs=[]
    for _ in range(1500):
        bi=RNG.integers(0,len(yy),len(yy))
        if yy[bi].min()!=yy[bi].max(): bs.append(roc_auc_score(yy[bi],ss[bi]))
    return base,np.percentile(bs,5),np.percentile(bs,95)
yt=y[te]
for lab,mask in [("ALL test",np.ones(len(te_i),bool)),("NOVEL patterns",novel),("SEEN patterns",~novel)]:
    yy=yt[mask]; ss=s[mask]
    r=auc_ci(yy,ss)
    n=mask.sum(); pos=yy.sum() if len(yy) else 0
    if r: print(f"  {lab:16s} n={n:5d} pos={pos:4d}  AUC={r[0]:.3f} 90%CI[{r[1]:.3f},{r[2]:.3f}]")
    else: print(f"  {lab:16s} n={n:5d} pos={pos:4d}  (too few to score)")

print("\n=== P3: P&L on UNIQUE (non-farm) fires only ===")
for lab,mask in [("ALL fires",fire),("FARM fires",fire&is_repeat),("UNIQUE/organic fires",fire&~is_repeat)]:
    idx=np.where(mask)[0]; nets=[net_tp100t120(mints[i]) for i in idx]; nets=[x for x in nets if x is not None]
    if not nets: print(f"  {lab:22s}: no fires"); continue
    mu=np.mean(nets);
    bs=[np.mean(np.array(nets)[RNG.integers(0,len(nets),len(nets))]) for _ in range(2000)] if len(nets)>=2 else [mu]
    print(f"  {lab:22s}: n={len(nets):4d}  mean net={mu:+.3f}  90%CI[{np.percentile(bs,5):+.3f},{np.percentile(bs,95):+.3f}]  win={np.mean([x>0 for x in nets]):.0%}")
print("\nverdict: if NOVEL-pattern AUC stays high AND unique-fire net CI clears 0 -> real edge beyond farms.")
print("if unique fires are too few or net~0 -> the measured edge is farm-dependent (the concern is valid).")
