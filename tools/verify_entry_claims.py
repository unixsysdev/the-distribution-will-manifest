#!/usr/bin/env python3
"""verify_entry_claims.py — put error bars on tonight's entry conclusions.
Bootstrap CIs over DISTINCT PATTERNS (resample patterns, recompute deduped
mean) so 'inconclusive' is quantified, not asserted. Clean score<->net pairing.
Also re-confirms the tp_100_t120 OOS profit number across an independent path.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/the-distribution-will-manifest")
JUN10 = calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE = 0.1,250.0,0.0015
REG=dict(max_depth=3,max_iter=150,learning_rate=0.05,l2_regularization=5.0,random_state=42)
RNG=np.random.default_rng(7)

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

def dedup_groups(scores, nets):
    by={}
    for s,x in zip(scores,nets): by.setdefault(round(float(s),6),[]).append(x)
    return [np.mean(v) for v in by.values()]

def boot_ci(pattern_means, B=2000):
    pm=np.array(pattern_means)
    if len(pm)<2: return (float(pm.mean()), float("nan"), float("nan"))
    bs=[pm[RNG.integers(0,len(pm),len(pm))].mean() for _ in range(B)]
    return float(pm.mean()), float(np.percentile(bs,5)), float(np.percentile(bs,95))

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); M=d["mints"]; mints=list(M)
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=frr(fs[0]),frr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
X=np.array([M[m]["feats"] for m in mints]); TT=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])

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

NET={m:net_tp100t120(m) for m in mints}
PEAK=np.array([(max((r[2]/r[3] for r in M[m]["fwd"]),default=0)/(M[m]["decision"]["vsol"]/M[m]["decision"]["vtok"])-1) if M[m]["fwd"] else -1 for m in mints])
valid=np.array([NET[m] is not None for m in mints]); tr=valid&(TT<JUN10); te=valid&(TT>=JUN10)

# sanity: independent recompute of OOS tp_100_t120 deduped (should ~match compute_profit +0.170)
te_idx=np.where(te)[0]
clf0=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
s0=clf0.predict_proba(X)[:,1]
fired0=[(mints[i],s0[i]) for i in te_idx if s0[i]>=0.50]
gm=dedup_groups([sc for _,sc in fired0],[NET[m] for m,_ in fired0])
m_,lo_,hi_=boot_ci(gm)
print(f"SANITY deployed tp_100_t120 OOS deduped = {m_:+.3f} [90% CI {lo_:+.3f},{hi_:+.3f}] n_pat={len(gm)}")
print("  (should ~match compute_profit's +0.170; confirms the profit pipeline)\n")

print("ENTRY TARGET, deduped tp_100_t120 net with 90% bootstrap CI over patterns:")
for tname,y in [("peak>=2x (current)",(PEAK>=2).astype(int)),("peak>=1x (exit-matched)",(PEAK>=1).astype(int))]:
    clf=HistGradientBoostingClassifier(**REG).fit(X[tr],y[tr]); s=clf.predict_proba(X[te])[:,1]
    auc=roc_auc_score(y[te],s)
    print(f"  --- {tname}  test_auc={auc:.3f}")
    for pct in (3.0,1.5,0.7):
        thr=np.quantile(s,1-pct/100)
        fired=[(mints[te_idx[k]],s[k]) for k in range(len(te_idx)) if s[k]>=thr]
        nn=[NET[m] for m,_ in fired if NET[m] is not None]; ss=[sc for m,sc in fired if NET[m] is not None]
        gm=dedup_groups(ss,nn); mm,lo,hi=boot_ci(gm)
        print(f"      top-{pct}%: fires={len(nn):3d} n_pat={len(gm):2d}  DEDUP={mm:+.3f} [90% CI {lo:+.3f},{hi:+.3f}]")
print("\nverdict logic: peak>=1x BEATS peak>=2x only if its CI lower bound exceeds the other's mean.")
