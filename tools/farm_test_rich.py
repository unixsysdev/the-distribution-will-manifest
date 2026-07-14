#!/usr/bin/env python3
"""farm_test_rich.py — repeat the farm-independence test for the RICH model,
head-to-head with 22-feat, on the SAME novel/organic token split.

Join: candidates.parquet (CLEAN rich features, by mint) + sweep_k3v03.pkl
(forward paths + the 22-feat vector used as the physical farm signature).
Farm cluster = exact-repeat 22-feat vector (same physical scripted launch for
BOTH models, so 'novel' is identical -> fair comparison).

Tests, train Jun7-9 / test Jun10-11:
 P2 detection AUC(reach +100%) on NOVEL vs SEEN patterns, RICH vs 22feat.
 P3 organic-only (unique) tp_100_t120 net for each model's own top fires.
Clean rich feature list from bot_artifacts_rich_shadow (leak cols excluded).
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
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
def auc_ci(yy,ss,B=1500):
    if len(yy)<30 or yy.min()==yy.max(): return None
    base=roc_auc_score(yy,ss); bs=[]
    for _ in range(B):
        bi=RNG.integers(0,len(yy),len(yy))
        if yy[bi].min()!=yy[bi].max(): bs.append(roc_auc_score(yy[bi],ss[bi]))
    return base,np.percentile(bs,5),np.percentile(bs,95)

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); M=d["mints"]; mints=list(M)
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=frr(fs[0]),frr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps

# rich features (clean) from candidates
cand=pd.read_parquet(ROOT/"data/rich_crossday_20260610/candidates.parquet")
cand=cand[(cand.k==3)&(cand.v_sol==0.3)].set_index("mint")
rich_feats=[c for c in json.load(open(ROOT/"bot_artifacts_rich_shadow/model_spec.json"))["entry"]["features"] if c in cand.columns]

# physical farm signature = exact 22-feat vector
vk={m:tuple(np.round(M[m]["feats"],6)) for m in mints}
vc=Counter(vk.values())

# common mints (in both rich candidates and sweep paths)
common=[m for m in mints if m in cand.index]
TT={m:s2t(M[m]["decision"]["slot"]) for m in common}
PEAK={m:(max((r[2]/r[3] for r in M[m]["fwd"]),default=0)/(M[m]["decision"]["vsol"]/M[m]["decision"]["vtok"])-1) if M[m]["fwd"] else -1 for m in common}
common=[m for m in common if PEAK[m]>-1]
tr=[m for m in common if TT[m]<JUN10]; te=[m for m in common if TT[m]>=JUN10]
print(f"common mints (rich+paths): {len(common)}  train {len(tr)} test {len(te)}")

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

yall={m:int(PEAK[m]>=1.0) for m in common}
train_vecs=set(vk[m] for m in tr)
novel_te=np.array([vk[m] not in train_vecs for m in te])
ytr=np.array([yall[m] for m in tr]); yte=np.array([yall[m] for m in te])

FEATS={"22feat":[np.array([M[m]["feats"] for m in tr]), np.array([M[m]["feats"] for m in te])],
       "RICH":[cand.loc[tr,rich_feats].values, cand.loc[te,rich_feats].values]}
print(f"\n=== P2 DETECTION AUC(reach+100%), train Jun7-9 / test Jun10-11 ===")
for name,(Xtr,Xte) in FEATS.items():
    clf=HistGradientBoostingClassifier(**REG).fit(Xtr,ytr); s=clf.predict_proba(Xte)[:,1]
    print(f"  --- {name} ({Xtr.shape[1]} feats)")
    for lab,mask in [("ALL",np.ones(len(te),bool)),("NOVEL",novel_te),("SEEN",~novel_te)]:
        r=auc_ci(yte[mask],s[mask])
        print(f"      {lab:6s} n={mask.sum():5d} pos={int(yte[mask].sum()):4d}  "+(f"AUC={r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]" if r else "(too few)"))

print(f"\n=== P3 organic-only tp_100_t120 net, each model's top-3% test fires ===")
for name,(Xtr,Xte) in FEATS.items():
    clf=HistGradientBoostingClassifier(**REG).fit(Xtr,ytr); s=clf.predict_proba(Xte)[:,1]
    thr=np.quantile(s,0.97)
    fired=[te[k] for k in range(len(te)) if s[k]>=thr]
    org=[m for m in fired if vc[vk[m]]==1]
    for lab,sub in [("all fires",fired),("ORGANIC only",org)]:
        nets=[net_tp100t120(m) for m in sub]; nets=[x for x in nets if x is not None]
        if len(nets)<2: print(f"  {name} {lab:14s}: n={len(nets)} (too few)"); continue
        mu=np.mean(nets); bs=[np.mean(np.array(nets)[RNG.integers(0,len(nets),len(nets))]) for _ in range(2000)]
        print(f"  {name} {lab:14s}: n={len(nets):3d} mean={mu:+.3f} 90%CI[{np.percentile(bs,5):+.3f},{np.percentile(bs,95):+.3f}] win={np.mean([x>0 for x in nets]):.0%}")
print("\nverdict: does RICH novel-AUC exceed 22feat's 0.853? and is rich organic-net CI higher/cleaner?")
