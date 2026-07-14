#!/usr/bin/env python3
"""farm_curation.py — should we avoid farms, trade all, or CURATE?
Per-repeated-pattern realized tp_100_t120 P&L, then a CAUSAL (past-only)
reputation rule vs the two extremes. Farm = exact-repeat 22-feat vector.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT=Path("/root/the-distribution-will-manifest"); Q,CB,FEE=0.1,250.0,0.0015
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
clf=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
X=np.array([M[m]["feats"] for m in mints]); S=clf.predict_proba(X)[:,1]
def net(m):
    f=M[m]["fwd"]
    if not f: return None
    j=min(1,len(f)-1); evs,evt=f[j][2],f[j][3]; tok=bt(evs,evt,Q*1e9); em=evs/evt; rest=f[j+1:]
    if not rest: return None
    t0=f[j][0]; xi=0
    for i in range(len(rest)):
        xi=i
        if (rest[i][2]/rest[i][3])/em-1>=1.0 or rest[i][0]-t0>=120: break
    return sl(rest[xi][2],rest[xi][3],tok)/(Q*1e9)-1-CB/1e4-2*FEE/Q
vk={m:tuple(np.round(M[m]["feats"],6)) for m in mints}
vc=Counter(vk.values())
fires=[(s2t(M[m]["decision"]["slot"]),m) for i,m in enumerate(mints) if S[i]>=0.50 and M[m]["fwd"] and net(m) is not None]
fires.sort()
NET={m:net(m) for _,m in fires}

print("=== per-REPEATED-pattern realized tp_100_t120 (farms, n>=2 fires) ===")
byp=defaultdict(list)
for t,m in fires: byp[vk[m]].append(NET[m])
rep=[(k,v) for k,v in byp.items() if len(v)>=2]
rep.sort(key=lambda kv:-np.mean(kv[1]))
print(f"  {len(rep)} repeated patterns; mean range {np.mean(rep[0][1]):+.2f} .. {np.mean(rep[-1][1]):+.2f}")
for k,v in rep:
    print(f"    n={len(v):3d}  mean={np.mean(v):+.3f}  win={np.mean([x>0 for x in v]):.0%}")

print("\n=== three policies (total + per-fire net) ===")
# A trade all
alln=[NET[m] for _,m in fires]
# B avoid all farms (organic only, n==1 vector)
orgn=[NET[m] for _,m in fires if vc[vk[m]]==1]
# C CAUSAL curate: fire a pattern unless its PRIOR realized mean<0 after >=K prior fires
K=2; run=defaultdict(lambda:[0,0.0]); cur=[]
for t,m in fires:
    key=vk[m]; c=run[key]
    blocked = (c[0]>=K and c[1]/c[0] < 0.0)
    if not blocked: cur.append(NET[m])
    c[0]+=1; c[1]+=NET[m]   # update reputation AFTER (causal)
for lab,arr in [("A trade ALL",alln),("B avoid ALL farms (organic)",orgn),("C CURATE (block pattern after >=2 priors avg<0)",cur)]:
    print(f"  {lab:46s} n={len(arr):3d}  per-fire={np.mean(arr):+.3f}  TOTAL={np.sum(arr)*Q:+.3f} SOL")
print("\nread: if C's TOTAL >= A and per-fire >= A, curation beats trading-all; if B<A, avoiding-all loses money.")
