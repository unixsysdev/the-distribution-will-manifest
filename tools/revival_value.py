#!/usr/bin/env python3
"""revival_value.py — does the 300s/event-driven design earn its keep vs a flat
120s WALL-CLOCK cap? The only tokens where they differ: a snap gap straddling
120s (token quiet before 120, then a trade at/after 120).
  CURRENT (deployed): stop30 + EVENT cap (first trade at t>=120) + stale ride.
                      -> sells INTO the post-120 trade at its fresh price.
  WALLCLOCK120 (user):exit at earliest of TP / -30% stop / last snap with t<=120.
                      -> never acts on a post-120 trade; closes at frozen pre-120 mark.
diff = CURRENT - WALLCLOCK = the revival option's value (can be +pump or -dump).
OOS Jun10-11, lat1, fees, deduped by pattern.
"""
import calendar, glob, gzip, json, pickle
from pathlib import Path
import numpy as np

ROOT=Path("/root/the-distribution-will-manifest"); JUN10=calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE=0.1,250.0,0.0015; RNG=np.random.default_rng(5)
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
    def out(s,i): return sl(s.vs[i],s.vt[i],s.tok)/(Q*1e9)-1-CB/1e4-2*FEE/Q

fires=[(m,S[i]) for i,m in enumerate(mints) if S[i]>=0.50 and M[m]["fwd"] and len(M[m]["fwd"])>2 and TT[i]>=JUN10]
F={m:Fire(m) for m,_ in fires}

def net(m, allow_post120):
    f=F[m]; n=len(f.ret)
    if n==0: return -CB/1e4-2*FEE/Q, "empty"
    last_le=None
    for i in range(n):
        if f.ret[i]>=0.5: return f.out(i), "tp"
        if f.ret[i]<=-0.30: return f.out(i), "stop"
        if f.t[i]>=120:
            if allow_post120: return f.out(i), "evt_cap_post120"      # sell into the post-120 trade
            return (f.out(last_le) if last_le is not None else f.out(i)), "wallclock_frozen"
        last_le=i
    return f.out(n-1), "stale_lastsnap"                              # rode to end (same both rules)

cur=[net(m,True)[0]  for m,_ in fires]
wc =[net(m,False)[0] for m,_ in fires]
scs=[sc for _,sc in fires]
print(f"OOS Jun10-11 fires={len(fires)}  (deduped by pattern)\n")
for lab,arr in [("CURRENT (event-cap + stale ride)",cur),("WALLCLOCK120 (your rule, frozen pre-120)",wc)]:
    mu,lo,hi,npat=dedup(scs,arr); print(f"  {lab:42s} {mu:+.3f} [{lo:+.3f},{hi:+.3f}] np={npat}")

# isolate the ONLY tokens where the rules diverge: a post-120 trade preceded by a pre-120 snap
div=[]
for m,_ in fires:
    f=F[m]
    if F[m].ret.size==0: continue
    # do current & wallclock disagree?
    if abs(net(m,True)[0]-net(m,False)[0])>1e-9:
        # gap size at the divergence + pump/dump sign
        last_le=None; post=None
        for i in range(len(f.ret)):
            if f.ret[i]>=0.5 or f.ret[i]<=-0.30: break
            if f.t[i]>=120: post=i; break
            last_le=i
        if post is not None and last_le is not None:
            gap=f.t[post]-f.t[last_le]; dpnl=f.out(post)-f.out(last_le)
            div.append((m,gap,dpnl))
print(f"\n  tokens where rules DIVERGE (quiet-then-trade straddling 120s): {len(div)} / {len(fires)}")
if div:
    gaps=np.array([g for _,g,_ in div]); dp=np.array([d for _,_,d in div])
    print(f"    silence-gap straddling 120s: median {np.median(gaps):.0f}s  max {gaps.max():.0f}s")
    print(f"    revival value (CURRENT-WALLCLOCK) on these: sum={dp.sum()*Q:+.4f} SOL  mean={dp.mean():+.3f}/fire")
    print(f"    of these: {int((dp>0).sum())} pumped after 120 (current wins), {int((dp<0).sum())} dumped (wallclock wins)")
print("\nread: if CURRENT~=WALLCLOCK and revival-value~0, the 300s buys nothing -> a flat 120s")
print("age cap is the cleaner rule (P&L-neutral, recycles the slot at 120s not 300s).")
