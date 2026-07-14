#!/usr/bin/env python3
"""compute_profit.py — tp_100_t120 profitability under AVERAGE execution (lat1,
the live-reconciled landing) on ALL fired data we have (Jun7-11), plus the
OOS slice (Jun10-11). Fractional + absolute SOL at 0.1 bet + daily extrapolation.
Raw (what the book actually banks incl. farm repeats) AND deduped (what
generalizes). Per-tranche fees, 250bps cost.
"""
import calendar, glob, gzip, json, pickle, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026,6,10,0,0,0))
Q,CB,FEE = 0.1,250.0,0.0015
THR=0.50

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

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); M=d["mints"]
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=fr(fs[0]),fr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
clf=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
mints=list(M); SC=clf.predict_proba(np.array([M[m]["feats"] for m in mints]))[:,1]
TT=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])

def net_tp100_t120(f, lat=1):
    if not f: return None
    j=min(lat,len(f)-1); evs,evt=f[j][2],f[j][3]; tok=bt(evs,evt,Q*1e9); em=evs/evt
    rest=f[j+1:]
    if not rest: return None
    t0=f[j][0]; xi=0
    for i in range(len(rest)):
        r=(rest[i][2]/rest[i][3])/em-1.0; xi=i
        if r>=1.0 or (rest[i][0]-t0)>=120: break
    return sl(rest[xi][2],rest[xi][3],tok)/(Q*1e9)-1.0-CB/1e4-2*FEE/Q

fires=[(m,s,t) for m,s,t in zip(mints,SC,TT) if s>=THR and M[m]["fwd"]]
def report(label, sub):
    nets=[]; scs=[]; days={}
    for m,s,t in sub:
        nv=net_tp100_t120(M[m]["fwd"])
        if nv is None: continue
        nets.append(nv); scs.append(s)
        dk=time.strftime("%m-%d", time.gmtime(t)); days[dk]=days.get(dk,0)+1
    if not nets: print(f"{label}: no fires"); return
    raw=float(np.mean(nets)); ded,npat=dedup(scs,nets); win=float(np.mean([x>0 for x in nets]))
    p25=float(np.percentile(nets,25)); tot=float(np.sum(nets))
    print(f"\n=== {label}: {len(nets)} fires, {npat} distinct patterns ===")
    print(f"  per-fire net:  raw {raw:+.3f}  |  deduped {ded:+.3f}  | win {win:.0%}  p25 {p25:+.3f}")
    print(f"  per-fire SOL (bet {Q}):  raw {raw*Q:+.4f}  |  deduped {ded*Q:+.4f}")
    print(f"  window total (raw, what the book banks): {tot*Q:+.3f} SOL over {len(days)} days {sorted(days)}")
    nd=max(len(days),1)
    print(f"  ~per day: {len(nets)/nd:.0f} fires -> RAW {tot*Q/nd:+.3f} SOL/day"
          f"  |  generalizing (deduped x patterns/day {npat/nd:.0f}): {ded*Q*npat/nd:+.3f} SOL/day")

report("ALL FIRED DATA Jun7-11 (entry Jun7-8 in-sample)", fires)
report("OOS Jun10-11 only (clean forward)", [(m,s,t) for m,s,t in fires if t>=JUN10])
print("\nnote: lat1 == average/live-reconciled execution (live bot realized +0.151/fire ~ this).")
print("raw includes launch-farm repetition (real income while the farm runs); deduped is the")
print("conservative per-independent-pattern expectation. Both are paper/dry-run.")
