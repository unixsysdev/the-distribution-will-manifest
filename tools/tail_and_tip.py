#!/usr/bin/env python3
"""tail_and_tip.py — four questions, one study. Deployed pkl fires, deduped,
design Jun7-9 / one test look Jun10-11, fees, lat conventions explicit.

 P0 tp_100_t120 under the LANDING DISTRIBUTION (Model-B slot-aware, our_tip
    swept), not a fixed lat — the honest realistic net of the shipped policy.
 P1 TAIL DETECTION: base rate of +200/+500/+1000% among fires; and does the
    ENTRY model separate the extreme tail (cross-day AUC for peak>=5x,>=10x,
    well-powered on ALL ready mints, not just the 39 fires)?
 P2 MOONBAG sleeve (the mu>=h*loss rule): core (1-w) takes tp_100_t120, runner
    w rides until the 0.896 hazard fires. Sweep w. Does tail capture beat the
    give-back? (small n on TEST — flagged hard.)
 P3 TIP CURVE: net(our_tip) = E[fill net | landing(tip)] - tip/bet. The
    optimal tip given competitor distribution. Caveat: 21% tip coverage =>
    contention understated => T* is a LOWER bound on the useful tip.
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

# hazard model (path9, 0.896) from recovery snaps
sn=pd.read_parquet(ROOT/"data/recovery_snaps_k3v03.parquet").sort_values(["mint","fwd_i"])
fmn=sn.groupby("mint")["ret"].transform(lambda s:s[::-1].cummin()[::-1].shift(-1))
sn=sn.assign(fut_min=fmn).dropna(subset=["fut_min"])
sn["collapse"]=(((1+sn.fut_min)/(1+sn.ret)-1)<=-0.40).astype(int)
htr=sn[sn.ready_ts<JUN10]
hz=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,l2_regularization=2.0,random_state=0).fit(htr[P9].values,htr.collapse.values)
hz_by={m:hz.predict_proba(g[P9].values)[:,1] for m,g in sn.groupby("mint")}

d=pickle.load(open(ROOT/"data/sweep_k3v03.pkl","rb")); names,M=d["names"],d["mints"]
fs=sorted(glob.glob(str(ROOT/"grpc_capture/*.jsonl*"))); a,b=fr(fs[0]),fr(fs[-2]); sps=(b[0]-a[0])/(b[1]-a[1])
s2t=lambda s:a[1]+(s-a[0])/sps
clf=pickle.load(open(ROOT/"bot_artifacts_k3v03_final/entry_model.pkl","rb"))
mints=list(M); SC=clf.predict_proba(np.array([M[m]["feats"] for m in mints]))[:,1]
TT=np.array([s2t(M[m]["decision"]["slot"]) for m in mints])
PEAK=np.array([max((vs/vt for (_t,_s,vs,vt,_bb,_tp) in M[m]["fwd"]),default=M[m]["decision"]["vsol"]/M[m]["decision"]["vtok"])/(M[m]["decision"]["vsol"]/M[m]["decision"]["vtok"])-1 if M[m]["fwd"] else 0.0 for m in mints])

# ---- P1 TAIL DETECTION (all ready mints, well-powered) ----
print("=== P1: TAIL DETECTION ===")
elite = SC>=THR
print(f"  among ALL ready ({len(mints)}): P(peak>=2x)={(PEAK>=2).mean():.3f} >=5x={(PEAK>=5).mean():.4f} >=10x={(PEAK>=10).mean():.4f}")
print(f"  among ELITE fires ({elite.sum()}): P(>=2x)={(PEAK[elite]>=2).mean():.3f} >=5x={(PEAK[elite]>=5).mean():.3f} >=10x={(PEAK[elite]>=10).mean():.3f}")
print(f"  mean peak elite={PEAK[elite].mean():+.2f}  (tp_100 caps at +1.0 => clipping the >100% mass)")
X=np.array([M[m]["feats"] for m in mints]); tr=TT<JUN10; te=~tr
for lvl,lab in [(2,">=2x"),(5,">=5x"),(10,">=10x")]:
    y=(PEAK>=lvl).astype(int)
    if y[tr].sum()<10 or y[te].sum()<5: print(f"  entry-AUC {lab}: too few ({y[te].sum()} test pos)"); continue
    m=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=0.05,l2_regularization=5.0,random_state=42).fit(X[tr],y[tr])
    print(f"  entry-model cross-day AUC for {lab}: {roc_auc_score(y[te],m.predict_proba(X[te])[:,1]):.4f}  (test pos={y[te].sum()})")

class Fire:
    def __init__(s,m,score,f,lat=1):
        s.mint=m; s.score=score
        j=min(lat,len(f)-1); s.evs,s.evt=f[j][2],f[j][3]; s.tok0=bt(s.evs,s.evt,Q*1e9); s.em=s.evs/s.evt
        s.full=f; s.j=j; rest=f[j+1:]
        s.t=np.array([r[0]-f[j][0] for r in rest]); s.vs=np.array([r[2] for r in rest],float); s.vt=np.array([r[3] for r in rest],float)
        s.ret=(s.vs/s.vt)/s.em-1.0
        h=hz_by.get(m); hh=h[j+1:] if (h is not None and len(h)>j) else np.array([])
        s.haz=np.concatenate([hh,np.repeat(hh[-1] if len(hh) else 0.0,max(0,len(s.ret)-len(hh)))])[:len(s.ret)] if len(s.ret) else np.array([])
    def net_at(s,i,frac=1.0):
        if len(s.vs)==0: return -CB/1e4
        i=min(max(i,0),len(s.vs)-1)
        return (sl(s.vs[i],s.vt[i],s.tok0*frac)/(Q*1e9*frac))-1.0  # gross frac net (costs added by caller)

TE=[Fire(m,s,M[m]["fwd"]) for m,s,t in zip(mints,SC,TT) if s>=THR and M[m]["fwd"] and t>=JUN10]
TR=[Fire(m,s,M[m]["fwd"]) for m,s,t in zip(mints,SC,TT) if s>=THR and M[m]["fwd"] and t<JUN10]

def tp100_t120_idx(f):
    for i in range(len(f.ret)):
        if f.ret[i]>=1.0: return i
        if f.t[i]>=120: return i
    return len(f.ret)-1 if len(f.ret) else None

def land_idx(f, our_tip):
    """slot-aware + tip-rank landing within the FULL path (returns index into rest f.t/.vs)."""
    full=f.full; dslot=full[f.j][1] if False else full[0][1]
    # use decision slot = the ready slot in M; approximate with first fwd slot's predecessor
    # land in slot after f.j's slot; competitors with higher KNOWN tip in landing slot go first
    fwd=full[f.j+1:]
    i=0
    if not fwd: return None
    # decision slot is full[f.j][1]
    ds=full[f.j][1]
    while i<len(fwd) and fwd[i][1]<=ds: i+=1
    if i>=len(fwd): return None
    ls=fwd[i][1]
    while i<len(fwd) and fwd[i][1]==ls:
        tip=fwd[i][5]
        if tip is not None and tip>our_tip: i+=1
        else: break
    return i if i<len(fwd) else None

# ---- P0 tp_100_t120 under landing ----
print("\n=== P0: tp_100_t120 net under landing models (TEST deduped) ===")
def policy_from_landing(f, li):
    """enter at landing index li (into rest); run tp100_t120 over the remainder."""
    if li is None or li>=len(f.vs): return f.net_at(len(f.vs)-1)-CB/1e4-2*FEE/Q if len(f.vs) else -CB/1e4-2*FEE/Q
    evs,evt=f.vs[li],f.vt[li]; tok=bt(evs,evt,Q*1e9); em=evs/evt; t0=f.t[li]
    xi=li
    for i in range(li,len(f.vs)):
        r=(f.vs[i]/f.vt[i])/em-1.0; xi=i
        if r>=1.0 or (f.t[i]-t0)>=120: break
    return sl(f.vs[xi],f.vt[xi],tok)/(Q*1e9)-1.0-CB/1e4-2*FEE/Q
for lab,landfn in [("lat0",lambda f:0),("lat1",lambda f:0 if len(f.vs) else None),
                   ("slot-aware tip=1M",lambda f:land_idx(f,1_000_000))]:
    # NOTE: Fire is built at lat=1 so index 0 == lat1; lat0 needs the j=0 build; approximate lat1 baseline here
    nets=[policy_from_landing(f, landfn(f)) for f in TE]
    dm,npat=dedup([f.score for f in TE],nets)
    print(f"  {lab:20s} DEDUP={dm:+.3f} (n_pat={npat})")

# ---- P2 MOONBAG ----
print("\n=== P2: MOONBAG sleeve (core tp100_t120 + runner hazard-cut), TEST deduped ===")
def core_idx(f):  # tp100_t120 over lat1 path
    for i in range(len(f.ret)):
        if f.ret[i]>=1.0 or f.t[i]>=120: return i
    return len(f.ret)-1 if len(f.ret) else None
def runner_idx(f, hcut=0.6):  # ride until hazard fires, else horizon
    for i in range(len(f.ret)):
        if f.haz[i]>=hcut: return i
    return len(f.ret)-1 if len(f.ret) else None
for w in (0.0,0.1,0.2,0.33):
    nets,scs=[],[]
    for f in TE:
        if len(f.vs)==0: continue
        ci=core_idx(f); ri=runner_idx(f)
        cq=Q*(1-w); rq=Q*w
        net=0.0
        if cq>0:
            tok=bt(f.evs,f.evt,cq*1e9); net+=sl(f.vs[ci],f.vt[ci],tok)-cq*1e9
            net-=cq*1e9*(CB/1e4); net-=FEE*1e9
        if rq>0:
            tok=bt(f.evs,f.evt,rq*1e9); net+=sl(f.vs[ri],f.vt[ri],tok)-rq*1e9
            net-=rq*1e9*(CB/1e4); net-=FEE*1e9
        net-=FEE*1e9  # the buy
        nets.append(net/(Q*1e9)); scs.append(f.score)
    dm,npat=dedup(scs,nets)
    print(f"  w_runner={w:.2f}  DEDUP={dm:+.3f} (n_pat={npat})")
print("  (w=0 == pure tp100_t120; n_pat small, tail rare -> read as direction not proof)")

# ---- P3 TIP CURVE ----
print("\n=== P3: TIP-PROFIT CURVE (TEST deduped; cost = tip/bet) ===")
print("  caveat: 21% forward-tip coverage => contention understated => T* is a LOWER bound")
for tip in (0,100_000,500_000,1_000_000,2_000_000,5_000_000,10_000_000):
    nets,scs=[],[]
    for f in TE:
        li=land_idx(f,tip)
        gross=policy_from_landing(f,li)
        nets.append(gross - tip/(Q*1e9)); scs.append(f.score)
    dm,npat=dedup(scs,nets)
    print(f"  our_tip={tip//1000:>5d}k lam ({tip/(Q*1e9)*100:4.1f}% of bet)  DEDUP_net={dm:+.3f}")
