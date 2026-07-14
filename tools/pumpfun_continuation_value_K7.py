"""pumpfun_continuation_value.py  —  the early free boundary the project never built.

The management search (Findings 8/9/10) tested re-separation at HIGH rungs (2x..10x),
the signal-dead zone (AUC ~0.59). The strong signal (0.75) lives at ENTRY. This builds
the conditional manifold / optimal-stopping objects in the EARLY window where the signal
actually is, with three never-modeled targets:

  A  TIME-TO-PEAK exit clock   : regress slots-to-peak on early-path features; backtest
                                 "exit at predicted peak time" vs passive vs flat TP.
  B  CONDITIONAL RECOVERY      : P(recover to break-even | in drawdown, state_t); loss-cut
                                 when P(recover) < c. Checks it doesn't amputate dipped-
                                 then-mooned winners (the Finding-4 failure mode).
  C  EARLY FREE BOUNDARY (LSM) : Longstaff-Schwartz backward induction of the continuation
                                 value on realized early paths; stopping boundary where
                                 close-value > continuation-value.

ZERO new API calls. One streaming pass over local trades.csv → two cached parquets,
then all three analyses run from cache.

  python pumpfun_continuation_value.py --extract [--cap N]   # stage 1 (stream)
  python pumpfun_continuation_value.py --analyze             # stages A/B/C from cache
"""
from __future__ import annotations
import argparse, csv, json, os, time
from collections import deque
import numpy as np
import pandas as pd
from pathlib import Path

TRADES   = "data_pull/coherent_417495154/trades.csv"
OUT_DIR  = Path("data/pumpfun_continuation")
import os as _os
K = int(_os.getenv("K_WINDOW", "7"))   # entry window (first K trades)
# Default 7 matches wide_v2 production. Set K_WINDOW=5 for earlier-trigger
# experiments (smaller TP targets tolerate noisier entry).
MIN_FWD    = 5         # need >= this many post-window trades
SNAP_EVERY = 3         # snapshot cadence in forward trades
MAX_SNAP   = 80        # cap snapshots per token (early window is what matters)
W          = 15        # rolling window (trades) for flow/seller features
POS_SOL    = 0.5
COST_BPS   = 250.0


def buy_tokens(vs, vt, dsol): return vt - (vs * vt) / (vs + dsol)
def sell_sol(vs, vt, dtok):  return vs - (vs * vt) / (vt + dtok)
def fill_k(vsol_lam):        return max(0.0, min(1.0, (vsol_lam/1e9 - 30.0) / 85.0))


class M:
    __slots__ = ("n","mid0","midK","vsK","vtK","first_slot","first_ts","last_ts","mids",
                 "users","user_sol","n_buy","net_sol","tot_sol","entry_sol","win_dup","win_ddown",
                 "vsC","vtC","peakmax","fwd","run_max_ret","peak_fwd_i","peak_slot","peak_ts",
                 "win","snaps","window_last_ts")
    def __init__(s, mid, slot, ts, sol, is_buy, uid, vs, vt):
        s.n=1; s.mid0=mid; s.midK=mid; s.vsK=0.0; s.vtK=0.0
        s.first_slot=slot; s.first_ts=ts; s.last_ts=ts; s.window_last_ts=ts; s.mids=[mid]
        s.users={uid}; s.user_sol={uid:sol}; s.n_buy=1 if is_buy else 0
        s.net_sol=sol if is_buy else -sol; s.tot_sol=sol; s.entry_sol=sol
        s.win_dup=0.0; s.win_ddown=0.0; s.vsC=vs; s.vtC=vt; s.peakmax=mid
        s.fwd=0; s.run_max_ret=0.0; s.peak_fwd_i=0; s.peak_slot=slot; s.peak_ts=ts
        s.win=deque(maxlen=W)               # rolling (uid, sol, is_buy, ts)
        s.win.append((uid, sol, is_buy, ts))
        s.snaps=[]                          # list of snapshot tuples


def entry_feats(m):
    mids=m.mids; d=[mids[i]-mids[i-1] for i in range(1,len(mids))]
    sa=sum(abs(x) for x in d); dir_eff=abs(sum(d))/sa if sa>0 else 0.0
    win_ret=m.midK/m.mid0-1.0 if m.mid0>0 else 0.0
    span=max(1e-6,m.window_last_ts-m.first_ts); sas=max(m.user_sol.values())/m.tot_sol if m.tot_sol>0 else 0.0
    return [win_ret,dir_eff,m.n_buy/m.n,len(m.users),m.net_sol,m.tot_sol,sas,m.n/span,
            m.entry_sol,m.win_dup,m.win_ddown]

ENTRY_NAMES=["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]


def window_feats(m):
    """rolling flow/seller features at the current moment from the last-W deque."""
    wl=list(m.win); nb=sum(1 for _,_,b,_ in wl if b); nw=len(wl)
    buy_frac=nb/nw if nw else 0.0
    sellers={}; sell_tot=0.0
    for uid,sol,b,_ in wl:
        if not b:
            sellers[uid]=sellers.get(uid,0.0)+sol; sell_tot+=sol
    nsell=len(sellers)
    solo_sell=(max(sellers.values())/sell_tot) if sell_tot>0 else 0.0
    dt=max(1e-6, wl[-1][3]-wl[0][3])
    net=sum((sol if b else -sol) for _,sol,b,_ in wl)
    vel=net/dt
    return buy_frac, nsell, solo_sell, vel


def stream(trades, cap=0, fresh_rsol_lam=0):
    """When fresh_rsol_lam > 0, only keep tokens whose FIRST observed real_sol_reserves
    is strictly below the threshold. Matches the live bot's FRESH_RSOL filter."""
    states={}; skipped=set()
    with open(trades) as fh:
        r=csv.reader(fh); hdr=next(r); c={n:i for i,n in enumerate(hdr)}
        im,isl,its=c["mint"],c["slot"],c["event_timestamp"]
        ivs,ivt=c["virtual_sol_reserves"],c["virtual_token_reserves"]
        irs=c["real_sol_reserves"]
        iso,ib,iu=c["sol_amount_lamports"],c["is_buy"],c["user"]
        n=0
        for row in r:
            n+=1
            if cap and n>cap: break
            if n%2_000_000==0: print(f"   .. {n:,} trades ({len(states)} mints, {len(skipped)} skipped)",flush=True)
            mint=row[im]
            if mint in skipped: continue
            try:
                vs=float(row[ivs]); vt=float(row[ivt]); slot=int(float(row[isl]))
                ts=float(row[its]); sol=float(row[iso])/1e9
                rs=float(row[irs])
            except (ValueError,TypeError): continue
            if vs<=0 or vt<=0: continue
            mid=vs/vt; is_buy=str(row[ib]).lower() in ("true","1","t"); uid=row[iu]
            m=states.get(mint)
            if m is None:
                if fresh_rsol_lam > 0 and rs >= fresh_rsol_lam:
                    skipped.add(mint); continue
                states[mint]=M(mid,slot,ts,sol,is_buy,uid,vs,vt); continue
            m.win.append((uid,sol,is_buy,ts))
            if m.n<K:
                m.n+=1; m.mids.append(mid); m.users.add(uid)
                m.user_sol[uid]=m.user_sol.get(uid,0.0)+sol
                if is_buy: m.n_buy+=1; m.net_sol+=sol
                else: m.net_sol-=sol
                m.tot_sol+=sol; m.last_ts=ts
                rr=mid/m.mid0-1.0 if m.mid0>0 else 0.0
                m.win_dup=max(m.win_dup,rr); m.win_ddown=min(m.win_ddown,rr)
                if m.n==K: m.midK=mid; m.vsK=vs; m.vtK=vt; m.window_last_ts=ts   # FREEZE span (causal)
            else:
                if m.midK<=0: continue
                m.fwd+=1; m.vsC=vs; m.vtC=vt; m.last_ts=ts
                ret=mid/m.midK-1.0
                if mid>m.peakmax:
                    m.peakmax=mid; m.peak_fwd_i=m.fwd; m.peak_slot=slot; m.peak_ts=ts
                if ret>m.run_max_ret: m.run_max_ret=ret
                # snapshot at cadence
                if (m.fwd%SNAP_EVERY==1 or SNAP_EVERY==1) and len(m.snaps)<MAX_SNAP:
                    bf,nsell,solo,vel=window_feats(m)
                    dd=(mid/(m.midK*(1+m.run_max_ret))-1.0) if m.run_max_ret>-1 else 0.0
                    m.snaps.append((m.fwd, slot-m.first_slot, ts-m.first_ts, ret,
                                    m.run_max_ret, dd, fill_k(vs), bf, nsell, solo, vel, vs, vt))
    return states


def do_extract(cap=0, trades=TRADES, out_dir=OUT_DIR, min_launch_slot=0, fresh_rsol_lam=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] streaming {trades} (cap={cap or 'none'}, min_launch_slot={min_launch_slot or 'none'}, "
          f"fresh_rsol_lam={fresh_rsol_lam or 'none'}) ...")
    t0=time.time()
    states=stream(trades, cap, fresh_rsol_lam=fresh_rsol_lam)
    print(f"[i] streamed in {(time.time()-t0)/60:.1f}min; {len(states)} mints")

    POS=POS_SOL*1e9
    tok_rows=[]; snap_rows=[]
    for mint,m in states.items():
        if m.n<K or m.fwd<MIN_FWD or m.midK<=0: continue
        if min(m.vsK,m.vtK,m.vsC,m.vtC)<=0: continue
        if min_launch_slot and m.first_slot <= min_launch_slot: continue   # forward OOS slice only
        pos_tok=buy_tokens(m.vsK,m.vtK,POS)
        terminal=sell_sol(m.vsC,m.vtC,pos_tok)/POS-1.0-COST_BPS/1e4
        peak_ret=m.peakmax/m.midK-1.0
        tok_rows.append({
            "mint":mint,"first_slot":m.first_slot,
            **{ENTRY_NAMES[i]:v for i,v in enumerate(entry_feats(m))},
            "n_fwd":m.fwd,"trades_to_peak":m.peak_fwd_i,
            "slots_to_peak":m.peak_slot-m.first_slot,"secs_to_peak":m.peak_ts-m.first_ts,
            "peak_ret":peak_ret,"terminal_ret":terminal,
            "vsK":m.vsK,"vtK":m.vtK,"pos_tok":pos_tok,
            "vsC":m.vsC,"vtC":m.vtC,           # terminal reserves (for passive exit at any size)
        })
        for (fwd,dslot,dts,ret,runmax,dd,fk,bf,nsell,solo,vel,vs,vt) in m.snaps:
            # realizable close-value if you exit at this snapshot (AMM impact + cost)
            cv=sell_sol(vs,vt,pos_tok)/POS-1.0-COST_BPS/1e4
            snap_rows.append((mint,m.first_slot,fwd,dslot,dts,ret,runmax,dd,fk,
                              bf,nsell,solo,vel,cv,vs,vt))   # store raw reserves for size sweep

    tok=pd.DataFrame(tok_rows)
    snap=pd.DataFrame(snap_rows,columns=["mint","first_slot","fwd","dslot","dts","ret","run_max_ret",
                                         "dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w",
                                         "close_val","vs","vt"])
    tok.to_parquet(out_dir/"token_level.parquet",index=False)
    snap.to_parquet(out_dir/"path_snapshots.parquet",index=False)
    print(f"[i] usable tokens {len(tok)}  | snapshots {len(snap)}  "
          f"(median {len(snap)/max(1,len(tok)):.0f}/token)")
    print(f"    peak>=2x {(tok.peak_ret>=1).mean():.1%}  terminal>0 {(tok.terminal_ret>0).mean():.1%}  "
          f"terminal>=5x {(tok.terminal_ret>=5).mean():.1%}")
    print(f"Saved: {out_dir}/token_level.parquet + path_snapshots.parquet")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--extract",action="store_true")
    ap.add_argument("--analyze",action="store_true")
    ap.add_argument("--cap",type=int,default=0)
    ap.add_argument("--trades",default=TRADES)
    ap.add_argument("--out-dir",default=str(OUT_DIR))
    ap.add_argument("--min-launch-slot",type=int,default=0)
    ap.add_argument("--fresh-rsol-lam",type=int,default=0,
                    help="if >0, only keep tokens whose first observed real_sol_reserves is strictly below this (lamports). 3000000000 = 3 SOL to match the live bot.")
    a=ap.parse_args()
    if a.extract: do_extract(a.cap, trades=a.trades, out_dir=a.out_dir, min_launch_slot=a.min_launch_slot, fresh_rsol_lam=a.fresh_rsol_lam)
    if a.analyze:
        import pumpfun_continuation_analyze as an
        an.run()
