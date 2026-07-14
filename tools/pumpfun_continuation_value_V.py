"""Volume-windowed extractor: window closes at first cumulative buy SOL >= V.

Same output schemas as pumpfun_continuation_value.py (token_level.parquet +
path_snapshots.parquet), just with the window trigger changed from K trades to
V SOL of cumulative buy volume. Causal: window_last_ts frozen at the V-crossing
trade.
"""
from __future__ import annotations
import argparse, csv, json, time
from collections import deque
import numpy as np, pandas as pd
from pathlib import Path

V_TRIGGER  = 0.5         # SOL of cumulative buy volume
MIN_N      = 3           # minimum trades in window (skip degenerate)
MIN_FWD    = 5
SNAP_EVERY = 3
MAX_SNAP   = 80
W          = 15
POS_SOL    = 0.5
COST_BPS   = 250.0


def buy_tokens(vs,vt,d): return vt - (vs*vt)/(vs+d)
def sell_sol(vs,vt,d):   return vs - (vs*vt)/(vt+d)
def fill_k(vsol_lam):    return max(0.0, min(1.0, (vsol_lam/1e9 - 30.0)/85.0))


class M:
    __slots__=("n","mid0","midV","vsV","vtV","first_slot","first_ts","last_ts",
               "window_last_ts","mids","users","user_sol","n_buy","cum_buy_sol",
               "net_sol","tot_sol","entry_sol","win_dup","win_ddown","windowed",
               "fwd","run_max_ret","peakmax","peak_fwd_i","peak_slot","peak_ts",
               "vsC","vtC","win","snaps")
    def __init__(s,mid,slot,ts,sol,is_buy,uid,vs,vt):
        s.n=1; s.mid0=mid; s.midV=mid; s.vsV=0.0; s.vtV=0.0
        s.first_slot=slot; s.first_ts=ts; s.last_ts=ts; s.window_last_ts=ts
        s.mids=[mid]; s.users={uid}; s.user_sol={uid:sol}
        s.n_buy=1 if is_buy else 0
        s.cum_buy_sol=sol if is_buy else 0.0
        s.net_sol=sol if is_buy else -sol
        s.tot_sol=sol; s.entry_sol=sol
        s.win_dup=0.0; s.win_ddown=0.0
        s.windowed=False; s.fwd=0; s.run_max_ret=0.0
        s.peakmax=mid; s.peak_fwd_i=0; s.peak_slot=slot; s.peak_ts=ts
        s.vsC=vs; s.vtC=vt
        s.win=deque(maxlen=W); s.win.append((uid,sol,is_buy,ts))
        s.snaps=[]


def entry_feats(m):
    d=[m.mids[i]-m.mids[i-1] for i in range(1,len(m.mids))]
    sa=sum(abs(x) for x in d); dir_eff=abs(sum(d))/sa if sa>0 else 0.0
    win_ret=m.midV/m.mid0-1.0 if m.mid0>0 else 0.0
    span=max(1e-6,m.window_last_ts-m.first_ts)
    sas=max(m.user_sol.values())/m.tot_sol if m.tot_sol>0 else 0.0
    return [win_ret,dir_eff,m.n_buy/m.n,len(m.users),m.net_sol,m.tot_sol,sas,
            m.n/span,m.entry_sol,m.win_dup,m.win_ddown]

ENTRY_NAMES=["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]


def window_feats(m):
    wl=list(m.win); nb=sum(1 for _,_,b,_ in wl if b); nw=len(wl)
    buy_frac=nb/nw if nw else 0.0
    sellers={}; sell_tot=0.0
    for u,s,b,_ in wl:
        if not b:
            sellers[u]=sellers.get(u,0.0)+s; sell_tot+=s
    solo_sell=(max(sellers.values())/sell_tot) if sell_tot>0 else 0.0
    dt=max(1e-6,wl[-1][3]-wl[0][3]); net=sum((s if b else -s) for _,s,b,_ in wl)
    return buy_frac,len(sellers),solo_sell,net/dt


def stream(trades, cap, v_trigger, min_launch_slot, fresh_rsol_lam=0):
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
            if n%2_000_000==0: print(f"  .. {n:,} trades ({len(states)} mints, {len(skipped)} skipped)",flush=True)
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
            if not m.windowed:
                m.n+=1; m.mids.append(mid); m.users.add(uid)
                m.user_sol[uid]=m.user_sol.get(uid,0.0)+sol
                if is_buy:
                    m.n_buy+=1; m.net_sol+=sol; m.cum_buy_sol+=sol
                else:
                    m.net_sol-=sol
                m.tot_sol+=sol; m.last_ts=ts
                rr=mid/m.mid0-1.0 if m.mid0>0 else 0.0
                m.win_dup=max(m.win_dup,rr); m.win_ddown=min(m.win_ddown,rr)
                # V trigger
                if m.cum_buy_sol >= v_trigger and m.n >= MIN_N:
                    m.midV=mid; m.vsV=vs; m.vtV=vt; m.window_last_ts=ts
                    m.windowed=True
            else:
                if m.midV<=0: continue
                m.fwd+=1; m.vsC=vs; m.vtC=vt; m.last_ts=ts
                ret=mid/m.midV-1.0
                if mid>m.peakmax:
                    m.peakmax=mid; m.peak_fwd_i=m.fwd; m.peak_slot=slot; m.peak_ts=ts
                if ret>m.run_max_ret: m.run_max_ret=ret
                if (m.fwd%SNAP_EVERY==1 or SNAP_EVERY==1) and len(m.snaps)<MAX_SNAP:
                    bf,nsell,solo,vel=window_feats(m)
                    dd=(mid/(m.midV*(1+m.run_max_ret))-1.0) if m.run_max_ret>-1 else 0.0
                    m.snaps.append((m.fwd, slot-m.first_slot, ts-m.first_ts, ret,
                                    m.run_max_ret, dd, fill_k(vs), bf, nsell, solo, vel, vs, vt))
    return states


def do_extract(trades, out_dir, v_trigger=V_TRIGGER, cap=0, min_launch_slot=0, fresh_rsol_lam=0):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] V-extract V={v_trigger} SOL  trades={trades}  out={out_dir}  "
          f"min_launch_slot={min_launch_slot}  fresh_rsol_lam={fresh_rsol_lam}", flush=True)
    t0=time.time()
    states=stream(trades, cap, v_trigger, min_launch_slot, fresh_rsol_lam=fresh_rsol_lam)
    print(f"[i] streamed in {(time.time()-t0)/60:.1f}min  states={len(states)}", flush=True)

    POS=POS_SOL*1e9
    tok_rows=[]; snap_rows=[]
    for mint, m in states.items():
        if not m.windowed or m.fwd < MIN_FWD or m.midV <= 0: continue
        if min(m.vsV,m.vtV,m.vsC,m.vtC) <= 0: continue
        if min_launch_slot and m.first_slot <= min_launch_slot: continue
        pos_tok = buy_tokens(m.vsV, m.vtV, POS)
        terminal = sell_sol(m.vsC, m.vtC, pos_tok)/POS - 1.0 - COST_BPS/1e4
        peak_ret = m.peakmax/m.midV - 1.0
        tok_rows.append({
            "mint": mint, "first_slot": m.first_slot,
            **{ENTRY_NAMES[i]: v for i,v in enumerate(entry_feats(m))},
            "n_fwd": m.fwd, "trades_to_peak": m.peak_fwd_i,
            "slots_to_peak": m.peak_slot - m.first_slot,
            "secs_to_peak": m.peak_ts - m.first_ts,
            "peak_ret": peak_ret, "terminal_ret": terminal,
            "vsK": m.vsV, "vtK": m.vtV, "pos_tok": pos_tok,
            "vsC": m.vsC, "vtC": m.vtC,
            "cum_buy_sol_at_trigger": m.cum_buy_sol,
            "n_at_trigger": m.n,
        })
        for (fwd,dslot,dts,ret,runmax,dd,fk,bf,nsell,solo,vel,vs,vt) in m.snaps:
            cv = sell_sol(vs, vt, pos_tok)/POS - 1.0 - COST_BPS/1e4
            snap_rows.append((mint, m.first_slot, fwd, dslot, dts, ret, runmax,
                              dd, fk, bf, nsell, solo, vel, cv, vs, vt))

    tok = pd.DataFrame(tok_rows)
    snap = pd.DataFrame(snap_rows, columns=[
        "mint","first_slot","fwd","dslot","dts","ret","run_max_ret","dd","fill_k",
        "buy_frac_w","nsell_w","solo_sell_w","vel_w","close_val","vs","vt"])
    tok.to_parquet(out_dir/"token_level.parquet", index=False)
    snap.to_parquet(out_dir/"path_snapshots.parquet", index=False)
    print(f"[i] usable {len(tok)} mints | snapshots {len(snap)}", flush=True)
    if len(tok):
        print(f"    median n_at_trigger={tok.n_at_trigger.median():.0f}  "
              f"peak>=2x {(tok.peak_ret>=1).mean():.1%}  "
              f"terminal>=5x {(tok.terminal_ret>=5).mean():.1%}", flush=True)
    print(f"Saved: {out_dir}/token_level.parquet + path_snapshots.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--V", type=float, default=V_TRIGGER)
    ap.add_argument("--min-launch-slot", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--fresh-rsol-lam", type=int, default=0,
                    help="if >0, only keep tokens whose first observed real_sol_reserves is strictly below this (lamports). 3000000000 = 3 SOL to match the live bot.")
    a = ap.parse_args()
    do_extract(a.trades, a.out_dir, v_trigger=a.V, cap=a.cap, min_launch_slot=a.min_launch_slot, fresh_rsol_lam=a.fresh_rsol_lam)
