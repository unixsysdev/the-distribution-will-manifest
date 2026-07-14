"""Capture-to-extractor pipeline.

Replays grpc_capture/*.jsonl(.gz) chronologically through the SAME K=7 and V=0.5
trigger + feature-extraction logic the offline trades.csv extractors use, emits
token_level.parquet + path_snapshots.parquet to a target dir.

Once we have 3-7 days of capture data this lets us re-extract training tables from
the regime we actually trade in (the firehose, not a historical snapshot). Feed the
output to build_bot_artifacts_K7V.py to refresh production artifacts.

CRITICAL: this script's accumulator MUST produce byte-identical features to
pumpfun_continuation_value_K7.py and _V.py for any token both see. We import the
M class from each extractor so we never drift apart. Only the input source differs.
"""
from __future__ import annotations
import argparse, glob, gzip, json, time
from pathlib import Path
import pandas as pd

# Import the M class + helpers from the existing K7 + V extractors so the
# feature math is GUARANTEED identical. We just wire a different input source.
import pumpfun_continuation_value_K7 as ek7
import pumpfun_continuation_value_V  as eV


def iter_capture(capture_dir: Path):
    """Yield capture rows in chronological order across all .jsonl(.gz) files in dir."""
    files = sorted(glob.glob(str(capture_dir / "*.jsonl*")))
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            for ln in f:
                try: rec = json.loads(ln)
                except Exception: continue
                yield rec


def stream_k7(rows, fresh_rsol_lam=0):
    """Same shape as pumpfun_continuation_value_K7.stream() but reads dicts instead of CSV.
    Returns states dict that ek7.do_extract's post-processing knows how to consume."""
    K = ek7.K
    states = {}; skipped = set(); n = 0
    for rec in rows:
        n += 1
        if n % 200_000 == 0:
            print(f"  .. {n:,} events ({len(states)} mints, {len(skipped)} skipped)", flush=True)
        mint = rec.get("mint")
        if not mint or mint in skipped: continue
        vs = float(rec.get("vsol", 0)); vt = float(rec.get("vtok", 0))
        if vs <= 0 or vt <= 0: continue
        ts = float(rec.get("ev_ts", 0)); slot = int(rec.get("slot", 0))
        sol = float(rec.get("sol", 0)) / 1e9
        is_buy = bool(rec.get("is_buy")); uid = rec.get("user", "")
        rs = float(rec.get("rsol", 0))
        mid = vs / vt
        m = states.get(mint)
        if m is None:
            if fresh_rsol_lam > 0 and rs >= fresh_rsol_lam:
                skipped.add(mint); continue
            states[mint] = ek7.M(mid, slot, ts, sol, is_buy, uid, vs, vt); continue
        m.win.append((uid, sol, is_buy, ts))
        if m.n < K:
            m.n += 1; m.mids.append(mid); m.users.add(uid)
            m.user_sol[uid] = m.user_sol.get(uid, 0.0) + sol
            if is_buy: m.n_buy += 1; m.net_sol += sol
            else: m.net_sol -= sol
            m.tot_sol += sol; m.last_ts = ts
            rr = mid/m.mid0 - 1.0 if m.mid0 > 0 else 0.0
            m.win_dup = max(m.win_dup, rr); m.win_ddown = min(m.win_ddown, rr)
            if m.n == K:
                m.midK = mid; m.vsK = vs; m.vtK = vt; m.window_last_ts = ts
        else:
            if m.midK <= 0: continue
            m.fwd += 1; m.vsC = vs; m.vtC = vt; m.last_ts = ts
            ret = mid/m.midK - 1.0
            if mid > m.peakmax:
                m.peakmax = mid; m.peak_fwd_i = m.fwd; m.peak_slot = slot; m.peak_ts = ts
            if ret > m.run_max_ret: m.run_max_ret = ret
            if (m.fwd % ek7.SNAP_EVERY == 1 or ek7.SNAP_EVERY == 1) and len(m.snaps) < ek7.MAX_SNAP:
                bf, nsell, solo, vel = ek7.window_feats(m)
                dd = (mid/(m.midK*(1+m.run_max_ret)) - 1.0) if m.run_max_ret > -1 else 0.0
                m.snaps.append((m.fwd, slot-m.first_slot, ts-m.first_ts, ret,
                                m.run_max_ret, dd, ek7.fill_k(vs), bf, nsell, solo, vel, vs, vt))
    return states


def stream_v(rows, v_trigger, fresh_rsol_lam=0):
    """Same shape as pumpfun_continuation_value_V.stream() but reads dicts."""
    states = {}; skipped = set(); n = 0
    MIN_N = eV.MIN_N
    for rec in rows:
        n += 1
        if n % 200_000 == 0:
            print(f"  .. {n:,} events ({len(states)} mints, {len(skipped)} skipped)", flush=True)
        mint = rec.get("mint")
        if not mint or mint in skipped: continue
        vs = float(rec.get("vsol", 0)); vt = float(rec.get("vtok", 0))
        if vs <= 0 or vt <= 0: continue
        ts = float(rec.get("ev_ts", 0)); slot = int(rec.get("slot", 0))
        sol = float(rec.get("sol", 0)) / 1e9
        is_buy = bool(rec.get("is_buy")); uid = rec.get("user", "")
        rs = float(rec.get("rsol", 0))
        mid = vs / vt
        m = states.get(mint)
        if m is None:
            if fresh_rsol_lam > 0 and rs >= fresh_rsol_lam:
                skipped.add(mint); continue
            states[mint] = eV.M(mid, slot, ts, sol, is_buy, uid, vs, vt); continue
        m.win.append((uid, sol, is_buy, ts))
        if not m.windowed:
            m.n += 1; m.mids.append(mid); m.users.add(uid)
            m.user_sol[uid] = m.user_sol.get(uid, 0.0) + sol
            if is_buy:
                m.n_buy += 1; m.net_sol += sol; m.cum_buy_sol += sol
            else:
                m.net_sol -= sol
            m.tot_sol += sol; m.last_ts = ts
            rr = mid/m.mid0 - 1.0 if m.mid0 > 0 else 0.0
            m.win_dup = max(m.win_dup, rr); m.win_ddown = min(m.win_ddown, rr)
            if m.cum_buy_sol >= v_trigger and m.n >= MIN_N:
                m.midV = mid; m.vsV = vs; m.vtV = vt; m.window_last_ts = ts
                m.windowed = True
        else:
            if m.midV <= 0: continue
            m.fwd += 1; m.vsC = vs; m.vtC = vt; m.last_ts = ts
            ret = mid/m.midV - 1.0
            if mid > m.peakmax:
                m.peakmax = mid; m.peak_fwd_i = m.fwd; m.peak_slot = slot; m.peak_ts = ts
            if ret > m.run_max_ret: m.run_max_ret = ret
            if (m.fwd % eV.SNAP_EVERY == 1 or eV.SNAP_EVERY == 1) and len(m.snaps) < eV.MAX_SNAP:
                bf, nsell, solo, vel = eV.window_feats(m)
                dd = (mid/(m.midV*(1+m.run_max_ret)) - 1.0) if m.run_max_ret > -1 else 0.0
                m.snaps.append((m.fwd, slot-m.first_slot, ts-m.first_ts, ret,
                                m.run_max_ret, dd, eV.fill_k(vs), bf, nsell, solo, vel, vs, vt))
    return states


def emit_k7(states, out_dir: Path):
    """Same finalization as pumpfun_continuation_value_K7.do_extract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    POS = ek7.POS_SOL * 1e9
    tok_rows = []; snap_rows = []
    for mint, m in states.items():
        if m.midK <= 0 or m.fwd < ek7.MIN_FWD: continue
        if min(m.vsK, m.vtK, m.vsC, m.vtC) <= 0: continue
        pos_tok = ek7.buy_tokens(m.vsK, m.vtK, POS)
        terminal = ek7.sell_sol(m.vsC, m.vtC, pos_tok)/POS - 1.0 - ek7.COST_BPS/1e4
        peak_ret = m.peakmax/m.midK - 1.0
        tok_rows.append({"mint": mint, "first_slot": m.first_slot,
                          **{ek7.ENTRY_NAMES[i]: v for i, v in enumerate(ek7.entry_feats(m))},
                          "n_fwd": m.fwd, "trades_to_peak": m.peak_fwd_i,
                          "slots_to_peak": m.peak_slot - m.first_slot,
                          "secs_to_peak": m.peak_ts - m.first_ts,
                          "peak_ret": peak_ret, "terminal_ret": terminal,
                          "vsK": m.vsK, "vtK": m.vtK, "pos_tok": pos_tok,
                          "vsC": m.vsC, "vtC": m.vtC})
        for (fwd,dslot,dts,ret,runmax,dd,fk,bf,nsell,solo,vel,vs,vt) in m.snaps:
            cv = ek7.sell_sol(vs, vt, pos_tok)/POS - 1.0 - ek7.COST_BPS/1e4
            snap_rows.append((mint, m.first_slot, fwd, dslot, dts, ret, runmax,
                               dd, fk, bf, nsell, solo, vel, cv, vs, vt))
    tok = pd.DataFrame(tok_rows)
    snap = pd.DataFrame(snap_rows, columns=["mint","first_slot","fwd","dslot","dts","ret",
                                              "run_max_ret","dd","fill_k","buy_frac_w","nsell_w",
                                              "solo_sell_w","vel_w","close_val","vs","vt"])
    tok.to_parquet(out_dir/"token_level.parquet", index=False)
    snap.to_parquet(out_dir/"path_snapshots.parquet", index=False)
    print(f"  K7 -> {out_dir} : {len(tok)} tokens / {len(snap)} snaps")
    if len(tok):
        print(f"    peak>=2x {(tok.peak_ret>=1).mean():.1%}  "
              f"terminal>0 {(tok.terminal_ret>0).mean():.1%}  "
              f"terminal>=5x {(tok.terminal_ret>=5).mean():.1%}")


def emit_v(states, out_dir: Path):
    """Same finalization as pumpfun_continuation_value_V.do_extract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    POS = eV.POS_SOL * 1e9
    tok_rows = []; snap_rows = []
    for mint, m in states.items():
        if not m.windowed or m.fwd < eV.MIN_FWD or m.midV <= 0: continue
        if min(m.vsV, m.vtV, m.vsC, m.vtC) <= 0: continue
        pos_tok = eV.buy_tokens(m.vsV, m.vtV, POS)
        terminal = eV.sell_sol(m.vsC, m.vtC, pos_tok)/POS - 1.0 - eV.COST_BPS/1e4
        peak_ret = m.peakmax/m.midV - 1.0
        tok_rows.append({"mint": mint, "first_slot": m.first_slot,
                          **{eV.ENTRY_NAMES[i]: v for i, v in enumerate(eV.entry_feats(m))},
                          "n_fwd": m.fwd, "trades_to_peak": m.peak_fwd_i,
                          "slots_to_peak": m.peak_slot - m.first_slot,
                          "secs_to_peak": m.peak_ts - m.first_ts,
                          "peak_ret": peak_ret, "terminal_ret": terminal,
                          "vsK": m.vsV, "vtK": m.vtV, "pos_tok": pos_tok,
                          "vsC": m.vsC, "vtC": m.vtC,
                          "cum_buy_sol_at_trigger": m.cum_buy_sol,
                          "n_at_trigger": m.n})
        for (fwd,dslot,dts,ret,runmax,dd,fk,bf,nsell,solo,vel,vs,vt) in m.snaps:
            cv = eV.sell_sol(vs, vt, pos_tok)/POS - 1.0 - eV.COST_BPS/1e4
            snap_rows.append((mint, m.first_slot, fwd, dslot, dts, ret, runmax,
                               dd, fk, bf, nsell, solo, vel, cv, vs, vt))
    tok = pd.DataFrame(tok_rows)
    snap = pd.DataFrame(snap_rows, columns=["mint","first_slot","fwd","dslot","dts","ret",
                                              "run_max_ret","dd","fill_k","buy_frac_w","nsell_w",
                                              "solo_sell_w","vel_w","close_val","vs","vt"])
    tok.to_parquet(out_dir/"token_level.parquet", index=False)
    snap.to_parquet(out_dir/"path_snapshots.parquet", index=False)
    print(f"  V  -> {out_dir} : {len(tok)} tokens / {len(snap)} snaps")
    if len(tok):
        print(f"    median n_at_trigger={tok.n_at_trigger.median():.0f}  "
              f"peak>=2x {(tok.peak_ret>=1).mean():.1%}  "
              f"terminal>=5x {(tok.terminal_ret>=5).mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True, help="grpc_capture dir on the host")
    ap.add_argument("--k7-out", required=True, help="output dir for K7 parquets")
    ap.add_argument("--v-out",  required=True, help="output dir for V parquets")
    ap.add_argument("--V", type=float, default=eV.V_TRIGGER)
    ap.add_argument("--fresh-rsol-lam", type=int, default=3_000_000_000,
                    help="default 3e9 = 3 SOL; matches live FRESH_RSOL filter")
    args = ap.parse_args()

    print(f"=== capture-to-extractor (V+K7) ===")
    print(f"capture dir: {args.capture_dir}")
    print(f"fresh_rsol_lam: {args.fresh_rsol_lam}")
    t0 = time.time()
    # Two passes (one per stream class) so we don't have to merge two state classes.
    # Faster alternative would be single-pass with both M instances per mint; do that
    # later if we hit IO bottleneck.
    print("\npass 1/2: K7")
    states_k = stream_k7(iter_capture(Path(args.capture_dir)),
                          fresh_rsol_lam=args.fresh_rsol_lam)
    emit_k7(states_k, Path(args.k7_out))
    print(f"\npass 2/2: V={args.V}")
    states_v = stream_v(iter_capture(Path(args.capture_dir)), args.V,
                         fresh_rsol_lam=args.fresh_rsol_lam)
    emit_v(states_v, Path(args.v_out))
    print(f"\ntotal wallclock: {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
