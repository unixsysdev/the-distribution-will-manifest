#!/usr/bin/env python3
"""GRADUATION (PumpSwap AMM) 2x-continuation panel extractor.

Counterpart of cont_2x_aug_extract.py but for POST-graduation tokens trading on
the PumpSwap AMM (program pAMMBay...). Single causal pass over the structured
grpc_capture jsonl (which already tags PumpSwap.Buy/SellEvent + CreatePoolEvent).

Price/size come from the Anchor event payload (`raw` b64), offsets locked from
/tmp/pump_amm.json:
    BuyEvent  : pool_base_token_reserves@48  pool_quote_token_reserves@56  quote_amount_in@64   pool@120 user@152
    SellEvent : pool_base_token_reserves@48  pool_quote_token_reserves@56  quote_amount_out@64  pool@120 user@152
    CreatePool: base_mint@50  pool@173  (graduation moment)
mid = quote_res / base_res (scale-invariant for the 2x cross + all RICH ratios);
SOL size = quote_amount / 1e9; pool SOL depth (mcap analog) = quote_res / 1e9.

Trigger axis (swept in one pass, independent per pool, first qualifying cross only):
  mirror  : ref = first observed AMM price for the pool; fire at first 2x. (parity w/ cont_2x)
  at_grad : same, but only pools whose CreatePoolEvent we saw in-window; records t_since_grad.
  rolling : ref = min price over the trailing --rolling-window-s; fire at first 2x-from-trailing-min.

Label (all modes): from the realistic post-cross fill (next trade's mid),
  y=1 if mid reaches +TP (0.5x) before -STOP (0.3x); else y=0; censored=1 on horizon timeout.
Every feature is frozen at-or-before the cross; the label uses only trades strictly after the fill.

Features = the continuation_tracker_rich RICH set (22, ported verbatim) + AMM-native
(pool depth, age, t_since_grad). Output: one jsonl row per (pool,mode) resolved event.
"""
from __future__ import annotations
import argparse, glob, gzip, json, math, os, random, struct, sys, time
from collections import deque, Counter

RNG = random.Random(1)   # fixed seed: reproducible wallet-id null

WSOL = "So11111111111111111111111111111111111111112"
DISC_BUY    = bytes.fromhex("67f4521f2cf57777")
DISC_SELL   = bytes.fromhex("3e2f370aa503dc2a")
DISC_CREATE = bytes.fromhex("b1310cd2a076a774")

TP, STOP = 0.50, 0.30
RECENT_LAG = 10
STALE_SEC = 1800.0


def _u64(b, off): return struct.unpack_from("<Q", b, off)[0]


def decode_amm(raw_b64):
    """Return dict for Buy/Sell/CreatePool PumpSwap events, else None."""
    try:
        b = __import__("base64").b64decode(raw_b64)
    except Exception:
        return None
    if len(b) < 16:
        return None
    disc = bytes(b[:8])
    if disc == DISC_BUY or disc == DISC_SELL:
        if len(b) < 184:
            return None
        # Offsets locked EMPIRICALLY vs post_tb (IDL base/quote order is reversed on-wire):
        #   off48 = pool quote (WSOL) reserve,  off56 = pool base (memecoin) reserve.
        # Pubkey offsets confirmed by the u64 ladder ending at 112 (pool@120, user@152).
        # Trade SOL size is taken from the quote-reserve delta in the main loop (the IDL
        # quote_amount offset reads implausibly large, so it is not trusted).
        return {
            "kind": "buy" if disc == DISC_BUY else "sell",
            "quote_res": _u64(b, 48),
            "base_res": _u64(b, 56),
            "pool": bytes(b[120:152]).hex(),
            "user": bytes(b[152:184]).hex(),
        }
    if disc == DISC_CREATE:
        if len(b) < 205:
            return None
        return {"kind": "create", "base_mint": bytes(b[50:82]).hex(),
                "pool": bytes(b[173:205]).hex()}
    return None


def tb_pool_mid(row):
    """Independent price from post_tb: largest WSOL bal / largest base bal (pool vaults
    dwarf the trader). Used only to self-validate the event decode. Returns mid or None."""
    post = row.get("post_tb") or []
    wsol = [e for e in post if e.get("mint") == WSOL]
    base = [e for e in post if e.get("mint") and e.get("mint") != WSOL]
    if not wsol or not base:
        return None
    def amt(e):
        try: return int(e.get("amt_str", "0"))
        except Exception: return 0
    q = max(amt(e) for e in wsol); b = max(amt(e) for e in base)
    return (q / b) if b else None


# ---- RICH feature set, ported verbatim from continuation_tracker_rich._features ----
BASE6 = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq"]
RICH = BASE6 + ["t_to_2x", "log_t_to_2x", "accel", "last_gap", "mcap_sol", "vol_sol",
                "sol_per_trade", "max_buy_sol", "whale_frac", "net_flow", "n_buyers",
                "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]


def features_window(W, real_sol):
    """W = list of trades [{ts,mid,sol,is_buy,user}] from ref..cross inclusive."""
    nt = len(W); cross = W[-1]; mid = cross["mid"]; ts = cross["ts"]
    p0 = W[0]["mid"]; t0 = W[0]["ts"]
    mn = min(t["mid"] for t in W); mx = max(t["mid"] for t in W)
    nb = sum(1 for t in W if t["is_buy"])
    buyers = set(t["user"] for t in W if t["is_buy"])
    sellers = set(t["user"] for t in W if not t["is_buy"])
    buy_sol = sum(t["sol"] for t in W if t["is_buy"])
    sell_sol = sum(t["sol"] for t in W if not t["is_buy"])
    vol_sol = buy_sol + sell_sol
    max_buy_sol = max((t["sol"] for t in W if t["is_buy"]), default=0.0)
    sigc = Counter(t["user"] for t in W)
    n_up = 0; prev = p0
    for t in W[1:]:
        if t["mid"] > prev: n_up += 1
        prev = t["mid"]
    recent_w = [t["mid"] for t in W[-(RECENT_LAG + 1):]]
    ref = recent_w[0] if recent_w else mid
    tsw = [t["ts"] for t in W[-21:]]
    span = (tsw[-1] - tsw[0]) if len(tsw) >= 2 else 0.0
    tps = (len(tsw) - 1) / max(span, 0.1) if len(tsw) >= 2 else 0.0
    uniq = len(set(t["user"] for t in W[-21:]))
    t_to = max(ts - t0, 1e-3); overall_tps = nt / t_to
    vol = max(vol_sol, 1e-9)
    last_gap = (ts - W[-2]["ts"]) if nt >= 2 else 0.0
    return {
        "dd": mn / p0 - 1.0, "buy_frac": nb / nt, "ntr": nt,
        "recent": (mid / ref - 1.0) if ref else 0.0, "tps": tps, "uniq": uniq,
        "t_to_2x": t_to, "log_t_to_2x": math.log(t_to), "accel": tps / max(overall_tps, 1e-9),
        "last_gap": last_gap, "mcap_sol": real_sol / 1e9, "vol_sol": vol_sol,
        "sol_per_trade": vol_sol / nt, "max_buy_sol": max_buy_sol, "whale_frac": max_buy_sol / vol,
        "net_flow": (buy_sol - sell_sol) / vol, "n_buyers": len(buyers), "n_sellers": len(sellers),
        "bs_ratio": len(buyers) / (len(sellers) + 1), "signer_conc": (max(sigc.values()) / nt) if sigc else 0.0,
        "up_frac": n_up / nt, "max_runup": mx / p0 - 1.0,
    }


def barrier_outcome(up_env, dn_env, final_ret, final_dt, tp, stop):
    """First-touch outcome for a (tp, stop) policy from the monotonic envelopes.
    up_env: list[(dt,ret)] new-max (increasing ret); dn_env: new-min (decreasing ret).
    Returns (y, realized_ret, dur_s, censored)."""
    t_tp = next((dt for dt, r in up_env if r >= tp), None)
    t_st = next((dt for dt, r in dn_env if r <= -stop), None)
    if t_tp is not None and (t_st is None or t_tp <= t_st):
        return 1, tp, t_tp, 0
    if t_st is not None:
        return 0, -stop, t_st, 0
    return 0, final_ret, final_dt, 1   # neither barrier in-window -> exit at final (censored)


CURVE_FEATS = ["curve_has", "curve_max_runup", "curve_ntr", "curve_nbuy_frac",
               "curve_vol_sol", "curve_mcap_grad", "curve_age_s"]


def curve_feats(cs, cross_t):
    """Pre-graduation bonding-curve priors for a mint (orthogonal to AMM momentum).
    cs = [first_t, launch_mid, max_mid, ntr, nbuy, vol_sol, last_rsol] accumulated from
    the mint's TradeEvents (all strictly before graduation => before the AMM cross => causal)."""
    if not cs:
        return {"curve_has": 0, "curve_max_runup": 0.0, "curve_ntr": 0, "curve_nbuy_frac": 0.0,
                "curve_vol_sol": 0.0, "curve_mcap_grad": 0.0, "curve_age_s": 0.0}
    first_t, launch_mid, max_mid, ntr, nbuy, vol, last_rsol = cs
    return {"curve_has": 1,
            "curve_max_runup": (max_mid / launch_mid - 1.0) if launch_mid else 0.0,
            "curve_ntr": ntr,
            "curve_nbuy_frac": (nbuy / ntr) if ntr else 0.0,
            "curve_vol_sol": vol,
            "curve_mcap_grad": last_rsol / 1e9,
            "curve_age_s": max(0.0, cross_t - first_t)}


class RepTable:
    """As-of wallet reputation, Laplace (win+1)/(seen+2). features() BEFORE update().
    Ported verbatim from cont_aug_features.ShredRep; here keyed on PumpSwap `user`
    (the swapper) instead of shred signer. rep[w] = [seen, win]."""
    __slots__ = ("rep",)
    def __init__(self): self.rep = {}

    def features(self, wallets, prefix=""):
        sset = [w for w in wallets if w]
        known = []; nsmart = 0
        for w in sset:
            rc = self.rep.get(w)
            if rc and rc[0] > 0:
                v = (rc[1] + 1.0) / (rc[0] + 2.0)
                known.append(v)
                if rc[0] >= 3 and v >= 0.6:
                    nsmart += 1
        return {
            prefix + "rep_mean": (sum(known) / len(known)) if known else 0.5,
            prefix + "rep_max": max(known) if known else 0.0,
            prefix + "rep_nknown": len(known),
            prefix + "rep_frac_known": (len(known) / len(sset)) if sset else 0.0,
            prefix + "rep_frachigh": (sum(1 for r in known if r > 0.5) / len(known)) if known else 0.0,
            prefix + "rep_nsmart": nsmart,
        }

    def update(self, wallets, y):
        y = int(y)
        for w in wallets:
            if not w:
                continue
            rc = self.rep.get(w)
            if rc is None:
                self.rep[w] = [1, y]
            else:
                rc[0] += 1; rc[1] += y


class ModeState:
    __slots__ = ("phase", "cross_t", "fill", "fill_t", "window", "cross_qres", "first_t",
                 "grad_t", "rep", "rep_buyers", "curve",
                 "up_env", "dn_env", "max_ret", "min_ret", "final_ret", "final_dt",
                 "path", "last_path_dt", "t05", "below05", "t05_slot", "below05_slot", "last_slot")
    def __init__(self):
        self.phase = "pre"; self.cross_t = None
        self.fill = None; self.fill_t = None
        self.window = None; self.cross_qres = 0; self.first_t = None; self.grad_t = None
        self.rep = None; self.rep_buyers = None; self.curve = None
        self.up_env = None; self.dn_env = None
        self.max_ret = -9e9; self.min_ret = 9e9; self.final_ret = 0.0; self.final_dt = 0.0
        self.path = None; self.last_path_dt = -1e9
        self.t05 = None; self.below05 = None   # exact: first +0.5x, first drop back below +0.5x
        self.t05_slot = None; self.below05_slot = None; self.last_slot = None   # SLOT-space window (tradeable)


class Pool:
    __slots__ = ("trades", "grad_t", "first_t", "first_mid", "last_t", "modes", "min_mono")
    def __init__(self, modes):
        self.trades = deque(maxlen=4000)
        self.grad_t = None; self.first_t = None; self.first_mid = None; self.last_t = 0.0
        self.modes = {m: ModeState() for m in modes}
        self.min_mono = deque()   # (ts, mid) increasing-mid monotonic deque for sliding-window min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dirs", nargs="+", default=[
        "/root/the-distribution-will-manifest/grpc_capture",
        "/mnt/storagebox/backup/archive/grpc_capture",
    ])
    ap.add_argument("--since", default="20260624")
    ap.add_argument("--until", default="99999999")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--tp", type=float, default=TP)
    ap.add_argument("--stop", type=float, default=STOP)
    ap.add_argument("--rolling-window-s", type=float, default=600.0)
    ap.add_argument("--label-horizon-s", type=float, default=1800.0)
    ap.add_argument("--record-bound-up", type=float, default=20.0, help="stop recording once ret reaches +this (so the up-tail is captured, not clipped)")
    ap.add_argument("--record-bound-dn", type=float, default=0.6, help="stop recording once ret reaches -this (any STOP<=this determined)")
    ap.add_argument("--record-max-pts", type=int, default=400, help="cap envelope/path points per event")
    ap.add_argument("--path-sample-s", type=float, default=5.0, help="record a full-path point at most every N seconds (for trailing/time-stop sim)")
    ap.add_argument("--entry-lag-s", type=float, default=0.0,
                    help="delay the fill until this many seconds AFTER the cross (execution-latency + look-ahead stress test). Features stay frozen at the cross; only entry timing moves.")
    ap.add_argument("--lead-offset-s", type=float, default=0.0,
                    help="freeze feature snapshot at cross_t - lead_offset_s (approx shred-fire moment; shred lead ~= 0.16s vs gRPC). Truncates the upper end of the feature window; label unchanged.")
    ap.add_argument("--rep", action="store_true", default=True, help="emit as-of wallet-reputation features (rep_* + shuf_* null) keyed on PumpSwap user")
    ap.add_argument("--no-rep", dest="rep", action="store_false")
    ap.add_argument("--curve", action="store_true", default=True, help="emit pre-graduation bonding-curve priors (curve_*) joined by mint")
    ap.add_argument("--no-curve", dest="curve", action="store_false")
    ap.add_argument("--modes", default="mirror,at_grad,rolling")
    ap.add_argument("--out", default="/root/the-distribution-will-manifest/bot_data/grad_cont_panel.jsonl")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--stdin", action="store_true", help="read pre-filtered capture lines from stdin (e.g. zcat|grep) instead of globbing files")
    ap.add_argument("--validate-only", action="store_true", help="decode self-check vs post_tb on first N events, then exit")
    ap.add_argument("--validate-n", type=int, default=400)
    ap.add_argument("--progress-every", type=int, default=500000)
    ap.add_argument("--entry-sweep", action="store_true",
                    help="sweep entry-trigger growth on AGED graduated pools (avoid the crowded graduation event)")
    ap.add_argument("--sweep-ks", default="1.3,1.5,1.75,2.0,2.5,3.0",
                    help="entry trigger multiples X (fire when price >= X * trailing-window min)")
    ap.add_argument("--min-age-s", type=float, default=300.0,
                    help="only fire after the pool has traded this long since first AMM print (skip graduation frenzy)")
    a = ap.parse_args()
    if a.entry_sweep:
        _ks = [float(x) for x in a.sweep_ks.split(",")]
        modes = [f"k{x:g}" for x in _ks]
        MODE_K = {f"k{x:g}": x for x in _ks}
        MODE_KIND = {m: "rolling_aged" for m in modes}
    else:
        modes = [m for m in a.modes.split(",") if m]
        MODE_K = {m: a.k for m in modes}
        MODE_KIND = {"mirror": "launch", "at_grad": "grad", "rolling": "rolling"}

    # collect capture files by date
    seen = {}
    for d in a.capture_dirs:
        for fn in glob.glob(os.path.join(d, "capture_*.jsonl*")):
            base = os.path.basename(fn)
            try: stamp = base.split("capture_")[1][:8]
            except Exception: continue
            if stamp < a.since or stamp > a.until: continue
            key = base.replace(".gz", "")
            if key not in seen or (seen[key].endswith(".gz") and not fn.endswith(".gz")):
                seen[key] = fn
    files = [seen[k] for k in sorted(seen)]
    if a.max_files: files = files[:a.max_files]
    sys.stderr.write(f"[grad] {len(files)} capture files {a.since}->{a.until} modes={modes}\n"); sys.stderr.flush()

    pools = {}
    rep_real = RepTable(); rep_shuf = RepTable(); all_wallets = []   # all_wallets: surrogate pool for the wallet-id null
    curve = {}        # mint -> [first_t, launch_mid, max_mid, ntr, nbuy, vol_sol, last_rsol]
    pool_mint = {}    # pool(hex) -> base mint (b58, from post_tb) for the curve join
    out = None if a.validate_only else open(a.out, "w")
    n_seen = n_amm = n_cross = n_rows = 0
    vN = vagree = vhave = 0
    vdiffs = []
    t_start = time.time()

    def emit(pool_key, mode, mst, y, ret, dur, censored):
        nonlocal n_rows
        W = mst.window; real_sol = mst.cross_qres
        # CAUSAL GUARD: every feature trade must be at/before the cross.
        assert W[-1]["ts"] <= mst.cross_t + 1e-6, "LOOKAHEAD: feature window extends past cross"
        assert mst.fill is not None, "no fill recorded"
        row = features_window(W, real_sol)
        row.update({
            "pool": pool_key, "mode": mode, "y": y, "ret": ret,
            "cross_t": mst.cross_t, "dur_s": dur, "censored": censored,
            "fill_mid": mst.fill, "entry_slip": (mst.fill / W[-1]["mid"] - 1.0) if W[-1]["mid"] else 0.0,
            "depth_sol": real_sol / 1e9,
            "first_seen_age_s": mst.cross_t - (mst.first_t or mst.cross_t),
            "t_since_grad": (mst.cross_t - mst.grad_t) if mst.grad_t is not None else None,
            # forward-path envelopes for the exit/sizing sweep (compact, monotonic):
            "up_env": mst.up_env or [], "dn_env": mst.dn_env or [], "path": mst.path or [],
            "final_ret": round(mst.final_ret, 4), "final_dt": round(mst.final_dt, 1),
            "t05_s": (round(mst.t05, 2) if mst.t05 is not None else None),
            "window05_slots": ((mst.below05_slot - mst.t05_slot) if (mst.t05_slot is not None and mst.below05_slot is not None)
                               else ((mst.last_slot - mst.t05_slot) if mst.t05_slot is not None else None)),
            "window05_s": (round((mst.below05 - mst.t05), 2) if (mst.t05 is not None and mst.below05 is not None)
                           else (round(mst.final_dt - mst.t05, 2) if mst.t05 is not None else None)),
        })
        if mst.rep:
            row.update(mst.rep)
        if mst.curve:
            row.update(mst.curve)
        out.write(json.dumps(row, separators=(",", ":")) + "\n"); n_rows += 1

    def iter_lines():
        if a.stdin:
            for ln in sys.stdin:
                yield ln
            return
        for fi, fn in enumerate(files):
            try:
                fh = gzip.open(fn, "rt") if fn.endswith(".gz") else open(fn, "rt")
            except Exception as e:
                sys.stderr.write(f"[grad] open fail {fn}: {e}\n"); continue
            for ln in fh:
                yield ln
            fh.close()

    for ln in iter_lines():
        n_seen += 1
        if (n_seen % a.progress_every) == 0:
            sys.stderr.write(f"[grad] seen={n_seen} amm={n_amm} cross={n_cross} rows={n_rows} "
                             f"pools={len(pools)} ({n_seen/max(1,time.time()-t_start):.0f}/s)\n")
            sys.stderr.flush()
        if ("PumpSwap.BuyEvent" not in ln and "PumpSwap.SellEvent" not in ln
                and "CreatePoolEvent" not in ln and "TradeEvent" not in ln):
            continue
        try: row = json.loads(ln)
        except Exception: continue
        ts = row.get("t") or 0.0
        if row.get("event") == "TradeEvent":
            # accumulate the mint's bonding-curve history (pre-graduation prior). All
            # curve trades precede graduation -> precede the AMM cross -> causal.
            if a.curve:
                m = row.get("mint"); vs = row.get("vsol"); vt = row.get("vtok")
                if m and vs and vt:
                    mid = vs / vt
                    cs = curve.get(m)
                    if cs is None:
                        if len(curve) < 800000:
                            curve[m] = [ts, mid, mid, 1, 1 if row.get("is_buy") else 0,
                                        float(row.get("sol") or 0.0), int(row.get("rsol") or 0)]
                    else:
                        if mid > cs[2]: cs[2] = mid
                        cs[3] += 1
                        if row.get("is_buy"): cs[4] += 1
                        cs[5] += float(row.get("sol") or 0.0)
                        cs[6] = int(row.get("rsol") or cs[6])
            continue
        raw = row.get("raw")
        if not raw: continue
        d = decode_amm(raw)
        if d is None: continue
        if d["kind"] == "create":
            p = pools.get(d["pool"])
            if p is None: p = pools[d["pool"]] = Pool(modes)
            if p.grad_t is None: p.grad_t = ts
            continue
        base_res = d["base_res"]; quote_res = d["quote_res"]
        if base_res <= 0 or quote_res <= 0: continue
        mid = quote_res / base_res
        is_buy = d["kind"] == "buy"
        user = d["user"]; pool_key = d["pool"]; cur_slot = int(row.get("slot") or 0)
        n_amm += 1

        if a.validate_only:
            if vN < a.validate_n:
                vN += 1
                tbm = tb_pool_mid(row)
                if tbm:
                    vhave += 1
                    rel = abs(mid - tbm) / max(tbm, 1e-12)
                    vdiffs.append(rel)
                    if rel < 0.05: vagree += 1
                continue
            else:
                break

        p = pools.get(pool_key)
        if p is None: p = pools[pool_key] = Pool(modes)
        if a.curve and pool_key not in pool_mint:
            for e in (row.get("post_tb") or []):
                mm = e.get("mint")
                if mm and mm != WSOL:
                    pool_mint[pool_key] = mm; break
        sol = abs(quote_res - p.trades[-1]["qres"]) / 1e9 if p.trades else 0.0
        if p.first_t is None: p.first_t = ts; p.first_mid = mid
        p.last_t = ts
        tr = {"ts": ts, "mid": mid, "sol": sol, "is_buy": is_buy, "user": user, "qres": quote_res, "slot": cur_slot}
        p.trades.append(tr)

        # sliding-window min (for rolling): monotonic increasing deque of (ts,mid)
        mm = p.min_mono
        while mm and mm[-1][1] >= mid: mm.pop()
        mm.append((ts, mid))
        lo = ts - a.rolling_window_s
        while mm and mm[0][0] < lo: mm.popleft()
        roll_min = mm[0][1] if mm else mid

        snap = None  # list(p.trades), built lazily only when a cross fires (O(1) per-trade otherwise)
        for mode in modes:
            mst = p.modes[mode]
            if mst.phase == "pre":
                kind = MODE_KIND[mode]; kk = MODE_K[mode]
                if kind == "grad" and p.grad_t is None:
                    continue
                if kind == "rolling_aged" and (ts - (p.first_t or ts)) < a.min_age_s:
                    continue  # skip the crowded graduation window — only fire on AGED pools
                ref_from_min = False
                if kind in ("launch", "grad"):
                    crossed = mid >= p.first_mid * kk
                else:  # rolling / rolling_aged: trigger kk-x from the trailing-window local min
                    crossed = mid >= roll_min * kk
                    ref_from_min = crossed
                if crossed:
                    n_cross += 1
                    if snap is None: snap = list(p.trades)
                    ref_idx = 0
                    if ref_from_min:
                        ref_idx = len(snap) - 1; best = mid
                        for j in range(len(snap) - 1, -1, -1):
                            if snap[j]["ts"] < lo: break
                            if snap[j]["mid"] <= best:
                                best = snap[j]["mid"]; ref_idx = j
                    # LEAD-OFFSET (Version-A replica): truncate window's upper end to
                    # ts <= cross_t - lead_offset_s so features approximate the confirmed
                    # state at the shred-fire moment (~160ms before gRPC cross).
                    end_idx = len(snap)
                    if a.lead_offset_s > 0.0:
                        lead_ts = ts - a.lead_offset_s
                        while end_idx > 0 and snap[end_idx - 1]["ts"] > lead_ts:
                            end_idx -= 1
                        if end_idx - ref_idx < 2:
                            mst.phase = "done"  # not enough pre-cross history at lead-offset
                            continue
                    mst.phase = "await_fill"; mst.cross_t = ts
                    mst.window = snap[ref_idx:end_idx]
                    mst.cross_qres = mst.window[-1]["qres"] if a.lead_offset_s > 0.0 else quote_res
                    mst.first_t = p.first_t; mst.grad_t = p.grad_t
                    if a.rep:
                        # as-of reputation of the wallets that bought during launch->cross
                        buyers = list({t["user"] for t in mst.window if t["is_buy"]})
                        mst.rep_buyers = buyers
                        rf = rep_real.features(buyers)
                        rf.update(rep_shuf.features(buyers, prefix="shuf_"))
                        mst.rep = rf
                    if a.curve:
                        mst.curve = curve_feats(curve.get(pool_mint.get(pool_key)), ts)
            elif mst.phase == "await_fill":
                if ts < mst.cross_t + a.entry_lag_s:
                    continue  # not yet filled — wait out the entry lag (other modes proceed)
                mst.fill = mid; mst.fill_t = ts
                mst.up_env = []; mst.dn_env = []; mst.path = []; mst.last_path_dt = -1e9
                mst.max_ret = -9e9; mst.min_ret = 9e9; mst.final_ret = 0.0; mst.final_dt = 0.0
                mst.t05 = None; mst.below05 = None
                mst.t05_slot = None; mst.below05_slot = None; mst.last_slot = None
                mst.phase = "record"
            elif mst.phase == "record":
                ret = mid / mst.fill - 1.0
                dt = ts - mst.fill_t
                if ret > mst.max_ret:
                    mst.max_ret = ret; mst.up_env.append((round(dt, 1), round(ret, 4)))
                if ret < mst.min_ret:
                    mst.min_ret = ret; mst.dn_env.append((round(dt, 1), round(ret, 4)))
                # EXACT sell-window: first +0.5x cross, then first drop back below +0.5x
                if mst.t05 is None and ret >= 0.5:
                    mst.t05 = dt; mst.t05_slot = cur_slot
                elif mst.t05 is not None and mst.below05 is None and ret < 0.5:
                    mst.below05 = dt; mst.below05_slot = cur_slot
                mst.last_slot = cur_slot
                # fine path early (0.5s for first 30s, then path_sample_s) for sub-2s sell sim
                samp = 0.5 if dt < 30.0 else a.path_sample_s
                if dt - mst.last_path_dt >= samp and len(mst.path) < a.record_max_pts:
                    mst.path.append((round(dt, 2), round(ret, 4))); mst.last_path_dt = dt
                mst.final_ret = ret; mst.final_dt = dt
                if (dt > a.label_horizon_s or ret >= a.record_bound_up or ret <= -a.record_bound_dn
                        or (len(mst.up_env) + len(mst.dn_env)) >= a.record_max_pts):
                    y, rr, dd, cens = barrier_outcome(mst.up_env, mst.dn_env,
                                                      mst.final_ret, mst.final_dt, a.tp, a.stop)
                    if mst.window and len(mst.window) >= 2:
                        emit(pool_key, mode, mst, y, rr, dd, cens)
                    if a.rep and mst.rep_buyers:
                        # update AT RESOLUTION (outcome now known) -> only affects FUTURE crosses (causal).
                        rep_real.update(mst.rep_buyers, y)
                        k = len(mst.rep_buyers)
                        if all_wallets:               # wallet-id null: credit the SAME outcome to k RANDOM wallets
                            surro = RNG.sample(all_wallets, k) if len(all_wallets) >= k \
                                    else [RNG.choice(all_wallets) for _ in range(k)]
                            rep_shuf.update(surro, y)
                        all_wallets.extend(mst.rep_buyers)
                        if len(all_wallets) > 400000:
                            all_wallets[:] = RNG.sample(all_wallets, 200000)
                    mst.phase = "done"; mst.window = None; mst.rep_buyers = None
                    mst.up_env = None; mst.dn_env = None; mst.path = None

        if (n_seen % 300000) == 0:
            # drop pools with no trade in >horizon (dead coins); any open record on them
            # would only ever time out, so losing it is harmless and bounds memory.
            dead = [k for k, pp in pools.items()
                    if (ts - pp.last_t) > max(a.label_horizon_s, STALE_SEC)]
            for k in dead: pools.pop(k, None)

    if a.validate_only:
        med = sorted(vdiffs)[len(vdiffs) // 2] if vdiffs else float("nan")
        sys.stderr.write(f"[grad] VALIDATE: n={vN} with_tb={vhave} agree<5%={vagree} "
                         f"({100*vagree/max(1,vhave):.1f}%) median_rel_diff={med:.4g}\n")
        return
    out.close()
    sys.stderr.write(f"[grad] DONE rows={n_rows} cross={n_cross} amm_trades={n_amm} seen={n_seen}\n")


if __name__ == "__main__":
    main()
