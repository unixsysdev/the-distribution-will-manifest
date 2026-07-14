#!/usr/bin/env python3
"""PROPER exit/sizing sweep on the graduation model's top-decile.

Recording now captures the up-tail to +20x + a 5s-sampled forward path, so we can ask:
  (A) PEAK-RETURN distribution  -> where does the up-tail actually die? (is +1.5x the top?)
  (B) FIXED TP x STOP grid (TP out to +8x)  -> where does fixed-TP net peak / roll over?
  (C) TRAILING STOP (arm, trail)  -> ride the up-move, exit on a pullback (don't sell mid-rally)
  (D) TIME STOP                   -> cap holding time

All net numbers are realizable (gap-0 instant pops booked as no-fill) minus a flat fee.
CAVEAT printed inline: "price touched X" overstates what you'd realize on a market-sell exit.

Usage: ./venv/bin/python grad_cont_sweep.py [panel.jsonl] [fee]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/root/the-distribution-will-manifest/bot_data/grad_panel_sweep.jsonl"
FEE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
GAP0_S = 1.0
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EXTRA = ["depth_sol", "first_seen_age_s"]
TPS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
STOPS = [0.3, 0.5]
TRAILS = [(0.3, 0.2), (0.3, 0.3), (0.5, 0.3), (1.0, 0.3), (1.0, 0.5)]   # (arm, trail_frac)
TIMES = [60, 300, 900]


def barrier_outcome(up_env, dn_env, final_ret, final_dt, tp, stop):
    t_tp = next((dt for dt, r in up_env if r >= tp), None)
    t_st = next((dt for dt, r in dn_env if r <= -stop), None)
    if t_tp is not None and (t_st is None or t_tp <= t_st):
        return tp, t_tp
    if t_st is not None:
        return -stop, t_st
    return final_ret, final_dt


def trailing_outcome(path, final_ret, final_dt, arm, trail, hard_stop):
    """Hard stop at -hard_stop until peak>=arm; then exit on a `trail` pullback from the peak price."""
    peak = -9e9; armed = False
    for dt, r in path:
        if not armed and r <= -hard_stop:
            return -hard_stop, dt
        if r > peak:
            peak = r
        if not armed and peak >= arm:
            armed = True
        if armed:
            exit_lvl = (1.0 + peak) * (1.0 - trail) - 1.0
            if r <= exit_lvl:
                return r, dt
    return final_ret, final_dt


def time_outcome(path, up_env, dn_env, final_ret, final_dt, tp, stop, T):
    rr, dd = barrier_outcome(up_env, dn_env, final_ret, final_dt, tp, stop)
    if dd <= T:
        return rr, dd
    last = final_ret
    for dt, r in path:
        if dt > T:
            break
        last = r
    return last, min(T, final_dt)


def mean_net(picks, fn):
    rets = []
    for r in picks:
        rr, dd = fn(r)
        if dd < GAP0_S:
            continue
        rets.append(rr - FEE)
    return (float(np.mean(rets)), len(rets)) if rets else (float("nan"), 0)


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "path" in r and "up_env" in r]
print(f"panel={PANEL}  rows_with_path={len(rows)}  fee={FEE}")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

for m in sorted(set(r["mode"] for r in rows)):
    rs = [r for r in rows if r["mode"] == m]
    rs.sort(key=lambda r: r.get("cross_t", 0.0))
    n = len(rs)
    if n < 400:
        print(f"\n=== {m}: n={n} too small ==="); continue
    y = np.array([int(r["y"]) for r in rs])
    cols = RICH + EXTRA + (["t_since_grad"] if m == "at_grad" else [])
    X = np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rs]); X[~np.isfinite(X)] = 0.0
    split = int(n * 0.7)
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06,
                                         l2_regularization=1.0, min_samples_leaf=40)
    clf.fit(X[:split], y[:split])
    p = clf.predict_proba(X[split:])[:, 1]
    auc = roc_auc_score(y[split:], p)
    te = rs[split:]
    k = max(1, int(len(te) * 0.10))
    picks = [te[i] for i in np.argsort(-p)[:k]]
    depth = np.median([float(r.get("depth_sol", 0.0)) for r in picks])
    # (A) peak-return distribution among the top-decile picks
    peaks = np.array([(r["up_env"][-1][1] if r["up_env"] else r.get("final_ret", 0.0)) for r in picks])
    qs = {q: np.quantile(peaks, q) for q in (0.25, 0.5, 0.75, 0.9, 0.95)}
    reach = {lvl: float(np.mean(peaks >= lvl)) for lvl in (0.5, 1.0, 2.0, 3.0, 5.0)}
    print(f"\n=== {m}  AUC={auc:.3f}  top-decile={len(picks)}  median_depth={depth:.0f} SOL ===")
    print(f"   (A) PEAK ret reached: p50={qs[0.5]:+.2f} p75={qs[0.75]:+.2f} p90={qs[0.9]:+.2f} p95={qs[0.95]:+.2f}"
          f"  | reach +0.5x={reach[0.5]:.2f} +1x={reach[1.0]:.2f} +2x={reach[2.0]:.2f} +3x={reach[3.0]:.2f} +5x={reach[5.0]:.2f}")
    # (B) fixed TP x STOP
    print("   (B) FIXED TP net/trade (cols=STOP " + " ".join(f"s{int(s*100)}" for s in STOPS) + "):")
    best = (None, -9)
    for tp in TPS:
        cells = []
        for stop in STOPS:
            net, ntr = mean_net(picks, lambda r, tp=tp, stop=stop: barrier_outcome(r["up_env"], r["dn_env"], r.get("final_ret", 0), r.get("final_dt", 0), tp, stop))
            cells.append(net)
            if net == net and net > best[1]: best = ((f"TP{tp}/S{stop}",), net)
        print(f"       TP={tp:<4} " + "  ".join(f"{c:+.3f}" if c == c else "  nan" for c in cells))
    # (C) trailing stop
    print("   (C) TRAILING stop (arm, trail) -> net:")
    for arm, trail in TRAILS:
        net, ntr = mean_net(picks, lambda r, arm=arm, trail=trail: trailing_outcome(r["path"], r.get("final_ret", 0), r.get("final_dt", 0), arm, trail, 0.3))
        print(f"       arm=+{arm:<3} trail={int(trail*100)}%:  net={net:+.3f}")
    # (D) time stop at TP0.5/STOP0.3
    print("   (D) TIME stop @ TP0.5/STOP0.3 -> net:")
    for T in TIMES:
        net, ntr = mean_net(picks, lambda r, T=T: time_outcome(r["path"], r["up_env"], r["dn_env"], r.get("final_ret", 0), r.get("final_dt", 0), 0.5, 0.3, T))
        print(f"       T={T:<4}s:  net={net:+.3f}")
    bn = best[1]
    print(f"   BEST fixed cell: {best[0][0]} net={bn:+.3f}  (reminder: 'touched' >> 'realized' at high TP)")
