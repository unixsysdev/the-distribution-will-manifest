#!/usr/bin/env python3
"""Rigorous TIMING + time-stop analysis on the model's top-decile.

Answers: how long to reach +0.5x? how many winners arrive late (cost of cutting early)?
when do losers stop out? and the time-stop sweep: if not resolved by T, exit at ret(T) ('fizzle cut').
Time resolution is the 5s path sampling; t05/tstop come from the exact envelopes.

Usage: ./venv/bin/python grad_cont_time_analysis.py [panel.jsonl] [fee] [clean_pk] [spike_book]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pp.jsonl"
FEE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
CLEAN_PK = float(sys.argv[3]) if len(sys.argv) > 3 else 0.60
SPIKE_BOOK = float(sys.argv[4]) if len(sys.argv) > 4 else 0.20
TGRID = [None, 900, 600, 300, 180, 120, 60, 30]
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EXTRA = ["depth_sol", "first_seen_age_s"]


def clamp(x): return max(-1.0, min(10.0, x))
def first_touch(env, lvl, up):
    for dt, r in env:
        if (r >= lvl) if up else (r <= lvl):
            return dt
    return None
def ret_at_T(path, final_ret, T):
    last = None
    for dt, r in path:
        if dt <= T: last = r
        else: break
    return clamp(last if last is not None else (final_ret if not path else 0.0))


def ts_realized_and_hold(r, T):
    up, dn, path = r["up_env"], r["dn_env"], r.get("path", [])
    peak = up[-1][1] if up else 0.0
    t05 = first_touch(up, 0.5, True); tst = first_touch(dn, -0.3, False)
    evs = [(t, k) for t, k in ((t05, "tp"), (tst, "stop")) if t is not None and (T is None or t <= T)]
    if evs:
        t, k = min(evs)
        if k == "tp":
            return (0.5 if peak >= CLEAN_PK else SPIKE_BOOK), t
        return -0.3, t
    return ret_at_T(path, r.get("final_ret", 0.0), T if T else 1e9), (T if T else r.get("final_dt", 0.0))


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "up_env" in r]
print(f"panel={PANEL}  n={len(rows)}  fee={FEE}")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

for m in sorted(set(r["mode"] for r in rows)):
    rs = [r for r in rows if r["mode"] == m]; rs.sort(key=lambda r: r.get("cross_t", 0.0))
    n = len(rs)
    if n < 400:
        print(f"\n=== {m}: n={n} too small ==="); continue
    y = np.array([int(r["y"]) for r in rs])
    cols = RICH + EXTRA + (["t_since_grad"] if m == "at_grad" else [])
    X = np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rs]); X[~np.isfinite(X)] = 0.0
    sp = int(n * 0.7)
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06,
                                         l2_regularization=1.0, min_samples_leaf=40)
    clf.fit(X[:sp], y[:sp]); p = clf.predict_proba(X[sp:])[:, 1]
    auc = roc_auc_score(y[sp:], p)
    te = rs[sp:]; k = max(1, int(len(te) * 0.10))
    picks = [te[i] for i in np.argsort(-p)[:k]]
    # time-to-+0.5x for eventual winners (t05 before tstop); time-to--0.3x for losers
    t05s = []; tstops = []
    for r in picks:
        t05 = first_touch(r["up_env"], 0.5, True); tst = first_touch(r["dn_env"], -0.3, False)
        if t05 is not None and (tst is None or t05 <= tst): t05s.append(t05)
        elif tst is not None: tstops.append(tst)
    t05s = np.array(t05s); print(f"\n=== {m}  AUC={auc:.3f}  fired={len(picks)}  winners={len(t05s)} ===")
    if len(t05s):
        q = lambda a: np.quantile(t05s, a)
        print(f"   time-to-+0.5x (winners): p25={q(.25):.0f}s  p50={q(.5):.0f}s  p75={q(.75):.0f}s  p90={q(.9):.0f}s  p95={q(.95):.0f}s")
        print("   winners reached +0.5x by:  " + "  ".join(f"{T}s={np.mean(t05s<=T):.2f}" for T in (30,60,120,300,600)))
    if len(tstops):
        ts = np.array(tstops); print(f"   time-to--0.3x (losers): p50={np.quantile(ts,.5):.0f}s  p90={np.quantile(ts,.9):.0f}s")
    print("   TIME-STOP sweep:  T      net    clean-win%  avg-hold  winners-cut%")
    for T in TGRID:
        rr = []; holds = []; wins = 0; cut = 0
        for r in picks:
            val, hold = ts_realized_and_hold(r, T)
            rr.append(val - FEE); holds.append(hold)
            if val >= 0.5 - 1e-9: wins += 1
        # winners sacrificed: eventual winners whose t05 > T
        if T is not None and len(t05s):
            cut = float(np.mean(t05s > T))
        label = "none" if T is None else f"{T}s"
        print(f"     {label:>5}  {np.mean(rr):+.3f}   {wins/len(picks):.2f}      {np.median(holds):.0f}s    {cut:.2f}")
