#!/usr/bin/env python3
"""REALIZABLE profit-target (TP) sweep — sell-into-strength model at every TP level.

For each target TP, per top-decile fire (time-stop T, hard stop -0.3x):
  reach TP before stop, by T?
    YES + peak >= TP with ROOM above (price kept rising) -> CLEAN: book +TP (sold into strength)
    YES + no room (TP ~ the local top)                    -> SPIKE: book TP*(1-SPIKE_FRAC) (mistimed turn)
    NO, stop first                                         -> -0.3
    NO, neither by T                                       -> ret at T (fizzle cut), clamped
net = realized - fee. Shows where realizable net PEAKS (vs the naive 'touched' sweep that ran to +8x).

Usage: ./venv/bin/python grad_cont_tp_sweep.py [panel.jsonl] [fee] [Tstop_s]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pp.jsonl"
FEE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
TSTOP = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
ROOM_MULT = 1.10     # "room above" = price went >=10% past your exit (sell into strength)
SPIKE_FRAC = 0.50    # mistimed top: capture (1-this) of the target
TPS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
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
    return clamp(last if last is not None else 0.0)


def realized(r, tp, T):
    up, dn, path = r["up_env"], r["dn_env"], r.get("path", [])
    peak = up[-1][1] if up else 0.0
    t_tp = first_touch(up, tp, True); tst = first_touch(dn, -0.3, False)
    evs = [(t, k) for t, k in ((t_tp, "tp"), (tst, "stop")) if t is not None and t <= T]
    if evs:
        t, k = min(evs)
        if k == "tp":
            clean = peak >= (1 + tp) * ROOM_MULT - 1
            return (tp if clean else tp * (1 - SPIKE_FRAC)), clean
        return -0.3, False
    return ret_at_T(path, r.get("final_ret", 0.0), T), False


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "up_env" in r]
print(f"panel={PANEL}  n={len(rows)}  fee={FEE}  time_stop={TSTOP:.0f}s  room=+{int((ROOM_MULT-1)*100)}%  spike_frac={SPIKE_FRAC}")
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
    te = rs[sp:]; k = max(1, int(len(te) * 0.10)); picks = [te[i] for i in np.argsort(-p)[:k]]
    print(f"\n=== {m}  AUC={auc:.3f}  fired={len(picks)} ===")
    print("   TP     clean-fill%   net/trade")
    best = (None, -9)
    for tp in TPS:
        vals = []; cleans = 0
        for r in picks:
            v, c = realized(r, tp, TSTOP); vals.append(v - FEE); cleans += c
        net = float(np.mean(vals))
        if net > best[1]: best = (tp, net)
        mark = "  <-- current" if abs(tp - 0.5) < 1e-9 else ""
        print(f"   {tp:<5}  {cleans/len(picks):.2f}          {net:+.3f}{mark}")
    print(f"   => realizable net PEAKS at TP={best[0]}  (net {best[1]:+.3f})")
