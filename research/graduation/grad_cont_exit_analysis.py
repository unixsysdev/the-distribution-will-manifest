#!/usr/bin/env python3
"""Rigorous '+0.5x sold into strength' realization analysis on the model's top-decile.

The user's policy: aim for a clean +0.5x by SELLING INTO THE UP-MOVE (fire the market-sell
the instant price ticks through +0.5x). That fills cleanly ONLY if the price is still rising
there (buyers to sell into). So per fired event we classify the outcome:

  reach +0.5x before -0.3x ?
    YES + peak >= CLEAN_PK  -> CLEAN: crossed +0.5x with room above => sell into strength => book +0.50
    YES + peak <  CLEAN_PK  -> SPIKE: +0.5x was ~the top, we fire as it turns => book SPIKE_BOOK (haircut)
    NO, hit -0.3x first      -> STOP: book -0.30
    NO, neither (fizzle)     -> book final_ret
net = realized - FEE. Compares 100%@0.5x vs a 50/50 scale-out (half clean @0.5x, half trailing).

Usage: ./venv/bin/python grad_cont_exit_analysis.py [panel.jsonl] [fee] [clean_pk] [spike_book]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pp.jsonl"
FEE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
CLEAN_PK = float(sys.argv[3]) if len(sys.argv) > 3 else 0.60   # peak needed to call the +0.5x cross "into strength"
SPIKE_BOOK = float(sys.argv[4]) if len(sys.argv) > 4 else 0.20  # what we realize when +0.5x was a spike top
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EXTRA = ["depth_sol", "first_seen_age_s"]


def first_touch(env, lvl, up):
    for dt, r in env:
        if (r >= lvl) if up else (r <= lvl):
            return dt
    return None


def clamp(x):
    # thin-pool reserve-ratio spikes produce absurd ret artifacts (base->0); cap to a
    # sane/realizable band. >+10x is unrealizable anyway, so this is still optimistic.
    return max(-1.0, min(10.0, x))


def trailing(path, final_ret, arm, trail, hard):
    peak = -9e9; armed = False
    for dt, r in path:
        if not armed and r <= -hard:
            return -hard
        if r > peak: peak = r
        if not armed and peak >= arm: armed = True
        if armed and r <= (1 + peak) * (1 - trail) - 1:
            return clamp(r)
    return clamp(final_ret)


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "up_env" in r]
print(f"panel={PANEL}  n={len(rows)}  fee={FEE}  clean_peak_threshold=+{CLEAN_PK}x  spike_book=+{SPIKE_BOOK}x")
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
    clf.fit(X[:sp], y[:sp])
    p = clf.predict_proba(X[sp:])[:, 1]
    auc = roc_auc_score(y[sp:], p)
    te = rs[sp:]; k = max(1, int(len(te) * 0.10))
    picks = [te[i] for i in np.argsort(-p)[:k]]
    clean = spike = stop = fizzle = 0
    net100 = []; net_so = []
    for r in picks:
        up, dn = r["up_env"], r["dn_env"]
        peak = up[-1][1] if up else r.get("final_ret", 0.0)
        t05 = first_touch(up, 0.5, True); tst = first_touch(dn, -0.3, False)
        opp = (t05 is not None) and (tst is None or t05 <= tst)
        if opp:
            if peak >= CLEAN_PK:
                clean += 1; leg05 = 0.50
            else:
                spike += 1; leg05 = SPIKE_BOOK
        elif tst is not None:
            stop += 1; leg05 = -0.30
        else:
            fizzle += 1; leg05 = clamp(r.get("final_ret", 0.0))
        net100.append(leg05 - FEE)
        leg_tr = trailing(r.get("path", []), r.get("final_ret", 0.0), 1.0, 0.3, 0.3)
        net_so.append(0.5 * leg05 + 0.5 * leg_tr - FEE)
    N = len(picks)
    print(f"\n=== {m}  AUC={auc:.3f}  fired(top-decile)={N} ===")
    print(f"   reach +0.5x CLEAN (sell into strength): {clean/N:.2f}   "
          f"spike-to-0.5x-then-revert: {spike/N:.2f}   hit -0.3x stop: {stop/N:.2f}   fizzle: {fizzle/N:.2f}")
    print(f"   => 'sure +0.5x' (clean) rate = {clean/N:.0%};  +0.5x-or-better opportunity = {(clean+spike)/N:.0%}")
    print(f"   NET/trade  100%@+0.5x = {np.mean(net100):+.3f}   |   50/50 scale-out (half@0.5x, half trailing) = {np.mean(net_so):+.3f}")
