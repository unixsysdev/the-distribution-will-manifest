#!/usr/bin/env python3
"""Min-sell-latency constraint: we can't sell before L seconds after entry (detect+land+detect+land).
So a +0.5x reached in < L is NOT cleanly capturable -> we hold and sell at the first chance >= L.

Reports: fraction of winners that hit +0.5x in <1/<2/<5s (exact, from envelope times), and the net
under the constraint (sub-L pops booked at the realistic ret>=L from the path), bracketed by
optimistic(sub-L still books +0.5) and pessimistic(sub-L books 0 = total miss).

Usage: ./venv/bin/python grad_cont_latency.py [panel.jsonl] [fee] [L_s] [Tstop_s]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pp.jsonl"
FEE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
L = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
TSTOP = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0
CLEAN_PK = 0.60
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
def ret_after(path, L, final_ret):
    for dt, r in path:
        if dt >= L:
            return clamp(r)
    return clamp(final_ret)
def ret_at(path, T):
    last = None
    for dt, r in path:
        if dt <= T: last = r
        else: break
    return clamp(last if last is not None else 0.0)


def outcome(r, L, T, book_subL):
    """book_subL: 'real' (sell at ret>=L), 'opt' (+0.5), 'pess' (0)."""
    up, dn, path = r["up_env"], r["dn_env"], r.get("path", [])
    peak = up[-1][1] if up else 0.0
    t05 = first_touch(up, 0.5, True); tst = first_touch(dn, -0.3, False)
    evs = sorted([(t, k) for t, k in ((t05, "tp"), (tst, "stop")) if t is not None and t <= T])
    if evs:
        t, k = evs[0]
        if k == "stop":
            return -0.3
        if t >= L:
            return 0.5 if peak >= CLEAN_PK else 0.2
        # +0.5x too fast to capture cleanly
        if book_subL == "opt": return 0.5
        if book_subL == "pess": return 0.0
        return ret_after(path, L, r.get("final_ret", 0.0))
    return ret_at(path, T)


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "up_env" in r]
print(f"panel={PANEL}  n={len(rows)}  fee={FEE}  min_sell_latency={L}s  time_stop={TSTOP}s")
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
    # exact time-to-+0.5x among picks that reach it before stop
    t05s = []
    for r in picks:
        t05 = first_touch(r["up_env"], 0.5, True); tst = first_touch(r["dn_env"], -0.3, False)
        if t05 is not None and (tst is None or t05 <= tst):
            t05s.append(t05)
    t05s = np.array(t05s) if t05s else np.array([1e9])
    nr = lambda mode: float(np.mean([outcome(r, L, TSTOP, mode) - FEE for r in picks]))
    print(f"\n=== {m}  AUC={auc:.3f}  fired={len(picks)}  reach+0.5x={len(t05s)} ===")
    print(f"   +0.5x in <1s={np.mean(t05s<1):.2f}  <2s={np.mean(t05s<2):.2f}  <5s={np.mean(t05s<5):.2f}  "
          f"(these can't be cleanly captured)")
    print(f"   NET/trade @L={L}s:  realistic={nr('real'):+.3f}   [optimistic(sub-L=+0.5)={nr('opt'):+.3f}  "
          f"pessimistic(sub-L=0)={nr('pess'):+.3f}]   vs no-latency(+0.5 always)≈ the optimistic col")
