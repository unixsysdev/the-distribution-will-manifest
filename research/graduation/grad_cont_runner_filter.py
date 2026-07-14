#!/usr/bin/env python3
"""Spike-vs-runner filter test. Uses exact window05_s + fine path.

  runner = reached +0.5x before -0.3x AND price stayed >= +0.5x for >= W sec (sellable window)
  spike  = reached +0.5x but window < W (violent, can't sell into)

Q1: is 'clean runner' predictable from PRE-CROSS RICH features? (AUC) -> can we filter ex-ante?
Q2: does firing on the clean-runner model (vs the plain reach-+0.5x model) raise REALIZABLE net?
    realized = sell-into-strength at t05+DELTA from the fine path (clamped) ; stop=-0.3 ; fizzle=final.

Usage: ./venv/bin/python grad_cont_runner_filter.py [panel.jsonl] [W_s] [delta_s] [fee]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pp.jsonl"
W = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
FEE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
CAP = 1.5
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EXTRA = ["depth_sol", "first_seen_age_s"]


def clamp(x, c=CAP): return max(-1.0, min(c, x))
def ftouch(env, lvl, up):
    for dt, r in env:
        if (r >= lvl) if up else (r <= lvl): return dt
    return None


def reached_05(r):
    t05 = r.get("t05_s"); tst = ftouch(r["dn_env"], -0.3, False)
    return (t05 is not None) and (tst is None or t05 <= tst), t05, tst


def realized_sellinto(r):
    ok, t05, tst = reached_05(r)
    if ok:
        for dt, rr in r.get("path", []):
            if dt >= t05 + DELTA:
                return clamp(rr)
        return clamp(r.get("final_ret", 0.0))
    if tst is not None:
        return -0.3
    return clamp(r.get("final_ret", 0.0))


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "window05_s" in r]
print(f"panel={PANEL}  n={len(rows)}  runner_window>={W}s  sell_latency={DELTA}s  fee={FEE}  cap={CAP}")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def fit_score(rs, target, sp, cols):
    X = np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rs]); X[~np.isfinite(X)] = 0.0
    yt = np.array(target)
    if yt[:sp].sum() < 10 or yt[sp:].sum() < 5: return None, None
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06,
                                         l2_regularization=1.0, min_samples_leaf=40)
    clf.fit(X[:sp], yt[:sp]); return clf.predict_proba(X[sp:])[:, 1], yt[sp:]


for m in sorted(set(r["mode"] for r in rows)):
    rs = [r for r in rows if r["mode"] == m]; rs.sort(key=lambda r: r.get("cross_t", 0.0))
    n = len(rs)
    if n < 400: continue
    cols = RICH + EXTRA + (["t_since_grad"] if m == "at_grad" else [])
    reach = []; runner = []; w05 = []
    for r in rs:
        ok, _, _ = reached_05(r); reach.append(int(ok))
        win = r.get("window05_s")
        runner.append(int(ok and (win is not None) and win >= W))
        if ok and win is not None: w05.append(win)
    reach = np.array(reach); runner = np.array(runner); w05 = np.array(w05) if w05 else np.array([0.0])
    sp = int(n * 0.7)
    print(f"\n=== {m}  n={n}  reach+0.5x={reach.mean():.2f}  clean-runner(base)={runner.mean():.2f} ===")
    print(f"   window05 among reachers: p25={np.quantile(w05,.25):.0f}s p50={np.quantile(w05,.5):.0f}s "
          f"p75={np.quantile(w05,.75):.0f}s  | >=2s={np.mean(w05>=2):.2f} >=5s={np.mean(w05>=5):.2f} >=10s={np.mean(w05>=10):.2f}")
    # Q1: predict clean-runner from pre-cross features
    p_run, yte_run = fit_score(rs, runner, sp, cols)
    p_reach, yte_reach = fit_score(rs, reach, sp, cols)
    if p_run is None or p_reach is None:
        print("   (too few positives for model)"); continue
    auc_run = roc_auc_score(yte_run, p_run)
    te = rs[sp:]
    # Q2: realizable sell-into net, top-decile by reach-model vs by runner-model
    realN = np.array([realized_sellinto(r) - FEE for r in te])
    k = max(1, int(len(te) * 0.10))
    top_reach = np.argsort(-p_reach)[:k]; top_run = np.argsort(-p_run)[:k]
    run_frac_reach = runner[sp:][top_reach].mean(); run_frac_run = runner[sp:][top_run].mean()
    print(f"   Q1 predict clean-runner: AUC={auc_run:.3f}  (>0.6 => filterable ex-ante)")
    print(f"   Q2 realizable sell-into net/trade @top-decile:")
    print(f"      plain reach-model:  net={realN[top_reach].mean():+.3f}   (runners in fires={run_frac_reach:.2f})")
    print(f"      runner-filter model:net={realN[top_run].mean():+.3f}   (runners in fires={run_frac_run:.2f})")
