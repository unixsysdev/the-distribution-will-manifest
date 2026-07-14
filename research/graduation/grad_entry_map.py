#!/usr/bin/env python3
"""Continuous entry map for AGED graduated tokens.

(1) entry-X curve (near-continuous): per trigger X, P(+0.5x before -0.3x), freq, sell-into net.
(2) lifespan: P(+0.5x before -0.3x) by token AGE bucket (does graduation-time differ from later?).
(3) the CONTINUOUS scorer: pool ALL candidates, features = RICH + depth + age (recent-growth is
    captured by max_runup/recent), predict the outcome -> AUC + top-decile net. 'Best entry' = high P.

Usage: ./venv/bin/python grad_entry_map.py [panel.jsonl] [days] [delta_s] [fee]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/root/the-distribution-will-manifest/bot_data/grad_entry_alllife_2d.jsonl"
DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
FEE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EXTRA = ["depth_sol", "first_seen_age_s"]
AGE_BUCKETS = [(0, 60), (60, 300), (300, 1800), (1800, 7200), (7200, 1e9)]


def clamp(x): return max(-1.0, min(1.5, x))
def sellinto(r):
    t05 = r.get("t05_s")
    if t05 is None:
        # didn't reach +0.5x: stop=-0.3 if it hit, else final
        dn = r.get("dn_env", [])
        if any(rr <= -0.3 for _, rr in dn): return -0.3
        return clamp(r.get("final_ret", 0.0))
    for dt, rr in r.get("path", []):
        if dt >= t05 + DELTA: return clamp(rr)
    return clamp(r.get("final_ret", 0.0))


rows = [json.loads(l) for l in open(PANEL)]
print(f"panel={PANEL}  n={len(rows)}  days~{DAYS}  sell_latency={DELTA}s  fee={FEE}")

def kval(m):
    try: return float(m[1:])
    except: return 0.0
modes = sorted(set(r["mode"] for r in rows), key=kval)

# (1) entry-X curve
print("\n(1) ENTRY-X CURVE  (trigger = price >= X * trailing-min):")
print("    X      n     freq/day   base_rate(+0.5 before -0.3)   sell-into net")
for m in modes:
    rs = [r for r in rows if r["mode"] == m]
    y = np.array([int(r["y"]) for r in rs])
    net = np.mean([sellinto(r) - FEE for r in rs])
    print(f"   {kval(m):<5} {len(rs):5d}   {len(rs)/DAYS:6.0f}     {y.mean():.3f}                       {net:+.3f}")

# (2) lifespan / age dependence (pooled across X)
print("\n(2) LIFESPAN — base_rate by token AGE at entry (pooled over X):")
ages = np.array([float(r.get("first_seen_age_s", 0.0) or 0.0) for r in rows])
yall = np.array([int(r["y"]) for r in rows])
for lo, hi in AGE_BUCKETS:
    mask = (ages >= lo) & (ages < hi)
    if mask.sum() == 0: continue
    lbl = f"{int(lo)}-{int(hi) if hi < 1e9 else '+'}s"
    print(f"   age {lbl:<14} n={mask.sum():5d}  base_rate={yall[mask].mean():.3f}  "
          f"sell-into net={np.mean([sellinto(r)-FEE for r,mk in zip(rows,mask) if mk]):+.3f}")

# (3) continuous scorer: pool all candidates, predict outcome from continuous features
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rows.sort(key=lambda r: r.get("cross_t", 0.0))
    cols = RICH + EXTRA
    X = np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rows]); X[~np.isfinite(X)] = 0.0
    y = np.array([int(r["y"]) for r in rows]); sp = int(len(rows) * 0.7)
    clf = HistGradientBoostingClassifier(max_iter=250, max_depth=3, learning_rate=0.06,
                                         l2_regularization=1.0, min_samples_leaf=60)
    clf.fit(X[:sp], y[:sp]); p = clf.predict_proba(X[sp:])[:, 1]
    auc = roc_auc_score(y[sp:], p)
    te = rows[sp:]; net = np.array([sellinto(r) - FEE for r in te])
    print(f"\n(3) CONTINUOUS SCORER (all candidates pooled, RICH+depth+age):  holdout AUC={auc:.3f}")
    for tier, frac in (("top-5%", 0.05), ("top-10%", 0.10), ("top-25%", 0.25)):
        k = max(1, int(len(te) * frac)); idx = np.argsort(-p)[:k]
        print(f"      {tier:7s} fires/day~{k/DAYS*0.3:.0f}  base_rate={y[sp:][idx].mean():.3f}  sell-into net={net[idx].mean():+.3f}")
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(clf, X[sp:], y[sp:], n_repeats=4, random_state=0, scoring="roc_auc")
        top = np.argsort(-imp.importances_mean)[:7]
        print("      top features:", ", ".join(f"{cols[i]}({imp.importances_mean[i]:+.3f})" for i in top))
    except Exception: pass
except Exception as e:
    print("  (model skipped:", e, ")")
