#!/usr/bin/env python3
"""Slow-climber vs intra-block-spike filter (SLOT-space). Needs window05_slots from the extractor.

  reach        = price hit +0.5x before -0.3x
  spike        = reach but window05_slots == 0  (above +0.5x only WITHIN one block -> untradeable)
  slow-climber = reach and window05_slots >= K  (holds +0.5x across >=K slot boundaries -> sellable)

(1) slot-space bimodality among winners.
(2) CHARACTERISTICS: which pre-cross features separate slow-climbers from spikes (a priori).
(3) a-priori model: predict slow-climber from pre-cross RICH+depth+age -> AUC + importances.
(4) realizable: top-decile by that score -> % actually slow + sell-into net + fire rate.

Usage: ./venv/bin/python grad_climber_filter.py [panel.jsonl] [K_slots] [age_max] [days]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/root/the-distribution-will-manifest/bot_data/grad_climber_2d.jsonl"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
AGE_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9   # default: ALL ages (find where catchable edge lives)
DAYS = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EX = ["depth_sol", "first_seen_age_s"]


def clamp(x): return max(-1.0, min(1.5, x))
def reached(r):
    t05 = r.get("t05_s"); dn = r.get("dn_env", [])
    if t05 is None: return False
    tst = next((dt for dt, rr in dn if rr <= -0.3), None)
    return tst is None or t05 <= tst
def sellinto(r):
    t05 = r.get("t05_s")
    if t05 is None:
        return -0.3 if any(rr <= -0.3 for _, rr in r.get("dn_env", [])) else clamp(r.get("final_ret", 0.0))
    for dt, rr in r.get("path", []):
        if dt >= t05 + 1.5: return clamp(rr)
    return clamp(r.get("final_ret", 0.0))


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if (r.get("first_seen_age_s") or 0) < AGE_MAX and "window05_slots" in r]
print(f"panel={PANEL}  fresh(age<{AGE_MAX}s) n={len(rows)}  K(slots)={K}")
rows.sort(key=lambda r: r.get("cross_t", 0.0))

reach = np.array([reached(r) for r in rows])
wsl = np.array([(r.get("window05_slots") if r.get("window05_slots") is not None else -1) for r in rows])
win = rows  # alias
slow = np.array([reach[i] and wsl[i] >= K for i in range(len(rows))])
spike = np.array([reach[i] and wsl[i] == 0 for i in range(len(rows))])

# (1) slot bimodality among reachers
wr = wsl[reach]
print(f"\n(1) SLOT-window among winners (n={reach.sum()}):  ==0(intra-block spike)={np.mean(wr==0):.2f}  "
      f"==1={np.mean(wr==1):.2f}  >=2={np.mean(wr>=2):.2f}  >=5={np.mean(wr>=5):.2f}  median={np.median(wr):.0f} slots")
print(f"    base rates: reach+0.5x={reach.mean():.2f}  slow-climber(>={K} slots)={slow.mean():.2f}  intra-block-spike={spike.mean():.2f}")

# (1b) CATCHABLE edge BY AGE — does the slow-climber (sellable) edge live later than the raw edge?
print("\n(1b) CATCHABLE-WINNER edge by token AGE at entry:")
print("   age band        n     reach+0.5x   slow|reach   catchable-winner(reach&>=Kslots)   sell-into net")
ages = np.array([float(r.get("first_seen_age_s") or 0.0) for r in rows])
for lo, hi in [(0, 60), (60, 300), (300, 1800), (1800, 7200), (7200, 1e9)]:
    m = (ages >= lo) & (ages < hi)
    if m.sum() < 20:
        continue
    rr = reach[m]; sl = slow[m]
    slow_given_reach = (sl.sum() / rr.sum()) if rr.sum() else 0.0
    net = np.mean([sellinto(r) for r, mk in zip(rows, m) if mk])
    lbl = f"{int(lo)}-{int(hi) if hi < 1e9 else '+'}s"
    print(f"   {lbl:14s} {int(m.sum()):5d}   {rr.mean():.3f}        {slow_given_reach:.3f}        "
          f"{sl.mean():.3f}                          {net:+.3f}")

# (2) characteristics: slow vs spike means
print("\n(2) CHARACTERISTICS (mean for slow-climber vs intra-block-spike, among winners):")
feats = ["depth_sol", "tps", "whale_frac", "n_buyers", "sol_per_trade", "vol_sol", "max_runup", "accel", "signer_conc", "ntr"]
for f in feats:
    sv = np.array([float(r.get(f, 0) or 0) for r in rows])
    a, b = sv[slow], sv[spike]
    if len(a) and len(b):
        ma, mb = np.median(a), np.median(b)
        print(f"   {f:14s} slow={ma:10.3f}   spike={mb:10.3f}   ratio={ma/mb if mb else float('nan'):.2f}")

# (3) a-priori model: predict slow-climber
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance
    cols = RICH + EX
    X = np.array([[float(r.get(c, 0) or 0) for c in cols] for r in rows]); X[~np.isfinite(X)] = 0
    y = slow.astype(int); sp = int(len(rows) * 0.7)
    if y[:sp].sum() >= 10 and y[sp:].sum() >= 5:
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=3, learning_rate=0.06,
                                             l2_regularization=1.0, min_samples_leaf=60)
        clf.fit(X[:sp], y[:sp]); p = clf.predict_proba(X[sp:])[:, 1]
        auc = roc_auc_score(y[sp:], p)
        print(f"\n(3) A-PRIORI slow-climber model:  AUC={auc:.3f}")
        imp = permutation_importance(clf, X[sp:], y[sp:], n_repeats=5, random_state=0, scoring="roc_auc")
        top = np.argsort(-imp.importances_mean)[:8]
        print("    characteristics (perm-importance):", ", ".join(f"{cols[i]}({imp.importances_mean[i]:+.3f})" for i in top))
        # (4) realizable on top-decile by slow-score
        te = rows[sp:]; net = np.array([sellinto(r) for r in te]); slow_te = slow[sp:]
        for tier, frac in (("top-10%", 0.10), ("top-25%", 0.25)):
            k = max(1, int(len(te) * frac)); idx = np.argsort(-p)[:k]
            print(f"    {tier}: fires/day~{k/DAYS:.0f}  %slow-climber={slow_te[idx].mean():.2f}  sell-into net={net[idx].mean():+.3f}")
    else:
        print("\n(3) too few slow-climbers for a model")
except Exception as e:
    print("  (model skipped:", e, ")")
