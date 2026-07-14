#!/usr/bin/env python3
"""Train + serialize the SLOW-CLIMBER FILTER (2nd gate for the graduation bot).

Replicates grad_climber_filter.py EXACTLY — same features (RICH + [depth_sol, first_seen_age_s]) and same
label (reached +0.5x before -0.3x AND window05_slots >= K = holds >=K slot boundaries = sellable) — so the
live bot's gate is train==live. Emits:
  bot_data/grad_climber_model.pkl   (sklearn HGB, predict_proba)
  bot_data/grad_climber_spec.json   {"features":[RICH+EX], "threshold":<Youden J>, ...}

Usage: ./venv/bin/python grad_climber_train.py [panel] [K_slots] [age_max]
"""
import json, sys, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = "/root/the-distribution-will-manifest"
PANEL = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/bot_data/grad_climber_2d.jsonl"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
AGE_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0   # train on the deployable age window (where catchable lives)
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EX = ["depth_sol", "first_seen_age_s"]
COLS = RICH + EX


def reached(r):
    t05 = r.get("t05_s"); dn = r.get("dn_env", [])
    if t05 is None:
        return False
    tst = next((dt for dt, rr in dn if rr <= -0.3), None)
    return tst is None or t05 <= tst


rows = [json.loads(l) for l in open(PANEL)]
rows = [r for r in rows if "window05_slots" in r and (r.get("first_seen_age_s") or 0) < AGE_MAX]
rows.sort(key=lambda r: r.get("cross_t", 0.0))
reach = np.array([reached(r) for r in rows])
wsl = np.array([(r.get("window05_slots") if r.get("window05_slots") is not None else -1) for r in rows])
y = np.array([bool(reach[i] and wsl[i] >= K) for i in range(len(rows))]).astype(int)
X = np.array([[float(r.get(c, 0) or 0) for c in COLS] for r in rows]); X[~np.isfinite(X)] = 0.0
n = len(rows); sp = int(n * 0.7)
print(f"panel={PANEL} n={n} (age<{AGE_MAX}s)  slow_base_rate={y.mean():.3f}  train={sp} test={n-sp}")

HGB = dict(max_iter=250, max_depth=3, learning_rate=0.06, l2_regularization=1.0,
           min_samples_leaf=60, random_state=0)
# time-ordered 70/30 to get honest OOS AUC + pick the threshold
clf0 = HistGradientBoostingClassifier(**HGB).fit(X[:sp], y[:sp])
p = clf0.predict_proba(X[sp:])[:, 1]; yte = y[sp:]
auc = roc_auc_score(yte, p)
fpr, tpr, thr = roc_curve(yte, p)
j = int(np.argmax(tpr - fpr)); threshold = float(thr[j])
fired = p >= threshold
prec = float(yte[fired].mean()) if fired.sum() else 0.0
recall = float(yte[fired].sum() / max(1, yte.sum()))
print(f"OOS AUC={auc:.3f}  Youden thr={threshold:.4f}")
print(f"  at thr: fires={fired.mean():.2f} of candidates  precision(slow|fired)={prec:.3f}  recall(slow kept)={recall:.3f}")
# deploy model: retrain on ALL rows (more data) with the same params + threshold
clf = HistGradientBoostingClassifier(**HGB).fit(X, y)
pickle.dump(clf, open(f"{ROOT}/bot_data/grad_climber_model.pkl", "wb"))
json.dump({"features": COLS, "threshold": round(threshold, 5), "k_slots": K, "age_max_train": AGE_MAX,
           "oos_auc": round(auc, 4), "slow_base_rate": round(float(y.mean()), 4), "n": n,
           "precision_slow_given_fired": round(prec, 4), "recall": round(recall, 4)},
          open(f"{ROOT}/bot_data/grad_climber_spec.json", "w"), indent=2)
print(f"SAVED grad_climber_model.pkl + grad_climber_spec.json  ({len(COLS)} features, thr={threshold:.4f})")
