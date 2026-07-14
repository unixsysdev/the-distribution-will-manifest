"""Decisive cap-aware test on the FULL panel (cont_multi_panel.jsonl, ~4.5 days, 9945 2x crosses).
Question: of the model's top tier, how many can we actually FILL (slip<=25%), and do those
fillable entries have a realized edge -- or does the edge live only in the unfillable runners?
By-coin OOS. Joins cross(features+cross_mid)+fill(fill_mid)+outcome(y,ret) per mint.
"""
import json, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

P = "/root/the-distribution-will-manifest/bot_data/cont_multi_panel.jsonl"
FE = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq"]
BET = 0.1; PUMP_RT = 0.02; FIXED_RT = 0.0017; CAP = 0.25; REVERT = 0.0006

cross = {}; fill = {}; outc = {}
for l in open(P):
    r = json.loads(l)
    if r.get("mult") not in (2, 2.0):
        continue
    m = r["mint"]; k = r["kind"]
    if k == "cross":
        cross.setdefault(m, r)
    elif k == "fill":
        fill.setdefault(m, r)
    elif k == "outcome":
        outc.setdefault(m, r)

recs = []
for m, o in outc.items():
    c = cross.get(m); f = fill.get(m)
    if not c or not f:
        continue
    cm = c.get("cross_mid", 0.0); fm = f.get("fill_mid", cm)
    slip = (fm / cm - 1.0) if cm > 0 else 0.0
    recs.append({**{k: c[k] for k in FE}, "y": o["y"], "ret": o["ret"], "slip": slip, "mint": m})

print(f"full-panel matched crosses: {len(recs)}")


def cb(m):
    return int(hashlib.md5(m.encode()).hexdigest(), 16) % 100


X = np.array([[r[k] for k in FE] for r in recs], float)
y = np.array([r["y"] for r in recs], int)
ret = np.array([r["ret"] for r in recs], float)
slip = np.array([r["slip"] for r in recs], float)
bk = np.array([cb(r["mint"]) for r in recs])
tr = bk < 70; te = bk >= 70

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4, l2_regularization=1.0)
clf.fit(X[tr], y[tr])
p = clf.predict_proba(X[te])[:, 1]
yte = y[te]; rte = ret[te]; ste = slip[te]
print(f"OOS AUC {roc_auc_score(yte, p):.3f}   base win {yte.mean():.0%}   n_test {int(te.sum())}")
print(f"{'tier':<6}{'n':>5}{'fill%':>6}{'win_all':>8}{'win_fill':>9}{'ret_fill':>9}{'netfill':>9}{'netsel':>9}{'med_slip':>9}")
for tag, q in (("top5", 0.95), ("top10", 0.90), ("all", 0.0)):
    cut = np.quantile(p, q); m = p >= cut
    sl = ste[m]; rr = rte[m]; yy = yte[m]
    filled = sl <= CAP; nrev = int((~filled).sum())
    nf = BET * rr[filled] - PUMP_RT * BET - FIXED_RT
    netsel = (nf.sum() + nrev * (-REVERT)) / m.sum()
    wf = (rr[filled] > 0).mean() if filled.sum() else float("nan")
    rf = rr[filled].mean() if filled.sum() else float("nan")
    print(f"{tag:<6}{int(m.sum()):>5}{filled.mean():>6.0%}{yy.mean():>8.0%}{wf:>9.0%}"
          f"{rf:>+9.2f}{nf.mean() if filled.sum() else float('nan'):>+9.4f}{netsel:>+9.4f}{np.median(sl):>+9.0%}")


# --- cap-aware model: select FOR fillable-winners (target = filled AND win), not just winners ---
yreal = ((slip <= CAP) & (ret > 0)).astype(int)
clf2 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4, l2_regularization=1.0)
clf2.fit(X[tr], yreal[tr])
p2 = clf2.predict_proba(X[te])[:, 1]
yreal_te = ((ste <= CAP) & (rte > 0)).astype(int)
print(f"\ncap-aware model (target=fillable-win)  OOS AUC {roc_auc_score(yreal_te, p2):.3f}  "
      f"base fillable-win {yreal_te.mean():.0%}")
print(f"{'tier':<6}{'n':>5}{'fill%':>6}{'win_fill':>9}{'ret_fill':>9}{'netfill':>9}{'netsel':>9}{'med_slip':>9}")
for tag, q in (("top5", 0.95), ("top10", 0.90)):
    cut = np.quantile(p2, q); m = p2 >= cut
    sl = ste[m]; rr = rte[m]
    filled = sl <= CAP; nrev = int((~filled).sum())
    nf = BET * rr[filled] - PUMP_RT * BET - FIXED_RT
    netsel = (nf.sum() + nrev * (-REVERT)) / m.sum()
    wf = (rr[filled] > 0).mean() if filled.sum() else float("nan")
    rf = rr[filled].mean() if filled.sum() else float("nan")
    nfm = nf.mean() if filled.sum() else float("nan")
    print(f"{tag:<6}{int(m.sum()):>5}{filled.mean():>6.0%}{wf:>9.0%}{rf:>+9.2f}{nfm:>+9.4f}{netsel:>+9.4f}{np.median(sl):>+9.0%}")
