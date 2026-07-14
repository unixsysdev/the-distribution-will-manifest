"""A1/A2/A2.5 eval of the multi-milestone panel. Pooled HistGradientBoosting model
(6 trade features + milestone as a feature), cross-period OOS (train early days /
test late days), per-milestone top-tier EV. Mid-based ret (gap-0-assumed); tip drag
shown at 0.5-bet (1%) and 0.05-bet (10%). Also a 2x-only model for comparison."""
import json, numpy as np, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PANEL = "/root/the-distribution-will-manifest/bot_data/cont_multi_panel.jsonl"
cross = {}; out = {}
for l in open(PANEL):
    try: e = json.loads(l)
    except Exception: continue
    k = e.get("kind")
    if k == "cross": cross[(e["mint"], e["mult"])] = e
    elif k == "outcome": out[(e["mint"], e["mult"])] = e
rows = []
for key, o in out.items():
    c = cross.get(key)
    if c is None: continue
    rows.append([c["mult"], time.strftime("%m-%d", time.gmtime(c["t"])),
                 c["dd"], c["buy_frac"], c["ntr"], c["recent"], c.get("tps", 0), c.get("uniq", 0),
                 o["y"], o["ret"]])
print(f"joined {len(rows)} (mint,milestone) outcomes")
mult = np.array([r[0] for r in rows]); day = np.array([r[1] for r in rows])
X = np.array([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows], float)
Xm = np.column_stack([X, mult])
y = np.array([r[8] for r in rows], float); ret = np.array([r[9] for r in rows], float)

days = sorted(set(day)); print("days:", days)
trd = set(days[:3]); ted = set(days[3:])
tr = np.array([d in trd for d in day]); te = ~tr
print(f"train {sorted(trd)} n={tr.sum()}   test {sorted(ted)} n={te.sum()}")

clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(Xm[tr], y[tr])
p = clf.predict_proba(Xm[te])[:, 1]
auc = roc_auc_score(y[te], p) if len(set(y[te])) > 1 else float("nan")
print(f"\n=== A2.5 pooled model (6 feats + milestone), OOS AUC across ALL multiples = {auc:.3f} ===")

print("\n=== A1/A2 per-milestone, TEST set, tiered within each milestone by pooled p ===")
print(f"{'mult':>5}{'n':>7}{'base':>7} | {'top10%: n  hit   mid   EV.5  EV.05':<38} | {'top5%: n  hit   mid   EV.5  EV.05':<36}")
mt = mult[te]; yt = y[te]; rt = ret[te]
def tier(pk, yk, rk, q):
    t = pk >= np.quantile(pk, q)
    if t.sum() < 3: return "      (too few)"
    return f"{int(t.sum()):>3} {yk[t].mean():>4.0%} {rk[t].mean():>+6.2f} {rk[t].mean()-0.01:>+6.2f} {rk[t].mean()-0.10:>+6.2f}"
for k in sorted(set(mult)):
    s = mt == k
    if s.sum() < 20:
        print(f"{k:>5}{int(s.sum()):>7}   (too few in test)"); continue
    print(f"{k:>5}{int(s.sum()):>7}{yt[s].mean():>7.0%} | {tier(p[s],yt[s],rt[s],0.90):<38} | {tier(p[s],yt[s],rt[s],0.95)}")

# 2x-only model for comparison (is pooling better than a dedicated 2x model on 2x?)
m2 = mult == 2.0
clf2 = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(X[tr & m2], y[tr & m2])
p2 = clf2.predict_proba(X[te & m2])[:, 1]
a2 = roc_auc_score(y[te & m2], p2) if len(set(y[te & m2])) > 1 else float("nan")
t2 = p2 >= np.quantile(p2, 0.90)
print(f"\n=== compare on 2x test: pooled AUC {roc_auc_score(y[te&m2],p[mt==2.0]):.3f} vs 2x-only AUC {a2:.3f}; "
      f"2x-only top10% n={int(t2.sum())} hit={y[te&m2][t2].mean():.0%} mid={ret[te&m2][t2].mean():+.3f} ===")
