"""Daily eval of the continuation dry-run. Joins continuation_shadow.jsonl
(cross+fill+outcome per mint), time-splits (train early -> test late), fits the model
on features, and reports the model's top-tier REALIZED EV on the held-out test set.

Realized PnL uses the logged exit `ret` (gap-aware: losses can be worse than -30%,
wins ~+50%), minus the tip. This is the honest live read the capstone could only
estimate. Run daily; as data accumulates the train/test split lengthens (the refit).
"""
import json, sys, collections, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
ROOT = "/root/the-distribution-will-manifest"
TIP_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01   # 0.005 SOL on 0.5 bet

rec = collections.defaultdict(dict)
for l in open(f"{ROOT}/bot_data/continuation_shadow.jsonl"):
    try: e = json.loads(l)
    except Exception: continue
    m = e.get("mint"); k = e.get("kind")
    if not m: continue
    if k == "cross": rec[m]["cross"] = e
    elif k == "fill": rec[m]["fill"] = e
    elif k == "outcome": rec[m]["out"] = e

rows = []
for m, d in rec.items():
    if "cross" in d and "out" in d:
        c = d["cross"]; o = d["out"]; f = d.get("fill", {})
        rows.append((c["t"], c["dd"], c["buy_frac"], c["ntr"], c["recent"],
                     c.get("tps", 0.0), c.get("uniq", 0),
                     f.get("slip_vs_cross") or 0.0, o["y"], o["ret"]))
if len(rows) < 50:
    print(f"only {len(rows)} resolved crosses so far — let the shadow accumulate (need a few hundred)."); sys.exit()
arr = np.array(rows, float)
arr = arr[arr[:, 0].argsort()]
# cols: 0=t 1=dd 2=bf 3=ntr 4=recent 5=tps 6=uniq 7=slip 8=y 9=ret
FE = [1, 2, 3, 4, 5, 6]   # live-computable features (NO lookahead inslot)
n = len(arr); base_hit = arr[:, 8].mean()
print(f"resolved crosses={n}  base hit={base_hit:.1%}  base mean-ret={arr[:,9].mean():+.3f}  median entry-slip={np.median(arr[:,7]):.1%}")
cut = np.median(arr[:, 0]); tr = arr[arr[:,0] < cut]; te = arr[arr[:,0] >= cut]
if len(tr) < 30 or len(te) < 30:
    print("not enough for a time-split yet."); sys.exit()
clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05)
clf.fit(tr[:, FE], tr[:, 8]); p = clf.predict_proba(te[:, FE])[:, 1]
auc = roc_auc_score(te[:,8], p) if len(set(te[:,8])) > 1 else float("nan")
print(f"time-split: train={len(tr)} test={len(te)}  OOS_AUC={auc:.3f}")
print(f"{'tier':>7} {'n':>5} {'hit':>6} {'realized_EV(net tip)':>20}")
for q in [1.0, 0.25, 0.10, 0.05]:
    sel = p >= np.quantile(p, 1 - q)
    if sel.sum() < 5: continue
    hit = te[sel,8].mean(); rEV = te[sel,9].mean() - TIP_FRAC   # mean realized exit-ret minus tip
    print(f"  top{q:>4.0%} {int(sel.sum()):5d} {hit:6.1%} {rEV:+20.3f}")
print("\nrealized_EV = mean(exit ret, gap-aware) - tip. If top-decile stays + across days -> arm small live.")
