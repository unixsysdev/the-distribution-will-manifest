"""B1/B2: join 4-day buy-intent to every multi-milestone cross and test whether
intent/Jito add to the FILTER. CAUSAL by default: counts only intents RECEIVED BEFORE
the cross (recv_ns <= cross_t, within W lookback) = what's in our ring at decision.
Also tracks a symmetric +/-W count to expose how much of any lift is lookahead.
Usage: ls intent files >=Jun9 | xargs zcat -f | python continuation_intent_multi.py [W]"""
import json, sys, collections, numpy as np, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PANEL = "/root/the-distribution-will-manifest/bot_data/cont_multi_panel.jsonl"
W = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
cross = {}; out = {}
for l in open(PANEL):
    try: e = json.loads(l)
    except Exception: continue
    k = e.get("kind")
    if k == "cross": cross[(e["mint"], e["mult"])] = e
    elif k == "outcome": out[(e["mint"], e["mult"])] = e
rec = {}; bymint = collections.defaultdict(list)
for key in out:
    c = cross.get(key)
    if c is None: continue
    rec[key] = {"mult": key[1], "t": c["t"], "day": time.strftime("%m-%d", time.gmtime(c["t"])),
                "f": [c["dd"], c["buy_frac"], c["ntr"], c["recent"], c.get("tps", 0), c.get("uniq", 0)],
                "y": out[key]["y"], "ret": out[key]["ret"], "icp": 0, "ics": 0, "tip": 0}
    bymint[key[0]].append(key)
cross_mints = set(bymint.keys())
print(f"{len(rec)} crosses, {len(cross_mints)} mints; scanning 4-day buy-intents (causal <=cross, W={W}s)", flush=True)
n = 0
for line in sys.stdin:
    n += 1
    try: r = json.loads(line)
    except Exception: continue
    if not r.get("type", "").startswith("buy"): continue
    accs = r.get("ix_accounts")
    if not accs: continue
    hit = cross_mints.intersection(accs)
    if not hit: continue
    rt = r.get("recv_ns", 0) / 1e9; tip = r.get("jito_tip_lam", 0)
    for mm in hit:
        for key in bymint[mm]:
            dt = rec[key]["t"] - rt          # >0 => intent before the cross (causal)
            if -W <= dt <= W:
                rec[key]["ics"] += 1
                if dt >= 0:
                    rec[key]["icp"] += 1
                    if tip > rec[key]["tip"]: rec[key]["tip"] = tip
print(f"scanned {n} lines", flush=True)
K = list(rec.keys())
mult = np.array([rec[k]["mult"] for k in K]); day = np.array([rec[k]["day"] for k in K])
X = np.array([rec[k]["f"] for k in K], float)
ICP = np.array([rec[k]["icp"] for k in K], float)   # causal
ICS = np.array([rec[k]["ics"] for k in K], float)   # symmetric (leaky)
TIP = np.array([rec[k]["tip"] / 1e9 for k in K])
y = np.array([rec[k]["y"] for k in K], float); ret = np.array([rec[k]["ret"] for k in K], float)
days = sorted(set(day)); trd = set(days[:3]); tr = np.array([d in trd for d in day]); te = ~tr
print(f"causal intent: mean={ICP.mean():.1f}  with_intent={(ICP>0).mean():.0%}")
print("per-milestone corr(CAUSAL intent_count, ret) and corr(max_tip, ret):")
for k in sorted(set(mult)):
    s = mult == k
    print(f"  {k}x: corr_ic={np.corrcoef(ICP[s],ret[s])[0,1]:+.3f}  corr_tip={np.corrcoef(TIP[s],ret[s])[0,1]:+.3f}")

def auc_with(cols):
    Xx = np.column_stack([X, mult] + cols)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(Xx[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(Xx[te])[:, 1])
base = auc_with([]); caus = auc_with([ICP, TIP]); leak = auc_with([ICS, TIP])
print(f"\nB1/B2 OOS AUC: base={base:.3f}  +CAUSAL intent={caus:.3f} (lift {caus-base:+.3f})  "
      f"+leaky-symmetric={leak:.3f} (lift {leak-base:+.3f} <- the lookahead inflation)")
# within top-10% by base p, split by CAUSAL intent
Xm = np.column_stack([X, mult])
clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(Xm[tr], y[tr])
p = clf.predict_proba(Xm[te])[:, 1]
print("within top-10% (test, by base p): CAUSAL intent HIGH vs LOW ret:")
for k in sorted(set(mult)):
    s = mult[te] == k
    if s.sum() < 40: continue
    pk = p[s]; ick = ICP[te][s]; rk = ret[te][s]
    top = pk >= np.quantile(pk, 0.90)
    if top.sum() < 6: continue
    med = np.median(ick[top]); hi = top & (ick > med); lo = top & (ick <= med)
    print(f"  {k}x: HIGH n={int(hi.sum())} ret={rk[hi].mean():+.3f} | LOW n={int(lo.sum())} ret={rk[lo].mean():+.3f}")
