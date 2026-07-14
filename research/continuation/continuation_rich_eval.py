"""Rich eval: by-coin OOS, compare BASE6 vs +TRADE vs +TRADE+INTENT on the rich panel. Reports
OOS AUC, the honest cap-aware net-per-SELECTED-cross at top-5%/top-10%, and permutation feature
importance. Answers: do the new trade features lift the filter, and does intent add anything on top?
"""
import json, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

ROOT = "/root/the-distribution-will-manifest"
import os as _os, sys as _sys
PANEL = _sys.argv[1] if len(_sys.argv) > 1 else (f"{ROOT}/bot_data/cont_rich_rep_panel.jsonl" if _os.path.exists(f"{ROOT}/bot_data/cont_rich_rep_panel.jsonl") else f"{ROOT}/bot_data/cont_rich_intent_panel.jsonl")
BET = 0.1; PUMP_RT = 0.02; FIXED_RT = 0.0017; REVERT = 0.0006; CAP = 0.25
BASE6 = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq"]
TRADE = BASE6 + ["t_to_2x", "log_t_to_2x", "accel", "last_gap", "mcap_sol", "vol_sol", "sol_per_trade",
                 "max_buy_sol", "whale_frac", "net_flow", "n_buyers", "n_sellers", "bs_ratio",
                 "signer_conc", "up_frac", "max_runup"]
INTENT = ["int_n", "int_buy", "int_sell", "int_buy_frac", "int_uniq", "int_prio_p90", "int_tip_rate", "int_tip_p90"]
ALL = TRADE + INTENT
REP = ["cre_n_launch", "cre_n_2x_res", "cre_winrate", "buy_n", "buy_known_frac", "buy_rep_mean", "buy_rep_max"]

cr = {}; fl = {}; oc = {}
for l in open(PANEL):
    r = json.loads(l); k = r.get("kind"); m = r.get("mint")
    if k == "cross": cr[m] = r
    elif k == "fill": fl[m] = r
    elif k == "outcome": oc[m] = r
keys = [m for m in oc if m in cr]


def slip(m):
    c = cr[m]; f = fl.get(m); cm = c.get("cross_mid", 0.0); fm = (f or {}).get("fill_mid", cm)
    return (fm / cm - 1.0) if cm > 0 else 0.0


ret = np.array([oc[m]["ret"] for m in keys], float)
y = np.array([1 if oc[m]["y"] == 1 else 0 for m in keys], int)
sl = np.array([slip(m) for m in keys], float)
bk = np.array([int(hashlib.md5(m.encode()).hexdigest(), 16) % 100 for m in keys])
tr = bk < 70; te = bk >= 70
print(f"rich panel: {len(keys)} crosses  base win {y.mean():.0%}  n_test {int(te.sum())}  "
      f"crosses-with-intent {sum(1 for m in keys if cr[m].get('int_n', 0) > 0)}")

# join the LAUNCH model's per-coin score (shadow_run.jsonl entry_decision) -- the user's "first filter"
LS = {}
for l in open(f"{ROOT}/bot_data/shadow_run.jsonl"):
    try: rr = json.loads(l)
    except Exception: continue
    if rr.get("kind") == "entry_decision" and "score" in rr:
        LS[rr["mint"]] = rr["score"]
_have = [LS[m] for m in keys if m in LS]
_med = float(np.median(_have)) if _have else 0.0
for m in keys:
    cr[m]["launch_score"] = LS.get(m, _med)
print(f"launch_score joined: {len(_have)}/{len(keys)} (missing -> median {_med:.3f})")


def netsel_at(p, tier):
    cut = np.quantile(p[te], tier); m = p[te] >= cut
    if m.sum() < 10:
        return None
    rr = ret[te][m]; ss = sl[te][m]; filled = ss <= CAP; nrev = int((~filled).sum())
    nf = BET * rr[filled] - PUMP_RT * BET - FIXED_RT
    ns = (nf.sum() + nrev * (-REVERT)) / m.sum()
    return ns, int(m.sum()), float(filled.mean()), float((rr[filled] > 0).mean()) if filled.sum() else float("nan")


def run(name, FE):
    X = np.array([[cr[m].get(f, 0.0) for f in FE] for m in keys], float)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0).fit(X[tr], y[tr])
    p = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y[te], p[te])
    line = f"{name:<16} {len(FE):>3} feat   OOS AUC {auc:.3f}"
    for tier, lbl in ((0.95, "top5"), (0.90, "top10")):
        r = netsel_at(p, tier)
        if r:
            line += f"   {lbl} netsel {r[0]:+.4f} (fill {r[2]:.0%} win {r[3]:.0%} n{r[1]})"
    print(line)
    return clf, X


# LEAN = base6 + only the trade features that carried permutation signal (>~0.004); drops the
# ~0-importance timing/flow features and ALL intent (which hurt). The candidate deployable set.
LEAN = BASE6 + ["vol_sol", "mcap_sol", "signer_conc", "max_buy_sol", "max_runup", "bs_ratio", "whale_frac"]
RICHALL = ALL + ["launch_score"] + REP

print("\n--- feature-set comparison (by-coin OOS, cap 25%, bet 0.1) ---")
print("ceiling so far: LEAN ~0.59.  +launch = launch model score.  +REP = creator+buyer reputation (orthogonal).")
for name, FE in (("BASE6", BASE6), ("LEAN", LEAN), ("LEAN+launch", LEAN + ["launch_score"]),
                 ("LEAN+REP", LEAN + REP), ("LEAN+launch+REP", LEAN + ["launch_score"] + REP),
                 ("ALL+launch+REP", RICHALL)):
    run(name, FE)

clf, X = run("ALL+launch+REP (imp)", RICHALL)
imp = permutation_importance(clf, X[te], y[te], n_repeats=6, random_state=0, scoring="roc_auc")
order = np.argsort(imp.importances_mean)[::-1]
print("\n--- permutation importance (OOS AUC drop when shuffled), top 18 ---")
for i in order[:18]:
    nm = RICHALL[i]
    tag = ("LAUNCH" if nm == "launch_score" else "REP" if nm in REP else "INTENT" if nm in INTENT
           else "base" if nm in BASE6 else "trade")
    print(f"  {nm:>14} [{tag:>6}]: {imp.importances_mean[i]:+.4f}")

# --- BY-DAY split (cross-period): does the REP lift survive, or was it within-window burst correlation? ---
print("\n--- BY-DAY split (train EARLY / test LATE by cross time) -- the leak test for reputation ---")
t_arr = np.array([cr[m]["t"] for m in keys])
tcut = float(np.quantile(t_arr, 0.65))
trd = t_arr < tcut; ted = t_arr >= tcut
print(f"  train {int(trd.sum())} early / test {int(ted.sum())} late")
for name, FE in (("LEAN", LEAN), ("LEAN+REP", LEAN + REP), ("LEAN+launch+REP", LEAN + ["launch_score"] + REP)):
    Xd = np.array([[cr[m].get(f, 0.0) for f in FE] for m in keys], float)
    cld = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0).fit(Xd[trd], y[trd])
    pd_ = cld.predict_proba(Xd[ted])[:, 1]
    print(f"  {name:<16} by-DAY OOS AUC {roc_auc_score(y[ted], pd_):.3f}")
