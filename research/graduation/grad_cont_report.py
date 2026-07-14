#!/usr/bin/env python3
"""Report 'is there signal' for the graduation (PumpSwap) 2x-continuation panel — HONEST economics
+ reputation A/B with the wallet-identity null.

Per trigger mode:
  - base rate, censored, depth, resolution time
  - REALIZABILITY: fraction of WINS that are single-trade overshoot (<1s) + entry-slip dist
  - ECON: TP-capped net/trade (win=+TP, loss=-STOP, minus cost). Raw mean ret shown only as
    the (unrealizable) overshoot for contrast.
  - MODEL A/B on a time-ordered holdout (train early, test late):
      RICH            : the 22 momentum features (+ depth, age [, t_since_grad])
      RICH+REP        : + as-of wallet reputation (6)
      RICH+REPshuf    : + reputation computed under a wallet-IDENTITY permutation (the null)
    REP is real only if RICH+REP beats BOTH RICH and RICH+REPshuf.

Usage: ./venv/bin/python grad_cont_report.py [panel.jsonl] [cost]
"""
import json, sys
import numpy as np

PANEL = sys.argv[1] if len(sys.argv) > 1 else "/root/the-distribution-will-manifest/bot_data/grad_cont_panel.jsonl"
COST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
TP_CAP, STOP_CAP = 0.50, 0.30
GAP0_S = 1.0

RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel",
        "last_gap", "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac",
        "net_flow", "n_buyers", "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
EXTRA = ["depth_sol", "first_seen_age_s"]
REP = ["rep_mean", "rep_max", "rep_nknown", "rep_frac_known", "rep_frachigh", "rep_nsmart"]
REP_SHUF = ["shuf_" + c for c in REP]
CURVE = ["curve_has", "curve_max_runup", "curve_ntr", "curve_nbuy_frac", "curve_vol_sol",
         "curve_mcap_grad", "curve_age_s"]

rows = []
with open(PANEL) as f:
    for ln in f:
        try: rows.append(json.loads(ln))
        except Exception: pass
print(f"panel={PANEL}  total_rows={len(rows)}  cost/trade={COST}  (win=+{TP_CAP} loss=-{STOP_CAP})")
have_rep = bool(rows) and all(c in rows[0] for c in REP)
have_curve = bool(rows) and ("curve_has" in rows[0])
print(f"reputation features present: {have_rep}  |  curve-prior features present: {have_curve}")
modes = sorted(set(r["mode"] for r in rows))

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception as e:
    HAVE_SK = False
    print("  (sklearn unavailable:", e, ")")


def feat_matrix(rs, cols):
    X = np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rs], dtype=float)
    X[~np.isfinite(X)] = 0.0
    return X


def capped_net(yv):
    return np.where(yv == 1, TP_CAP, -STOP_CAP) - COST


def train_eval(rs, cols, y, dur, split):
    X = feat_matrix(rs, cols)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    dte = dur[split:]
    if ytr.sum() < 10 or yte.sum() < 5 or (yte == 0).sum() < 5:
        return None
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06,
                                         l2_regularization=1.0, min_samples_leaf=40)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    k = max(1, int(len(p) * 0.10))
    top = np.argsort(-p)[:k]
    tprec = yte[top].mean()
    tk = ~((yte[top] == 1) & (dte[top] < GAP0_S))   # book gap-0 wins as non-fills
    tnet = capped_net(yte[top][tk]).mean() if tk.any() else float("nan")
    return auc, tprec, tnet, len(yte), int(yte.sum())


for m in modes:
    rs = [r for r in rows if r["mode"] == m]
    rs.sort(key=lambda r: r.get("cross_t", 0.0))
    n = len(rs)
    if n == 0:
        continue
    y = np.array([int(r["y"]) for r in rs])
    dur = np.array([float(r.get("dur_s", 0.0)) for r in rs])
    ret = np.array([float(r.get("ret", 0.0)) for r in rs])
    eslip = np.array([float(r.get("entry_slip", 0.0)) for r in rs])
    depth = np.array([float(r.get("depth_sol", 0.0)) for r in rs])
    base = y.mean()
    wins = y == 1
    wdur = dur[wins] if wins.any() else np.array([0.0])
    net_all = capped_net(y).mean()
    print(f"\n=== mode={m}  n={n}  base_rate={base:.3f}  median_depth_sol={np.median(depth):.0f}  "
          f"median_dur_s={np.median(dur):.0f}")
    print(f"   REALIZABILITY: wins<1s(overshoot)={float((wdur<1).mean()):.2f}  "
          f"entry_slip med={np.median(eslip):+.2f} p90={np.quantile(eslip,0.9):+.2f}  "
          f"|  ECON capped_net/trade(all)={net_all:+.3f}")
    if not HAVE_SK or n < 300:
        print("   (n too small for a stable model)")
        continue
    split = int(n * 0.7)
    base_cols = RICH + EXTRA + (["t_since_grad"] if m == "at_grad" else [])
    if have_curve:
        cov = float(np.mean([float(r.get("curve_has", 0) or 0) for r in rs]))
        print(f"   CURVE-JOIN coverage (curve_has=1): {cov:.2f}")
    sets = [("RICH", base_cols)]
    if have_curve:
        sets.append(("RICH+CURVE", base_cols + CURVE))
    if have_rep:
        sets.append(("RICH+REP", base_cols + REP))
    if have_curve and have_rep:
        sets.append(("RICH+CURVE+REP", base_cols + CURVE + REP))
        sets.append(("ALL+REPshuf(null)", base_cols + CURVE + REP_SHUF))
    res = {}
    for name, cols in sets:
        r = train_eval(rs, cols, y, dur, split)
        if r is None:
            print(f"   {name:20s} (too few pos/neg in split)")
            continue
        auc, tprec, tnet, ntest, pos = r
        res[name] = (auc, tnet)
        print(f"   {name:20s} AUC={auc:.3f}  top10_prec={tprec:.3f}  top10_net={tnet:+.3f}  (test n={ntest} pos={pos})")
    if "RICH" in res:
        b = res["RICH"][0]
        if "RICH+CURVE" in res:
            d = res["RICH+CURVE"][0] - b
            print(f"   >> CURVE lift over RICH: {d:+.3f}  -> {'REAL' if d > 0.005 else 'marginal/none'}")
        if have_rep and "RICH+CURVE+REP" in res and "ALL+REPshuf(null)" in res:
            d_real = res["RICH+CURVE+REP"][0] - res.get("RICH+CURVE", res["RICH"])[0]
            d_null = res["ALL+REPshuf(null)"][0] - res.get("RICH+CURVE", res["RICH"])[0]
            verdict = "REAL (beats wallet-id null)" if (d_real > 0.005 and d_real > d_null + 0.005) else "not convincing"
            print(f"   >> REP lift over RICH+CURVE: real={d_real:+.3f}  null={d_null:+.3f}  -> {verdict}")
