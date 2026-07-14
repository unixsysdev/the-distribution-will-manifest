#!/usr/bin/env python3
"""rich_exec_compare.py — does the RICH model's execution-adjusted net beat the
22-feature model's, on the IDENTICAL population and fires?

Joins rich features + labels (candidates.parquet) with forward paths + slot/tip
(sweep_k3v03.pkl) by mint. Trains BOTH a rich and a 22-feat peak_ge_200 head on
the SAME train split (Jun 9), tests Jun 10-11, takes each model's top-3% fires,
and computes the DEDUPED execution-adjusted net under fixed-latency (lat 0/1/2)
and slot-aware landing. Same cost model as the other sims.

Answers leg-3: rich's better selection (high-headroom rockets) should lift the
realistic-latency net back toward positive where the 22-feat went negative.

Caveats: train is Jun 9 only (rich-schema capture starts Jun 9); population is
the intent-present-filtered candidates set (both models see the same bias, so
the COMPARISON is fair; absolute numbers are on that subset).
"""
import calendar
import glob
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
REG = dict(max_iter=150, max_depth=3, learning_rate=0.05, l2_regularization=5.0, random_state=42)
Q_SOL, COST_BPS, FEE_TX, TP = 0.1, 250.0, 0.0015, 0.50
TOP_PCT = 3.0
DENY = ("peak", "future", "label", "target", "tp", "terminal", "net_exit")
BOOK = {"mint", "decision_ts", "k", "v_sol", "first_ts", "decision_slot", "decision_idx",
        "k_idx", "v_idx", "n_total_trades_seen"}


def buy_tokens(vs, vt, d):
    return vt - (vs * vt) / (vs + d)


def sell_sol(vs, vt, d):
    return vs - (vs * vt) / (vt + d)


def walk(evs, evt, path):
    q = Q_SOL * 1e9
    tok = buy_tokens(evs, evt, q)
    em = evs / evt
    xvs, xvt = evs, evt
    for (_t, _s, vs, vt, _b, _tp) in path:
        xvs, xvt = vs, vt
        if (vs / vt) / em - 1.0 >= TP:
            break
    return sell_sol(xvs, xvt, tok) / q - 1.0 - COST_BPS / 1e4 - (FEE_TX * 2) / Q_SOL


def land_b(dec, f, our_tip=1_000_000):
    i = 0
    while i < len(f) and f[i][1] <= dec["slot"]:
        i += 1
    if i >= len(f):
        return None
    ls = f[i][1]
    while i < len(f) and f[i][1] == ls:
        if f[i][5] is not None and f[i][5] > our_tip:
            i += 1
        else:
            break
    return i if i < len(f) else None


def dedup_mean(scores, nets):
    by = {}
    for s, x in zip(scores, nets):
        by.setdefault(round(float(s), 6), []).append(x)
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def exec_block(label, fired, sweep):
    out = []
    for lab, lat in (("lat0", 0), ("lat1", 1), ("lat2", 2), ("slot", None)):
        nets, scs = [], []
        for mm, sc in fired:
            f = sweep[mm]["fwd"]
            if not f:
                continue
            if lat is None:
                li = land_b(sweep[mm]["decision"], f)
            else:
                li = min(lat, len(f) - 1)
            if li is None:
                continue
            nets.append(walk(f[li][2], f[li][3], f[li + 1:]))
            scs.append(sc)
        if nets:
            raw = float(np.mean(nets))
            ded, npat = dedup_mean(scs, nets)
            out.append(f"  {label}/{lab:5s} raw={raw:+.3f} DEDUPED={ded:+.3f} (n_pat={npat})")
    return out


def main():
    cand = pd.read_parquet(ROOT / "data/rich_crossday_20260610/candidates.parquet")
    cand = cand[(cand.k == 3) & (cand.v_sol == 0.3)].set_index("mint")
    sweep = pickle.load(open(ROOT / "data/sweep_k3v03.pkl", "rb"))
    snames, smints = sweep["names"], sweep["mints"]
    rich_feats = json.loads((ROOT / "bot_artifacts_rich_shadow/model_spec.json").read_text())["entry"]["features"]
    rich_feats = [c for c in rich_feats if c in cand.columns]

    both = [m for m in cand.index if m in smints]
    cj = cand.loc[both]
    peak_col = [c for c in cand.columns if c.startswith("peak_ret_h")][0]
    y = (cj[peak_col].values >= 2.0).astype(int)
    tr = (cj.decision_ts < JUN10).values
    te = ~tr
    print(f"joined mints: {len(both)}  train(Jun9)={tr.sum()} test(Jun10-11)={te.sum()}  "
          f"peak200 base tr={y[tr].mean():.3f} te={y[te].mean():.3f}\n")

    # 22-feat from sweep features (rebuild the frame aligned to `both`)
    f22 = pd.DataFrame({n: [smints[m]["feats"][i] for m in both] for i, n in enumerate(snames)},
                       index=both)

    results = {}
    for label, X in (("22feat", f22[snames].values), ("RICH", cj[rich_feats].values)):
        clf = HistGradientBoostingClassifier(**REG).fit(X[tr], y[tr])
        s_te = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], s_te)
        thr = float(np.quantile(s_te, 1.0 - TOP_PCT / 100.0))
        te_m = [both[i] for i in range(len(both)) if te[i]]
        fired = [(mm, sc) for mm, sc in zip(te_m, s_te) if sc >= thr]
        print(f"=== {label}: test_auc={auc:.4f}  fires={len(fired)}/{te.sum()} (top {TOP_PCT:.0f}%) ===")
        for line in exec_block(label, fired, smints):
            print(line)
        results[label] = auc
    print("\nverdict: if RICH/lat1 and RICH/slot DEDUPED are positive while 22feat's are negative,")
    print("rich's selection rescues the realistic-latency economics (leg-3 confirmed).")


if __name__ == "__main__":
    main()
