#!/usr/bin/env python3
"""sweep_compare.py — head-to-head a K/V challenger vs the deployed K=3/V=0.3,
on the SAME June cross-day split, judged on execution-adjusted DEDUPED net
(not raw AUC). Pre-registered single hypothesis; no grid search here.

For each cell: train peak_ge_200 head on Jun 7-9, test Jun 10-11; report
cross-day AUC, then over the test fires (score>=thr) simulate fills under
Model A (lat 0/1/2) and Model B (slot-aware + tip-rank), and report the
DEDUPED (by score-pattern) execution-adjusted net. Same cost model as exec_sim.

Usage: python tools/sweep_compare.py data/sweep_k7v05.pkl
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
TOP_PCT = 3.0  # per-model top-% threshold (matched selectivity, not a shared cutoff)


def slot_map():
    def first_row(p):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "slot" in r and "t" in r:
                        return float(r["slot"]), float(r["t"])
                except Exception:
                    pass
    files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a, b = first_row(files[0]), first_row(files[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])
    return lambda s: a[1] + (s - a[0]) / sps


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
    """mean over distinct score-patterns (launch-farm replays count ~once)."""
    by = {}
    for s, x in zip(scores, nets):
        by.setdefault(round(float(s), 6), []).append(x)
    pm = [np.mean(v) for v in by.values()]
    return float(np.mean(pm)), len(pm)


def evaluate(pkl_path, s2t):
    d = pickle.load(open(pkl_path, "rb"))
    K, Vv, names, mints = d["K"], d["V"], d["names"], d["mints"]
    rows = []
    for m, r in mints.items():
        rows.append({"mint": m, "ts": s2t(r["decision"]["slot"]),
                     "peak_ret": r["peak_ret"], **dict(zip(names, r["feats"]))})
    df = pd.DataFrame(rows).set_index("mint")
    tr = (df.ts < JUN10).values
    te = ~tr
    y = (df.peak_ret >= 2.0).astype(int).values
    if y[tr].sum() < 10 or y[te].sum() < 5:
        return f"K={K}/V={Vv}: too few positives (tr {y[tr].sum()}, te {y[te].sum()})"
    clf = HistGradientBoostingClassifier(**REG).fit(df[names].values[tr], y[tr])
    s_te = clf.predict_proba(df[names].values[te])[:, 1]
    auc = roc_auc_score(y[te], s_te)
    te_mints = df.index[te]
    thr = float(np.quantile(s_te, 1.0 - TOP_PCT / 100.0))
    fired = [(mm, sc) for mm, sc in zip(te_mints, s_te) if sc >= thr]
    out = {"K": K, "V": Vv, "test_auc": auc, "n_fires": len(fired),
           "thr": thr, "n_test": int(te.sum())}
    for lab, lat in (("A_lat0", 0), ("A_lat1", 1), ("A_lat2", 2)):
        nets, scs = [], []
        for mm, sc in fired:
            f = mints[mm]["fwd"]
            if not f:
                continue
            j = min(lat, len(f) - 1)
            nets.append(walk(f[j][2], f[j][3], f[j + 1:])); scs.append(sc)
        if nets:
            dm, npat = dedup_mean(scs, nets)
            out[lab] = (float(np.mean(nets)), dm, npat)
    nets, scs = [], []
    for mm, sc in fired:
        f = mints[mm]["fwd"]
        li = land_b(mints[mm]["decision"], f)
        if li is not None:
            nets.append(walk(f[li][2], f[li][3], f[li + 1:])); scs.append(sc)
    if nets:
        dm, npat = dedup_mean(scs, nets)
        out["B_slotaware"] = (float(np.mean(nets)), dm, npat)
    return out


def main():
    s2t = slot_map()
    paths = sys.argv[1:] or [str(ROOT / "data/sweep_k7v05.pkl")]
    print("cross-day: train Jun7-9, test Jun10-11. metric = execution-adjusted DEDUPED net.\n")
    for p in paths:
        r = evaluate(p, s2t)
        if isinstance(r, str):
            print(r); continue
        print(f"=== K={r['K']} / V={r['V']}  test_auc={r['test_auc']:.4f}  "
              f"fires={r['n_fires']}/{r['n_test']} (top {TOP_PCT:.0f}%, thr {r['thr']:.3f}) ===")
        for k in ("A_lat0", "A_lat1", "A_lat2", "B_slotaware"):
            if k in r:
                raw, ded, npat = r[k]
                print(f"  {k:12s} raw_mean={raw:+.3f}  DEDUPED={ded:+.3f} (n_pat={npat})")
    print("\nNOTE: incumbent K=3/V=0.3 reference (June cross-day, from today): "
          "A_lat1 raw ~+0.08, deduped ~+0.10 (n_pat~12). Judge the challenger on "
          "DEDUPED B_slotaware / A_lat1, not AUC. Beat the incumbent by a margin "
          "(winner's-curse discount) before considering deployment.")


if __name__ == "__main__":
    main()
