#!/usr/bin/env python3
"""train_recovery_k3v03.py — K=3 recovery head + death-cut book comparison.

Replicates the validated recovery convention (drawdown snaps, recovers-to-
breakeven label, 9 path + 11 K features, HGB d3/250/0.05/l2=1) at the live
trigger, split chronologically (train Jun 7-9, test Jun 10), then replays
level_tp_50 WITH vs WITHOUT the death-cut on the deployed entry model's fired
bets (lat=1 convention: enter at first forward snap reserves, harness cost
model). Saves the candidate head; does NOT touch the live spec.
"""
import calendar
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
PATH9 = ["ret", "run_max_ret", "dd", "fill_k", "buy_frac_w", "nsell_w",
         "solo_sell_w", "vel_w", "dts"]
K11 = ["win_ret", "dir_eff", "buy_frac", "uniq", "net_sol", "tot_sol",
       "single_actor_share", "trades_per_sec", "entry_sol", "win_drawup", "win_drawdown"]
Q_SOL = 0.1
COST_BPS = 250.0
FEE_TX = 0.0015
TP = 0.50


def buy_tokens(vs, vt, dsol):
    return vt - (vs * vt) / (vs + dsol)


def sell_sol(vs, vt, dtok):
    return vs - (vs * vt) / (vt + dtok)


def main():
    df = pd.read_parquet(ROOT / "data/recovery_snaps_k3v03.parquet")
    feats = PATH9 + K11
    dd = df[df.ret < 0]
    tr = dd[dd.ready_ts < JUN10]
    te = dd[dd.ready_ts >= JUN10]
    print(f"drawdown snaps: train {len(tr):,} (recover base {(tr.fm>=0).mean():.3f})  "
          f"test {len(te):,} (base {(te.fm>=0).mean():.3f})")
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(tr[feats].values, (tr.fm >= 0).astype(int).values)
    for name, part in (("train", tr), ("test Jun10", te)):
        y = (part.fm >= 0).astype(int).values
        s = clf.predict_proba(part[feats].values)[:, 1]
        print(f"  {name} AUC {roc_auc_score(y, s):.4f}")
    s_te = clf.predict_proba(te[feats].values)[:, 1]
    bins = pd.cut(s_te, [0, 0.05, 0.10, 0.20, 0.40, 1.0])
    cal = pd.DataFrame({"b": bins, "y": (te.fm >= 0).astype(int)}).groupby(
        "b", observed=True).agg(n=("y", "size"), recover=("y", "mean"))
    print("calibration (test):\n" + cal.to_string(float_format=lambda v: f"{v:.3f}"))

    # entry scores for the fired-bet set
    ent = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    spec = json.loads((ROOT / "bot_artifacts_k3v03_final/model_spec.json").read_text())
    e22 = spec["entry"]["features"]
    lm = pd.read_parquet(ROOT / "data/live_matched_k3v03_all2.parquet")
    lm["escore"] = ent.predict_proba(lm[e22].values)[:, 1]
    fired = set(lm[lm.escore >= 0.50].mint)
    print(f"\nfired mints (entry thr 0.50): {len(fired)}")

    snaps = {m: g.sort_values("fwd_i") for m, g in df[df.mint.isin(fired)].groupby("mint")}

    def sim(g, cut_thr=None):
        rows = g[PATH9 + K11 + ["vsol", "vtok", "ret"]].to_dict("records")
        if len(rows) < 1:
            return None, None
        evs, evt = rows[0]["vsol"], rows[0]["vtok"]   # lat=1: enter at first fwd snap
        q = Q_SOL * 1e9
        tok = buy_tokens(evs, evt, q)
        e_mid = evs / evt
        xvs, xvt, cut = evs, evt, 0
        for r in rows[1:]:
            ret = (r["vsol"] / r["vtok"]) / e_mid - 1.0
            xvs, xvt = r["vsol"], r["vtok"]
            if ret >= TP:
                break
            if cut_thr is not None and ret < 0:
                p = clf.predict_proba([[r[f] for f in PATH9 + K11]])[0, 1]
                if p < cut_thr:
                    cut = 1
                    break
        net = sell_sol(xvs, xvt, tok) / q - 1.0 - COST_BPS / 1e4 - (FEE_TX * 2) / Q_SOL
        return net, cut

    for era, lo, hi in (("Jun7-9 (head in-train)", 0, JUN10), ("Jun10 OOS", JUN10, 9e18)):
        bets = [g for m, g in snaps.items()
                if lo <= g.ready_ts.iloc[0] < hi and len(g) >= 1]
        if not bets:
            continue
        base = [sim(g)[0] for g in bets]
        base = [x for x in base if x is not None]
        print(f"\n=== {era}: n={len(base)}  level_tp_50 alone: mean {np.mean(base):+.4f} "
              f"p25 {np.percentile(base,25):+.3f} es10 {np.mean(np.sort(base)[:max(1,len(base)//10)]):+.3f} "
              f"win {(np.array(base)>0).mean():.0%} ===")
        for thr in (0.05, 0.10, 0.15, 0.20):
            res = [sim(g, thr) for g in bets]
            nets = [x[0] for x in res if x[0] is not None]
            cuts = sum(x[1] for x in res if x[1] is not None)
            print(f"  +death_cut@{thr:.2f}: mean {np.mean(nets):+.4f} "
                  f"p25 {np.percentile(nets,25):+.3f} "
                  f"es10 {np.mean(np.sort(nets)[:max(1,len(nets)//10)]):+.3f} "
                  f"win {(np.array(nets)>0).mean():.0%}  cuts={cuts}")

    out = ROOT / "bot_artifacts_k3v03_final"
    with open(out / "recovery_candidate.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"\nsaved {out}/recovery_candidate.pkl (NOT wired into model_spec; deploy decision pending)")


if __name__ == "__main__":
    main()
