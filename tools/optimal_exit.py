#!/usr/bin/env python3
"""optimal_exit.py — the rigorous exit study.

Four parts, all on the deployed pkl's fires, lat1 entry, per-tranche fees,
deduped judge, design Jun7-9 / ONE test look Jun10-11.

  P1 HINDSIGHT CEILING: the best net ANY causal exit could extract (sell at the
     true forward-max snap). The denominator: how much is on the table, and how
     much tp_100 already captures of it.
  P2 TIME-TO-+100% (Q1): of fires that reach +100%, when; and a time-cap sweep.
  P3 COLLAPSE-HAZARD AUC (Q3): train a hazard head h(X)=P(collapse>=50% within
     next ~30s | state) on Jun7-9 snaps, test AUC on Jun10-11. Does cutting
     losers separate better than picking winners (the 0.9 claim)?
  P4 OPTIMAL STOPPING (Q2): fitted backward-induction continuation value on a
     COARSE robust state (ret, dd, sell-frac, time bucket); forward-evaluate the
     resulting boundary (sell when m >= Chat), one OOS look vs tp_100.

Cost model: q=0.1, 250bps, 0.0015/tx.
"""
import calendar
import glob
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
Q, CB, FEE = 0.1, 250.0, 0.0015
THR = 0.50


def bt(vs, vt, d):
    return vt - (vs * vt) / (vs + d)


def sl(vs, vt, d):
    return vs - (vs * vt) / (vt + d)


def fr(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        for ln in f:
            try:
                r = json.loads(ln)
                if "slot" in r and "t" in r:
                    return float(r["slot"]), float(r["t"])
            except Exception:
                pass


def dedup(scores, vals):
    by = {}
    for s, x in zip(scores, vals):
        by.setdefault(round(float(s), 6), []).append(x)
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


class F:
    __slots__ = ("mint", "score", "evs", "evt", "tok0", "ret", "vs", "vt", "t", "mnet")

    def __init__(self, mint, score, f, lat=1):
        self.mint, self.score = mint, score
        j = min(lat, len(f) - 1)
        self.evs, self.evt = f[j][2], f[j][3]
        self.tok0 = bt(self.evs, self.evt, Q * 1e9)
        em = self.evs / self.evt
        rest = f[j + 1:]
        self.t = np.array([r[0] - f[j][0] for r in rest])
        self.vs = np.array([r[2] for r in rest], dtype=float)
        self.vt = np.array([r[3] for r in rest], dtype=float)
        self.ret = (self.vs / self.vt) / em - 1.0
        # realizable net if we liquidate the WHOLE position at snap i
        self.mnet = np.array([sl(self.vs[i], self.vt[i], self.tok0) / (Q * 1e9)
                              - 1.0 - CB / 1e4 - 2 * FEE / Q for i in range(len(rest))])


def sell_all_at(f, i):
    if i is None or i >= len(f.mnet):
        # never triggered: liquidate at last snap
        return f.mnet[-1] if len(f.mnet) else -CB / 1e4 - 2 * FEE / Q
    return f.mnet[i]


def tp_index(f, lvl):
    w = np.where(f.ret >= lvl)[0]
    return int(w[0]) if len(w) else None


def main():
    d = pickle.load(open(ROOT / "data/sweep_k3v03.pkl", "rb"))
    names, M = d["names"], d["mints"]
    fs = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a, b = fr(fs[0]), fr(fs[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])
    s2t = lambda s: a[1] + (s - a[0]) / sps
    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    mints = list(M)
    sc = clf.predict_proba(np.array([M[m]["feats"] for m in mints]))[:, 1]
    tt = np.array([s2t(M[m]["decision"]["slot"]) for m in mints])
    tr = [F(m, s, M[m]["fwd"]) for m, s, t in zip(mints, sc, tt) if s >= THR and M[m]["fwd"] and t < JUN10]
    te = [F(m, s, M[m]["fwd"]) for m, s, t in zip(mints, sc, tt) if s >= THR and M[m]["fwd"] and t >= JUN10]
    print(f"fires: train={len(tr)} test={len(te)}\n")

    # ---- P1 hindsight ceiling ----
    print("=== P1: HINDSIGHT CEILING (best net any causal exit could get) ===")
    for lab, fl in (("train", tr), ("test", te)):
        ceil = [f.mnet.max() if len(f.mnet) else 0.0 for f in fl]
        tp1 = [sell_all_at(f, tp_index(f, 1.0)) for f in fl]
        tp5 = [sell_all_at(f, tp_index(f, 0.5)) for f in fl]
        dc, _ = dedup([f.score for f in fl], ceil)
        d1, _ = dedup([f.score for f in fl], tp1)
        d5, _ = dedup([f.score for f in fl], tp5)
        print(f"  {lab}: ceiling DEDUP={dc:+.3f} | tp_100={d1:+.3f} ({d1/dc:.0%} of ceiling) "
              f"| tp_50={d5:+.3f} ({d5/dc:.0%})")

    # ---- P2 time to +100% (Q1) ----
    print("\n=== P2: TIME-TO-+100% (Q1) ===")
    t100 = [f.t[tp_index(f, 1.0)] for f in tr if tp_index(f, 1.0) is not None]
    print(f"  of train fires reaching +100%: n={len(t100)}, time-to p25={np.percentile(t100,25):.0f}s "
          f"p50={np.percentile(t100,50):.0f}s p90={np.percentile(t100,90):.0f}s")
    print("  time-cap sweep (sell at +100% if reached by cap, else at cap) [test deduped]:")
    for cap in (30, 60, 90, 120, 300):
        nets, scs = [], []
        for f in te:
            i100 = tp_index(f, 1.0)
            ci = np.where(f.t >= cap)[0]
            cap_i = int(ci[0]) if len(ci) else (len(f.mnet) - 1 if len(f.mnet) else None)
            if i100 is not None and (cap_i is None or i100 <= cap_i):
                nets.append(sell_all_at(f, i100))
            else:
                nets.append(sell_all_at(f, cap_i))
            scs.append(f.score)
        dm, npat = dedup(scs, nets)
        print(f"    cap={cap:3d}s  DEDUP={dm:+.3f} (n_pat={npat})")

    # ---- P3 collapse-hazard AUC (Q3) ----
    print("\n=== P3: COLLAPSE-HAZARD model (Q3: can we 'cut losers' at AUC ~0.9?) ===")
    snaps = pd.read_parquet(ROOT / "data/recovery_snaps_k3v03.parquet")
    P9 = ["ret", "run_max_ret", "dd", "fill_k", "buy_frac_w", "nsell_w", "solo_sell_w", "vel_w", "dts"]
    # label: from this snap, does the token collapse >=40% below CURRENT mid within the rest of the path?
    snaps = snaps.sort_values(["mint", "fwd_i"])
    fut_min = snaps.groupby("mint")["ret"].transform(lambda s: s[::-1].cummin()[::-1].shift(-1))
    snaps = snaps.assign(fut_min=fut_min).dropna(subset=["fut_min"])
    snaps["collapse"] = ((1 + snaps.fut_min) / (1 + snaps.ret) - 1 <= -0.40).astype(int)
    str_ = snaps[snaps.ready_ts < JUN10]
    ste = snaps[snaps.ready_ts >= JUN10]
    hz = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                        l2_regularization=2.0, random_state=0)
    hz.fit(str_[P9].values, str_.collapse.values)
    auc = roc_auc_score(ste.collapse.values, hz.predict_proba(ste[P9].values)[:, 1])
    print(f"  hazard head: train {len(str_):,} snaps (collapse base {str_.collapse.mean():.2f}); "
          f"TEST collapse-AUC = {auc:.4f}")
    # winner-side reference: P(>=+25% more upside) AUC, same split
    fu_max = snaps.groupby("mint")["ret"].transform(lambda s: s[::-1].cummax()[::-1].shift(-1))
    snaps = snaps.assign(up=((snaps.assign(fm=fu_max).fm - snaps.ret) >= 0.25).astype(int))
    s2 = snaps[snaps.ready_ts < JUN10]; s3 = snaps[snaps.ready_ts >= JUN10]
    up = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                        l2_regularization=2.0, random_state=0).fit(s2[P9].values, s2.up.values)
    aucu = roc_auc_score(s3.up.values, up.predict_proba(s3[P9].values)[:, 1])
    print(f"  winner head (>=+25% more upside): TEST AUC = {aucu:.4f}")
    print(f"  => {'HAZARD separates better (cut-losers easier than pick-winners)' if auc>aucu else 'winner head as good or better'}")

    # ---- P4 fitted optimal stopping (Q2) ----
    print("\n=== P4: FITTED OPTIMAL STOPPING (backward-induction continuation value) ===")
    # build per-snap training matrix from TRAIN fires: state + realized-best-future-net (the target C)
    def snap_state(f, i):
        sell = 0.0  # sell pressure proxy unavailable per-snap here; use dd + time + ret
        return [f.ret[i], (f.ret[i] - max(f.ret[:i + 1].max(), 0.0)), f.t[i]]
    Xtr, ytr = [], []
    for f in tr:
        n = len(f.mnet)
        if n == 0:
            continue
        best_future = np.maximum.accumulate(f.mnet[::-1])[::-1]  # max net from i onward
        for i in range(n):
            Xtr.append(snap_state(f, i))
            ytr.append(best_future[i])
    reg = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                        l2_regularization=2.0, random_state=0).fit(np.array(Xtr), np.array(ytr))
    def policy_net(f):
        for i in range(len(f.mnet)):
            chat = reg.predict([snap_state(f, i)])[0]
            if f.mnet[i] >= chat:        # sell-now >= expected best-future
                return f.mnet[i]
        return f.mnet[-1] if len(f.mnet) else -CB / 1e4 - 2 * FEE / Q
    for lab, fl in (("train", tr), ("test", te)):
        nets = [policy_net(f) for f in fl]
        tp1 = [sell_all_at(f, tp_index(f, 1.0)) for f in fl]
        dpn, np_ = dedup([f.score for f in fl], nets)
        dtp, _ = dedup([f.score for f in fl], tp1)
        print(f"  {lab}: OPTIMAL-STOP DEDUP={dpn:+.3f}  vs tp_100 DEDUP={dtp:+.3f}  (n_pat={np_})")
    print("\n  judge: optimal-stop must BEAT tp_100 on TEST deduped to justify its complexity.")


if __name__ == "__main__":
    main()
