#!/usr/bin/env python3
"""exit_lab.py — exit-policy tournament on the deployed model's fires.

Phase 1 (anatomy, TRAIN days Jun7-9 only): what the paths of elite-fired
tokens actually do: crossing probabilities, suffix upside after each level,
retrace after peak, worst single-snap gap (trailing feasibility).

Phase 2 (tournament): fixed policy formulas + one trained continuation head
(Longstaff-Schwartz-style optimal stopping using the recovery snaps' fm
label, trained on Jun7-9 snaps only). All policies on the IDENTICAL fires,
entry at lat1 (paper-book convention), per-tranche tx fees charged, deduped.
Design/selection on TRAIN fires; ONE look at TEST (Jun10-11).
Pre-registered judge: TEST deduped mean net; tiebreak p25.

Data: data/sweep_k3v03.pkl (paths+slots), bot_artifacts_k3v03_final pkl
(exact deployed scorer), data/recovery_snaps_k3v03.parquet (fm labels).
"""
import calendar
import glob
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
Q, CB, FEE = 0.1, 250.0, 0.0015
THR = 0.50
HORIZON = 300.0
PATH9 = ["ret", "run_max_ret", "dd", "fill_k", "buy_frac_w", "nsell_w",
         "solo_sell_w", "vel_w", "dts"]
K11 = ["win_ret", "dir_eff", "buy_frac", "uniq", "net_sol", "tot_sol",
       "single_actor_share", "trades_per_sec", "entry_sol", "win_drawup", "win_drawdown"]


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
    pm = [np.mean(v) for v in by.values()]
    return float(np.mean(pm)), len(pm)


class Fire:
    __slots__ = ("mint", "score", "entry_vs", "entry_vt", "rets", "vs", "vt", "ts", "lsm")

    def __init__(self, mint, score, f, lat=1):
        self.mint = mint
        self.score = score
        j = min(lat, len(f) - 1)
        self.entry_vs, self.entry_vt = f[j][2], f[j][3]
        em = self.entry_vs / self.entry_vt
        rest = f[j + 1:]
        self.ts = np.array([r[0] - f[j][0] for r in rest])
        self.vs = np.array([r[2] for r in rest])
        self.vt = np.array([r[3] for r in rest])
        self.rets = (self.vs / self.vt) / em - 1.0
        self.lsm = None


def simulate(fire, plan):
    """plan: list of (kind, params, frac_of_initial). kinds:
       tp(level) | time(t) | trail(arm, retrace) | lsm(pstar) | horizon
       Sells frac of INITIAL tokens at trigger snap; leftovers at last snap."""
    q = Q * 1e9
    tok0 = bt(fire.entry_vs, fire.entry_vt, q)
    n = len(fire.rets)
    if n == 0:
        return -CB / 1e4 - 2 * FEE / Q
    proceeds = 0.0
    sold = 0.0
    ntx = 1  # the buy
    pending = list(plan)
    runmax = 0.0
    armed = {}
    for i in range(n):
        r = fire.rets[i]
        runmax = max(runmax, r)
        fired_now = []
        for p in pending:
            kind, prm, frac = p
            hit = False
            if kind == "tp" and r >= prm:
                hit = True
            elif kind == "time" and fire.ts[i] >= prm:
                hit = True
            elif kind == "trail":
                arm, retr = prm
                if runmax >= arm:
                    armed[id(p)] = True
                if armed.get(id(p)) and r <= (1 + runmax) * (1 - retr) - 1:
                    hit = True
            elif kind == "lsm" and fire.lsm is not None and fire.lsm[i] < prm:
                hit = True
            if hit:
                fired_now.append(p)
        for p in fired_now:
            frac = p[2]
            sell_tok = tok0 * frac
            proceeds += sl(fire.vs[i], fire.vt[i], sell_tok)
            sold += frac
            ntx += 1
            pending.remove(p)
        if sold >= 0.999:
            break
    if sold < 0.999:
        rem = tok0 * (1 - sold)
        proceeds += sl(fire.vs[-1], fire.vt[-1], rem)
        ntx += 1
    return proceeds / q - 1.0 - CB / 1e4 - ntx * FEE / Q


def main():
    d = pickle.load(open(ROOT / "data/sweep_k3v03.pkl", "rb"))
    names, M = d["names"], d["mints"]
    fs = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a, b = fr(fs[0]), fr(fs[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])
    s2t = lambda s: a[1] + (s - a[0]) / sps
    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    mints = list(M)
    X = np.array([M[m]["feats"] for m in mints])
    sc = clf.predict_proba(X)[:, 1]
    ts = np.array([s2t(M[m]["decision"]["slot"]) for m in mints])
    fires = [(m, s, t) for m, s, t in zip(mints, sc, ts) if s >= THR and M[m]["fwd"]]
    train = [Fire(m, s, M[m]["fwd"]) for m, s, t in fires if t < JUN10]
    test = [Fire(m, s, M[m]["fwd"]) for m, s, t in fires if t >= JUN10]
    print(f"fires: train(Jun7-9)={len(train)}  test(Jun10-11)={len(test)}\n")

    print("=== PHASE 1: path anatomy (TRAIN fires, lat1 entry) ===")
    peaks = np.array([f.rets.max() if len(f.rets) else 0.0 for f in train])
    for lv in (0.5, 1.0, 2.0, 4.0):
        cross = peaks >= lv
        print(f"  P(cross +{lv*100:.0f}%) = {cross.mean():.0%}", end="")
        if lv < 4.0:
            nxt = {0.5: 1.0, 1.0: 2.0, 2.0: 4.0}[lv]
            cond = peaks[cross] >= nxt
            print(f"   P(then reach +{nxt*100:.0f}% | crossed) = {cond.mean():.0%}" if cross.any() else "")
        else:
            print()
    give = []
    gaps = []
    for f in train:
        if len(f.rets) and f.rets.max() >= 0.5:
            i50 = int(np.argmax(f.rets >= 0.5))
            after = f.rets[i50:]
            give.append(after.max() - after[-1])
            gaps.append(np.min(np.diff(after)) if len(after) > 1 else 0.0)
    print(f"  after crossing +50%: give-back peak->horizon p50={np.median(give):.2f} "
          f"p90={np.percentile(give,90):.2f}; worst single-snap gap p50={np.median(gaps):.2f} "
          f"p10={np.percentile(gaps,10):.2f}  (gap severity = trailing's enemy)")
    t2p = [f.ts[int(np.argmax(f.rets))] for f in train if len(f.rets)]
    print(f"  time-to-peak p50={np.median(t2p):.0f}s p90={np.percentile(t2p,90):.0f}s\n")

    # continuation head for LSM policy (train snaps Jun7-9 only)
    snaps = pd.read_parquet(ROOT / "data/recovery_snaps_k3v03.parquet")
    str_ = snaps[snaps.ready_ts < JUN10]
    ylsm = ((str_.fm - str_.ret) >= 0.25).astype(int).values   # >=25% more upside ahead
    lsm = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)
    lsm.fit(str_[PATH9 + K11].values, ylsm)
    print(f"LSM continuation head: trained on {len(str_):,} Jun7-9 snaps "
          f"(label: >=+25% further upside ahead; base {ylsm.mean():.2f})")

    # attach per-snap LSM scores to fires (recompute path feats from snaps parquet by mint)
    bymint = {m: g.sort_values("fwd_i") for m, g in snaps.groupby("mint")}
    for flist in (train, test):
        for f in flist:
            g = bymint.get(f.mint)
            if g is None or len(g) == 0:
                continue
            p = lsm.predict_proba(g[PATH9 + K11].values)[:, 1]
            # align: snaps are decision-anchored; fire path is lat1-anchored (1 shorter)
            k = min(len(p) - 1, len(f.rets))
            if k > 0:
                f.lsm = np.concatenate([p[1:1 + k], np.repeat(p[-1], max(0, len(f.rets) - k))])

    POLICIES = {
        "tp_50 (incumbent)": [("tp", 0.5, 1.0)],
        "tp_100": [("tp", 1.0, 1.0)],
        "tp_200": [("tp", 2.0, 1.0)],
        "time_60s": [("time", 60.0, 1.0)],
        "ladder 1/3@50,150 +ride": [("tp", 0.5, 1 / 3), ("tp", 1.5, 1 / 3)],
        "ladder 1/4@50,100,200,400": [("tp", 0.5, .25), ("tp", 1.0, .25), ("tp", 2.0, .25), ("tp", 4.0, .25)],
        "bank50%@50 + trail(arm100,re50)": [("tp", 0.5, 0.5), ("trail", (1.0, 0.5), 0.5)],
        "trail only (arm50,re40)": [("trail", (0.5, 0.4), 1.0)],
        "lsm p*<0.30": [("lsm", 0.30, 1.0)],
        "lsm p*<0.50": [("lsm", 0.50, 1.0)],
        "score-cond: <0.52 tp50 else bank+trail": None,  # special-cased
    }

    def run(flist, label):
        rows = []
        for name, plan in POLICIES.items():
            nets, scs = [], []
            for f in flist:
                if plan is None:
                    pl = [("tp", 0.5, 1.0)] if f.score < 0.52 else [("tp", 0.5, 0.5), ("trail", (1.0, 0.5), 0.5)]
                else:
                    pl = plan
                nets.append(simulate(f, pl))
                scs.append(f.score)
            ded, npat = dedup(scs, nets)
            rows.append((name, float(np.mean(nets)), ded, float(np.percentile(nets, 25)),
                         float(np.median(nets)), npat))
        rows.sort(key=lambda r: -r[2])
        print(f"\n=== {label}: {len(flist)} fires ===")
        print(f"  {'policy':34s} {'raw':>7s} {'DEDUP':>7s} {'p25':>7s} {'med':>7s} npat")
        for name, raw, ded, p25, med, npat in rows:
            print(f"  {name:34s} {raw:+7.3f} {ded:+7.3f} {p25:+7.3f} {med:+7.3f} {npat:4d}")
        return rows

    run(train, "TRAIN Jun7-9 (design/selection)")
    run(test, "TEST Jun10-11 (the one look; judge = DEDUP, tiebreak p25)")


if __name__ == "__main__":
    main()
