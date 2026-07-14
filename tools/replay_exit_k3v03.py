#!/usr/bin/env python3
"""replay_exit_k3v03.py — book economics for the cross-day K3/V0.3 candidate.

Streams the full gRPC capture with the live gates + joint K=3/V=0.3 trigger,
scores every ready mint with BOTH candidate heads (peak_ge_50 / peak_ge_200),
collects 300s forward paths, then simulates level-TP exits with the harness
cost model (q=0.1 SOL, cost_bps=250, fee 0.0015/tx, stale/horizon 300s).

Main table: Jun 9 bets only (cross-day OOS for the candidate). Jun 7-8 shown
as in-sample reference. Entry latency 0/1/2 forward trades; the live paper
book runs entry_lat_snaps=1, so lat=1 is the PLAUSIBLE operating row.

Pre-stated adoption rule (anti winner's-curse):
  adopt (head, thr, tp) iff at lat=1 Jun9 mean_net>0 with bootstrap
  P(mean>0)>=0.90, the same (head,tp) is positive at neighboring thresholds,
  and lat=2 mean >= -0.01. Prefer the simplest TP satisfying; tiebreak on
  p25, not max mean.
No recovery/death-cut head exists for K=3 yet; exits are TP/stale only,
matching the deployed level_tp family.
"""
import calendar
import glob
import gzip
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

assert os.environ.get("K_TRIGGER") == "3" and os.environ.get("V_TRIGGER") == "0.3", \
    "run with K_TRIGGER=3 V_TRIGGER=0.3"
from feature_accum import TokenState, ENTRY_FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
import sys
ART = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bot_artifacts_k3v03_crossday"
TAG = sys.argv[2] if len(sys.argv) > 2 else "crossday"
CACHE = ROOT / "data/replay_paths_k3v03.pkl"
FRESH = 3_000_000_000
HORIZON = 300.0
Q_SOL = 0.1
COST_BPS = 250.0
FEE_TX = 0.0015
JUN9 = calendar.timegm((2026, 6, 9, 0, 0, 0))


def is_classic(vsol, rsol):
    return abs(vsol - 30_000_000_000 - rsol) < 50_000_000


def buy_tokens(vs, vt, dsol):
    return vt - (vs * vt) / (vs + dsol)


def sell_sol(vs, vt, dtok):
    return vs - (vs * vt) / (vt + dtok)


def unwrap(p):
    obj = pickle.load(open(p, "rb"))
    return obj["model"] if isinstance(obj, dict) else obj


def main():
    h50 = unwrap(ART / "entry_model.pkl")
    h200 = unwrap(ART / "entry_model_tp200.pkl")

    if CACHE.exists():
        print(f"loading cached paths {CACHE}", flush=True)
        ready, fwd, max_ts = pickle.load(open(CACHE, "rb"))
        run_stream = False
    else:
        run_stream = True

    states, first_seen = {}, {}
    if run_stream:
        ready = {}   # mint -> dict(ts, vs, vt, feats)
        fwd = {}     # mint -> list[(ts, vs, vt)]
    t0 = time.time()
    files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*"))) if run_stream else []
    n = 0
    if run_stream:
        max_ts = 0.0
    for path in files:
        op = gzip.open if path.endswith(".gz") else open
        try:
            fh = op(path, "rt")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "vsol" not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                mint = r.get("mint")
                if not mint or "vsol" not in r:
                    continue
                try:
                    vsol = float(r["vsol"]); vtok = float(r["vtok"]); rsol = float(r["rsol"])
                    sol = float(r["sol"]) / 1e9; ts = float(r["ev_ts"])
                except (KeyError, TypeError, ValueError):
                    continue
                if vsol <= 0 or vtok <= 0:
                    continue
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  .. {n/1e6:.0f}M trades | ready={len(ready)} | {time.time()-t0:.0f}s", flush=True)
                if ts > max_ts:
                    max_ts = ts
                if not is_classic(vsol, rsol):
                    continue
                if mint in ready and mint in fwd:
                    f = fwd[mint]
                    if f is not None and ts - ready[mint]["ts"] <= HORIZON:
                        f.append((ts, vsol, vtok))
                    continue
                if mint not in first_seen:
                    first_seen[mint] = 1e18 if rsol >= FRESH else rsol
                    if first_seen[mint] >= FRESH:
                        continue
                if first_seen[mint] >= FRESH:
                    continue
                is_buy = bool(r.get("is_buy")); user = r.get("user", "")
                st = states.get(mint)
                if st is None:
                    states[mint] = TokenState(vsol, vtok, sol, is_buy, user, ts)
                    continue
                st.update(vsol, vtok, sol, is_buy, user, ts)
                if st.k_fired and st.v_fired and mint not in ready:
                    feats = st.combined_entry_features()
                    ready[mint] = {"ts": ts, "vs": vsol, "vt": vtok, "feats": feats}
                    fwd[mint] = []
                    del states[mint]
    if run_stream:
        print(f"streamed {n:,} trades in {(time.time()-t0)/60:.1f}min; ready mints={len(ready)}", flush=True)
        pickle.dump((ready, fwd, max_ts), open(CACHE, "wb"))
        print(f"cached paths -> {CACHE}", flush=True)

    mints = [m for m in ready if ready[m]["ts"] <= max_ts - HORIZON - 10]
    X = np.array([ready[m]["feats"] for m in mints], dtype=float)
    s50 = h50.predict_proba(X)[:, 1]
    s200 = h200.predict_proba(X)[:, 1]
    rts = np.array([ready[m]["ts"] for m in mints])
    is_test = rts >= JUN9
    print(f"scored {len(mints)} (tail-guarded) | Jun9 share {is_test.mean():.2f}", flush=True)

    def simulate(mint, lat, tp):
        e = ready[mint]
        path = fwd[mint]
        if lat == 0 or len(path) == 0:
            evs, evt = e["vs"], e["vt"]
            start = 0
        else:
            j = min(lat, len(path)) - 1
            evs, evt = path[j][1], path[j][2]
            start = j + 1
        q_lam = Q_SOL * 1e9
        tok = buy_tokens(evs, evt, q_lam)
        e_mid = evs / evt
        xvs, xvt = evs, evt
        for (ts_, vs_, vt_) in path[start:]:
            ret = (vs_ / vt_) / e_mid - 1.0
            xvs, xvt = vs_, vt_
            if ret >= tp:
                break
        proceeds = sell_sol(xvs, xvt, tok)
        return proceeds / q_lam - 1.0 - COST_BPS / 1e4 - (FEE_TX * 2.0) / Q_SOL

    rng = np.random.default_rng(42)

    def boot_p(nets):
        if len(nets) < 5:
            return float("nan")
        a = np.array(nets)
        means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
        return float(np.mean(np.array(means) > 0))

    grids = {
        "tp50_head": (s50, [0.35, 0.45, 0.55, 0.65]),
        "tp200_head": (s200, [0.20, 0.30, 0.40, 0.50]),
    }
    results = []
    print(f"\n{'head':10s} {'thr':>5s} {'tp':>4s} {'lat':>3s} | {'n9':>4s} {'mean9':>7s} {'med9':>7s} "
          f"{'p25':>7s} {'win9':>5s} {'P>0':>5s} | {'n78':>5s} {'mean78':>7s}")
    for head, (sc, thrs) in grids.items():
        for thr in thrs:
            sel = sc >= thr
            for tp in (0.5, 1.0, 2.0):
                for lat in (0, 1, 2):
                    nets9, nets78 = [], []
                    for i, m in enumerate(mints):
                        if not sel[i]:
                            continue
                        net = simulate(m, lat, tp)
                        (nets9 if is_test[i] else nets78).append(net)
                    if not nets9:
                        continue
                    a9 = np.array(nets9)
                    row = {
                        "head": head, "thr": thr, "tp": tp, "lat": lat,
                        "n_jun9": len(a9), "mean_jun9": float(a9.mean()),
                        "median_jun9": float(np.median(a9)),
                        "p25_jun9": float(np.percentile(a9, 25)),
                        "win_jun9": float((a9 > 0).mean()),
                        "p_gt0": boot_p(nets9),
                        "n_jun78": len(nets78),
                        "mean_jun78": float(np.mean(nets78)) if nets78 else None,
                    }
                    results.append(row)
                    print(f"{head:10s} {thr:5.2f} {tp:4.1f} {lat:3d} | {row['n_jun9']:4d} "
                          f"{row['mean_jun9']:+7.3f} {row['median_jun9']:+7.3f} {row['p25_jun9']:+7.3f} "
                          f"{row['win_jun9']:5.1%} {row['p_gt0']:5.2f} | {row['n_jun78']:5d} "
                          f"{(row['mean_jun78'] if row['mean_jun78'] is not None else float('nan')):+7.3f}", flush=True)

    out = ROOT / f"data/replay_exit_k3v03_{TAG}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
