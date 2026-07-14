#!/usr/bin/env python3
"""verify_deployed_elite.py — the strictest version of the reconciliation:
score the EXACT deployed pkl (bot_artifacts_k3v03_final, trained May+Jun7-8)
over the sweep forward-path mints, take its thr-0.50 fires on Jun 10-11, and
compute execution-adjusted nets RAW and DEDUPED. Then independently recompute
the live realized numbers from position_close records (not via the diag tool)
and confirm the paper book's entry convention.
"""
import calendar
import glob
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/the-distribution-will-manifest")
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
Q, CB, FT, TP = 0.1, 250.0, 0.0015, 0.50


def bt(vs, vt, d):
    return vt - (vs * vt) / (vs + d)


def sl(vs, vt, d):
    return vs - (vs * vt) / (vt + d)


def walk(evs, evt, path):
    q = Q * 1e9
    tok = bt(evs, evt, q)
    em = evs / evt
    xvs, xvt = evs, evt
    for (_t, _s, vs, vt, _b, _tp) in path:
        xvs, xvt = vs, vt
        if (vs / vt) / em - 1 >= TP:
            break
    return sl(xvs, xvt, tok) / q - 1 - CB / 1e4 - (FT * 2) / Q


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


def dedup(scores, nets):
    by = {}
    for s, x in zip(scores, nets):
        by.setdefault(round(float(s), 6), []).append(x)
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def main():
    d = pickle.load(open(ROOT / "data/sweep_k3v03.pkl", "rb"))
    names, M = d["names"], d["mints"]
    fs = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a, b = fr(fs[0]), fr(fs[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])
    s2t = lambda s: a[1] + (s - a[0]) / sps

    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    spec = json.load(open(ROOT / "bot_artifacts_k3v03_final/model_spec.json"))
    assert spec["entry"]["features"] == names, "feature order mismatch"

    mints = list(M)
    X = np.array([M[m]["feats"] for m in mints], dtype=float)
    s = clf.predict_proba(X)[:, 1]
    ts = np.array([s2t(M[m]["decision"]["slot"]) for m in mints])
    te = ts >= JUN10
    fired = [(m, sc) for m, sc, t in zip(mints, s, te) if t and sc >= 0.50]
    print(f"DEPLOYED pkl, thr 0.50, test Jun10-11: fires={len(fired)} "
          f"of {int(te.sum())} test mints ({len(fired)/te.sum():.2%})")

    for lab, lat in (("lat0", 0), ("lat1", 1), ("lat2", 2), ("slot", None)):
        nets, scs = [], []
        for m, sc in fired:
            f = M[m]["fwd"]
            if not f:
                continue
            if lat is None:
                i = 0
                while i < len(f) and f[i][1] <= M[m]["decision"]["slot"]:
                    i += 1
                if i >= len(f):
                    continue
                li = i
            else:
                li = min(lat, len(f) - 1)
            nets.append(walk(f[li][2], f[li][3], f[li + 1:]))
            scs.append(sc)
        if nets:
            dm, npat = dedup(scs, nets)
            print(f"  {lab:5s} raw mean={np.mean(nets):+.3f} win={np.mean([x>0 for x in nets]):.0%} "
                  f"n={len(nets)}  |  DEDUPED={dm:+.3f} (n_pat={npat})")

    # independent live realized recomputation from raw close records
    cut = 1781047096.0
    pol, book = [], []
    with open(ROOT / "bot_data/shadow_run.jsonl") as f:
        for ln in f:
            if '"position_close"' not in ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if (r.get("t") or 0) < cut:
                continue
            if r.get("live_policy_net") is not None:
                pol.append(float(r["live_policy_net"]))
            if r.get("net") is not None:
                book.append(float(r["net"]))
    print(f"\nLIVE realized (recomputed from raw position_close, era n={len(pol)}):")
    print(f"  policy net/fire = {np.mean(pol):+.4f}   book net/fire = {np.mean(book):+.4f}")
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    print(f"  paper_book.entry_lat_snaps = {cfg['paper_book']['entry_lat_snaps']} "
          f"(book entry convention; 1 == the offline lat1 row by construction)")


if __name__ == "__main__":
    main()
