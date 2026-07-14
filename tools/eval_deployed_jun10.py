#!/usr/bin/env python3
"""eval_deployed_jun10.py — offline forward test of the DEPLOYED model on its
true forward day. Scores the live pkl (bot_artifacts_k3v03_final) over the
honest live-matched extraction and reports Jun 10 (never seen in training:
trained May + Jun 7-8, holdout Jun 9) with Jun 9 as reference, including
pattern dedup so launch-farm replays can't inflate the read.
"""
import calendar
import glob
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN9 = calendar.timegm((2026, 6, 9, 0, 0, 0))
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
THR = 0.50


def first_row(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
                if "slot" in r and "t" in r:
                    return float(r["slot"]), float(r["t"])
            except Exception:
                pass


def main():
    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    spec = json.loads((ROOT / "bot_artifacts_k3v03_final/model_spec.json").read_text())
    feats = spec["entry"]["features"]

    files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a, b = first_row(files[0]), first_row(files[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])
    df = pd.read_parquet(ROOT / "data/live_matched_k3v03_all2.parquet")
    df["ts"] = a[1] + (df.first_slot - a[0]) / sps
    df = df[df.ts < df.ts.max() - 600].reset_index(drop=True)
    s = clf.predict_proba(df[feats].values)[:, 1]
    df["score"] = s

    for name, lo, hi in [("Jun 9 (holdout ref)", JUN9, JUN10), ("Jun 10 (TRUE forward)", JUN10, 1e18)]:
        d = df[(df.ts >= lo) & (df.ts < hi)]
        if len(d) < 50:
            print(f"{name}: only {len(d)} rows, skipping")
            continue
        y50 = (d.peak_ret >= 0.5).astype(int).values
        y200 = (d.peak_ret >= 2.0).astype(int).values
        sc = d.score.values
        fired = d[sc >= THR]
        print(f"\n=== {name}: n={len(d)}  base50={y50.mean():.3f} base200={y200.mean():.3f} ===")
        print(f"AUC: peak50 {roc_auc_score(y50, sc):.4f}  peak200 {roc_auc_score(y200, sc):.4f}")
        print(f"fired @ {THR}: {len(fired)} ({len(fired)/len(d):.2%})  "
              f"hit50 {float((fired.peak_ret >= 0.5).mean()) if len(fired) else float('nan'):.1%}  "
              f"hit200 {float((fired.peak_ret >= 2.0).mean()) if len(fired) else float('nan'):.1%}  "
              f"mean_peak {fired.peak_ret.mean() if len(fired) else float('nan'):+.2f}")
        if len(fired):
            pats = fired.groupby(fired.score.round(6))
            print(f"pattern dedup: {len(fired)} fires / {pats.ngroups} distinct patterns")
            rep = {k: g for k, g in pats if len(g) > 1}
            for k, g in sorted(rep.items(), key=lambda kv: -len(kv[1]))[:4]:
                print(f"  repeated score={k:.6f}: n={len(g)} hit50={float((g.peak_ret>=0.5).mean()):.0%}")
            dd50 = pats.apply(lambda g: (g.peak_ret >= 0.5).mean())
            print(f"  deduped hit50/pattern: {dd50.mean():.1%} (n_pat={pats.ngroups})")
        band = d[(sc >= 0.4)]
        print(f"band >=0.40: n={len(band)} hit50 {float((band.peak_ret>=0.5).mean()) if len(band) else float('nan'):.1%}")


if __name__ == "__main__":
    main()
