#!/usr/bin/env python3
"""train_crossera_k3v03.py — the month-gap test and the final candidate.

Eval 1 (pure cross-era): train on May (Apr 29 - May 5, 70,669 ready mints,
live-matched, honest population) -> test on ALL of June (Jun 7-9, 21,567).
A month of regime drift sits between train and test. If buckets hold here,
the signal is structural, not a day artifact.

Eval 2 (final candidate): train May + Jun 7-8 -> test Jun 9. Saves
bot_artifacts_k3v03_final/ (both target heads). Does NOT touch the symlink.
"""
import calendar
import glob
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
REG = dict(max_iter=150, max_depth=3, learning_rate=0.05, l2_regularization=5.0, random_state=42)
JUN9 = calendar.timegm((2026, 6, 9, 0, 0, 0))
OUT = ROOT / "bot_artifacts_k3v03_final"


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


def bucket_table(s, y, peak):
    d = pd.DataFrame({"s": s, "y": y, "peak": peak})
    d["b"] = pd.cut(d.s, np.arange(0, 1.01, 0.1))
    return d.groupby("b", observed=True).agg(
        n=("y", "size"), win=("y", "mean"), mean_peak=("peak", "mean")).query("n>0")


def main():
    may = pd.read_parquet(ROOT / "data/live_matched_k3v03_may.parquet")
    jun = pd.read_parquet(ROOT / "data/live_matched_k3v03_all.parquet")
    files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a = first_row(files[0])
    b = first_row(files[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])
    jun["ready_ts"] = a[1] + (jun.first_slot - a[0]) / sps
    t_end = jun.ready_ts.max()
    jun = jun[jun.ready_ts < t_end - 600].reset_index(drop=True)

    feats = [c for c in may.columns if c not in ("mint", "first_slot", "ready_ts", "peak_ret", "n_fwd")]
    assert all(f in jun.columns for f in feats)
    print(f"May n={len(may)}  June n={len(jun)} (Jun9+ share {(jun.ready_ts>=JUN9).mean():.2f})")

    Xm, Xj = may[feats].values, jun[feats].values
    for tgt, tname in [(0.5, "peak_ge_50"), (2.0, "peak_ge_200")]:
        ym = (may.peak_ret >= tgt).astype(int).values
        yj = (jun.peak_ret >= tgt).astype(int).values
        clf = HistGradientBoostingClassifier(**REG).fit(Xm, ym)
        s = clf.predict_proba(Xj)[:, 1]
        print(f"\n=== EVAL1 {tname}: train MAY (base {ym.mean():.3f}) -> test ALL JUNE "
              f"(base {yj.mean():.3f})  AUC {roc_auc_score(yj, s):.4f} ===")
        print(bucket_table(s, yj, jun.peak_ret.values).to_string(float_format=lambda v: f"{v:.3f}"))

    tr_jun = (jun.ready_ts < JUN9).values
    comb_X = np.vstack([Xm, Xj[tr_jun]])
    te_X = Xj[~tr_jun]
    OUT.mkdir(exist_ok=True)
    heads = {}
    for tgt, tname in [(0.5, "peak_ge_50"), (2.0, "peak_ge_200")]:
        y_all = np.concatenate([
            (may.peak_ret >= tgt).astype(int).values,
            (jun.peak_ret[tr_jun] >= tgt).astype(int).values,
        ])
        y_te = (jun.peak_ret[~tr_jun] >= tgt).astype(int).values
        clf = HistGradientBoostingClassifier(**REG).fit(comb_X, y_all)
        s = clf.predict_proba(te_X)[:, 1]
        print(f"\n=== EVAL2 {tname}: train MAY+Jun7-8 (n={len(comb_X)}) -> test Jun9 "
              f"(n={len(te_X)}, base {y_te.mean():.3f})  AUC {roc_auc_score(y_te, s):.4f} ===")
        print(bucket_table(s, y_te, jun.peak_ret[~tr_jun].values).to_string(float_format=lambda v: f"{v:.3f}"))
        heads[tname] = clf
        fname = "entry_model.pkl" if tname == "peak_ge_50" else "entry_model_tp200.pkl"
        with open(OUT / fname, "wb") as f:
            pickle.dump(clf, f)   # BARE classifier: ModelServer contract

    (OUT / "TRAIN_NOTE.json").write_text(json.dumps({
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train": "May coherent span (Apr29-May5) + June capture Jun7-8, live-matched honest population (min_fwd=0)",
        "features": feats,
        "pending": "model_spec.json written at packaging time after economics replay",
    }, indent=1))
    print(f"\nwrote {OUT}/ (bare pkls for both heads)")


if __name__ == "__main__":
    main()
