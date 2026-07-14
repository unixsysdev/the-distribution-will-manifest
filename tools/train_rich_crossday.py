#!/usr/bin/env python3
"""train_rich_crossday.py — the queued rich/intent question, answered honestly.

The june_causal sweep claimed big lift from 189 rich+intent features, but on a
single intraday split with 20-fire selection. Now that rich-schema capture
spans a day boundary (Jun 9 04:33 -> Jun 10), this runs the apples-to-apples
CROSS-DAY comparison at the deployed trigger (K=3/V=0.3), NO selection:
fixed model, fixed cells, day-boundary split, report everything.

Cells: 22f-equivalent baseline | rich without intent | rich + intent
       (intent rows now carry _present flags; zero-fill ambiguity removed)
x targets peak_ge_50 / peak_ge_200.
"""
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data/rich_crossday_20260610/candidates.parquet"
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
REG = dict(max_iter=150, max_depth=3, learning_rate=0.05, l2_regularization=5.0,
           random_state=42)
DENY = ("peak", "future", "label", "target", "tp", "terminal", "net_exit")
BASE22 = [
    "win_ret", "dir_eff", "buy_frac", "uniq", "net_sol", "tot_sol",
    "single_actor_share", "trades_per_sec", "entry_sol", "win_drawup", "win_drawdown",
]


def main():
    df = pd.read_parquet(CAND)
    df = df[(df.k == 3) & (df.v_sol == 0.3)].reset_index(drop=True)
    peak_col = [c for c in df.columns if c.startswith("peak_ret_h")][0]
    tr = (df.decision_ts < JUN10).values
    te = ~tr
    print(f"candidates: {len(df)}  train Jun9 n={tr.sum()}  test Jun10 n={te.sum()}")

    all_feats = [c for c in df.columns
                 if pd.api.types.is_numeric_dtype(df[c])
                 and c not in ("k", "v_sol", "decision_ts", "first_slot")
                 and not any(t in c for t in DENY)]
    f22 = [f"k_{b}" for b in BASE22] + [f"v_{b}" for b in BASE22]
    f22 = [c for c in f22 if c in df.columns]
    rich_noint = [c for c in all_feats if not c.startswith("intent_")]
    rich_int = all_feats
    print(f"feature sets: 22f-equiv={len(f22)}  rich={len(rich_noint)}  rich+intent={len(rich_int)}")

    for tgt, tname, net_col in [(0.5, "peak_ge_50", "tp50_net"), (2.0, "peak_ge_200", "tp200_net")]:
        y = (df[peak_col].values >= tgt).astype(int)
        if y[tr].sum() < 10 or y[te].sum() < 10:
            print(f"{tname}: too few positives")
            continue
        print(f"\n=== {tname}: train base {y[tr].mean():.3f}  test base {y[te].mean():.3f} ===")
        print(f"{'cell':14s} {'trainAUC':>8s} {'testAUC':>8s} | top-5% band: {'n':>4s} {'prec':>6s} {'net/bet':>8s}")
        for cell, feats in (("22f-equiv", f22), ("rich", rich_noint), ("rich+intent", rich_int)):
            X = df[feats].astype(float).replace([np.inf, -np.inf], np.nan).values
            clf = HistGradientBoostingClassifier(**REG).fit(X[tr], y[tr])
            s_tr = clf.predict_proba(X[tr])[:, 1]
            s_te = clf.predict_proba(X[te])[:, 1]
            thr5 = np.quantile(s_te, 0.95)
            m = s_te >= thr5
            net = df[net_col].values[te][m] if net_col in df.columns else np.array([])
            print(f"{cell:14s} {roc_auc_score(y[tr], s_tr):8.4f} {roc_auc_score(y[te], s_te):8.4f} | "
                  f"{m.sum():4d} {y[te][m].mean():6.1%} "
                  f"{(net.mean() if len(net) else float('nan')):+8.3f}")
    print("\nNote: NaN passthrough (no zero-fill); intent cols carry _present flags.")


if __name__ == "__main__":
    main()
