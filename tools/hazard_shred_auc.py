#!/usr/bin/env python3
"""hazard_shred_auc.py — does pre-execution SHRED sell-intent improve the
collapse-hazard AUC over path-only? Same collapse label, same split, same HGB.
"""
import calendar
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JUN10 = calendar.timegm((2026, 6, 10, 0, 0, 0))
P9 = ["ret", "run_max_ret", "dd", "fill_k", "buy_frac_w", "nsell_w", "solo_sell_w", "vel_w", "dts"]
SHRED = ["shred_sell_2s", "shred_buy_2s", "shred_sellfrac_2s", "shred_tip_p90_2s"]
REG = dict(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=2.0, random_state=0)


def main():
    sn = pd.read_parquet(ROOT / "data/recovery_snaps_shred_k3v03.parquet").sort_values(["mint", "fwd_i"])
    fmn = sn.groupby("mint")["ret"].transform(lambda s: s[::-1].cummin()[::-1].shift(-1))
    sn = sn.assign(fut_min=fmn).dropna(subset=["fut_min"])
    sn["collapse"] = (((1 + sn.fut_min) / (1 + sn.ret) - 1) <= -0.40).astype(int)
    tr = sn[sn.ready_ts < JUN10]
    te = sn[sn.ready_ts >= JUN10]
    cov = (sn[["shred_sell_2s", "shred_buy_2s"]].sum(axis=1) > 0).mean()
    print(f"collapse-hazard AUC (test Jun10-11); shred coverage {cov:.0%} of snaps\n")
    for lab, feats in [("path9 (baseline)", P9), ("path9 + shred", P9 + SHRED), ("shred only", SHRED)]:
        m = HistGradientBoostingClassifier(**REG).fit(tr[feats].values, tr.collapse.values)
        auc = roc_auc_score(te.collapse.values, m.predict_proba(te[feats].values)[:, 1])
        print(f"  {lab:20s} AUC={auc:.4f}")
    # near-term collapse (next ~30s) where shreds should help most
    sn30 = sn.copy()
    print("\n  (sanity: collapse base rate train", f"{tr.collapse.mean():.3f})")


if __name__ == "__main__":
    main()
