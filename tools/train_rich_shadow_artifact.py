#!/usr/bin/env python3
"""train_rich_shadow_artifact.py — train + save a rich model for SHADOW scoring.

Saves bot_artifacts_rich_shadow/ with the peak_ge_200 ranking head and a
feature list restricted to features the LIVE builders can reproduce
(build_entry_features non-intent + shred_window.intent_features), so the
shadow scorer never references a feature it cannot compute live. NOT wired to
act; for parallel live scoring + intent-timing measurement only.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path("/root/the-distribution-will-manifest")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shred_bot"))

from rich_entry_features import build_entry_features
from shred_window import ShredWindow

OUT = ROOT / "bot_artifacts_rich_shadow"
DENY = ("peak", "future", "label", "target", "tp", "terminal", "net_exit")
BOOKKEEP = {"mint", "decision_ts", "k", "v_sol", "first_ts", "decision_slot",
            "decision_idx", "k_idx", "v_idx", "n_total_trades_seen"}
REG = dict(max_iter=220, max_depth=3, learning_rate=0.045, l2_regularization=5.0,
           random_state=42)


def live_producible_feature_names():
    """Names the live builders emit (so the shadow scorer can reproduce them)."""
    # minimal synthetic trade history that reaches the k=3/v=0.3 trigger
    rows = []
    t0 = 1_000_000.0
    for i in range(6):
        rows.append({"ts": t0 + i, "slot": i, "mid": 3e-5, "vsol": 3.0e10 + i * 1e8,
                     "vtok": 1.0e15, "rsol": 1.0e8, "sol": 0.2, "is_buy": True,
                     "user": f"u{i}", "fee_lam": 1.0, "cu": 1.0, "cu_limit": 1.0,
                     "priority_fee_micro": 1.0, "jito_tip_lam": 0.0, "route_present": 0.0,
                     "n_inner_ix": 1.0, "n_keys": 1.0, "failed": 0.0})
    _, full = build_entry_features(rows, k=3, v_sol=0.3, expected_features=[])
    sw = ShredWindow(ring_name="unused")
    intent = sw.intent_features("x", now_ns=int(t0 * 1e9))
    return set(full.keys()) | set(intent.keys())


def main():
    cand = pd.read_parquet(ROOT / "data/rich_crossday_20260610/candidates.parquet")
    cand = cand[(cand.k == 3) & (cand.v_sol == 0.3)].reset_index(drop=True)
    live_names = live_producible_feature_names()
    feats = [c for c in cand.columns
             if c not in BOOKKEEP and not any(t in c for t in DENY)
             and pd.api.types.is_numeric_dtype(cand[c]) and c in live_names]
    feats = sorted(feats)
    print(f"candidates: {len(cand)}  live-reproducible rich features: {len(feats)} "
          f"({sum(1 for f in feats if f.startswith('intent_'))} intent)")

    peak_col = [c for c in cand.columns if c.startswith("peak_ret_h")][0]
    y = (cand[peak_col].values >= 2.0).astype(int)
    X = cand[feats].astype(float).replace([np.inf, -np.inf], np.nan).values
    clf = HistGradientBoostingClassifier(**REG).fit(X, y)
    print(f"trained peak_ge_200 head on {len(cand)} rows, base {y.mean():.3f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "entry_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    import sklearn
    spec = {
        "sklearn_version": sklearn.__version__,
        "artifact_kind": "rich_shadow_scorer",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry": {
            "features": feats,
            "n_features": len(feats),
            "n_intent": sum(1 for f in feats if f.startswith("intent_")),
            "target": "peak_ret>=2.0 ranking head",
            "k": 3, "v_sol": 0.3,
            "rich_features": True,
        },
        "purpose": "SHADOW ONLY: parallel live scoring + intent-timing measurement. Never acts.",
        "trained_on": "data/rich_crossday_20260610/candidates.parquet (k3/v03, all days)",
    }
    (OUT / "model_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"saved {OUT}/  (features list in model_spec.json)")


if __name__ == "__main__":
    main()
