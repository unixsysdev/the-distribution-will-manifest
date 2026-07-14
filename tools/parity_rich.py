#!/usr/bin/env python3
"""parity_rich.py — serve-path parity for the rich feature set.

Proves the LIVE builder (rich_entry_features.build_entry_features) reproduces
the OFFLINE training features (tools/train_june_causal_sweep) given identical
trades. Method: reuse the offline load_trades (so the trade input is the EXACT
sequence that produced candidates.parquet), convert each mint's g-arrays to the
live row schema, run the live builder, and diff every shared non-intent feature
against candidates.parquet.

Non-intent only by design: intent features come from different sources live
(SHM ring, real-time, 49% coverage) vs offline (intent_capture jsonl by ts);
they are handled separately and cannot be byte-identical.

Read-only. No model, no live change.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/the-distribution-will-manifest")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import train_june_causal_sweep as off  # offline builder (import-safe; main is guarded)
from rich_entry_features import build_entry_features  # live builder

K, V = 3, 0.3
FRESH_RSOL_LAM = 3_000_000_000
MIN_STEM = "capture_20260609T043337Z"   # the rich-schema-era default the build used
N_SAMPLE = 200
TOL = 1e-6

ROW_KEYS = ["ts", "slot", "mid", "vsol", "vtok", "rsol", "sol", "is_buy", "user",
            "fee_lam", "cu", "cu_limit", "priority_fee_micro", "jito_tip_lam",
            "route_present", "n_inner_ix", "n_keys", "failed"]


def g_to_rows(g):
    n = len(g["mid"])
    rows = []
    for i in range(n):
        rows.append({key: g[key][i] for key in ROW_KEYS})
    return rows


def main():
    cand = pd.read_parquet(ROOT / "data/rich_crossday_20260610/candidates.parquet")
    cand = cand[(cand.k == K) & (cand.v_sol == V)].set_index("mint")
    print(f"candidates (k={K},v={V}): {len(cand)} mints", flush=True)

    print("loading trades via offline load_trades (same fn that built candidates) ...", flush=True)
    groups, stats = off.load_trades(ROOT / "grpc_capture", MIN_STEM, False, FRESH_RSOL_LAM)
    print(f"loaded {len(groups):,} mints", flush=True)

    shared = [m for m in cand.index if m in groups]
    rng = np.random.default_rng(0)
    sample = [shared[i] for i in rng.choice(len(shared), min(N_SAMPLE, len(shared)), replace=False)]
    print(f"sampling {len(sample)} of {len(shared)} mints present in both\n", flush=True)

    feat_maxdiff = {}
    idx_match = 0
    idx_mismatch = []
    n_ok = 0
    trigger_fail = 0
    for m in sample:
        g = groups[m]
        rows = g_to_rows(g)
        try:
            _, live = build_entry_features(rows, k=K, v_sol=V, expected_features=[])
        except ValueError:
            trigger_fail += 1
            continue
        crow = cand.loc[m]
        # decision-index parity (recomputed live vs offline-recorded)
        from rich_entry_features import trigger_indices as live_trig
        _, _, live_dec = live_trig(rows, K, V)
        if int(live_dec) == int(crow["decision_idx"]):
            idx_match += 1
        else:
            idx_mismatch.append((m, int(live_dec), int(crow["decision_idx"])))
        n_ok += 1
        for k in live:
            if k.startswith("intent_") or k not in crow.index:
                continue
            try:
                cv = float(crow[k]); lv = float(live[k])
            except (TypeError, ValueError):
                continue
            d = abs(cv - lv)
            if (k not in feat_maxdiff) or d > feat_maxdiff[k][0]:
                feat_maxdiff[k] = (d, cv, lv, m)

    print(f"=== PARITY RESULT (non-intent rich features) ===")
    print(f"mints compared: {n_ok}  trigger-not-ready: {trigger_fail}")
    print(f"decision-index match: {idx_match}/{n_ok}")
    if idx_mismatch:
        for m, lv, cv in idx_mismatch[:5]:
            print(f"  MISMATCH {m[:12]} live_idx={lv} offline_idx={cv}")
    feats_cmp = sorted(feat_maxdiff)
    print(f"features compared: {len(feats_cmp)}")
    worst = sorted(feat_maxdiff.items(), key=lambda kv: -kv[1][0])[:12]
    n_clean = sum(1 for _, (d, *_ ) in feat_maxdiff.items() if d <= TOL)
    print(f"features within tol ({TOL:g}): {n_clean}/{len(feats_cmp)}")
    print("\nworst 12 by max abs diff:")
    print(f"  {'feature':28s} {'maxdiff':>12s} {'offline':>14s} {'live':>14s}")
    for k, (d, cv, lv, m) in worst:
        print(f"  {k:28s} {d:12.6g} {cv:14.6g} {lv:14.6g}")
    verdict = "PARITY OK" if (n_clean == len(feats_cmp) and not idx_mismatch and not trigger_fail) \
        else "DIVERGENCE FOUND"
    print(f"\n=> {verdict}")


if __name__ == "__main__":
    main()
