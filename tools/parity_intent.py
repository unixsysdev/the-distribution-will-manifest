#!/usr/bin/env python3
"""parity_intent.py — measure the intent-feature gap between training (jsonl,
hindsight-complete) and live (SHM ring, bounded deque).

Both consume the SAME producer stream (intent_recorder writes jsonl AND ring)
and both filter recv_ns <= decision_ts, so this isolates the STRUCTURAL gaps:
  - the ring's 200-record per-mint deque cap (jsonl has no cap)
  - SOL vs lamport unit handling in the two implementations
  - signer-vs-user identity for uniq counts
  - any formula drift between off.intent_features and shred_window.intent_features
It does NOT capture the live TIMING gap (intents in-flight at the decision
instant) -- that is irreducibly live and is what the shadow scorer measures.

Read-only.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/the-distribution-will-manifest")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "shred_bot"))

import train_june_causal_sweep as off
from shred_window import ShredWindow, PER_MINT_MAXLEN

N_SAMPLE = 400
TOL = 1e-6


def main():
    intent_dir = ROOT / "shred_bot/intent_capture"
    print("loading intents (offline/hindsight view) ...", flush=True)
    groups, stats = off.load_intents(intent_dir)
    print(f"  {len(groups):,} mints, {stats['trade_rows']:,} intent rows", flush=True)

    # ring-faithful per-mint records (ring stores lamports + user identity)
    ring_recs: dict[str, list[dict]] = {}
    for path in sorted(intent_dir.glob("intent-*.jsonl*")):
        import gzip
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("type") not in ("buy", "sell", "buy_quote", "buy_sol_in"):
                    continue
                m = r.get("mint")
                rn = off.fnum(r.get("recv_ns"))
                if not m or rn <= 0:
                    continue
                ring_recs.setdefault(m, []).append({
                    "recv_ns": rn,
                    "is_buy": bool(r.get("is_buy")),
                    "user": r.get("signer") or r.get("user") or "",
                    "sol_limit_lam": off.fnum(r.get("sol_limit_lam")),
                    "jito_tip_lam": off.fnum(r.get("jito_tip_lam")),
                    "priority_fee_micro": off.fnum(r.get("priority_fee_micro")),
                    "probable_spoof": bool(r.get("probable_spoof")),
                })
    for m in ring_recs:
        ring_recs[m].sort(key=lambda x: x["recv_ns"])

    cand = pd.read_parquet(ROOT / "data/rich_crossday_20260610/candidates.parquet")
    cand = cand[(cand.k == 3) & (cand.v_sol == 0.3)]
    # decisions that actually have intent activity (else trivially all-zero match)
    cand = cand[cand.mint.isin(ring_recs)]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cand), min(N_SAMPLE, len(cand)), replace=False)
    sample = cand.iloc[idx]
    print(f"sampling {len(sample)} decisions with intent data\n", flush=True)

    sw = ShredWindow(ring_name="unused")
    feat_diff = {}
    n_nonzero_intent = 0
    for _, crow in sample.iterrows():
        m = crow["mint"]
        dts = float(crow["decision_ts"])
        dns = int(dts * 1e9)
        hind = off.intent_features(groups, m, dts)
        # ring-faithful: last PER_MINT_MAXLEN records with recv_ns <= decision
        recs = [r for r in ring_recs[m] if r["recv_ns"] <= dns]
        dq = deque(recs[-PER_MINT_MAXLEN:], maxlen=PER_MINT_MAXLEN)
        sw._by_mint[m] = dq
        ringf = sw.intent_features(m, now_ns=dns)
        del sw._by_mint[m]
        if hind.get("intent_2p0s_n", 0) > 0:
            n_nonzero_intent += 1
        for k in hind:
            if k not in ringf:
                continue
            try:
                d = abs(float(hind[k]) - float(ringf[k]))
            except (TypeError, ValueError):
                continue
            if (k not in feat_diff) or d > feat_diff[k][0]:
                feat_diff[k] = (d, float(hind[k]), float(ringf[k]), m)

    print("=== INTENT PARITY (training/jsonl vs ring-faithful, structural only) ===")
    print(f"decisions: {len(sample)}  with nonzero 2s intent: {n_nonzero_intent}")
    clean = sum(1 for _, (d, *_ ) in feat_diff.items() if d <= TOL)
    print(f"features compared: {len(feat_diff)}  within tol ({TOL:g}): {clean}/{len(feat_diff)}")
    worst = sorted(feat_diff.items(), key=lambda kv: -kv[1][0])[:14]
    print(f"\n  {'feature':30s} {'maxdiff':>12s} {'jsonl':>12s} {'ring':>12s}")
    for k, (d, hv, rv, m) in worst:
        print(f"  {k:30s} {d:12.6g} {hv:12.6g} {rv:12.6g}")
    verdict = "INTENT STRUCTURAL PARITY OK" if clean == len(feat_diff) else "STRUCTURAL DIVERGENCE"
    print(f"\n=> {verdict}")
    print("(timing gap not measured here; needs the live shadow scorer)")


if __name__ == "__main__":
    main()
