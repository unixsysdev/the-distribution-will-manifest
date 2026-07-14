#!/usr/bin/env python3
"""extract_recovery_k3v03.py — recovery/death-cut training data at the live trigger.

For every joint-trigger-ready mint (live gates, honest population), walks the
300s forward window calling the ACTUAL live producer TokenState.path_features
after each update, so the 9 path features are train==live by construction.
Emits one row per forward snap (capped 240/mint) with the 11 frozen K-entry
features and the recovers-to-breakeven label convention of the validated
recovery head: fm = suffix max of future ret (seeded with horizon-terminal),
label = fm >= 0, trained on drawdown snaps (ret < 0) downstream.

Output: data/recovery_snaps_k3v03.parquet
Run: K_TRIGGER=3 V_TRIGGER=0.3 PYTHONPATH=. python tools/extract_recovery_k3v03.py
"""
import glob
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

assert os.environ.get("K_TRIGGER") == "3" and os.environ.get("V_TRIGGER") == "0.3"
from feature_accum import ENTRY_FEATURE_NAMES_K, TokenState  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRESH = 3_000_000_000
HORIZON = 300.0
MAX_SNAPS = 240
PATH_KEYS = ["ret", "run_max_ret", "dd", "fill_k", "buy_frac_w", "nsell_w",
             "solo_sell_w", "vel_w", "dts"]


def is_classic(vsol, rsol):
    return abs(vsol - 30_000_000_000 - rsol) < 50_000_000


def main():
    states, first_seen = {}, {}
    ready = {}   # mint -> dict(ts, k_feats); state stays in states for updates
    snaps = {}   # mint -> list[dict]
    t0 = time.time()
    n = 0
    files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
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
                if vsol <= 0 or vtok <= 0 or not is_classic(vsol, rsol):
                    continue
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  .. {n/1e6:.0f}M classic trades | ready={len(ready)} | {time.time()-t0:.0f}s",
                          flush=True)
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
                rd = ready.get(mint)
                if rd is None:
                    if st.k_fired and st.v_fired:
                        ready[mint] = {"ts": ts, "k_feats": tuple(st.k_feats or ())}
                        snaps[mint] = []
                    continue
                # forward snap (the update above already advanced the state)
                lst = snaps[mint]
                if ts - rd["ts"] <= HORIZON and len(lst) < MAX_SNAPS:
                    pf = st.path_features(vsol, vtok)
                    pf["snap_ts"] = ts
                    pf["vsol"] = vsol
                    pf["vtok"] = vtok
                    lst.append(pf)

    print(f"streamed {n:,} classic trades in {(time.time()-t0)/60:.1f}min; "
          f"ready={len(ready)}", flush=True)

    rows = []
    max_ts = max((rd["ts"] for rd in ready.values()), default=0)
    for mint, rd in ready.items():
        if rd["ts"] > max_ts - HORIZON - 10:
            continue
        lst = snaps.get(mint) or []
        if not lst:
            continue
        rets = [p["ret"] for p in lst]
        # suffix max of FUTURE ret, seeded with horizon-terminal (= last ret):
        # exactly the validated recovers-to-breakeven convention
        fm = np.empty(len(rets))
        run = rets[-1]
        for i in range(len(rets) - 1, -1, -1):
            fm[i] = run
            if rets[i] > run:
                run = rets[i]
        kf = rd["k_feats"]
        for i, p in enumerate(lst):
            row = {"mint": mint, "ready_ts": rd["ts"], "fwd_i": i, "fm": float(fm[i])}
            for k in PATH_KEYS + ["snap_ts", "vsol", "vtok"]:
                row[k] = float(p[k])
            for name, val in zip(ENTRY_FEATURE_NAMES_K, kf):
                row[name] = float(val)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / "data/recovery_snaps_k3v03.parquet"
    df.to_parquet(out, index=False)
    dd_rows = df[df.ret < 0]
    print(f"wrote {out}: {len(df):,} snaps over {df.mint.nunique():,} mints | "
          f"drawdown snaps {len(dd_rows):,} | recover base rate {(dd_rows.fm >= 0).mean():.3f}")


if __name__ == "__main__":
    main()
