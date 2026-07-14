"""extract_sophistication — per-mint K-window sophistication aggregates from
wide-capture data.

Reads grpc_capture/*.jsonl(.gz) files that have the wide-feature fields
(fee_lam, cu, n_inner_ix, n_keys, jito_tip_idx, jito_tip_lam, route — added
2026-06-07 in commit 76b3cae). For each mint, tracks the FIRST K BUYs and
computes the same aggregates the live harness's _sophistication_summary()
produces. Output: one parquet row per mint, ready to join onto the K7
token_level.parquet for training.

Output schema (per mint):
    mint                              str
    soph_n_buys                       int  (always K=7 for ready mints)
    soph_fee_p50_lam, _p90_lam        int
    soph_cu_p50, _mean                int
    soph_jito_tip_rate                float (0..1)
    soph_jito_tip_p50_lam             float (NaN if no tippers)
    soph_routed_rate                  float (0..1)
    soph_n_inner_ix_mean              float
    soph_n_keys_mean                  float
    soph_first_buy_ts                 float (for sanity)

NaN-friendly: HistGradientBoostingClassifier handles missing values via
split-on-NaN, so mints without sophistication data (e.g. May parquets) can
sit alongside June capture in the same training table.

Usage:
    python tools/extract_sophistication.py \\
        --capture-dir grpc_capture \\
        --out data/sophistication_capture_jun8.parquet \\
        --k 7 --fresh-rsol-lam 3000000000
"""
from __future__ import annotations
import argparse, glob, gzip, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="grpc_capture")
    ap.add_argument("--out", required=True,
                    help="output parquet path (e.g. data/sophistication_capture_jun8.parquet)")
    ap.add_argument("--k", type=int, default=7, help="K-window size (default 7)")
    ap.add_argument("--fresh-rsol-lam", type=int, default=3_000_000_000,
                    help="skip mints whose first-seen rsol >= this (= joined mid-curve)")
    return ap.parse_args()


def main():
    args = parse_args()
    K = args.k
    capture = Path(args.capture_dir)
    print(f"=== extract_sophistication @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"  K={K}  fresh_rsol_lam={args.fresh_rsol_lam:,}")

    # Per-mint accumulators. For each mint, collect the first K BUYS with their
    # sophistication fields. Stop collecting once we have K. Also track
    # first_seen_rsol to apply the fresh filter consistent with K7/V05 extractors.
    state: dict[str, dict] = {}   # mint -> {buys: list of dicts, n_buys: int, first_rsol, done}
    n_events = 0; n_skipped = 0; n_wide = 0
    files = sorted(glob.glob(str(capture / "*.jsonl*")))
    print(f"  scanning {len(files)} capture files ...")
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    n_events += 1
                    if n_events % 500_000 == 0:
                        print(f"    .. {n_events:,} events  ({len(state)} mints tracked, "
                              f"{n_wide} wide events seen)", flush=True)
                    try: r = json.loads(ln)
                    except Exception: continue
                    # Only wide-format events have fee_lam. Skip pre-widening events.
                    if "fee_lam" not in r:
                        continue
                    n_wide += 1
                    mint = r.get("mint")
                    if not mint: continue
                    is_buy = bool(r.get("is_buy"))
                    st = state.get(mint)
                    if st is None:
                        rs = float(r.get("rsol", 0))
                        if args.fresh_rsol_lam > 0 and rs >= args.fresh_rsol_lam:
                            n_skipped += 1
                            state[mint] = {"done": True, "skipped": True}
                            continue
                        st = state[mint] = {"buys": [], "n_buys": 0, "first_rsol": rs,
                                             "done": False, "skipped": False,
                                             "first_buy_ts": None}
                    if st.get("done"): continue
                    if not is_buy: continue   # K-window counts BUYS only
                    if st["n_buys"] >= K:
                        st["done"] = True; continue
                    st["n_buys"] += 1
                    if st["first_buy_ts"] is None:
                        st["first_buy_ts"] = r.get("ev_ts")
                    st["buys"].append({
                        "fee_lam":      r.get("fee_lam"),
                        "cu":           r.get("cu"),
                        "n_inner_ix":   r.get("n_inner_ix"),
                        "n_keys":       r.get("n_keys"),
                        "jito_tip_idx": r.get("jito_tip_idx"),
                        "jito_tip_lam": r.get("jito_tip_lam"),
                        "route":        r.get("route"),
                    })
                    if st["n_buys"] >= K:
                        st["done"] = True
        except Exception as e:
            print(f"  warn: {path}: {e}")

    print(f"  total events: {n_events:,}  wide events: {n_wide:,}  skipped (fresh filter): {n_skipped}")
    eligible = {m: s for m, s in state.items()
                if not s.get("skipped") and s.get("n_buys", 0) >= K}
    print(f"  mints with K={K} buys observed (wide): {len(eligible)}")

    # Aggregate per-mint sophistication
    rows = []
    for mint, s in eligible.items():
        buys = s["buys"]
        fees = [b["fee_lam"] for b in buys if b["fee_lam"] is not None]
        cus  = [b["cu"]      for b in buys if b["cu"]      is not None]
        nii  = [b["n_inner_ix"] for b in buys if b["n_inner_ix"] is not None]
        nks  = [b["n_keys"]     for b in buys if b["n_keys"]     is not None]
        tips = [b["jito_tip_lam"] for b in buys if b["jito_tip_lam"] is not None]
        n_tipped = sum(1 for b in buys if b.get("jito_tip_idx") is not None)
        n_routed = sum(1 for b in buys if b.get("route"))
        rec = {"mint": mint, "soph_n_buys": len(buys),
               "soph_first_buy_ts": s.get("first_buy_ts")}
        if fees:
            q = sorted(fees)
            rec["soph_fee_p50_lam"] = int(q[len(q)//2])
            rec["soph_fee_p90_lam"] = int(q[int(len(q)*0.9)] if len(q) > 1 else q[-1])
        if cus:
            q = sorted(cus)
            rec["soph_cu_p50"] = int(q[len(q)//2])
            rec["soph_cu_mean"] = int(sum(cus) / len(cus))
        rec["soph_jito_tip_rate"] = float(n_tipped) / len(buys)
        if tips:
            q = sorted(tips)
            rec["soph_jito_tip_p50_lam"] = int(q[len(q)//2])
        rec["soph_routed_rate"] = float(n_routed) / len(buys)
        if nii:
            rec["soph_n_inner_ix_mean"] = float(sum(nii) / len(nii))
        if nks:
            rec["soph_n_keys_mean"] = float(sum(nks) / len(nks))
        rows.append(rec)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"  wrote {out_path} ({len(df)} mints, {df.memory_usage(deep=True).sum()//1024} KB)")

    # Quick stats
    print()
    print("  =========================================")
    print("  Per-mint sophistication signature (K-window aggregates):")
    for col in ["soph_fee_p50_lam", "soph_fee_p90_lam", "soph_cu_mean",
                "soph_jito_tip_rate", "soph_routed_rate", "soph_n_inner_ix_mean",
                "soph_n_keys_mean"]:
        if col not in df.columns: continue
        v = df[col].dropna()
        if not len(v): continue
        print(f"    {col:30s}  n={len(v)}  p50={v.quantile(0.5):.2f}  "
              f"p90={v.quantile(0.9):.2f}  max={v.max():.2f}")


if __name__ == "__main__":
    main()
