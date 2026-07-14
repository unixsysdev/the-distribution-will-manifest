"""extract_live_matched.py — build a training set that matches the LIVE bot exactly.

Runs the ACTUAL live accumulator (feature_accum.TokenState) over the gRPC capture,
applying the SAME gates the harness applies in on_trade():
  - classic-curve only:  abs(vsol - 30e9 - rsol) < 50e6   (TradeEvent.is_classic_curve)
  - fresh launch:        first-seen real_sol_reserves < FRESH_RSOL_LAM (3 SOL)
At the joint 'ready' trigger (K=5 AND V=0.5) we record the 22 K+V features the model
scores live (combined_entry_features). peak_ret label = run_max_ret (K-anchored
forward max, the same target the K7 extractor builds, minus the mid0-seeding impurity).

train==live by construction: same class, same gates, same trigger, same feature math.

Usage:
  python tools/extract_live_matched.py --capture-dir grpc_capture --out data/live_matched_k5.parquet [--cap N]
"""
from __future__ import annotations
import argparse, glob, gzip, json, time
from pathlib import Path
import pandas as pd
from feature_accum import TokenState, ENTRY_FEATURE_NAMES   # 22 K+V names, K then V

FRESH_RSOL_LAM = 3_000_000_000
MIN_FWD = 5


def is_classic(vsol: float, rsol: float) -> bool:
    return abs(vsol - 30_000_000_000 - rsol) < 50_000_000


def iter_trade_rows(capture_dir: str):
    # Two capture schema eras: newer rows tag trades with "event":"TradeEvent"
    # and also log non-trade rows; older rows have no "event" key but carry the
    # same trade fields. Both eras put "vsol" ONLY on trade rows, so prefilter on
    # that substring (fast) and let the top-level mint/vsol checks do the rest.
    files = sorted(glob.glob(str(Path(capture_dir) / "*.jsonl*")))
    print(f"  {len(files)} capture files", flush=True)
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    if '"vsol"' not in ln:
                        continue
                    try:
                        yield json.loads(ln)
                    except Exception:
                        continue
        except Exception as e:
            print(f"  warn {path}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="grpc_capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="stop after N valid trades (0 = all)")
    ap.add_argument("--min-fwd", type=int, default=MIN_FWD, help="min forward trades to label (0 = keep insta-dead ready mints, labeled with their real ~0 peak)")
    args = ap.parse_args()

    states: dict[str, TokenState] = {}
    first_seen: dict[str, float] = {}
    first_slot: dict[str, int] = {}
    n = nclassic = 0
    t0 = time.time()
    for rec in iter_trade_rows(args.capture_dir):
        # works for both schema eras: trade rows carry top-level mint + vsol;
        # newer non-trade rows (NoEvent/failed) have no top-level mint/vsol.
        mint = rec.get("mint")
        if not mint or "vsol" not in rec:
            continue
        try:
            vsol = float(rec["vsol"]); vtok = float(rec["vtok"]); rsol = float(rec["rsol"])
            sol = float(rec["sol"]) / 1e9; ts = float(rec["ev_ts"]); slot = int(rec["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        if vsol <= 0 or vtok <= 0:
            continue
        n += 1
        if args.cap and n > args.cap:
            break
        if n % 500_000 == 0:
            print(f"  .. {n:,} trades | {nclassic:,} classic | {len(states)} mints | {time.time()-t0:.0f}s", flush=True)
        # gate 1: classic curve (live gates per-trade before anything else)
        if not is_classic(vsol, rsol):
            continue
        nclassic += 1
        # gate 2: fresh launch (first-seen rsol < 3 SOL); first_seen recorded on first classic trade
        if mint not in first_seen:
            first_seen[mint] = rsol
            if rsol >= FRESH_RSOL_LAM:
                first_seen[mint] = 1e18   # permanently dropped
                continue
        if first_seen[mint] >= FRESH_RSOL_LAM:
            continue
        is_buy = bool(rec.get("is_buy")); user = rec.get("user", "")
        st = states.get(mint)
        if st is None:
            states[mint] = TokenState(vsol, vtok, sol, is_buy, user, ts)
            first_slot[mint] = slot
            continue
        st.update(vsol, vtok, sol, is_buy, user, ts)

    print(f"  streamed {n:,} valid trades ({nclassic:,} classic) in {(time.time()-t0)/60:.1f}min; {len(states)} mints", flush=True)

    rows = []
    n_thin = 0
    for mint, st in states.items():
        if not (st.k_fired and st.v_fired):   # reached the JOINT trigger == live would fire
            continue
        if st.fwd < args.min_fwd:              # enough forward trades to label
            n_thin += 1
            continue
        feats = st.combined_entry_features()   # 22 K+V exactly as live scores
        row = {"mint": mint, "first_slot": first_slot.get(mint, 0),
               "peak_ret": st.run_max_ret, "n_fwd": st.fwd}
        for i, name in enumerate(ENTRY_FEATURE_NAMES):
            row[name] = feats[i]
        rows.append(row)
    if n_thin:
        print(f"  dropped {n_thin} ready mints with fwd < {args.min_fwd}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n  wrote {out}: {len(df):,} ready mints | peak>=0.5 rate {(df.peak_ret>=0.5).mean():.3f}")
    if len(df):
        print(f"  uniq (K) dist: {df['uniq'].value_counts().sort_index().to_dict()}")
        print(f"  tot_sol p50={df.tot_sol.quantile(.5):.2f} p90={df.tot_sol.quantile(.9):.2f}  "
              f"net_sol p50={df.net_sol.quantile(.5):.2f}")


if __name__ == "__main__":
    main()
