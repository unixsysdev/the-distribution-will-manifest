"""Measure the inter-trade gap distribution for pump.fun tokens from
grpc_capture, to empirically derive a sensible STALE_SEC value.

A "gap" = seconds between consecutive trade events for the same mint.

We split each token's gaps into two classes:
  - LIVE gap: the gap was followed by MORE trade events within a long
              window (DEATH_HORIZON_S) — the token kept trading
  - TERMINAL gap: nothing followed for DEATH_HORIZON_S — the token went
              quiet for good (from our observation window's POV)

The "correct" STALE_SEC sits BETWEEN p99(live_gap) and the typical
terminal_gap. Picking it too low = we stale-close live tokens
(false-positive close); too high = we sit on dead tokens too long.

Run-only-from-archived-data; no live changes.
"""
from __future__ import annotations
import argparse, gzip, json, time
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path("/root/the-distribution-will-manifest")
DEATH_HORIZON_S = 1800   # if no trade in next 30 min, token is "dead" relative to gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="grpc_capture")
    ap.add_argument("--min-trades-per-mint", type=int, default=10,
                    help="exclude mints with fewer than this many trades (noise floor)")
    args = ap.parse_args()

    cap = ROOT / args.capture_dir
    files = sorted(cap.glob("*.jsonl*"))
    print(f"=== stale_sec_audit @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"  scanning {len(files)} capture files in {cap}")

    # Per-mint sorted list of trade timestamps
    per_mint = defaultdict(list)
    n_evt = 0
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    n_evt += 1
                    if n_evt % 2_000_000 == 0:
                        print(f"    .. {n_evt:,} events  mints_so_far={len(per_mint):,}")
                    # Both old (no `event` field, has `is_buy`) and new
                    # (has `event: "TradeEvent"`) capture schemas. Trade
                    # records always have both `mint` and `is_buy`.
                    if '"mint"' not in ln: continue
                    if '"is_buy"' not in ln: continue
                    try: r = json.loads(ln)
                    except Exception: continue
                    # New-schema records may also have event="NoEvent" or
                    # PumpSwap.* — skip those, we only want bonding-curve
                    # TradeEvents
                    ev = r.get("event")
                    if ev not in (None, "TradeEvent"): continue
                    m = r.get("mint")
                    t = r.get("t") or r.get("ev_ts")
                    if not m or not t: continue
                    per_mint[m].append(float(t))
        except Exception as e:
            print(f"  warn: {path}: {e}")
    print(f"  total events scanned: {n_evt:,}")
    print(f"  distinct mints w/ trades: {len(per_mint):,}")

    # For each mint, sort timestamps + compute gaps
    live_gaps = []
    terminal_gaps = []
    n_mints_used = 0
    for m, ts in per_mint.items():
        if len(ts) < args.min_trades_per_mint: continue
        n_mints_used += 1
        ts.sort()
        for i in range(len(ts) - 1):
            gap = ts[i+1] - ts[i]
            # Was the gap followed by more activity in the window?
            # By construction: ts[i+1] exists, so something followed; this is
            # always a LIVE gap. The TERMINAL gap is the one AFTER ts[-1]
            # extending to either the next file boundary or the global
            # observation end.
            live_gaps.append(gap)
        # The trailing terminal gap: time from last trade to global end
        # (= max of all observed times). If this gap is < DEATH_HORIZON_S
        # we don't know if it's a real death or just truncated by file end.
        # We treat last_gap = obs_end - ts[-1] as the terminal gap.

    obs_end = max((max(ts) for ts in per_mint.values() if ts), default=0)
    for m, ts in per_mint.items():
        if len(ts) < args.min_trades_per_mint: continue
        terminal_gap = obs_end - ts[-1]
        # Only count as a true terminal gap if it's at least DEATH_HORIZON_S
        # (otherwise we're seeing a not-yet-dead token at the edge of the
        # observation window — exclude from terminal stats)
        if terminal_gap >= DEATH_HORIZON_S:
            terminal_gaps.append(terminal_gap)

    print(f"  mints used (>= {args.min_trades_per_mint} trades): {n_mints_used:,}")
    print(f"  live gaps (between consecutive trades): {len(live_gaps):,}")
    print(f"  terminal gaps (last trade -> observation end, >= {DEATH_HORIZON_S}s): {len(terminal_gaps):,}")

    if not live_gaps:
        print("no data"); return

    def _qs(arr, label):
        arr = sorted(arr)
        n = len(arr)
        print(f"\n  {label}: n={n:,}")
        for q in (0.50, 0.75, 0.90, 0.95, 0.99, 0.999):
            v = arr[int(n * q)] if int(n*q) < n else arr[-1]
            print(f"    p{q*100:5.1f}: {v:8.1f}s ({v/60:5.1f} min)")
        print(f"    max:    {arr[-1]:.0f}s ({arr[-1]/60:.1f} min)")

    _qs(live_gaps, "INTER-TRADE GAP (LIVE — gap was followed by more trades)")

    print(f"\n=== stale_sec sweep ===")
    print(f"At each candidate STALE_SEC, the FALSE-CLOSE rate = % of LIVE gaps that exceed it")
    print(f"(i.e., we'd have falsely closed a token that was just having a long pause)\n")
    for cand in (30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600):
        n_fp = sum(1 for g in live_gaps if g > cand)
        pct  = n_fp * 100 / len(live_gaps)
        # ALSO: how long would we be hanging on to dead tokens? Estimate from
        # terminal gaps that are < cand (we'd stay open that long unnecessarily)
        # Use mean of (cand - terminal_gap) clipped at 0 across terminal gaps.
        wasted = [min(cand, tg) for tg in terminal_gaps]
        mean_wait = (sum(wasted)/len(wasted)) if wasted else 0
        print(f"    STALE_SEC={cand:5d}s  ({cand/60:5.1f} min)  "
              f"false-close rate={pct:5.2f}%  "
              f"mean dead-hold time≤{mean_wait:.0f}s")

    # Recommendation: pick p99 of LIVE gaps as the smallest STALE_SEC that
    # closes < 1% of live tokens by accident
    p99 = sorted(live_gaps)[int(len(live_gaps) * 0.99)]
    p999= sorted(live_gaps)[int(len(live_gaps) * 0.999)]
    print(f"\n  reference:")
    print(f"    p99 (live gap):  {p99:.0f}s ({p99/60:.1f} min) -> closes 1% of live tokens accidentally")
    print(f"    p999(live gap):  {p999:.0f}s ({p999/60:.1f} min) -> closes 0.1% accidentally")
    print(f"    current STALE_SEC: 300s (5 min)")


if __name__ == "__main__":
    main()
