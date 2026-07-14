"""backfill_live_policy_net — replay each historical closed position under the
LIVE policy that was active at its close time, write live_policy_net into a
sidecar JSONL.

WHY: shadow_harness now writes live_policy_net inline on every new close
(commit 2eafcef). But the 200+ historical closes from before that commit only
have PaperBook's GREEN reference net_return — which underreports actual P&L
by ~3x because GREEN is "sell on first profit + death cut" while the live
policy was h_time_spaced / c_hybrid_t30 / k_combined etc.

This tool reconstructs each historical position's snap timeline from
positions.jsonl, looks up which policy was active at close time from
policy_decisions.jsonl, replays under that policy via
strategy_ab_replay.policy_via_registry, and writes the result to a sidecar
JSONL the dashboard merges in.

Output: bot_data/historical_live_net.jsonl with one record per backfilled close:
    {"mint": <mint>, "close_t": <t>, "live_policy_net": <float>,
     "live_policy_name": <str>}

The dashboard checks live_policy_net inline FIRST, then falls back to this
sidecar, then to net_return (GREEN ref) as last resort.

Usage:
    python tools/backfill_live_policy_net.py                  # full backfill
    python tools/backfill_live_policy_net.py --dry            # show plan, don't write
    python tools/backfill_live_policy_net.py --policy h_time_spaced  # override active-at-close
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from strategy_ab_replay import policy_via_registry


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dry", action="store_true",
                    help="don't write output, just print the plan")
    ap.add_argument("--policy", default=None,
                    help="override: replay ALL historical closes under this single "
                         "policy name instead of the per-close active-policy lookup")
    ap.add_argument("--out", default="bot_data/historical_live_net.jsonl",
                    help="output sidecar JSONL path")
    return ap.parse_args()


def build_policy_timeline(root: Path) -> list[tuple[float, str]]:
    """Read policy_decisions.jsonl swap events, return sorted [(t, policy)] timeline.
    Default before any swap = k_combined (bot's original default)."""
    timeline = [(0.0, "k_combined")]
    pd = root / "logs" / "policy_decisions.jsonl"
    if not pd.exists(): return timeline
    with open(pd) as f:
        for ln in f:
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("kind") == "swap" and r.get("to"):
                timeline.append((r.get("t", 0), r["to"]))
    timeline.sort(key=lambda x: x[0])
    return timeline


def policy_at(timeline: list[tuple[float, str]], t: float) -> str:
    """Look up which policy was active at time t. Bisect since timeline is sorted."""
    idx = bisect_right([x[0] for x in timeline], t) - 1
    if idx < 0: idx = 0
    return timeline[idx][1]


def build_cfg():
    """cfg-like object the registry policies can read at replay-time."""
    try:
        import yaml
        y = yaml.safe_load((ROOT / "config.yaml").read_text()) or {}
        ex = y.get("exit", {})
    except Exception:
        ex = {}
    class _Cfg:
        class exit:
            total_slices = int(ex.get("total_slices", 8))
            derisk_slices = int(ex.get("derisk_slices", 4))
            derisk_min_gap_s = float(ex.get("derisk_min_gap_s", 5.0))
            runner_min_gap_s = float(ex.get("runner_min_gap_s", 15.0))
            runner_retrace_frac = float(ex.get("runner_retrace_frac", 0.30))
            runner_min_arm_ret = float(ex.get("runner_min_arm_ret", 0.20))
            death_threshold = float(ex.get("death_threshold", 0.10))
            rl_artifact_dir = ex.get("rl_artifact_dir", "bot_artifacts_K7V_rl_layered")
            rl_q5_threshold = float(ex.get("rl_q5_threshold", 0.20))
    return _Cfg


def main():
    args = parse_args()
    root = Path(args.root)
    timeline = build_policy_timeline(root)
    print(f"=== backfill_live_policy_net @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"policy timeline ({len(timeline)} entries):")
    for t, p in timeline:
        iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) if t > 0 else "(default)"
        print(f"  {iso}  -> {p}")
    if args.policy:
        print(f"OVERRIDE: replay ALL closes under '{args.policy}' regardless of timeline")

    # Pass 1: gather per-mint open + ordered snaps + close from positions.jsonl
    opens: dict[str, dict] = {}
    snaps_by_mint: dict[str, list] = defaultdict(list)
    closes: dict[str, dict] = {}
    n_rows = 0
    with open(root / "bot_data" / "positions.jsonl") as f:
        for ln in f:
            try: r = json.loads(ln)
            except Exception: continue
            n_rows += 1
            m = r.get("mint")
            if not m: continue
            k = r.get("kind")
            if k == "open":
                opens[m] = r
            elif k == "snap":
                snaps_by_mint[m].append(r)
            elif k == "close":
                # keep LATEST close per mint (in case of restart-write dupes)
                if r.get("t", 0) > closes.get(m, {}).get("t", 0):
                    closes[m] = r
    print(f"loaded: {n_rows} rows  opens={len(opens)}  closed={len(closes)}")

    # Filter to closes that DON'T already have live_policy_net inline
    needs_backfill = [m for m, c in closes.items()
                       if not isinstance(c.get("live_policy_net"), (int, float))]
    print(f"needs backfill: {len(needs_backfill)} / {len(closes)} closes")

    cfg = build_cfg()
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0; n_failed = 0
    written = []
    for m in needs_backfill:
        op = opens.get(m); cl = closes[m]
        if op is None: n_failed += 1; continue
        try:
            vsK = float(op["vsK"]); vtK = float(op["vtK"])
        except Exception: n_failed += 1; continue
        if vsK <= 0 or vtK <= 0: n_failed += 1; continue
        snaps_recs = sorted(snaps_by_mint.get(m, []), key=lambda r: r.get("fwd", 0))
        if not snaps_recs:
            # No snaps — degenerate; use entry == terminal (no movement). Skip.
            n_failed += 1; continue
        snaps = [(float(s["vs"]), float(s["vt"])) for s in snaps_recs]
        # Reconstruct dts from snap timestamps if present, else fwd-based proxy
        if all("t" in s for s in snaps_recs):
            t0 = float(snaps_recs[0]["t"])
            dts = [float(s["t"]) - t0 for s in snaps_recs]
        else:
            # No t-per-snap available — approximate dts as fwd_index × snap_period.
            # Bot's SNAP_EVERY=3 fwd events per snap; typical bonding curve trade
            # rate is 0.5-2 trades/sec, so ~3-6s per snap. Use 4s/snap.
            dts = [4.0 * i for i in range(len(snaps))]
        vsC = float(cl.get("vsC", snaps[-1][0])) if "vsC" in cl else snaps[-1][0]
        vtC = float(cl.get("vtC", snaps[-1][1])) if "vtC" in cl else snaps[-1][1]
        # Which policy was active at close time?
        policy_name = args.policy or policy_at(timeline, cl.get("t", 0))
        try:
            net = policy_via_registry(vsK, vtK, vsC, vtC, snaps, dts,
                                       policy_name=policy_name, cfg=cfg, mint=m)
        except Exception as e:
            net = None
        if net is None:
            n_failed += 1; continue
        rec = {"mint": m, "close_t": cl.get("t", 0),
               "live_policy_net": float(net),
               "live_policy_name": policy_name,
               "ref_net_return": cl.get("net_return")}
        written.append(rec)
        n_written += 1
    print(f"backfilled: {n_written}  failed: {n_failed}")

    if args.dry:
        print(f"DRY mode — sample 10:")
        for r in written[:10]:
            print(f"  {r['mint'][:10]}  policy={r['live_policy_name']}  "
                  f"live={r['live_policy_net']:+.4f}  ref={r['ref_net_return']:+.4f}")
        return 0

    # Sort written records by close_t for stable output
    written.sort(key=lambda r: r["close_t"])
    with open(out_path, "w") as f:
        for r in written:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    # Aggregate stats
    if written:
        bs = 0.1   # bet_sol; matches config
        total_ref = sum(r["ref_net_return"] for r in written
                         if isinstance(r.get("ref_net_return"), (int, float)))
        total_live = sum(r["live_policy_net"] for r in written)
        print()
        print(f'Aggregate over {len(written)} backfilled closes:')
        print(f'  PaperBook GREEN reference net_return sum: {total_ref:+.4f} fractional  '
              f'(= {total_ref*bs:+.4f} SOL @ bet={bs})')
        print(f'  Live-policy replay  net_return sum:        {total_live:+.4f} fractional  '
              f'(= {total_live*bs:+.4f} SOL @ bet={bs})')
        print(f'  delta (live - ref):                        '
              f'{total_live-total_ref:+.4f} fractional ({(total_live-total_ref)*bs:+.4f} SOL)')


if __name__ == "__main__":
    sys.exit(main() or 0)
