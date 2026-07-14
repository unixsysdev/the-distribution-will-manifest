"""ac_backtest — Almgren-Chriss vs K_combined backtest on May parquets.

Loads the historical V+K7-anchored path snaps and runs both:
  - K_combined        — current live policy (paced de-risk + force-spaced runner)
  - A-C (kappa, T_s)  — Almgren-Chriss hyperbolic schedule, N=8 slices over horizon
                        T_s with risk-aversion kappa. kappa=0 = uniform schedule.

Output: per-policy mean/median/p25/win% across the full population.

Crucially this is APPLES-TO-APPLES: identical entry (K=7 reserves), identical
path snaps (replayed from parquet), identical AMM impact compounding (via the
proven ReplayContext math from tools/strategy_ab_replay.py), identical cost
model (250bps + fee_per_tx_sol). The only thing that differs across rows is
WHEN each policy decides to sell and HOW MUCH.

Usage:
  pumpfun_ctl.sh ac-backtest                              # default: kappa sweep
  pumpfun_ctl.sh ac-backtest --kappa 0.05 --horizon 60   # one config
  pumpfun_ctl.sh ac-backtest --subset 1000                # quick sanity sweep
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Reuse the proven AMM math + ReplayContext from the existing A/B framework
from strategy_ab_replay import (
    buy_tokens, sell_sol, ReplayContext, build_snap_array,
    SNAP_EVERY, COST_BPS, FEE_PER_TX_SOL, ENTRY_LAT_SNAPS, TOTAL_SLICES,
    policy_k_combined, policy_time_spaced, policy_hybrid_trail, policy_frontload,
)


def policy_almgren_chriss(vsK, vtK, vsC, vtC, snaps, dts,
                           *, kappa: float = 0.05, horizon_s: float = 60.0,
                           n_slices: int = TOTAL_SLICES) -> float | None:
    """Almgren-Chriss hyperbolic schedule. Holdings target at time t:
        x(t) = q_0 · sinh(κ(T-t)) / sinh(κT)
    At each scheduled slice time t_k = k·T/N, sell to bring holdings down to
    x(t_k). For κ=0 this is the linear (uniform) schedule; for κ→∞ this
    front-loads almost all the position into slice 1.

    NOTE: this is the pure SCHEDULE only — no state-aware death-cut, no ret-gated
    de-risk. That's intentional: we want to see how a pure mathematical scheduler
    compares to K_combined's hand-tuned heuristic. Bolt-on death-cut can be added
    later if the schedule itself looks promising.
    """
    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    T = horizon_s
    N = n_slices
    # Holdings target at each schedule point t_k for k=0..N
    if kappa < 1e-9:
        targets = [ctx.pos * (N - k) / N for k in range(N + 1)]
    else:
        denom = math.sinh(kappa * T)
        targets = [ctx.pos * math.sinh(kappa * max(0.0, T - k * T / N)) / denom
                   for k in range(N + 1)]
        targets[-1] = 0.0  # force liquidate by horizon end
    target_times = [k * T / N for k in range(1, N + 1)]
    n_sold = 0
    t0 = ctx.dts[0] if ctx.dts else 0.0
    for i in range(len(ctx.pool)):
        ts = (ctx.dts[i] if i < len(ctx.dts) else 0.0) - t0
        while n_sold < N and ts >= target_times[n_sold]:
            # slice size = holdings(t_k) - holdings(t_{k+1})
            slice_tok = max(0.0, targets[n_sold] - targets[n_sold + 1])
            slice_tok = min(slice_tok, ctx.pos - ctx.cum_sold)
            if slice_tok > 0:
                ctx.sell_at(i, slice_tok)
            n_sold += 1
        if n_sold >= N: break
    return ctx.finalize()


def _stats(name: str, rs: list[float]) -> dict:
    if not rs: return {"n": 0}
    arr = np.array(rs, dtype=float)
    return {
        "name": name,
        "n":      int(len(arr)),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "win_pct": float(100 * (arr > 0).mean()),
        "best":   float(arr.max()),
        "worst":  float(arr.min()),
        "total":  float(arr.sum()),
    }


def _print_table(stats_rows: list[dict], reference: str) -> None:
    ref_mean = next((s["mean"] for s in stats_rows if s["name"] == reference), 0.0)
    header = f"{'policy':38s} {'n':>5s} {'mean':>9s} {'median':>9s} {'p25':>9s} {'win%':>6s} {'best':>8s} {'worst':>8s}   uplift_vs_ref"
    print(header)
    print("-" * len(header))
    for s in stats_rows:
        if s.get("n", 0) == 0: continue
        uplift = s["mean"] - ref_mean
        marker = ""
        if uplift > 0.005: marker = "  <-- BEATS REF"
        elif uplift < -0.005: marker = "  (worse)"
        print(f"{s['name']:38s} {s['n']:>5d} {s['mean']:>+9.4f} {s['median']:>+9.4f} "
              f"{s['p25']:>+9.4f} {s['win_pct']:>5.1f}% {s['best']:>+8.3f} {s['worst']:>+8.3f}   "
              f"{uplift:+.4f}{marker}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pumpfun_continuation_K7_fresh",
                    help="directory with token_level.parquet + path_snapshots.parquet")
    ap.add_argument("--subset", type=int, default=None,
                    help="limit to first N mints (for quick sanity)")
    ap.add_argument("--kappa", type=float, default=None,
                    help="if set, run only this single kappa value")
    ap.add_argument("--horizon", type=float, default=60.0,
                    help="A-C horizon T in seconds (default 60)")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = ROOT / args.data_dir
    print(f"=== A-C vs K_combined backtest @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"data: {data_dir}")
    tk = pd.read_parquet(data_dir / "token_level.parquet")
    sk = pd.read_parquet(data_dir / "path_snapshots.parquet")
    print(f"loaded: {len(tk)} mints, {len(sk)} path snaps")
    # build per-mint snap arrays from parquet
    sk = sk.sort_values(["mint", "fwd"])
    mints = tk["mint"].tolist()
    if args.subset:
        mints = mints[:args.subset]
        print(f"subset to first {len(mints)} mints")
    tk = tk.set_index("mint")
    sk_groups = {m: g for m, g in sk.groupby("mint")}
    print(f"mints with path snaps: {sum(1 for m in mints if m in sk_groups)}")

    # Policies to compare. K_combined is the reference.
    policies = [
        ("K_combined (LIVE policy)",       lambda *a: policy_k_combined(*a)),
        ("H_time_spaced 15s",              lambda *a: policy_time_spaced(*a, gap_s=15.0)),
        ("B_frontload",                    lambda *a: policy_frontload(*a)),
        ("C_hybrid 4+4 t30",               lambda *a: policy_hybrid_trail(*a, runner_retrace=0.30)),
        ("F_hybrid 4+4 t50",               lambda *a: policy_hybrid_trail(*a, runner_retrace=0.50)),
    ]
    kappa_grid = [args.kappa] if args.kappa is not None else [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
    for kappa in kappa_grid:
        label = f"A-C kappa={kappa:.2f} T={args.horizon:.0f}s"
        policies.append((label, lambda *a, kk=kappa: policy_almgren_chriss(*a, kappa=kk, horizon_s=args.horizon)))

    results = {name: [] for name, _ in policies}
    n_analyzed = 0; n_skipped = 0
    for m in mints:
        if m not in sk_groups: n_skipped += 1; continue
        if m not in tk.index: n_skipped += 1; continue
        g = sk_groups[m]
        if len(g) < 2: n_skipped += 1; continue
        vsK = float(tk.at[m, "vsK"]) if "vsK" in tk.columns else None
        vtK = float(tk.at[m, "vtK"]) if "vtK" in tk.columns else None
        if vsK is None or vtK is None or vsK <= 0 or vtK <= 0:
            # Fall back to first snap reserves as a degenerate proxy
            n_skipped += 1; continue
        snaps = list(zip(g["vs"].tolist(), g["vt"].tolist()))
        dts = g["dts"].tolist()
        vsC, vtC = snaps[-1]
        n_analyzed += 1
        for name, fn in policies:
            try:
                pl = fn(vsK, vtK, vsC, vtC, snaps, dts)
                if pl is not None: results[name].append(pl)
            except Exception as e:
                pass
        if n_analyzed % 1000 == 0:
            print(f"  ... {n_analyzed} mints analyzed")

    print(f"\nanalyzed: {n_analyzed}  skipped (no snaps / missing vsK): {n_skipped}")
    print()
    stats_rows = [_stats(name, results[name]) for name, _ in policies]
    _print_table(stats_rows, reference="K_combined (LIVE policy)")
    print()
    print("Notes:")
    print(" - 'uplift_vs_ref' is absolute mean-per-bet uplift over K_combined.")
    print(" - All policies use the SAME entry (vsK/vtK) and the SAME path snaps.")
    print(" - All costs identical (250bps + per-tx fee).")
    print(" - Almgren-Chriss policies are pure SCHEDULES — no death-cut, no ret-gating.")
    print(" - kappa=0 ≈ H_time_spaced; higher kappa front-loads exit aggressively.")


if __name__ == "__main__":
    main()
