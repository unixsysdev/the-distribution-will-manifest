"""Unified policy backtest on the bot-selected universe.

Loads token_level + path_snapshots from the V+K7 corpus (May fresh + OOS fresh
+ June capture), computes the production entry score for every mint, filters
to mints where score >= tau, then replays every registered exit policy on
those mints' path_snapshots using the existing AMM-impact-correct ReplayContext
from strategy_ab_replay.py.

Apples-to-apples:
  - identical entry (K=7 reserves from token_level.vsK/vtK; entry_lat=1 snap)
  - identical path snaps (from path_snapshots.parquet's vs/vt, same cadence
    as the live bot's snap_every)
  - identical cost model (250bps + fee_per_tx_sol from strategy_ab_replay)
  - identical AMM impact accounting (cum_received / cum_sold tracked across
    sells)
  - the ONLY thing that differs across rows is when each policy decides to
    sell, and how much.

The selection filter (score >= tau) restricts the evaluation universe to mints
the live bot would actually fire on at threshold tau. This is the key research
hygiene piece: comparing exit policies on the full corpus mixes signal-carrying
mints with noise mints, which biases any policy that hedges differently across
the two populations.

Output:
  - stdout table per (tau, policy)
  - logs/policy_replay_full.json with full numerical breakdown

Usage:
  python tools/policy_replay_full.py \\
      --inputs _fresh _oos_fresh _capture_jun8 \\
      --thresholds 0.30 0.40 0.50 0.5108 \\
      --policies k_combined c_hybrid_t30 level_tp_50 level_tp_100 \\
                 level_tp_200 b_frontload h_time_spaced f_hybrid_t50 \\
      --max-mints-per-tau 0
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import strategy_ab_replay
from strategy_ab_replay import policy_via_registry

# Trigger registration of all registered policies (including the new level_tp)
import exit_policies  # noqa: F401
from exit_policies import list_policies

ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
ENTRY_V = [f"{c}_v" for c in ENTRY_K]


def load_corpus(inputs):
    """Load + concat token_level (K+V joined) and path_snapshots across suffixes.

    For each base suffix `_X`, tries BOTH `pumpfun_continuation_K7_X` (in-sample)
    AND `pumpfun_continuation_oos_K7_X` (chrono OOS holdout) — matching how
    build_bot_artifacts_K7V joins them for training.
    """
    tk_dfs, tv_dfs, sk_dfs = [], [], []
    for s in inputs:
        for prefix in ["", "oos_"]:
            tk_p = ROOT / f"data/pumpfun_continuation_{prefix}K7{s}/token_level.parquet"
            tv_p = ROOT / f"data/pumpfun_continuation_{prefix}V05{s}/token_level.parquet"
            sk_p = ROOT / f"data/pumpfun_continuation_{prefix}K7{s}/path_snapshots.parquet"
            if not (tk_p.exists() and tv_p.exists() and sk_p.exists()):
                continue
            tk = pd.read_parquet(tk_p); tv = pd.read_parquet(tv_p); sk = pd.read_parquet(sk_p)
            tag = f"{prefix}{s.lstrip('_')}"
            tk["_src"] = tag; tv["_src"] = tag; sk["_src"] = tag
            tk_dfs.append(tk); tv_dfs.append(tv); sk_dfs.append(sk)
            print(f"  loaded {tag!r}: tk={len(tk)}  tv={len(tv)}  sk={len(sk)}")
    if not tk_dfs:
        raise SystemExit("no input data found")
    tk = pd.concat(tk_dfs, ignore_index=True)
    tv = pd.concat(tv_dfs, ignore_index=True)
    sk = pd.concat(sk_dfs, ignore_index=True)
    # Dedupe by mint keeping the LAST occurrence (capture trumps fresh if both)
    tk = tk.drop_duplicates(subset=["mint"], keep="last").reset_index(drop=True)
    tv = tv.drop_duplicates(subset=["mint"], keep="last").reset_index(drop=True)
    return tk, tv, sk


def compute_scores(tk, tv, art_dir):
    """Score every mint with the production entry model. Returns joined table."""
    K_TO_V = {k: v for k, v in zip(ENTRY_K, ENTRY_V)}
    tv_f = tv[["mint"] + ENTRY_K].rename(columns=K_TO_V)
    joined = tk.merge(tv_f, on="mint", how="inner")
    clf = pickle.load(open(art_dir / "entry_model.pkl", "rb"))
    X = joined[ENTRY_K + ENTRY_V].values
    joined["score"] = clf.predict_proba(X)[:, 1]
    return joined


class _ExitCfg:
    """Minimal stand-in for cfg.exit so registered policies can read consts."""
    total_slices = 8
    derisk_slices = 4
    derisk_min_gap_s = 5.0
    runner_min_gap_s = 15.0
    runner_retrace_frac = 0.30
    runner_min_arm_ret = 0.20
    death_threshold = 0.10
    rl_artifact_dir = str(ROOT / "bot_artifacts_K7V_rl_layered")
    rl_q5_threshold = 0.20


class _MockCfg:
    exit = _ExitCfg()


def replay_one(row, snaps_df, cfg, policies):
    """Replay one mint through every policy. Returns dict policy -> pnl."""
    vsK = float(row["vsK"]); vtK = float(row["vtK"])
    vsC = float(row["vsC"]); vtC = float(row["vtC"])
    if vsK <= 0 or vtK <= 0 or vsC <= 0 or vtC <= 0:
        return None
    snaps_df = snaps_df.sort_values("fwd")
    snaps = list(zip(snaps_df["vs"].values.tolist(),
                     snaps_df["vt"].values.tolist()))
    dts = snaps_df["dts"].values.tolist()
    if not snaps:
        return None
    # Per-snap extras for policies that condition on flow features (lsm_*).
    # Aligned with snaps[] 1-to-1.
    extra_cols = ["fill_k", "buy_frac_w", "nsell_w", "solo_sell_w", "vel_w"]
    snap_extras = snaps_df[extra_cols].to_dict("records")
    entry_feats = {f: float(row[f]) for f in ENTRY_K if f in row}
    entry_feats.update({f"{f}_v": float(row[f"{f}_v"])
                        for f in ENTRY_K if f"{f}_v" in row})
    out = {}
    for p in policies:
        try:
            pl = policy_via_registry(vsK, vtK, vsC, vtC, snaps, dts,
                                      policy_name=p, cfg=cfg,
                                      entry_features=entry_feats,
                                      entry_score=float(row["score"]),
                                      mint=row["mint"],
                                      snap_extras=snap_extras)
            if pl is not None:
                out[p] = pl
        except Exception:
            pass
    return out


def stats(pl_list):
    a = np.array(pl_list, dtype=float)
    if len(a) == 0: return None
    return {
        "n":       int(len(a)),
        "mean":    float(np.mean(a)),
        "median":  float(np.median(a)),
        "p10":     float(np.percentile(a, 10)),
        "p25":     float(np.percentile(a, 25)),
        "p75":     float(np.percentile(a, 75)),
        "p90":     float(np.percentile(a, 90)),
        "p99":     float(np.percentile(a, 99)),
        "win_pct": float(100 * (a > 0).sum() / len(a)),
        "total":   float(np.sum(a)),
        "best":    float(np.max(a)),
        "worst":   float(np.min(a)),
    }


def bootstrap_mean_p05(pl_list, n_boot=200, seed=42):
    """Block-bootstrap 5th pct of the mean estimator."""
    a = np.array(pl_list, dtype=float)
    if len(a) < 50: return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        means.append(a[idx].mean())
    return float(np.percentile(means, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["_fresh", "_oos_fresh", "_capture_jun8"])
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.30, 0.40, 0.50, 0.5108])
    ap.add_argument("--artifact-dir", default="bot_artifacts_K7V")
    ap.add_argument("--policies", nargs="+", default=None,
                    help="policy names; default = all registered")
    ap.add_argument("--max-mints-per-tau", type=int, default=0,
                    help="quick-iteration cap; 0 = no cap")
    ap.add_argument("--entry-lat", type=int, default=None,
                    help="latency stress: snaps to wait after K7 trigger before "
                         "entering. Default = strategy_ab_replay.ENTRY_LAT_SNAPS "
                         "(=1). Use 2 to test '1 snap of slip' between policy "
                         "decision and execution.")
    ap.add_argument("--bet-sol", type=float, default=None,
                    help="capacity stress: position size in SOL. Default = 1.0 "
                         "(replay default; overstates AMM impact for our 0.1 SOL "
                         "live bet). Pass 0.1 to match live bet, or 2.0 / 5.0 "
                         "to find the capacity ceiling.")
    ap.add_argument("--out", default="logs/policy_replay_full.json")
    args = ap.parse_args()

    if args.entry_lat is not None:
        strategy_ab_replay.ENTRY_LAT_SNAPS = args.entry_lat
        print(f"  ENTRY_LAT_SNAPS overridden -> {args.entry_lat}")
    if args.bet_sol is not None:
        # qlam is the position size in lamports inside ReplayContext.
        original_qlam = 1e9
        new_qlam = args.bet_sol * 1e9
        print(f"  bet_sol overridden -> {args.bet_sol} (qlam={new_qlam:.0f})")
        # ReplayContext reads self.qlam = 1e9 at __init__; we need to patch the
        # class so all new instances pick up the override.
        _OrigInit = strategy_ab_replay.ReplayContext.__init__
        def _patched_init(self, *a, **kw):
            _OrigInit(self, *a, **kw)
            self.qlam = new_qlam
            if self.valid:
                from strategy_ab_replay import buy_tokens
                self.pos = buy_tokens(self.vse, self.vte, self.qlam)
        strategy_ab_replay.ReplayContext.__init__ = _patched_init

    print(f"=== policy_replay_full @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"  inputs     = {args.inputs}")
    print(f"  thresholds = {args.thresholds}")
    print(f"  art_dir    = {args.artifact_dir}")

    print(f"\n--- (1) load corpus ---")
    tk, tv, sk = load_corpus(args.inputs)
    print(f"  tk total mints={len(tk)}  sk total snaps={len(sk)}")

    print(f"\n--- (2) compute entry scores ---")
    joined = compute_scores(tk, tv, ROOT / args.artifact_dir)
    print(f"  joined K+V: {len(joined)} mints")
    qs = np.percentile(joined.score, [10, 50, 90, 95, 99])
    print(f"  score: p10={qs[0]:.4f} p50={qs[1]:.4f} p90={qs[2]:.4f} "
          f"p95={qs[3]:.4f} p99={qs[4]:.4f}")
    for tau in args.thresholds:
        n = (joined.score >= tau).sum()
        pct = 100 * n / len(joined)
        print(f"  τ={tau:.4f} -> {n} mints qualify ({pct:.1f}% of corpus)")

    # Build per-mint path index
    sk_by_mint = {m: g for m, g in sk.groupby("mint")}

    policies = args.policies or list_policies()
    print(f"\n--- (3) policies under test ({len(policies)}): {policies} ---")
    cfg = _MockCfg()

    all_results = {}
    for tau in args.thresholds:
        keep = joined[joined.score >= tau].copy()
        if args.max_mints_per_tau > 0:
            keep = keep.head(args.max_mints_per_tau)
        print(f"\n--- τ={tau}: replaying {len(keep)} mints ---")

        per_policy_pls = {p: [] for p in policies}
        t0 = time.time()
        ok = skip_no_path = skip_invalid = 0
        for _, row in keep.iterrows():
            mint = row["mint"]
            if mint not in sk_by_mint:
                skip_no_path += 1; continue
            res = replay_one(row, sk_by_mint[mint], cfg, policies)
            if res is None:
                skip_invalid += 1; continue
            ok += 1
            for p, pl in res.items():
                per_policy_pls[p].append(pl)
            if ok > 0 and ok % 5000 == 0:
                print(f"    ... {ok} replayed  ({time.time()-t0:.1f}s)", flush=True)

        print(f"  replayed: {ok}/{len(keep)} "
              f"(no_path={skip_no_path}, invalid={skip_invalid}) "
              f"in {time.time()-t0:.1f}s")

        print(f"\n  {'policy':18s} {'n':>5s} {'mean':>8s} {'median':>8s} "
              f"{'p25':>7s} {'p75':>7s} {'p99':>8s} {'win%':>5s} {'total':>8s}  {'boot p05':>9s}")
        per_policy_stats = {}
        for p in policies:
            s = stats(per_policy_pls[p])
            if s is None:
                print(f"  {p:18s} {'-':>5s}  (no replays)")
                continue
            boot = bootstrap_mean_p05(per_policy_pls[p])
            s["bootstrap_mean_p05"] = boot
            per_policy_stats[p] = s
            print(f"  {p:18s} {s['n']:>5d} {s['mean']:>+8.4f} {s['median']:>+8.4f} "
                  f"{s['p25']:>+7.3f} {s['p75']:>+7.3f} {s['p99']:>+8.3f} "
                  f"{s['win_pct']:>4.1f} {s['total']:>+8.2f}  "
                  f"{boot if boot is None else f'{boot:+9.4f}'}")
        all_results[f"tau_{tau:.4f}"] = per_policy_stats

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
               "inputs": args.inputs,
               "artifact_dir": args.artifact_dir,
               "thresholds": args.thresholds,
               "policies": policies,
               "results": all_results}, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
