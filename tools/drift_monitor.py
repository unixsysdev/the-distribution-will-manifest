"""Drift monitor — daily KS test on live scores vs frozen training distribution.

Runs daily (via systemd timer). Reads the last 24h of entry_decision rows from
bot_data/shadow_run.jsonl, computes summary stats and a Kolmogorov-Smirnov
statistic against a reference distribution loaded from a frozen training-sample
file. Emits a structured drift_alert event to logs/drift_log.jsonl when:
  (a) KS p-value (approx) < ALPHA_KS                        (distribution shape change)
  (b) live_p90 < training_p90 - SHIFT_THRESHOLD             (live materially lower)
  (c) fire-rate over the last N entry_decisions < MIN_FIRE_RATE_PCT
  (d) live winner rate (peak_ret>=2x via capture lookup) < MIN_WINNER_RATE
                                                            (data-driven calibration miss)

Exit codes:
  0 = no drift detected
  2 = drift detected (caller can use this to trigger a retrain check)

Read-only. Does not modify the running bot.
"""
from __future__ import annotations
import argparse, gzip, glob, json, sys, time
from pathlib import Path
import statistics as st

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")

# Thresholds for drift
ALPHA_KS              = 0.01    # ks_d threshold approx for n~100
SHIFT_THRESHOLD       = 0.10    # live_p90 must not drop more than this below training_p90
MIN_FIRE_RATE_PCT     = 1.0     # fire / ready < 1% triggers
MIN_WINNER_RATE_PCT   = 10.0    # winners / fires_with_outcome < 10% triggers


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--training-ref",
                    default="bot_artifacts_K7V/training_score_sample.json",
                    help="JSON file with reference training-set score samples")
    ap.add_argument("--lookback-hours", type=float, default=24.0)
    # Defaults left for backward compat but overridden in main() from the
    # CURRENT production model spec — see _dynamic_train_quantiles().
    ap.add_argument("--training-p50", type=float, default=None)
    ap.add_argument("--training-p90", type=float, default=None)
    ap.add_argument("--artifact-dir", default="bot_artifacts_K7V",
                    help="prod artifacts to pull dynamic training-set quantiles from")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress stdout if no drift detected (cron friendly)")
    return ap.parse_args()


def ks_d_2sample(a, b):
    """KS statistic (max abs diff of empirical CDFs)."""
    if not a or not b: return None
    a, b = sorted(a), sorted(b)
    i = j = 0; D = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        if a[i] <= b[j]: i += 1
        else: j += 1
        D = max(D, abs(i/na - j/nb))
    return D


def ks_p_approx(D, n, m):
    """Asymptotic p-value approximation for KS statistic.
    p ~ 2 * exp(-2 * D^2 * (nm/(n+m)))"""
    import math
    return 2.0 * math.exp(-2.0 * D**2 * (n * m / (n + m)))


def load_capture_winner_rate(root, fires, hours):
    """For each fire in fires, compute realized peak_ret from capture lookup.
    Returns (n_analyzed, winners_2x, total) so caller can compute the rate."""
    cap_dir = root / "grpc_capture"
    mints = set(f["mint"] for f in fires)
    cap_trades = {}
    for path in sorted(glob.glob(str(cap_dir / "*.jsonl*"))):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    try: rec = json.loads(ln)
                    except: continue
                    m = rec.get("mint")
                    if m in mints:
                        vs, vt = rec.get("vsol", 0), rec.get("vtok", 0)
                        if vs > 0 and vt > 0:
                            cap_trades.setdefault(m, []).append((rec.get("ev_ts", 0), vs, vt))
        except Exception:
            continue
    n = 0; w2x = 0
    for fire in fires:
        m = fire["mint"]; midK = fire.get("midK")
        trig = fire.get("k_window_last_ts") or fire.get("v_window_last_ts")
        if not midK or not trig: continue
        trades = [(ts, vs, vt) for ts, vs, vt in cap_trades.get(m, []) if ts >= trig]
        if len(trades) < 5: continue
        peak = max(vs/vt for _, vs, vt in trades)
        peak_ret = peak / midK - 1.0
        n += 1
        if peak_ret >= 1.0: w2x += 1
    return n, w2x


def _dynamic_train_quantiles(root: Path, artifact_dir: str) -> tuple[float, float]:
    """Score the production model's TRAINING data with itself, return (p50, p90).
    Eliminates the stale-hardcoded-reference bug. Quantiles match exactly what
    the LIVE model would say about its own training population."""
    import pickle
    import numpy as np
    import pandas as pd
    art = root / artifact_dir
    clf = pickle.load(open(art / "entry_model.pkl", "rb"))
    ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
               "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
    ENTRY_V = [f"{c}_v" for c in ENTRY_K]
    K_TO_V = dict(zip(ENTRY_K, ENTRY_V))
    # Load fresh + OOS (matches build_bot_artifacts_K7V default training set)
    dfs = []
    for prefix in ["", "oos_"]:
        tk_p = root / f"data/pumpfun_continuation_{prefix}K7_fresh/token_level.parquet"
        tv_p = root / f"data/pumpfun_continuation_{prefix}V05_fresh/token_level.parquet"
        if not (tk_p.exists() and tv_p.exists()): continue
        tk = pd.read_parquet(tk_p); tv = pd.read_parquet(tv_p)
        dfs.append(tk.merge(tv[["mint"]+ENTRY_K].rename(columns=K_TO_V),
                            on="mint", how="inner"))
    if not dfs:
        # Fallback: pull from model_spec if quantiles persisted there
        try:
            spec = json.loads((art / "model_spec.json").read_text())
            return (float(spec["entry"].get("train_p50", 0.189)),
                    float(spec["entry"].get("train_p90", 0.4453)))
        except Exception:
            return (0.189, 0.4453)
    tr = pd.concat(dfs, ignore_index=True).drop_duplicates("mint")
    X = tr[ENTRY_K + ENTRY_V].values
    p = clf.predict_proba(X)[:, 1]
    return float(np.percentile(p, 50)), float(np.percentile(p, 90))


def main():
    args = parse_args()
    root = Path(args.root)
    # Pull training quantiles DYNAMICALLY from the current production model
    # (overrides any stale CLI defaults).
    if args.training_p50 is None or args.training_p90 is None:
        try:
            dyn_p50, dyn_p90 = _dynamic_train_quantiles(root, args.artifact_dir)
            if args.training_p50 is None: args.training_p50 = dyn_p50
            if args.training_p90 is None: args.training_p90 = dyn_p90
            if not args.quiet:
                print(f"  training quantiles (dynamic from {args.artifact_dir}/entry_model.pkl): "
                      f"p50={args.training_p50:.4f}  p90={args.training_p90:.4f}")
        except Exception as e:
            print(f"  dynamic quantile lookup failed ({e}); using legacy defaults")
            args.training_p50 = args.training_p50 or 0.189
            args.training_p90 = args.training_p90 or 0.4453
    log_path = root / "logs" / "drift_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    cutoff = now - args.lookback_hours * 3600

    # Load entry_decisions in lookback window
    sr = root / "bot_data" / "shadow_run.jsonl"
    if not sr.exists():
        if not args.quiet: print("no shadow_run.jsonl")
        return 0
    eds = []
    with open(sr) as f:
        for ln in f:
            try: r = json.loads(ln)
            except: continue
            if r.get("kind") == "entry_decision" and r.get("t", 0) >= cutoff:
                eds.append(r)
    fires = [r for r in eds if r.get("fire")]
    if not args.quiet:
        print(f"drift monitor: {args.lookback_hours}h lookback, "
              f"{len(eds)} ready / {len(fires)} fires")
    if len(eds) < 20:
        if not args.quiet: print("not enough data for KS (< 20 ready)")
        return 0

    live_scores = [r["score"] for r in eds if r.get("score") is not None]
    live_quantiles = sorted(live_scores)
    def q(p): return live_quantiles[min(int(len(live_quantiles)*p), len(live_quantiles)-1)]
    live_p50, live_p90 = q(0.50), q(0.90)
    fire_rate_pct = 100 * len(fires) / max(len(eds), 1)

    # Load reference training-sample if available
    ref_path = root / args.training_ref
    ref_scores = None
    if ref_path.exists():
        try:
            ref = json.loads(ref_path.read_text())
            ref_scores = ref.get("scores")
        except Exception: pass

    ks_d = ks_p = None
    if ref_scores and len(ref_scores) >= 100:
        ks_d = ks_d_2sample(live_scores, ref_scores)
        ks_p = ks_p_approx(ks_d, len(live_scores), len(ref_scores))

    # Realized winner rate
    winner_n, winner_w2x = (0, 0)
    if fires:
        winner_n, winner_w2x = load_capture_winner_rate(root, fires, args.lookback_hours)
    winner_rate_pct = 100 * winner_w2x / max(winner_n, 1)

    # Detect drift
    flags = []
    if ks_d is not None and ks_p is not None and ks_p < ALPHA_KS:
        flags.append(f"shape_shift (KS_D={ks_d:.3f}, p~{ks_p:.2e})")
    if live_p90 < args.training_p90 - SHIFT_THRESHOLD:
        flags.append(f"location_shift (live_p90={live_p90:.4f} vs train_p90={args.training_p90:.4f})")
    if len(eds) >= 50 and fire_rate_pct < MIN_FIRE_RATE_PCT:
        flags.append(f"low_fire_rate ({fire_rate_pct:.1f}% < {MIN_FIRE_RATE_PCT}%)")
    if winner_n >= 30 and winner_rate_pct < MIN_WINNER_RATE_PCT:
        flags.append(f"low_winner_rate ({winner_rate_pct:.1f}% on n={winner_n})")

    alert = {
        "t": now, "kind": "drift_check",
        "lookback_h": args.lookback_hours,
        "n_ready": len(eds), "n_fires": len(fires),
        "fire_rate_pct": fire_rate_pct,
        "live_p50": live_p50, "live_p90": live_p90,
        "training_p50": args.training_p50, "training_p90": args.training_p90,
        "ks_d": ks_d, "ks_p_approx": ks_p,
        "winner_analyzed": winner_n, "winner_2x_count": winner_w2x,
        "winner_rate_pct": winner_rate_pct if winner_n else None,
        "flags": flags, "drift_detected": bool(flags),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(alert) + "\n")

    if not args.quiet or flags:
        print(f"  live: p50={live_p50:.4f} p90={live_p90:.4f}  fire_rate={fire_rate_pct:.1f}%  "
              f"winner_rate={winner_rate_pct:.1f}% (n={winner_n})")
        if ks_d is not None:
            print(f"  KS_D={ks_d:.3f} p~{ks_p:.2e}")
        if flags:
            print(f"  DRIFT_DETECTED: {', '.join(flags)}")
        else:
            print(f"  no drift detected")
    return 2 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
