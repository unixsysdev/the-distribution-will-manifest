"""Backtest the wide_v2 entry head on:
   (1) Its own OOS holdout (sanity, already measured)
   (2) Older May (_fresh) population — entry features only, no soph
   (3) Recent live entry_decision events from the bot's shadow_run.jsonl
       (the real "would-have-fired-on-what?" comparison)

And compute a calibrated entry threshold from the LIVE score distribution.

The current production threshold (0.5108) was the top-decile of TRAINING
data. Live score distribution has drifted lower (live_p90 ~0.225). At the
trained threshold the bot fires only ~0.5% of the time — below the 1%
drift-gate. The calibration step picks a threshold from the live score
distribution that gives a target fire rate.

Output:
  - Score histograms (text-printed) for all three populations
  - AUC numbers
  - Recommended calibrated threshold for several target fire rates

DOES NOT write any files or swap any symlinks.
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/the-distribution-will-manifest")

CLASSIC_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
CLASSIC_V = [f"{c}_v" for c in CLASSIC_K]
CLASSIC = CLASSIC_K + CLASSIC_V
SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]
WIDE = CLASSIC + SOPH


def _hist(arr, label, bins=10):
    """Print a quick text histogram of scores in [0,1]."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"  {label}: n={len(arr):,}  p10={qs[0]:.4f}  p25={qs[1]:.4f}  "
          f"p50={qs[2]:.4f}  p75={qs[3]:.4f}  p90={qs[4]:.4f}  p95={qs[5]:.4f}  p99={qs[6]:.4f}")
    h, _ = np.histogram(arr, bins=np.linspace(0, 1, bins+1))
    max_h = max(h) if max(h) else 1
    for i, c in enumerate(h):
        lo, hi = i/bins, (i+1)/bins
        bar = "#" * int(50 * c / max_h)
        print(f"    [{lo:.1f}-{hi:.1f}]  {c:6d}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="bot_artifacts_K7V_wide_v2")
    ap.add_argument("--soph", default="data/sophistication_current.parquet")
    ap.add_argument("--live-fires-window-hours", type=int, default=48)
    args = ap.parse_args()

    print(f"=== backtest_wide_v2 @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    cand_dir = ROOT / args.candidate
    spec = json.loads((cand_dir / "model_spec.json").read_text())

    wide_model     = pickle.load(open(cand_dir / "entry_model.pkl", "rb"))
    baseline_model = pickle.load(open(cand_dir / "entry_model_baseline.pkl", "rb"))
    print(f"  candidate: {cand_dir.name}")
    print(f"  wide features: {len(spec['entry']['features'])}, baseline: {len(spec['entry']['features_classic'])}")

    # -------- (1) MAY (_fresh) population — baseline only --------
    print("\n[1/3] MAY (_fresh) population, BASELINE 22-feat model")
    k7  = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_fresh/token_level.parquet")
    k7o = pd.read_parquet(ROOT / "data/pumpfun_continuation_oos_K7_fresh/token_level.parquet")
    v   = pd.read_parquet(ROOT / "data/pumpfun_continuation_V05_fresh/token_level.parquet")
    vo  = pd.read_parquet(ROOT / "data/pumpfun_continuation_oos_V05_fresh/token_level.parquet")
    k7m = pd.concat([k7, k7o], ignore_index=True)
    vm  = pd.concat([v,  vo],  ignore_index=True)
    v_rename = {c.removesuffix("_v"): c for c in CLASSIC_V if c.removesuffix("_v") in vm.columns}
    vm = vm.rename(columns=v_rename)
    may = k7m.merge(vm[["mint"]+CLASSIC_V], on="mint", how="inner").drop_duplicates(subset=["mint"], keep="last")
    target = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in may.columns)
    print(f"  rows: {len(may):,}")
    Xm = may[CLASSIC].values
    sm = baseline_model.predict_proba(Xm)[:,1]
    ym = (may[target] >= 1.0).astype(int).values
    print(f"  baseline OOS AUC on May = {roc_auc_score(ym, sm):.4f}")
    _hist(sm, "May score distribution (baseline)")

    # -------- (2) Recent live entry_decision events --------
    print(f"\n[2/3] LIVE entry_decision events (last {args.live_fires_window_hours}h)")
    since = time.time() - args.live_fires_window_hours * 3600
    rows = []
    with open(ROOT / "bot_data/shadow_run.jsonl") as f:
        for ln in f:
            try: r = json.loads(ln)
            except: continue
            if r.get("kind") != "entry_decision": continue
            if r.get("t", 0) < since: continue
            feats = r.get("features", {})
            if not feats: continue
            row = {"mint": r.get("mint"), "score_old": r.get("score"),
                   "fire_old": r.get("fire"), "threshold_old": r.get("threshold")}
            row.update(feats)
            rows.append(row)
    if not rows:
        print(f"  no live entry_decision events in last {args.live_fires_window_hours}h")
        return
    live = pd.DataFrame(rows)
    print(f"  live rows: {len(live):,}")
    # Score with BASELINE (22 features) — no soph in live records
    feat_cols = [c for c in CLASSIC if c in live.columns]
    missing = [c for c in CLASSIC if c not in live.columns]
    if missing:
        print(f"  warning: missing features in live data: {missing}")
    Xl = live[feat_cols].values
    sl = baseline_model.predict_proba(Xl)[:,1]
    live["score_new_baseline"] = sl
    # Score with WIDE if any soph data overlaps live mints
    soph = pd.read_parquet(args.soph)
    soph_cols = [c for c in SOPH if c in soph.columns]
    live_with_soph = live.merge(soph[["mint"]+soph_cols], on="mint", how="left")
    soph_present = live_with_soph[soph_cols[0]].notna()
    print(f"  live records with soph feature available: {soph_present.sum():,}/{len(live):,}")
    if soph_present.sum() > 0:
        sub = live_with_soph[soph_present].copy()
        Xsub = sub[feat_cols + soph_cols].values
        sub_scores = wide_model.predict_proba(Xsub)[:,1]
        sub["score_new_wide"] = sub_scores
        # Compute equivalents
        live_with_soph.loc[soph_present, "score_new_wide"] = sub_scores
    else:
        live_with_soph["score_new_wide"] = np.nan

    # Compare old vs new scores on the SAME live records
    print(f"\n  --- score distributions ---")
    _hist(live.score_old.dropna(),                 "old model score (production, 22-feat)")
    _hist(sl,                                       "new baseline score (22-feat from wide_v2)")
    if soph_present.sum() > 0:
        _hist(live_with_soph.score_new_wide.dropna(),  "new wide score (31-feat from wide_v2, soph-covered subset)")

    # -------- (3) CALIBRATED THRESHOLD --------
    print(f"\n[3/3] CALIBRATED THRESHOLDS for new model")
    for tgt_pct in (0.5, 1.0, 2.0, 5.0, 10.0):
        # Calibrated using BASELINE scores on live data (all rows)
        thr_b = float(np.quantile(sl, 1 - tgt_pct/100))
        # And wide scores on the soph-covered subset
        if soph_present.sum() > 30:
            wide_scores = live_with_soph.score_new_wide.dropna().values
            thr_w = float(np.quantile(wide_scores, 1 - tgt_pct/100))
        else:
            thr_w = None
        thr_w_str = f"  wide  thr={thr_w:.4f}" if thr_w else "  wide  (not enough soph-covered)"
        print(f"  target {tgt_pct:.1f}% fire rate ->  baseline thr={thr_b:.4f}  {thr_w_str}")

    # Current production threshold for reference
    prod_thr = spec["entry"]["entry_threshold_top_decile"]
    # Old prod threshold (what's actually live right now)
    old_prod_spec = json.loads((ROOT / "bot_artifacts_K7V/model_spec.json").read_text())
    real_prod_thr = old_prod_spec.get("entry",{}).get("entry_threshold_top_decile")
    print(f"\n  reference: current LIVE threshold = {real_prod_thr}")
    print(f"  reference: candidate's training-top-decile threshold = {prod_thr}")
    fire_at_prod = (sl >= real_prod_thr).mean() * 100
    print(f"  if we ship NEW BASELINE model with current LIVE threshold: live fire rate would be ~{fire_at_prod:.1f}%")

    # Hit rate (positive rate) at different thresholds — using May labels as proxy
    print(f"\n  May label ground truth at different baseline-score thresholds:")
    for thr in (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70):
        mask = sm >= thr
        if mask.sum() == 0:
            print(f"    thr={thr:.2f}  n=0")
            continue
        hit = ym[mask].mean()
        print(f"    thr={thr:.2f}  n_above={mask.sum():,}  >=2x hit rate={hit:.3f}")


if __name__ == "__main__":
    main()
