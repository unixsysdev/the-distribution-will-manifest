"""Train the LSM continuation-value regressor.

f(s_t)  ->  E[terminal_ret | s_t]   where s_t is the full per-snap state.

This is the rigorous answer to "what is the expected future value of holding
optimally from here". Once trained, the lsm_continuation exit policy uses
`sell_all iff ret_now > f(s_t) + epsilon` as its single decision rule.

Features (all from path_snapshots + per-mint entry_score):
    ret, run_max_ret, dd
    retracement_norm                — (run_max - ret) / (1 + run_max)
    velocity_ret_1                  — (ret_t - ret_{t-1}) / dt
    velocity_ret_3                  — slope over last 3 snaps
    accel_ret                       — Δvelocity (second derivative proxy)
    time_since_run_max              — fwd-snaps since run_max last updated
    dts                             — age since entry (s)
    fill_k                          — bonding curve fullness
    buy_frac_w                      — recent buy share
    nsell_w                         — recent sell count
    solo_sell_w                     — single-actor sell share (whale dump vs cascade)
    vel_w                           — trade velocity
    entry_score                     — per-mint, from production entry model

Target: terminal_ret (per-mint, from token_level — what passive-hold would realize).

NaN-friendly: HistGradientBoostingRegressor handles missing values natively, so
early snaps where velocity-3 isn't defined still train fine.

Usage:
    python tools/train_lsm_continuation.py \\
        --inputs _fresh _capture_jun8 \\
        --out bot_artifacts_K7V/lsm_continuation.pkl
"""
from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
ENTRY_V = [f"{c}_v" for c in ENTRY_K]

LSM_FEATURES = [
    "ret", "run_max_ret", "dd", "retracement_norm",
    "velocity_ret_1", "velocity_ret_3", "accel_ret",
    "time_since_run_max",
    "dts", "fill_k",
    "buy_frac_w", "nsell_w", "solo_sell_w", "vel_w",
    "entry_score",
]


def load_corpus(inputs):
    tk_dfs, tv_dfs, sk_dfs = [], [], []
    for s in inputs:
        for prefix in ["", "oos_"]:
            tk_p = ROOT / f"data/pumpfun_continuation_{prefix}K7{s}/token_level.parquet"
            tv_p = ROOT / f"data/pumpfun_continuation_{prefix}V05{s}/token_level.parquet"
            sk_p = ROOT / f"data/pumpfun_continuation_{prefix}K7{s}/path_snapshots.parquet"
            if not (tk_p.exists() and tv_p.exists() and sk_p.exists()):
                continue
            tk = pd.read_parquet(tk_p); tv = pd.read_parquet(tv_p); sk = pd.read_parquet(sk_p)
            tk_dfs.append(tk); tv_dfs.append(tv); sk_dfs.append(sk)
            print(f"  loaded {prefix}{s.lstrip('_')!r}: tk={len(tk)}  sk={len(sk)}")
    if not tk_dfs: raise SystemExit("no data")
    tk = pd.concat(tk_dfs, ignore_index=True).drop_duplicates("mint", keep="last")
    tv = pd.concat(tv_dfs, ignore_index=True).drop_duplicates("mint", keep="last")
    sk = pd.concat(sk_dfs, ignore_index=True)
    return tk, tv, sk


def compute_entry_score(tk, tv, art_dir):
    K_TO_V = {k: v for k, v in zip(ENTRY_K, ENTRY_V)}
    tv_f = tv[["mint"] + ENTRY_K].rename(columns=K_TO_V)
    joined = tk.merge(tv_f, on="mint", how="inner")
    clf = pickle.load(open(art_dir / "entry_model.pkl", "rb"))
    X = joined[ENTRY_K + ENTRY_V].values
    joined["entry_score"] = clf.predict_proba(X)[:, 1]
    return joined[["mint", "entry_score", "terminal_ret"]].copy()


def add_path_features(sk: pd.DataFrame) -> pd.DataFrame:
    """Compute path-local derivatives per mint."""
    sk = sk.sort_values(["mint", "fwd"]).reset_index(drop=True)
    g = sk.groupby("mint", sort=False)

    sk["ret_prev1"]  = g["ret"].shift(1)
    sk["ret_prev3"]  = g["ret"].shift(3)
    sk["dts_prev1"]  = g["dts"].shift(1)
    sk["dts_prev3"]  = g["dts"].shift(3)
    sk["vel_prev1"]  = g.apply(
        lambda d: (d["ret"] - d["ret"].shift(1)) /
                  (d["dts"] - d["dts"].shift(1)).replace(0, np.nan)
    ).reset_index(level=0, drop=True)

    sk["velocity_ret_1"] = (sk["ret"] - sk["ret_prev1"]) / \
                           (sk["dts"] - sk["dts_prev1"]).replace(0, np.nan)
    sk["velocity_ret_3"] = (sk["ret"] - sk["ret_prev3"]) / \
                           (sk["dts"] - sk["dts_prev3"]).replace(0, np.nan)
    sk["accel_ret"] = sk["velocity_ret_1"] - g["velocity_ret_1"].shift(1)

    sk["retracement_norm"] = ((sk["run_max_ret"] - sk["ret"]) /
                              (1 + sk["run_max_ret"]).clip(lower=1e-6))

    # time_since_run_max: snaps since the last new high. Use cumulative reset.
    # When ret == run_max_ret, this snap IS the new high → 0. Otherwise increment.
    is_new_high = (sk["ret"] >= sk["run_max_ret"] - 1e-9).astype(int)
    # within each mint, count snaps since last new high
    sk["_grp"] = is_new_high.groupby(sk["mint"]).cumsum()
    sk["time_since_run_max"] = sk.groupby(["mint", "_grp"]).cumcount()

    drop_cols = ["ret_prev1", "ret_prev3", "dts_prev1", "dts_prev3",
                 "vel_prev1", "_grp"]
    sk = sk.drop(columns=[c for c in drop_cols if c in sk.columns])
    return sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["_fresh", "_capture_jun8"])
    ap.add_argument("--artifact-dir", default="bot_artifacts_K7V")
    ap.add_argument("--out", default="bot_artifacts_K7V/lsm_continuation.pkl")
    ap.add_argument("--min-snaps-per-mint", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=300)
    args = ap.parse_args()

    print(f"=== train_lsm_continuation @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"  inputs={args.inputs}")

    tk, tv, sk = load_corpus(args.inputs)
    print(f"  total mints={len(tk)}  total snaps={len(sk)}")

    print(f"\n--- (1) compute entry scores ---")
    per_mint = compute_entry_score(tk, tv, ROOT / args.artifact_dir)
    print(f"  scored {len(per_mint)} mints")

    print(f"\n--- (2) compute path-local features ---")
    t0 = time.time()
    sk = sk.merge(per_mint, on="mint", how="inner")
    sk = add_path_features(sk)
    print(f"  done in {time.time()-t0:.1f}s; shape={sk.shape}")

    # Filter: at least N snaps per mint (so velocity_3 isn't trivially NaN for all)
    counts = sk.groupby("mint").size()
    eligible_mints = counts[counts >= args.min_snaps_per_mint].index
    sk = sk[sk["mint"].isin(eligible_mints)].copy()
    print(f"  after min-snaps filter: {len(sk)} rows from {len(eligible_mints)} mints")

    # Target = terminal_ret per mint (constant per mint, already joined)
    X = sk[LSM_FEATURES].values
    y = sk["terminal_ret"].values

    # Simple chrono-ish split: hash mint to {0..9}, use 8 for train, 2 for eval.
    # Avoids snap-level leakage (a mint's snaps are all-train or all-eval).
    h = sk["mint"].apply(lambda m: hash(m) % 10).values
    train_mask = h < 8
    eval_mask  = h >= 8
    print(f"  train rows={train_mask.sum()}  eval rows={eval_mask.sum()}")

    print(f"\n--- (3) train HGB regressor ---")
    t0 = time.time()
    reg = HistGradientBoostingRegressor(
        max_depth=4, max_iter=args.max_iter, learning_rate=0.05,
        l2_regularization=1.0, random_state=0,
    ).fit(X[train_mask], y[train_mask])
    print(f"  fit in {time.time()-t0:.1f}s")

    yhat_tr = reg.predict(X[train_mask])
    yhat_ev = reg.predict(X[eval_mask])
    print(f"  train: R^2={r2_score(y[train_mask], yhat_tr):.4f}  "
          f"MAE={mean_absolute_error(y[train_mask], yhat_tr):.4f}")
    print(f"  eval : R^2={r2_score(y[eval_mask], yhat_ev):.4f}  "
          f"MAE={mean_absolute_error(y[eval_mask], yhat_ev):.4f}")
    print(f"  target distribution: mean={y.mean():+.4f}  median={np.median(y):+.4f}  "
          f"p10={np.percentile(y,10):+.4f}  p90={np.percentile(y,90):+.4f}")
    print(f"  pred  distribution:  mean={yhat_tr.mean():+.4f}  median={np.median(yhat_tr):+.4f}")

    # Diagnostic: in regions where current ret is high (above some level), is the
    # model predicting LOWER continuation? That's what would let level-TP-style
    # exits emerge.
    df_eval = pd.DataFrame({
        "ret": X[eval_mask, 0],       # first feature = ret
        "y":   y[eval_mask],
        "yhat": yhat_ev,
    })
    print(f"\n  --- conditional-prediction diagnostic (eval set) ---")
    print(f"  {'ret_bucket':16s} {'n':>7s} {'y_actual_mean':>15s} {'yhat_mean':>10s} "
          f"{'sell_signal_pct':>16s}")
    for lo, hi in [(-1.0, 0.0), (0.0, 0.20), (0.20, 0.50), (0.50, 1.00),
                   (1.00, 2.00), (2.00, 5.00), (5.00, 100.0)]:
        m = (df_eval.ret >= lo) & (df_eval.ret < hi)
        if m.sum() == 0: continue
        # "sell_signal" = how often the rule would fire (ret > yhat)
        sell_pct = 100 * (df_eval.loc[m, "ret"] > df_eval.loc[m, "yhat"]).mean()
        print(f"  [{lo:>+5.2f},{hi:>+5.2f})  {m.sum():>7d} "
              f"{df_eval.loc[m,'y'].mean():>+13.4f}  "
              f"{df_eval.loc[m,'yhat'].mean():>+10.4f}  "
              f"{sell_pct:>15.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": reg,
        "features": LSM_FEATURES,
        "train_r2":  float(r2_score(y[train_mask], yhat_tr)),
        "eval_r2":   float(r2_score(y[eval_mask], yhat_ev)),
        "train_mae": float(mean_absolute_error(y[train_mask], yhat_tr)),
        "eval_mae":  float(mean_absolute_error(y[eval_mask], yhat_ev)),
        "n_train":   int(train_mask.sum()),
        "n_eval":    int(eval_mask.sum()),
        "target_dist": {
            "mean":   float(y.mean()),
            "median": float(np.median(y)),
        },
    }
    pickle.dump(payload, open(out, "wb"))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
