"""Clean-path model: predict tokens that reach +50% via a CLEAN trajectory
(no big drawdown before the peak), not just tokens that touched +50% at some
point in their life.

Defines clean_winner = (peak_ret >= 0.5) AND (min_ret_before_run_max_hits_50pct >= FLOOR)
Trains on K=5 features + V + soph (same 31 features as the current tp50_k5 model).
Reports positive base rate, AUC, precision sweep — and explicitly contrasts
against the loose "peak >= 0.5 ever" label so we can see the gap.
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path("/root/the-distribution-will-manifest")

CLASSIC_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
CLASSIC_V = [f"{c}_v" for c in CLASSIC_K]
CLASSIC = CLASSIC_K + CLASSIC_V
SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]
WIDE = CLASSIC + SOPH


def compute_min_dd_before_first_50(snaps: pd.DataFrame) -> pd.DataFrame:
    """For each mint, return min ret reached before the FIRST snap where
    ret >= 0.5. If ret never reaches 0.5, return min ret across all snaps
    (the token didn't make 0.5 — no execution-relevant constraint)."""
    snaps = snaps.sort_values(["mint", "fwd"]).reset_index(drop=True)
    out_rows = []
    for mint, g in snaps.groupby("mint", sort=False):
        rets = g["ret"].values
        # find first idx where ret >= 0.5
        hit = np.argmax(rets >= 0.5) if (rets >= 0.5).any() else None
        if hit is not None and rets[hit] >= 0.5:
            min_pre = rets[:hit+1].min() if hit > 0 else rets[0]
            hit_50 = True
        else:
            min_pre = rets.min()
            hit_50 = False
        out_rows.append({"mint": mint, "min_ret_before_50": min_pre, "hit_50": hit_50})
    return pd.DataFrame(out_rows)


def main():
    print(f"=== train_clean_path @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    k7 = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_k5_snap1/token_level.parquet")
    v  = pd.read_parquet(ROOT / "data/pumpfun_continuation_V05_k5_snap1/token_level.parquet")
    snaps = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_k5_snap1/path_snapshots.parquet")
    print(f"  K=5 tokens: {len(k7):,}  V tokens: {len(v):,}  path snaps: {len(snaps):,}")

    v = v.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                           if c.removesuffix("_v") in v.columns})
    df = k7.merge(v[["mint"]+CLASSIC_V], on="mint", how="inner")\
            .drop_duplicates(subset=["mint"], keep="last")
    soph = pd.read_parquet(ROOT/"data/sophistication_current.parquet")
    soph = soph[["mint"] + [c for c in SOPH if c in soph.columns]]\
              .drop_duplicates(subset=["mint"], keep="last")
    wide = df.merge(soph, on="mint", how="inner")
    print(f"  K+V+soph inner-join: {len(wide):,} mints")

    # Compute min drawdown before hitting +50%
    print("  computing min_ret_before_50 per mint...")
    md = compute_min_dd_before_first_50(snaps)
    wide = wide.merge(md, on="mint", how="left")
    n_with_path = wide["min_ret_before_50"].notna().sum()
    print(f"  mints with path data: {n_with_path:,}/{len(wide):,}")

    # Targets
    target_col = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in wide.columns)
    y_loose = (wide[target_col] >= 0.5).astype(int).values

    # Clean-path target at several floors
    print(f"\n  positive rate by target def:")
    print(f"    loose      peak >= 0.5 ever          : {y_loose.mean():.3f}")
    for floor in (-0.30, -0.20, -0.15, -0.10, -0.05):
        y_clean = ((wide[target_col] >= 0.5) &
                   (wide["min_ret_before_50"] >= floor)).astype(int).values
        print(f"    clean@{floor:+.2f} peak>=0.5 AND min_pre>={floor:+.2f}: {y_clean.mean():.3f}")

    # Train with floor = -0.15 (moderate clean-path)
    FLOOR = -0.15
    print(f"\n[TRAIN] clean-path target with floor={FLOOR}")
    wide_train = wide.dropna(subset=WIDE + ["min_ret_before_50"]).reset_index(drop=True)
    y = ((wide_train[target_col] >= 0.5) &
         (wide_train["min_ret_before_50"] >= FLOOR)).astype(int).values
    print(f"  rows with all features: {len(wide_train):,}")
    print(f"  clean-path positive rate: {y.mean():.3f}  (n_pos={int(y.sum()):,})")

    X = wide_train[WIDE].values
    idx_tr, idx_te = train_test_split(np.arange(len(wide_train)), test_size=0.20,
                                       random_state=42, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_depth=None, l2_regularization=1.0,
                                          random_state=42)
    clf.fit(X[idx_tr], y[idx_tr])
    s_tr = clf.predict_proba(X[idx_tr])[:,1]
    s_te = clf.predict_proba(X[idx_te])[:,1]
    auc_in  = roc_auc_score(y[idx_tr], s_tr)
    auc_oos = roc_auc_score(y[idx_te], s_te)
    print(f"  train AUC = {auc_in:.4f}   OOS AUC = {auc_oos:.4f}")

    # OOS precision sweep — both labels (clean and loose)
    y_loose_train = (wide_train[target_col] >= 0.5).astype(int).values
    peak_te = wide_train["peak_ret"].values[idx_te]
    min_pre_te = wide_train["min_ret_before_50"].values[idx_te]
    print(f"\n  OOS precision sweep (clean=peak>=0.5 AND min_pre>={FLOOR}):")
    print(f"  {'fire%':>6s} {'cutoff':>8s} {'n':>6s} {'clean_prec':>10s} {'loose_prec':>10s} {'mean_peak':>10s} {'mean_min_pre':>13s}")
    for pct in (0.5, 1, 2, 3, 5, 10, 15, 20):
        cut = float(np.quantile(s_te, 1 - pct/100))
        m = s_te >= cut
        nf = int(m.sum())
        if nf < 5: continue
        prec_clean = float(y[idx_te][m].mean())
        prec_loose = float(y_loose_train[idx_te][m].mean())
        mp = float(peak_te[m].mean())
        mm = float(min_pre_te[m].mean())
        print(f"  {pct:>5.1f}% {cut:>8.4f} {nf:>6d} {prec_clean*100:>9.1f}% {prec_loose*100:>9.1f}% {mp:>+10.2f} {mm:>+13.2f}")

    # Save
    OUT = ROOT / "bot_artifacts_K7V_clean_path"
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(clf, open(OUT/"entry_model.pkl","wb"))
    default_thr = float(np.quantile(s_te, 0.95))  # top-5% live target
    spec = {
        "sklearn_version": sklearn.__version__,
        "entry": {
            "features": WIDE,
            "features_classic": CLASSIC,
            "features_sophistication": SOPH,
            "target": f"peak_ret >= 0.5 AND min_ret_before_50 >= {FLOOR}",
            "clean_path_floor": FLOOR,
            "fire_if": "predict_proba[:,1] >= entry_threshold",
            "entry_threshold_top_decile": default_thr,
            "trigger": "K=5",
            "K_WINDOW": 5,
            "train_auc_in_sample": float(auc_in),
            "train_auc_peak2x":    float(auc_oos),
        },
        "exit_policy": "level_tp_50 (paired with this clean-path model)",
        "n_train_tokens": int(len(idx_tr)),
        "n_test_oos":     int(len(idx_te)),
        "positive_rate":  float(y.mean()),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Clean-path target: token reaches +50% from K=5 entry "
                "AND its lowest ret before doing so >= floor. Eliminates "
                "the dip-then-pump paths the bot would death_cut.",
    }
    (OUT/"model_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nSaved {OUT}/  threshold(top-5%)={default_thr:.4f}")


if __name__ == "__main__":
    main()
