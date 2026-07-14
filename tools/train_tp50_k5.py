"""Train a +50% target model on K=5 parquets, with sophistication
features inner-joined. Output goes to bot_artifacts_K7V_tp50_k5/.

Builds on tools/train_integrated_v2.py but with two differences:
  1. target = peak_ret >= 0.50  (was 1.0)
  2. K=5 inputs (data/pumpfun_continuation_K7_k5_snap1)
     — directory is named K7_k5_snap1 because the extractor still emits
       under the K7_ prefix even when K=5 (just a label, not load-bearing).

Then probes the score distribution at multiple top-percentile cutoffs
so we can pick a threshold that gives many fires + ~99% precision.

Does NOT swap any production artifact.
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
OUT = ROOT / "bot_artifacts_K7V_tp50_k5"

CLASSIC_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
CLASSIC_V = [f"{c}_v" for c in CLASSIC_K]
CLASSIC = CLASSIC_K + CLASSIC_V
SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]
WIDE = CLASSIC + SOPH

PATH = ["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w",
        "solo_sell_w","vel_w","dts"]
RECOVERY = PATH + CLASSIC_K   # 9 path + 11 K = 20 features


def main():
    print(f"=== train_tp50_k5 @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

    # Load K=5 parquets we just extracted, plus the K=7-fresh data for breadth
    # (the entry features extracted at K=5 vs K=7 are NOT directly comparable
    # — they're computed over different trade windows — so for the headline
    # K=5 experiment we use ONLY the K=5 _snap1 data. Older _fresh data
    # would muddy the picture).
    k7 = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_k5_snap1/token_level.parquet")
    v  = pd.read_parquet(ROOT / "data/pumpfun_continuation_V05_k5_snap1/token_level.parquet")
    print(f"  K=5 K-features rows: {len(k7):,}")
    print(f"  K=5 V-features rows: {len(v):,}")
    target_col = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in k7.columns)
    print(f"  target column: {target_col}")
    # Rename V cols to _v
    v = v.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                           if c.removesuffix("_v") in v.columns})
    df = k7.merge(v[["mint"]+CLASSIC_V], on="mint", how="inner")\
            .drop_duplicates(subset=["mint"], keep="last")
    print(f"  K+V inner-join: {len(df):,} mints")
    soph = pd.read_parquet(ROOT/"data/sophistication_current.parquet")
    soph = soph[["mint"] + [c for c in SOPH if c in soph.columns]]\
              .drop_duplicates(subset=["mint"], keep="last")
    wide = df.merge(soph, on="mint", how="inner")
    print(f"  K+V+soph inner-join: {len(wide):,} mints")

    # Target: peak_ret >= 0.5
    y = (wide[target_col] >= 0.5).astype(int).values
    print(f"  positive rate (peak >= +50%): {y.mean():.3f}")
    X = wide[WIDE].values
    idx = np.arange(len(wide))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)
    print(f"  train={len(idx_tr):,}  test={len(idx_te):,}")

    # Train entry head
    print("\n[1] ENTRY HEAD (target peak_ret >= 0.5)")
    clf_e = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=42)
    clf_e.fit(X[idx_tr], y[idx_tr])
    s_tr = clf_e.predict_proba(X[idx_tr])[:,1]
    s_te = clf_e.predict_proba(X[idx_te])[:,1]
    auc_in  = roc_auc_score(y[idx_tr], s_tr)
    auc_oos = roc_auc_score(y[idx_te], s_te)
    print(f"  train AUC = {auc_in:.4f}   OOS AUC = {auc_oos:.4f}")

    # Precision at multiple top-percentile cutoffs ON OOS
    print("\n[2] OOS precision vs fire rate sweep:")
    print(f"  {'fire %':>8s} {'cutoff':>9s} {'n_fires':>8s} {'precision':>10s} {'mean_peak':>10s}")
    peak_te = wide["peak_ret"].values[idx_te]
    for pct in (0.5, 1, 2, 3, 5, 10, 15, 20, 25, 30):
        cut = float(np.quantile(s_te, 1 - pct/100))
        m = s_te >= cut
        nf = int(m.sum())
        if not nf: continue
        prec = float((y[idx_te][m].mean()))
        mean_peak = float(peak_te[m].mean())
        print(f"  {pct:>7.1f}% {cut:>9.4f} {nf:>8d} {prec*100:>9.1f}% {mean_peak:>+10.2f}")

    # Recovery head (path snapshots)
    print("\n[3] RECOVERY HEAD")
    sk = pd.read_parquet(ROOT/"data/pumpfun_continuation_K7_k5_snap1/path_snapshots.parquet")
    print(f"  K=5 path-snapshots: {len(sk):,} rows")
    tk = wide[["mint"] + CLASSIC_K].copy()
    if "terminal_ret" in k7.columns:
        tk = tk.merge(k7[["mint","terminal_ret"]].drop_duplicates(subset=["mint"], keep="last"),
                       on="mint", how="left")
    else:
        tk["terminal_ret"] = wide[target_col]
    s2 = sk.sort_values(["mint","fwd"]).copy()
    tk_unique = tk.drop_duplicates(subset=["mint"], keep="last")
    s2["term"] = s2["mint"].map(tk_unique.set_index("mint")["terminal_ret"])
    def suf(g):
        r = g["ret"].values; f = np.empty(len(r)); run = g["term"].iloc[0]
        for i in range(len(r)-1, -1, -1):
            f[i] = run
            if r[i] > run: run = r[i]
        return pd.Series(f, index=g.index)
    s2["fm"] = s2.groupby("mint", group_keys=False).apply(suf)
    dd = s2[s2.ret < 0].merge(tk[["mint"]+CLASSIC_K], on="mint", how="left")
    dd = dd.dropna(subset=PATH + CLASSIC_K)
    print(f"  drawdown rows: {len(dd):,}")
    Xr = dd[PATH + CLASSIC_K].values
    yr = (dd["fm"] >= 0).astype(int).values
    clf_r = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.05,
        l2_regularization=1.0, random_state=42).fit(Xr, yr)
    auc_r = roc_auc_score(yr, clf_r.predict_proba(Xr)[:,1])
    print(f"  recovery train AUC = {auc_r:.4f}")

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(clf_e, open(OUT/"entry_model.pkl", "wb"))
    pickle.dump(clf_r, open(OUT/"recovery_model.pkl", "wb"))
    # Use top-2% as a reasonable default threshold (lots of fires, very high precision)
    default_thr = float(np.quantile(s_te, 0.98))
    spec = {
        "sklearn_version": sklearn.__version__,
        "entry": {
            "features": WIDE,
            "features_classic": CLASSIC,
            "features_sophistication": SOPH,
            "target": "peak_ret >= 0.5 (+50%, K=5-anchored)",
            "fire_if": "predict_proba[:,1] >= entry_threshold",
            "entry_threshold_top_decile": default_thr,   # called top_decile for compat with model_serve
            "trigger": "K=5 trade-count trigger",
            "K_WINDOW": 5,
            "train_auc_in_sample": float(auc_in),
            "train_auc_peak2x":    float(auc_oos),   # field name kept for compat; this is OOS for +50%
        },
        "recovery": {
            "features": RECOVERY,
            "target": "recovers to breakeven (future ret >= 0)",
            "death_cut_if": "predict_proba[:,1] < 0.10",
            "death_cut_threshold": 0.10,
            "train_auc": float(auc_r),
        },
        "exit_policy": "level_tp_50 (intended to be used with this model)",
        "n_train_tokens": int(len(idx_tr)),
        "n_test_oos":     int(len(idx_te)),
        "n_recovery_train_rows": int(len(dd)),
        "input_suffixes": ["_k5_snap1"],
        "fresh_rsol_filtered": True,
        "note": "+50% target model trained on K=5 features. Designed to pair "
                "with exit policy level_tp_50.",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT/"model_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nSaved {OUT}/  ({sklearn.__version__})")
    print(f"  entry threshold (top-2%): {default_thr:.4f}")


if __name__ == "__main__":
    main()
