"""Train V+K7 entry model with sophistication features — clean inner-join version.

Re-tests the wide_jun8 conclusion (sophistication features had 0.0 OOS uplift)
under two corrections:
  1. INNER JOIN K7+V tokens with sophistication parquet so every training row
     has real soph values (not NaN). Previously the wide_jun8 training had
     ~90% NaN soph rows because soph data only covered ~12% of K7 tokens.
  2. More data: 25,175 mints in sophistication_current.parquet (2.8x the
     8,868 jun8 had). Bigger sample = less noise on the comparison.

Splits:
  - 80/20 random train/test on the inner-join population
  - Same split applied to both baseline (22 features) and wide (31 features)
  - Apples-to-apples AUC comparison on the SAME 20% holdout

The recovery model is NOT retrained here (this script only tests if soph
features lift the entry head).

Output:
  - bot_artifacts_K7V_wide_v2/entry_model.pkl  (the wide head)
  - bot_artifacts_K7V_wide_v2/entry_model_baseline.pkl (22-feat baseline)
  - bot_artifacts_K7V_wide_v2/model_spec.json with both AUCs
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path("/root/the-distribution-will-manifest")

CLASSIC_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
CLASSIC_V = [f"{c}_v" for c in CLASSIC_K]
CLASSIC   = CLASSIC_K + CLASSIC_V                # 22 features

SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]  # 9 features

WIDE = CLASSIC + SOPH                            # 31 features


def _load_concat(suffixes, tag, kind):
    """tag = 'K7' or 'V05'; kind = 'token' or 'snap' (snap unused here)"""
    name = {"K7": "K7", "V05": "V05"}[tag]
    leaf = "token_level.parquet"
    dfs = []
    for s in suffixes:
        for prefix in (f"data/pumpfun_continuation_{name}{s}",
                       f"data/pumpfun_continuation_oos_{name}{s}"):
            p = ROOT / f"{prefix}/{leaf}"
            if p.exists():
                df = pd.read_parquet(p)
                df["_src"] = prefix
                dfs.append(df)
    if not dfs:
        raise SystemExit(f"no parquets for {tag} suffixes {suffixes}")
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soph", default="data/sophistication_current.parquet",
                    help="sophistication parquet (must have 'mint' + soph_ cols)")
    ap.add_argument("--inputs", nargs="+", default=["_fresh","_snap1"],
                    help="K7/V suffixes to merge for classic features")
    ap.add_argument("--out", default="bot_artifacts_K7V_wide_v2",
                    help="output artifact dir")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    print(f"=== train_wide_v2 @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"  inputs: {args.inputs}")
    print(f"  soph:   {args.soph}")

    # 1. Load K7 + V05 token-level data and merge by mint to get the V-window cols.
    k7 = _load_concat(args.inputs, "K7", "token")
    v  = _load_concat(args.inputs, "V05", "token")
    print(f"  K7 rows: {len(k7):,}  V rows: {len(v):,}")

    # Keep only feature columns + 'mint' + target. V uses _v suffix already on cols.
    target_col = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in k7.columns)
    print(f"  K7 target column: {target_col}")
    k7_keep = ["mint", target_col] + CLASSIC_K
    k7 = k7[[c for c in k7_keep if c in k7.columns]].copy()

    v_keep = ["mint"] + CLASSIC_V
    # The V parquet uses unsuffixed names; we rename to _v for join.
    v = v.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                          if c.removesuffix("_v") in v.columns})
    v = v[[c for c in v_keep if c in v.columns]].copy()

    # Join K7 + V on mint (inner — drop mints missing one side)
    df = k7.merge(v, on="mint", how="inner", suffixes=("","__dup"))
    print(f"  K7+V inner-join: {len(df):,} rows")

    # 2. Load soph and inner-join (drop mints with no soph data — this is the key
    # change vs wide_jun8 which left-joined and ended up with ~90% NaN soph)
    soph = pd.read_parquet(args.soph)
    soph = soph[["mint"] + [c for c in SOPH if c in soph.columns]].copy()
    print(f"  soph parquet rows: {len(soph):,}  cols available: {[c for c in SOPH if c in soph.columns]}")
    wide = df.merge(soph, on="mint", how="inner")
    print(f"  K7+V+soph inner-join: {len(wide):,} rows  (drop rate vs K7+V: {(1-len(wide)/max(len(df),1))*100:.0f}%)")

    # Build target binary: peak_ret >= 1.0  (>=2x return)
    y = (wide[target_col] >= 1.0).astype(int).values
    print(f"  positive rate (>=2x): {y.mean():.3f}  ({y.sum()}/{len(y)})")

    # 3. Train/test split — apply SAME split to baseline + wide for apples-to-apples
    idx = np.arange(len(wide))
    idx_tr, idx_te = train_test_split(idx, test_size=args.test_size,
                                       random_state=args.random_state,
                                       stratify=y)
    print(f"  split: train={len(idx_tr)}  test={len(idx_te)}")

    def _fit_score(name, feats):
        X = wide[feats].values
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            l2_regularization=1.0, random_state=args.random_state)
        clf.fit(Xtr, ytr)
        auc_in  = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
        auc_oos = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        return clf, auc_in, auc_oos

    print("\n[1/2] baseline (22 features)")
    clf_b, auc_b_in, auc_b_oos = _fit_score("baseline", CLASSIC)
    print(f"  train AUC: {auc_b_in:.4f}    test AUC: {auc_b_oos:.4f}")

    print("\n[2/2] wide (22 + 9 = 31 features)")
    clf_w, auc_w_in, auc_w_oos = _fit_score("wide", WIDE)
    print(f"  train AUC: {auc_w_in:.4f}    test AUC: {auc_w_oos:.4f}")

    uplift_in  = auc_w_in  - auc_b_in
    uplift_oos = auc_w_oos - auc_b_oos
    print(f"\n=== UPLIFT (wide - baseline) ===")
    print(f"  in-sample: {uplift_in:+.4f}")
    print(f"  OOS:       {uplift_oos:+.4f}    <-- the number that matters")

    # Save both pickles + spec
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "entry_model.pkl", "wb") as f: pickle.dump(clf_w, f)
    with open(out / "entry_model_baseline.pkl", "wb") as f: pickle.dump(clf_b, f)
    spec = {
        "features_classic": CLASSIC,
        "features_sophistication": SOPH,
        "features_wide": WIDE,
        "target": f"{target_col} >= 1.0  (>=2x)",
        "n_total_rows": len(wide),
        "n_train": int(len(idx_tr)),
        "n_test_oos": int(len(idx_te)),
        "positive_rate": float(y.mean()),
        "baseline_auc_in_sample": float(auc_b_in),
        "baseline_auc_oos": float(auc_b_oos),
        "wide_auc_in_sample": float(auc_w_in),
        "wide_auc_oos": float(auc_w_oos),
        "sophistication_uplift_in_sample": float(uplift_in),
        "sophistication_uplift_oos": float(uplift_oos),
        "inputs": args.inputs,
        "soph_source": args.soph,
        "note": "inner-join wide test (no NaN dilution). Joined K7+V05+soph on mint.",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "model_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nWrote {out}/entry_model.pkl + entry_model_baseline.pkl + model_spec.json")


if __name__ == "__main__":
    main()
