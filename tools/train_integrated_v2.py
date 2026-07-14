"""Integrated rebuild: wide entry head + recovery head, saved as one artifact dir.

Combines two earlier scripts:
  - tools/train_wide_v2.py (wide entry head: 22 classic + 9 soph features,
    inner-joined on the sophistication parquet so soph features are non-NaN)
  - tools/build_bot_artifacts_K7V.py (recovery head: 20 features = 9 path +
    11 K-entry; standard drawdown-snap formulation)

This is the ship-ready integrated rebuild for the snap_every=1 era.

Output: bot_artifacts_K7V_wide_v2/
  - entry_model.pkl          31-feature wide entry head
  - entry_model_baseline.pkl 22-feature classic baseline (for AUC comparison)
  - recovery_model.pkl       20-feature recovery head, trained on
                              {_fresh, _snap1} path-snapshots combined
  - model_spec.json          full spec with both AUCs + comparison vs the
                              production-deployed bot_artifacts_K7V/

The caller is expected to symlink-swap manually after reviewing the spec.
"""
from __future__ import annotations
import argparse, json, pickle, time
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
CLASSIC   = CLASSIC_K + CLASSIC_V                # 22

SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]                 # 9
WIDE = CLASSIC + SOPH                            # 31

PATH = ["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w",
        "solo_sell_w","vel_w","dts"]              # 9 path features for recovery
RECOVERY = PATH + CLASSIC_K                       # 20 = 9 path + 11 K-entry


def _load_concat(suffixes, tag, leaf):
    """Concat parquets across suffixes. tag in {'K7','V05','oos_K7','oos_V05'}.
    leaf in {'token_level.parquet','path_snapshots.parquet'}."""
    dfs = []
    for s in suffixes:
        p = ROOT / f"data/pumpfun_continuation_{tag}{s}" / leaf
        if p.exists():
            df = pd.read_parquet(p)
            df["_src"] = s
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["_fresh","_snap1"])
    ap.add_argument("--soph", default="data/sophistication_current.parquet")
    ap.add_argument("--out", default="bot_artifacts_K7V_wide_v2")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    print(f"=== train_integrated_v2 @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"  inputs: {args.inputs}")
    print(f"  soph:   {args.soph}")
    print(f"  out:    {args.out}")

    # ---------- ENTRY HEAD (wide, inner-join on soph) ----------
    print("\n[1/3] ENTRY HEAD")
    k7  = _load_concat(args.inputs, "K7",  "token_level.parquet")
    k7o = _load_concat(args.inputs, "oos_K7","token_level.parquet")
    v05 = _load_concat(args.inputs, "V05", "token_level.parquet")
    v05o= _load_concat(args.inputs, "oos_V05","token_level.parquet")
    print(f"  K7 train+oos: {len(k7)+len(k7o):,}  V05 train+oos: {len(v05)+len(v05o):,}")
    k7_all = pd.concat([k7, k7o], ignore_index=True) if len(k7o) else k7
    v_all  = pd.concat([v05, v05o], ignore_index=True) if len(v05o) else v05

    target_col = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in k7_all.columns)
    print(f"  target: {target_col}")
    k7_keep = ["mint", target_col] + CLASSIC_K
    k7_all = k7_all[[c for c in k7_keep if c in k7_all.columns]].copy()

    # V parquet uses unsuffixed names; rename to _v for join
    v_all = v_all.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                                   if c.removesuffix("_v") in v_all.columns})
    v_all = v_all[[c for c in ["mint"]+CLASSIC_V if c in v_all.columns]].copy()

    df = k7_all.merge(v_all, on="mint", how="inner").drop_duplicates(subset=["mint"], keep="last")
    print(f"  K7+V inner-join: {len(df):,} unique mints")

    soph = pd.read_parquet(args.soph)
    soph_cols = [c for c in SOPH if c in soph.columns]
    soph = soph[["mint"]+soph_cols].drop_duplicates(subset=["mint"], keep="last").copy()
    print(f"  soph parquet: {len(soph):,} mints with {len(soph_cols)} soph cols")

    wide = df.merge(soph, on="mint", how="inner")
    print(f"  K7+V+soph inner-join: {len(wide):,} mints (entry training set)")

    y_e = (wide[target_col] >= 1.0).astype(int).values
    idx = np.arange(len(wide))
    idx_tr, idx_te = train_test_split(idx, test_size=args.test_size,
                                       random_state=args.random_state,
                                       stratify=y_e)
    print(f"  split train={len(idx_tr)} test={len(idx_te)}  positive rate={y_e.mean():.3f}")

    def fit_entry(feats, label):
        X = wide[feats].values
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            l2_regularization=1.0, random_state=args.random_state)
        clf.fit(X[idx_tr], y_e[idx_tr])
        auc_in  = roc_auc_score(y_e[idx_tr], clf.predict_proba(X[idx_tr])[:,1])
        auc_oos = roc_auc_score(y_e[idx_te], clf.predict_proba(X[idx_te])[:,1])
        print(f"  {label}: in-sample AUC={auc_in:.4f}  OOS AUC={auc_oos:.4f}")
        return clf, auc_in, auc_oos

    clf_b, auc_b_in, auc_b_oos = fit_entry(CLASSIC, "baseline (22)")
    clf_e, auc_e_in, auc_e_oos = fit_entry(WIDE,    "wide     (31)")
    uplift_oos = auc_e_oos - auc_b_oos
    print(f"  ENTRY uplift OOS (wide - baseline) = {uplift_oos:+.4f}")

    # Top-decile threshold on the WIDE training distribution
    train_scores = clf_e.predict_proba(wide[WIDE].values[idx_tr])[:,1]
    thr_e = float(np.percentile(train_scores, 90))
    print(f"  wide entry top-decile threshold = {thr_e:.4f}")

    # ---------- RECOVERY HEAD ----------
    print("\n[2/3] RECOVERY HEAD")
    # Path-snapshots from {_fresh, _snap1, _oos_*} — use everything we have
    sk = _load_concat(args.inputs, "K7", "path_snapshots.parquet")
    sk_o = _load_concat(args.inputs, "oos_K7", "path_snapshots.parquet")
    sk_all = pd.concat([sk, sk_o], ignore_index=True) if len(sk_o) else sk
    print(f"  K7 path-snapshots train+oos: {len(sk_all):,} rows")
    print(f"  per-suffix breakdown:")
    for s in args.inputs:
        n = (sk_all["_src"] == s).sum()
        print(f"    {s}: {n:,} snaps")

    # Need entry features (CLASSIC_K) per mint + terminal_ret per mint
    tk_recover = k7_all[["mint"] + CLASSIC_K + [target_col]].copy()
    if "terminal_ret" in pd.concat([k7, k7o]).columns:
        # Pull terminal_ret from original (we dropped it above)
        all_tk = pd.concat([k7, k7o], ignore_index=True) if len(k7o) else k7
        tk_recover = tk_recover.merge(
            all_tk[["mint", "terminal_ret"]].drop_duplicates(subset=["mint"], keep="last"),
            on="mint", how="left"
        )
    else:
        # Fallback: use target_col as the "terminal" since we don't have anything else
        tk_recover["terminal_ret"] = tk_recover[target_col]

    # Suffix-max (future maximum ret after each snap) - mirror build_bot_artifacts_K7V.py
    s2 = sk_all.sort_values(["mint","fwd"]).copy()
    tk_unique = tk_recover.drop_duplicates(subset=["mint"], keep="last")
    s2["term"] = s2.mint.map(tk_unique.set_index("mint")["terminal_ret"])
    def suf(g):
        r = g["ret"].values; f = np.empty(len(r)); run = g["term"].iloc[0]
        for i in range(len(r)-1, -1, -1):
            f[i] = run
            if r[i] > run: run = r[i]
        return pd.Series(f, index=g.index)
    s2["fm"] = s2.groupby("mint", group_keys=False).apply(suf)

    # Recovery training set: drawdown snaps only (ret<0), target = future-max >= 0
    dd = s2[s2.ret < 0].merge(tk_recover[["mint"] + CLASSIC_K], on="mint", how="left")
    # Drop rows missing any feature (safer than HGB-NaN for the recovery model)
    dd = dd.dropna(subset=PATH + CLASSIC_K)
    print(f"  recovery training set: {len(dd):,} drawdown-snap rows")
    if len(dd) == 0:
        raise SystemExit("recovery training set empty — bailing")

    Xr = dd[PATH + CLASSIC_K].values
    yr = (dd["fm"] >= 0).astype(int).values
    print(f"  recovery base rate (recovers to breakeven) = {yr.mean():.3f}")

    clf_r = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.05,
        l2_regularization=1.0, random_state=args.random_state).fit(Xr, yr)
    auc_r = roc_auc_score(yr, clf_r.predict_proba(Xr)[:,1])
    print(f"  recovery train AUC = {auc_r:.4f}  death-cut threshold = 0.10")

    # ---------- SAVE ----------
    print("\n[3/3] SAVE")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(clf_e, open(out/"entry_model.pkl", "wb"))
    pickle.dump(clf_b, open(out/"entry_model_baseline.pkl", "wb"))
    pickle.dump(clf_r, open(out/"recovery_model.pkl", "wb"))

    # Compare against current production artifacts
    prod_spec_path = ROOT / "bot_artifacts_K7V/model_spec.json"
    prod = json.loads(prod_spec_path.read_text()) if prod_spec_path.exists() else {}
    prod_auc_e = prod.get("entry", {}).get("train_auc_peak2x")
    prod_auc_r = prod.get("recovery", {}).get("train_auc")
    prod_thr   = prod.get("entry", {}).get("entry_threshold_top_decile")

    spec = {
        "sklearn_version": sklearn.__version__,
        "entry": {
            "features": WIDE,
            "features_classic": CLASSIC,
            "features_sophistication": SOPH,
            "target": "peak_ret >= 1.0 (>=2x, K7-anchored)",
            "fire_if": "predict_proba[:,1] >= entry_threshold",
            "entry_threshold_top_decile": thr_e,
            "trigger": "BOTH K=7 trade-count AND V=0.5 cumulative buy SOL must fire (decide at max of the two)",
            "entry_reserves": "reserves at K=7 trigger time (vsK7, vtK7)",
            "train_auc_in_sample": float(auc_e_in),
            "train_auc_peak2x":    float(auc_e_oos),    # OOS now, matches prod's field name convention
            "baseline_auc_in_sample": float(auc_b_in),
            "baseline_auc_oos":       float(auc_b_oos),
            "sophistication_uplift_oos": float(uplift_oos),
        },
        "recovery": {
            "features": RECOVERY,
            "target": "recovers to breakeven (future ret >= 0)",
            "death_cut_if": "predict_proba[:,1] < 0.10",
            "death_cut_threshold": 0.10,
            "train_auc": float(auc_r),
        },
        "exit_policy": "scale-out-into-strength once ret>0 (cap 8 slices); precision death-cut when P(recover)<0.10",
        "n_train_tokens":      int(len(wide)),
        "n_test_oos":          int(len(idx_te)),
        "n_recovery_train_rows": int(len(dd)),
        "input_suffixes":      args.inputs,
        "output_suffix":       args.out,
        "fresh_rsol_filtered": True,
        "soph_source":         args.soph,
        "note":  "Integrated wide entry (inner-join on soph) + recovery rebuild. "
                 "snap_every=1 path-snapshots used for recovery. "
                 "DO NOT auto-swap — review spec, then manual symlink swap.",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparison_vs_production": {
            "prod_entry_auc": prod_auc_e,
            "prod_recovery_auc": prod_auc_r,
            "prod_entry_threshold": prod_thr,
            "new_entry_auc_oos": float(auc_e_oos),
            "new_recovery_auc": float(auc_r),
            "new_entry_threshold": thr_e,
            "delta_entry_auc": float(auc_e_oos - (prod_auc_e or 0)),
            "delta_recovery_auc": float(auc_r - (prod_auc_r or 0)),
            "delta_threshold": float(thr_e - (prod_thr or 0)),
        },
    }
    (out/"model_spec.json").write_text(json.dumps(spec, indent=2))

    print(f"\nSaved {out}/entry_model.pkl + entry_model_baseline.pkl + recovery_model.pkl + model_spec.json")
    print(f"\n=== COMPARISON vs production (bot_artifacts_K7V/) ===")
    print(f"  entry AUC:    prod={prod_auc_e!r}  new={auc_e_oos:.4f}  delta={auc_e_oos - (prod_auc_e or 0):+.4f}")
    print(f"  recovery AUC: prod={prod_auc_r!r}  new={auc_r:.4f}  delta={auc_r - (prod_auc_r or 0):+.4f}")
    print(f"  entry threshold (top-decile): prod={prod_thr!r}  new={thr_e:.4f}  delta={thr_e - (prod_thr or 0):+.4f}")
    print(f"\nTo swap (DO NOT RUN without user approval):")
    print(f"  mv bot_artifacts_K7V bot_artifacts_K7V_pre_wide_v2_swap_$(date +%s)")
    print(f"  ln -s {args.out} bot_artifacts_K7V")
    print(f"  sudo systemctl restart pumpfun-bot")


if __name__ == "__main__":
    main()
