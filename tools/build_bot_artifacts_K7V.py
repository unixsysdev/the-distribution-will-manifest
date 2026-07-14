"""Build production V+K7 artifacts (entry + recovery pickles + model_spec.json).

Reads OLD K7 + V parquets (and OOS for the training set; combine both for max data),
trains the 22-feature entry head and the 20-feature recovery head, computes the
TRAINING top-decile threshold for entry, saves to <out-dir>/.

Reusable for the fresh-population retrain: pass --suffix _fresh and the script reads
data/pumpfun_continuation_K7_fresh, _oos_K7_fresh, _V05_fresh, _oos_V05_fresh and
writes to bot_artifacts_K7V_fresh/.
"""
from __future__ import annotations
import argparse, json, pickle, sklearn
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
ENTRY_V = [f"{c}_v" for c in ENTRY_K]
PATH    = ["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w","dts"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="single-suffix mode (back-compat): reads+writes <suffix>")
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="multi-suffix mode: e.g. --inputs _fresh _capture "
                         "(concatenates training data from each suffix dir)")
    ap.add_argument("--out", default=None,
                    help="output bot_artifacts_K7V<out> dir (defaults to last input)")
    ap.add_argument("--include-oos-in-train", action="store_true", default=True,
                    help="include OOS tokens in training to maximize data (production default).")
    ap.add_argument("--no-oos-in-train", dest="include_oos_in_train", action="store_false",
                    help="train OLD-only (matches the strict analytical reference).")
    args = ap.parse_args()
    if args.inputs is None:
        args.inputs = [args.suffix]
    if args.out is None:
        args.out = args.suffix if args.suffix else args.inputs[-1]
    return args


def _load_concat(inputs, kind):
    """Load + concat tables across multiple input suffixes.
    kind: 'tk_old', 'tk_oos', 'tv_old', 'tv_oos', 'sk_old' """
    name = {"tk_old": "K7", "tk_oos": "oos_K7",
            "tv_old": "V05", "tv_oos": "oos_V05",
            "sk_old": "K7"}[kind]
    leaf = "path_snapshots.parquet" if kind == "sk_old" else "token_level.parquet"
    dfs = []
    for s in inputs:
        p = Path(f"data/pumpfun_continuation_{name}{s}/{leaf}")
        if p.exists():
            df = pd.read_parquet(p)
            df["_src_suffix"] = s
            dfs.append(df)
        else:
            print(f"  warning: {p} not present; skipping suffix {s!r} for {kind}")
    if not dfs: return None
    return pd.concat(dfs, ignore_index=True)


def main():
    args = parse_args()
    OUT_DIR = Path(f"bot_artifacts_K7V{args.out}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== build_bot_artifacts_K7V (inputs={args.inputs}  out={args.out!r}) ===")
    tk_old = _load_concat(args.inputs, "tk_old")
    tk_oos = _load_concat(args.inputs, "tk_oos")
    tv_old = _load_concat(args.inputs, "tv_old")
    tv_oos = _load_concat(args.inputs, "tv_oos")
    sk_old = _load_concat(args.inputs, "sk_old")
    if any(x is None for x in (tk_old, tk_oos, tv_old, tv_oos, sk_old)):
        raise SystemExit("missing required parquets for one or more inputs")
    print(f"  OLD K7={len(tk_old)} V={len(tv_old)} snaps={len(sk_old)}")
    print(f"  OOS K7={len(tk_oos)} V={len(tv_oos)}")
    if len(args.inputs) > 1:
        for s in args.inputs:
            n_in_s = (tk_old["_src_suffix"] == s).sum()
            print(f"    from suffix {s!r}: tk_old n={n_in_s}")
    # drop the source-tracking column before training
    for df in (tk_old, tk_oos, tv_old, tv_oos, sk_old):
        if "_src_suffix" in df.columns: df.drop(columns=["_src_suffix"], inplace=True)

    K_TO_V = {k: v for k, v in zip(ENTRY_K, ENTRY_V)}
    tv_old_f = tv_old[["mint"] + ENTRY_K].rename(columns=K_TO_V)
    tv_oos_f = tv_oos[["mint"] + ENTRY_K].rename(columns=K_TO_V)
    old_join = tk_old.merge(tv_old_f, on="mint", how="inner")
    oos_join = tk_oos.merge(tv_oos_f, on="mint", how="inner")
    print(f"  joined OLD={len(old_join)}  joined OOS={len(oos_join)}")

    if args.include_oos_in_train:
        train = pd.concat([old_join, oos_join], ignore_index=True)
        print(f"  training set = OLD + OOS = {len(train)} tokens (production default)")
    else:
        train = old_join
        print(f"  training set = OLD only = {len(train)} tokens (strict-OOS reference mode)")

    # Entry head — 22 features, target peak_ret >= 1.0 (2x)
    FEATS_ENTRY = ENTRY_K + ENTRY_V
    X_e = train[FEATS_ENTRY].values
    y_e = (train.peak_ret >= 1.0).astype(int).values
    print(f"  entry: training on {len(FEATS_ENTRY)} features, base rate peak>=2x = {y_e.mean():.1%}")
    clf_e = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.06,
        l2_regularization=1.0, random_state=0).fit(X_e, y_e)
    train_scores = clf_e.predict_proba(X_e)[:, 1]
    auc_e = roc_auc_score(y_e, train_scores)
    thr_e = float(np.percentile(train_scores, 90))   # top-decile on TRAINING distribution
    print(f"  entry train AUC = {auc_e:.4f}  top-decile threshold = {thr_e:.4f}")

    # Recovery head — 20 features (9 path + 11 K), target recovers-to-breakeven (suffix max >= 0)
    print("  building recovery training set ...")
    s2 = sk_old.sort_values(["mint","fwd"]).copy()
    # Some mints may appear in BOTH _fresh and _capture suffixes (rare but
    # possible when a token straddles the dataset boundary). Dedupe before
    # set_index so pandas doesn't reject the indexing.
    tk_unique = tk_old.drop_duplicates(subset=["mint"], keep="last")
    s2["term"] = s2.mint.map(tk_unique.set_index("mint")["terminal_ret"])
    def suf(g):
        r = g["ret"].values; f = np.empty(len(r)); run = g["term"].iloc[0]
        for i in range(len(r)-1, -1, -1):
            f[i] = run
            if r[i] > run: run = r[i]
        return pd.Series(f, index=g.index)
    s2["fm"] = s2.groupby("mint", group_keys=False).apply(suf)
    dd = s2[s2.ret < 0].merge(tk_old[ENTRY_K + ["mint"]], on="mint", how="left")
    Xr = dd[PATH + ENTRY_K].values
    yr = (dd["fm"] >= 0).astype(int).values
    print(f"  recovery: training on {len(PATH)+len(ENTRY_K)} features, "
          f"{len(dd)} drawdown snaps, base rate recover = {yr.mean():.1%}")
    clf_r = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.05,
        l2_regularization=1.0, random_state=0).fit(Xr, yr)
    auc_r = roc_auc_score(yr, clf_r.predict_proba(Xr)[:, 1])
    print(f"  recovery train AUC = {auc_r:.4f}  death-cut threshold = 0.10")

    # Save
    pickle.dump(clf_e, open(OUT_DIR/"entry_model.pkl", "wb"))
    pickle.dump(clf_r, open(OUT_DIR/"recovery_model.pkl", "wb"))
    spec = {
        "sklearn_version": sklearn.__version__,
        "entry": {
            "features": FEATS_ENTRY,
            "features_k7": ENTRY_K,
            "features_v":  ENTRY_V,
            "target": "peak_ret>=1.0 (>=2x, K7-anchored)",
            "fire_if": "predict_proba[:,1] >= entry_threshold",
            "entry_threshold_top_decile": thr_e,
            "trigger": "BOTH K=7 trade-count AND V=0.5 cumulative buy SOL must fire (decide at max of the two)",
            "entry_reserves": "reserves at K=7 trigger time (vsK7, vtK7)",
            "train_auc_peak2x": float(auc_e),
        },
        "recovery": {
            "features": PATH + ENTRY_K,
            "target": "recovers to breakeven (future ret>=0)",
            "death_cut_if": "predict_proba[:,1] < 0.10",
            "death_cut_threshold": 0.10,
            "train_auc": float(auc_r),
        },
        "exit_policy": "scale-out-into-strength once ret>0 (cap 8 slices); precision death-cut when P(recover)<0.10",
        "n_train_tokens": int(len(train)),
        "n_recovery_train_rows": int(len(dd)),
        "input_suffixes": args.inputs,
        "output_suffix": args.out,
        "fresh_rsol_filtered": any("_fresh" in x or "_capture" in x for x in args.inputs),
        "note": f"V+K7 stacked entry head at K=7 trigger, trained on inputs={args.inputs}",
    }
    json.dump(spec, open(OUT_DIR/"model_spec.json", "w"), indent=2)
    print(f"\nSaved {OUT_DIR}/{{entry_model.pkl,recovery_model.pkl,model_spec.json}}")
    print(f"  entry_threshold = {thr_e:.4f}  (was 0.4481 for the original non-fresh model)")


if __name__ == "__main__":
    main()
