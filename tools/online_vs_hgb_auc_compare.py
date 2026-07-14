"""HGB (batch) vs online learners on the IDENTICAL train/test split that
wide_v2 used. Pure read-only experiment. Doesn't touch the live bot.

Question: can an online learner get close enough to HGB's OOS AUC of
0.7923 that we'd consider swapping?

Method:
  1. Rebuild the wide_v2 inner-join population (K7+V05 inner-joined to
     sophistication parquet, dedupe by mint, 80/20 stratified split,
     same random_state=42 we used to train the live model).
  2. Train HGB (sklearn HistGradientBoostingClassifier) on the train
     half — sanity-check we land near 0.7923 OOS.
  3. Replay the training rows sequentially into each candidate online
     learner from `river`. Score the same OOS test set after EACH
     model has seen ALL training rows once.
  4. Compare OOS AUC.

Candidates:
  - river.forest.ARFClassifier      (online RF with drift detection)
  - river.ensemble.AdaBoostClassifier wrapping a HoeffdingTreeClassifier
                                      (online boosting)
  - river.tree.HoeffdingAdaptiveTreeClassifier  (single adaptive tree)

The HGB AUC is computed FRESH so the comparison is apples-to-apples
on the exact same data (not relying on the stored 0.7923).
"""
from __future__ import annotations
import json, pickle, time
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
CLASSIC = CLASSIC_K + CLASSIC_V
SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]
WIDE = CLASSIC + SOPH


def _load_concat(suffixes, tag):
    dfs = []
    for s in suffixes:
        for prefix in (f"data/pumpfun_continuation_{tag}{s}",
                       f"data/pumpfun_continuation_oos_{tag}{s}"):
            p = ROOT / f"{prefix}/token_level.parquet"
            if p.exists():
                dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    print(f"=== online_vs_hgb_auc_compare @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

    # Rebuild the wide_v2 training population
    k7  = _load_concat(["_fresh","_snap1"], "K7")
    v05 = _load_concat(["_fresh","_snap1"], "V05")
    v05 = v05.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                               if c.removesuffix("_v") in v05.columns})
    df = k7.merge(v05[["mint"]+CLASSIC_V], on="mint", how="inner")\
            .drop_duplicates(subset=["mint"], keep="last")
    target_col = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in df.columns)
    df = df[["mint", target_col] + CLASSIC].copy()
    soph = pd.read_parquet(ROOT/"data/sophistication_current.parquet")
    soph_cols = [c for c in SOPH if c in soph.columns]
    soph = soph[["mint"]+soph_cols].drop_duplicates(subset=["mint"], keep="last")
    wide = df.merge(soph, on="mint", how="inner")
    print(f"  inner-join population: {len(wide):,} mints")

    y = (wide[target_col] >= 1.0).astype(int).values
    idx = np.arange(len(wide))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20,
                                       random_state=42, stratify=y)
    print(f"  train={len(idx_tr):,}  test={len(idx_te):,}")
    print(f"  positive rate: {y.mean():.3f}")

    X = wide[WIDE].values
    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    # ---------- HGB baseline ----------
    print(f"\n[1] HGB (batch, sklearn HistGradientBoostingClassifier)")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=42)
    hgb.fit(Xtr, ytr)
    hgb_auc = roc_auc_score(yte, hgb.predict_proba(Xte)[:,1])
    print(f"  train_time={time.time()-t0:.1f}s   OOS AUC = {hgb_auc:.4f}")

    # ---------- river online learners ----------
    # Convert rows to dicts (river uses dict-of-features per sample)
    feat_names = WIDE
    def _row_to_dict(row):
        return {feat_names[i]: float(row[i]) if not np.isnan(row[i]) else 0.0
                for i in range(len(feat_names))}

    print(f"\n[2] river online learners (sequential replay of train, then score on test)")

    from river import forest, ensemble, tree, metrics

    candidates = {
        "ARFClassifier (10 trees)":
            lambda: forest.ARFClassifier(n_models=10, seed=42),
        "ARFClassifier (50 trees)":
            lambda: forest.ARFClassifier(n_models=50, seed=42),
        "HoeffdingAdaptiveTree":
            lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=42),
        "AdaBoost(HoeffdingTree x10)":
            lambda: ensemble.AdaBoostClassifier(
                model=tree.HoeffdingTreeClassifier(),
                n_models=10, seed=42),
    }

    results = []
    for name, ctor in candidates.items():
        model = ctor()
        t0 = time.time()
        # Replay training rows sequentially
        for i in range(len(Xtr)):
            x = _row_to_dict(Xtr[i])
            model.learn_one(x, int(ytr[i]))
        train_time = time.time() - t0
        # Score test set
        t1 = time.time()
        probs = []
        for i in range(len(Xte)):
            x = _row_to_dict(Xte[i])
            p = model.predict_proba_one(x)
            # river returns dict {0: p0, 1: p1}; default to 0.5 if class unseen
            probs.append(p.get(1, p.get(True, 0.5)))
        score_time = time.time() - t1
        auc = roc_auc_score(yte, probs)
        print(f"  {name:35s}  AUC={auc:.4f}   "
              f"train={train_time:5.1f}s  score={score_time:4.1f}s   "
              f"gap vs HGB={auc-hgb_auc:+.4f}")
        results.append((name, auc, train_time, score_time))

    # Summary table
    print(f"\n=== summary ===")
    print(f"  HGB (batch):                          AUC={hgb_auc:.4f}   (reference)")
    for name, auc, tt, st in results:
        gap = auc - hgb_auc
        verdict = "MATCH" if abs(gap) < 0.02 else ("CLOSE" if abs(gap) < 0.04 else "GAP")
        print(f"  {name:35s}  AUC={auc:.4f}   gap={gap:+.4f}   {verdict}")

    print(f"\nVerdict: {'HGB unchallenged' if not any(abs(a-hgb_auc) < 0.02 for _,a,_,_ in results) else 'at least one online model matches'}")
    print(f"  MATCH = within 0.02 OOS AUC of HGB  -> swap viable")
    print(f"  CLOSE = within 0.04                  -> consider with drift benefit")
    print(f"  GAP   = >= 0.04                      -> too lossy, keep HGB")


if __name__ == "__main__":
    main()
