"""Per-example agreement between HGB and ARFClassifier(50) on the
identical test set used in the AUC comparison. AUC measures ranking
quality; this script measures whether the two models actually pick
the SAME mints at any given threshold.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr

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
    print(f"=== HGB vs ARF50 agreement @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

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

    y = (wide[target_col] >= 1.0).astype(int).values
    idx = np.arange(len(wide))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20,
                                       random_state=42, stratify=y)
    X = wide[WIDE].values
    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    mints_te = wide["mint"].values[idx_te]
    print(f"  test rows: {len(idx_te):,}")

    # HGB scores
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=42)
    hgb.fit(Xtr, ytr)
    s_hgb = hgb.predict_proba(Xte)[:, 1]

    # ARF50 scores
    from river import forest
    arf = forest.ARFClassifier(n_models=50, seed=42)
    print(f"  training ARF50 (replay {len(Xtr):,} rows)...")
    t0 = time.time()
    for i in range(len(Xtr)):
        x = {WIDE[j]: float(Xtr[i, j]) if not np.isnan(Xtr[i, j]) else 0.0
             for j in range(len(WIDE))}
        arf.learn_one(x, int(ytr[i]))
    print(f"  ARF50 trained in {time.time()-t0:.1f}s")

    s_arf = np.zeros(len(Xte))
    for i in range(len(Xte)):
        x = {WIDE[j]: float(Xte[i, j]) if not np.isnan(Xte[i, j]) else 0.0
             for j in range(len(WIDE))}
        p = arf.predict_proba_one(x)
        s_arf[i] = p.get(1, p.get(True, 0.5))

    # ---------- agreement metrics ----------
    print(f"\n=== score-level agreement (do they output similar numbers?) ===")
    rho_pearson, _  = pearsonr(s_hgb, s_arf)
    rho_spearman, _ = spearmanr(s_hgb, s_arf)
    print(f"  Pearson r  (linear):  {rho_pearson:.4f}")
    print(f"  Spearman ρ (rank):    {rho_spearman:.4f}")
    print(f"  -> r=1.0 would mean identical outputs; r=0 would mean unrelated")

    # Score distribution comparison
    print(f"\n=== score distribution side-by-side ===")
    print(f"  {'pctile':>8s}  {'HGB':>8s}  {'ARF50':>8s}  {'diff':>8s}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00):
        h = float(np.quantile(s_hgb, q))
        a = float(np.quantile(s_arf, q))
        print(f"  {q*100:7.1f}%  {h:>8.4f}  {a:>8.4f}  {a-h:>+8.4f}")

    # ---------- fire-set agreement at the live threshold (0.4134) ----------
    print(f"\n=== fire-set agreement at the live threshold 0.4134 ===")
    THR = 0.4134
    fire_hgb = s_hgb >= THR
    fire_arf = s_arf >= THR
    n_h = int(fire_hgb.sum())
    n_a = int(fire_arf.sum())
    both = int((fire_hgb & fire_arf).sum())
    only_hgb = int((fire_hgb & ~fire_arf).sum())
    only_arf = int((~fire_hgb & fire_arf).sum())
    union = int((fire_hgb | fire_arf).sum())
    jaccard = both / union if union else 0
    print(f"  HGB fires:        {n_h:>5d}")
    print(f"  ARF50 fires:      {n_a:>5d}")
    print(f"  BOTH fire:        {both:>5d}")
    print(f"  ONLY HGB fires:   {only_hgb:>5d}")
    print(f"  ONLY ARF fires:   {only_arf:>5d}")
    print(f"  Jaccard overlap:  {jaccard*100:5.1f}%   (= |both| / |either|)")
    print(f"  HGB-fires-also-fired-by-ARF: {both*100/max(n_h,1):.1f}%")
    print(f"  ARF-fires-also-fired-by-HGB: {both*100/max(n_a,1):.1f}%")

    # ---------- top-K agreement (regardless of threshold) ----------
    print(f"\n=== top-K agreement (highest-conviction picks) ===")
    for k in (10, 50, 100, 200, 500):
        top_h = set(np.argsort(-s_hgb)[:k])
        top_a = set(np.argsort(-s_arf)[:k])
        overlap = len(top_h & top_a)
        print(f"  top-{k:>3d}: HGB and ARF agree on {overlap}/{k} = {overlap*100/k:.0f}%")

    # ---------- precision on label among each model's fires ----------
    print(f"\n=== precision (P(peak >= 2x | fired)) at threshold 0.4134 ===")
    if n_h: print(f"  HGB fires:           precision = {yte[fire_hgb].mean()*100:.1f}%  on n={n_h}")
    if n_a: print(f"  ARF50 fires:         precision = {yte[fire_arf].mean()*100:.1f}%  on n={n_a}")
    if both:
        print(f"  Agreement fires:     precision = {yte[fire_hgb & fire_arf].mean()*100:.1f}%  on n={both}")
    if only_hgb:
        print(f"  HGB-only fires:      precision = {yte[fire_hgb & ~fire_arf].mean()*100:.1f}%  on n={only_hgb}")
    if only_arf:
        print(f"  ARF-only fires:      precision = {yte[~fire_hgb & fire_arf].mean()*100:.1f}%  on n={only_arf}")


if __name__ == "__main__":
    main()
