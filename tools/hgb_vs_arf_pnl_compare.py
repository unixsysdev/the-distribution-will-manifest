"""Compare modeled level_tp_100 P&L between HGB and ARF50 fire sets on
the same wide_v2 test split. Reuses fires from each model at the live
threshold (0.4134) and applies the canonical TP rule:

  if peak_ret >= 1.0:  capture +1.00 - fees  (TP fired, sell-all)
  else:                eventually close via stale watchdog at
                       (approximately) terminal_ret - fees

Per-fire net ratio is then summed for total P&L. Bet size 0.1 SOL.

This is the same payout model I used in train_integrated_v2's
modeled-payout printout, applied independently to each model's
fire set.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
BET_SOL = 0.1
FEES_RATIO = 0.06   # 6% round-trip fee haircut as a ratio of bet


def _load(suffixes, tag):
    dfs = []
    for s in suffixes:
        for prefix in (f"data/pumpfun_continuation_{tag}{s}",
                       f"data/pumpfun_continuation_oos_{tag}{s}"):
            p = ROOT / f"{prefix}/token_level.parquet"
            if p.exists():
                dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _level_tp_100_pnl(peak, term):
    """Per-fire net ratio under level_tp_100. peak/term are arrays."""
    captured = (peak >= 1.0).astype(float) * 1.00         # TP capture at +100%
    not_capt = (peak <  1.0).astype(float) * term          # held to terminal
    return captured + not_capt - FEES_RATIO


def _stats(label, pnl, fires_n_pos):
    sol = pnl * BET_SOL
    print(f"  {label}: n={len(pnl):,}")
    print(f"    mean ratio:   {pnl.mean():+.4f}  ({pnl.mean()*100:+.2f}%)")
    print(f"    median:       {np.median(pnl):+.4f}")
    print(f"    win rate:     {(pnl>0).mean()*100:.1f}%")
    print(f"    TP captures:  {fires_n_pos:>4d}/{len(pnl)} ({fires_n_pos*100/len(pnl):.1f}%)")
    print(f"    SOL/bet:      {sol.mean():+.5f}")
    print(f"    total SOL:    {sol.sum():+.4f}")


def main():
    print(f"=== HGB vs ARF50 modeled-P&L (level_tp_100) @ "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

    k7  = _load(["_fresh","_snap1"], "K7")
    v05 = _load(["_fresh","_snap1"], "V05")
    v05 = v05.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                               if c.removesuffix("_v") in v05.columns})
    df = k7.merge(v05[["mint"]+CLASSIC_V], on="mint", how="inner")\
            .drop_duplicates(subset=["mint"], keep="last")
    target_col = next(c for c in ("peak_ret","peak_ret_v","peak_2x") if c in df.columns)
    term_col   = "terminal_ret" if "terminal_ret" in df.columns else target_col
    df = df[["mint", target_col, term_col] + CLASSIC].copy()
    soph = pd.read_parquet(ROOT/"data/sophistication_current.parquet")
    soph_cols = [c for c in SOPH if c in soph.columns]
    soph = soph[["mint"]+soph_cols].drop_duplicates(subset=["mint"], keep="last")
    wide = df.merge(soph, on="mint", how="inner")

    y = (wide[target_col] >= 1.0).astype(int).values
    idx = np.arange(len(wide))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20,
                                       random_state=42, stratify=y)
    X = wide[WIDE].values
    peak = wide[target_col].values
    term = wide[term_col].values
    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    peak_te, term_te = peak[idx_te], term[idx_te]
    print(f"  test rows: {len(idx_te):,}   bet_sol={BET_SOL}   fees_ratio={FEES_RATIO}")

    # HGB
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=42)
    hgb.fit(Xtr, ytr)
    s_hgb = hgb.predict_proba(Xte)[:, 1]

    # ARF50
    print(f"  training ARF50 (~130s)...")
    from river import forest
    arf = forest.ARFClassifier(n_models=50, seed=42)
    for i in range(len(Xtr)):
        x = {WIDE[j]: float(Xtr[i, j]) if not np.isnan(Xtr[i, j]) else 0.0
             for j in range(len(WIDE))}
        arf.learn_one(x, int(ytr[i]))
    s_arf = np.zeros(len(Xte))
    for i in range(len(Xte)):
        x = {WIDE[j]: float(Xte[i, j]) if not np.isnan(Xte[i, j]) else 0.0
             for j in range(len(WIDE))}
        p = arf.predict_proba_one(x)
        s_arf[i] = p.get(1, p.get(True, 0.5))

    THR = 0.4134
    fire_hgb = s_hgb >= THR
    fire_arf = s_arf >= THR
    fire_both = fire_hgb & fire_arf
    fire_either = fire_hgb | fire_arf

    print(f"\n=== fires by model at threshold {THR} ===")
    pnl_hgb = _level_tp_100_pnl(peak_te[fire_hgb], term_te[fire_hgb])
    pnl_arf = _level_tp_100_pnl(peak_te[fire_arf], term_te[fire_arf])
    pnl_both = _level_tp_100_pnl(peak_te[fire_both], term_te[fire_both])
    pnl_either = _level_tp_100_pnl(peak_te[fire_either], term_te[fire_either])

    _stats("HGB fires",
           pnl_hgb,
           int((peak_te[fire_hgb] >= 1.0).sum()))
    print()
    _stats("ARF50 fires",
           pnl_arf,
           int((peak_te[fire_arf] >= 1.0).sum()))
    print()
    _stats("BOTH-agree fires (intersection)",
           pnl_both,
           int((peak_te[fire_both] >= 1.0).sum()))
    print()
    _stats("EITHER-fire (union)",
           pnl_either,
           int((peak_te[fire_either] >= 1.0).sum()))

    # Direct head-to-head on the same number of fires per model? Probably
    # not necessary — main question is "would they make the same money"
    # at the same threshold (they pick different mints, different counts).

    # Disagreements: where each model alone fires
    fire_only_hgb = fire_hgb & ~fire_arf
    fire_only_arf = fire_arf & ~fire_hgb
    if fire_only_hgb.sum():
        pnl_h_only = _level_tp_100_pnl(peak_te[fire_only_hgb], term_te[fire_only_hgb])
        print(f"\n  HGB-only fires (ARF wouldn't): n={fire_only_hgb.sum()}  "
              f"mean ratio={pnl_h_only.mean():+.4f}  "
              f"TP rate={(peak_te[fire_only_hgb] >= 1.0).mean()*100:.1f}%  "
              f"SOL contribution={pnl_h_only.sum()*BET_SOL:+.4f}")
    if fire_only_arf.sum():
        pnl_a_only = _level_tp_100_pnl(peak_te[fire_only_arf], term_te[fire_only_arf])
        print(f"  ARF-only fires (HGB wouldn't): n={fire_only_arf.sum()}  "
              f"mean ratio={pnl_a_only.mean():+.4f}  "
              f"TP rate={(peak_te[fire_only_arf] >= 1.0).mean()*100:.1f}%  "
              f"SOL contribution={pnl_a_only.sum()*BET_SOL:+.4f}")

    print(f"\n=== summary ===")
    print(f"  Total SOL on OOS test set ({len(idx_te)} mints), 0.1 SOL/bet:")
    print(f"    HGB   alone: {pnl_hgb.sum()*BET_SOL:+.4f} SOL  ({len(pnl_hgb)} fires)")
    print(f"    ARF50 alone: {pnl_arf.sum()*BET_SOL:+.4f} SOL  ({len(pnl_arf)} fires)")
    print(f"    BOTH agree:  {pnl_both.sum()*BET_SOL:+.4f} SOL  ({len(pnl_both)} fires)")
    print(f"    EITHER fires:{pnl_either.sum()*BET_SOL:+.4f} SOL  ({len(pnl_either)} fires)")


if __name__ == "__main__":
    main()
