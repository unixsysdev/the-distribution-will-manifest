"""K-window sweep for TP50 model. Trains both:
  - LOOSE: peak_ret >= 0.5 ever (the training-set label we used)
  - CLEAN: peak_ret >= 0.5 AND min_ret_before_peak >= floor (execution-realistic)

For each K, reports OOS AUC, positive base rate, precision sweep, and
explicitly FLAGS if OOS AUC drops below 0.75 (signal too noisy for live).

Usage: python train_sweep_K.py 5 7 8 9 10
"""
from __future__ import annotations
import json, pickle, time, sys
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
CLEAN_FLOOR = -0.15


def k_dir(K: int) -> tuple[Path, Path]:
    """Return (K-features dir, V-features dir) for the given K window."""
    if K == 7:
        return ROOT / "data/pumpfun_continuation_K7_snap1", ROOT / "data/pumpfun_continuation_V05_snap1"
    return (ROOT / f"data/pumpfun_continuation_K7_k{K}_snap1",
            ROOT / f"data/pumpfun_continuation_V05_k{K}_snap1")


def min_ret_before_50(snaps: pd.DataFrame) -> pd.DataFrame:
    snaps = snaps.sort_values(["mint","fwd"]).reset_index(drop=True)
    rows = []
    for mint, g in snaps.groupby("mint", sort=False):
        rets = g["ret"].values
        if (rets >= 0.5).any():
            hit = int(np.argmax(rets >= 0.5))
            min_pre = float(rets[:hit+1].min())
            hit_50 = True
        else:
            min_pre = float(rets.min())
            hit_50 = False
        rows.append({"mint": mint, "min_ret_before_50": min_pre, "hit_50": hit_50})
    return pd.DataFrame(rows)


def train_one(K: int) -> dict:
    kd, vd = k_dir(K)
    if not kd.exists():
        return {"K": K, "status": "NO_DATA", "msg": f"missing {kd}"}
    print(f"\n========================== K={K} ==========================")
    k7 = pd.read_parquet(kd / "token_level.parquet")
    v  = pd.read_parquet(vd / "token_level.parquet")
    snaps = pd.read_parquet(kd / "path_snapshots.parquet")
    print(f"  K-features: {len(k7):,}  V-features: {len(v):,}  snaps: {len(snaps):,}")
    v = v.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V
                           if c.removesuffix("_v") in v.columns})
    df = k7.merge(v[["mint"]+CLASSIC_V], on="mint", how="inner").drop_duplicates(subset=["mint"], keep="last")
    soph = pd.read_parquet(ROOT/"data/sophistication_current.parquet")
    soph = soph[["mint"]+[c for c in SOPH if c in soph.columns]].drop_duplicates(subset=["mint"], keep="last")
    wide = df.merge(soph, on="mint", how="inner")
    md = min_ret_before_50(snaps)
    wide = wide.merge(md, on="mint", how="left")
    wide = wide.dropna(subset=WIDE+["min_ret_before_50"]).reset_index(drop=True)
    print(f"  trainable rows (K+V+soph+path): {len(wide):,}")

    target = "peak_ret"
    y_loose = (wide[target] >= 0.5).astype(int).values
    y_clean = ((wide[target] >= 0.5) & (wide["min_ret_before_50"] >= CLEAN_FLOOR)).astype(int).values
    print(f"  positive rate loose:  {y_loose.mean():.3f}")
    print(f"  positive rate clean@{CLEAN_FLOOR}: {y_clean.mean():.3f}")

    X = wide[WIDE].values
    idx_tr, idx_te = train_test_split(np.arange(len(wide)), test_size=0.20,
                                       random_state=42, stratify=y_clean)
    out = {"K": K, "n_rows": int(len(wide)),
           "positive_rate_loose": float(y_loose.mean()),
           "positive_rate_clean": float(y_clean.mean()),
           "models": {}}

    for label_name, y in (("loose", y_loose), ("clean", y_clean)):
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=42)
        clf.fit(X[idx_tr], y[idx_tr])
        s_tr = clf.predict_proba(X[idx_tr])[:,1]
        s_te = clf.predict_proba(X[idx_te])[:,1]
        auc_in = roc_auc_score(y[idx_tr], s_tr)
        auc_oos = roc_auc_score(y[idx_te], s_te)
        peak_te = wide["peak_ret"].values[idx_te]
        mp_te = wide["min_ret_before_50"].values[idx_te]
        sweep = []
        for pct in (0.5, 1, 2, 3, 5, 10, 15):
            cut = float(np.quantile(s_te, 1 - pct/100))
            m = s_te >= cut
            nf = int(m.sum())
            if nf < 5: continue
            prec = float(y[idx_te][m].mean())
            prec_clean = float(y_clean[idx_te][m].mean())
            prec_loose = float(y_loose[idx_te][m].mean())
            mp = float(peak_te[m].mean())
            mm = float(mp_te[m].mean())
            sweep.append({"pct": pct, "cutoff": cut, "n": nf,
                          "prec_self": prec, "prec_clean": prec_clean,
                          "prec_loose": prec_loose,
                          "mean_peak": mp, "mean_min_pre": mm})
        out["models"][label_name] = {"auc_in": float(auc_in), "auc_oos": float(auc_oos),
                                      "sweep": sweep, "flag_low_auc": auc_oos < 0.75}
        print(f"\n  [{label_name}] train_AUC={auc_in:.4f}  OOS_AUC={auc_oos:.4f}"
              + ("  *** FLAGGED: OOS < 0.75 ***" if auc_oos < 0.75 else ""))
        print(f"    {'pct':>5s} {'cut':>7s} {'n':>5s} {'prec_self':>10s} {'prec_clean':>10s} {'prec_loose':>10s} {'mean_peak':>10s} {'mean_min':>9s}")
        for s in sweep:
            print(f"    {s['pct']:>4.1f}% {s['cutoff']:>7.4f} {s['n']:>5d} "
                  f"{s['prec_self']*100:>9.1f}% {s['prec_clean']*100:>9.1f}% "
                  f"{s['prec_loose']*100:>9.1f}% {s['mean_peak']:>+10.2f} {s['mean_min_pre']:>+9.2f}")
    return out


def main(Ks):
    print(f"=== K-sweep @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    results = []
    for K in Ks:
        results.append(train_one(K))
    print("\n\n=== SWEEP SUMMARY ===")
    print(f"{'K':>4s} {'n_rows':>8s} {'pos_loose':>9s} {'pos_clean':>9s} "
          f"{'loose_OOS':>10s} {'clean_OOS':>10s} {'top5_clean_prec':>16s} {'top5_mean_peak':>15s} {'flag':>10s}")
    for r in results:
        if r.get("status") == "NO_DATA":
            print(f"{r['K']:>4d}  NO_DATA  ({r['msg']})")
            continue
        lm = r["models"]["loose"]; cm = r["models"]["clean"]
        flags = []
        if lm["flag_low_auc"]: flags.append("loose<.75")
        if cm["flag_low_auc"]: flags.append("clean<.75")
        top5_clean = next((s for s in cm["sweep"] if s["pct"] == 5), None)
        top5_prec = f"{top5_clean['prec_self']*100:.1f}%" if top5_clean else "-"
        top5_peak = f"{top5_clean['mean_peak']:+.2f}" if top5_clean else "-"
        print(f"{r['K']:>4d} {r['n_rows']:>8d} {r['positive_rate_loose']:>9.3f} "
              f"{r['positive_rate_clean']:>9.3f} {lm['auc_oos']:>10.4f} {cm['auc_oos']:>10.4f} "
              f"{top5_prec:>16s} {top5_peak:>15s}  {','.join(flags) if flags else 'ok':>10s}")
    json.dump(results, open(ROOT/"tools/sweep_K_results.json","w"), indent=2)
    print(f"\nResults JSON: tools/sweep_K_results.json")


if __name__ == "__main__":
    Ks = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 7]
    main(Ks)
