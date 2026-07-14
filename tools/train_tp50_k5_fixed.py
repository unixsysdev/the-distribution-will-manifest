"""train_tp50_k5_fixed.py — corrected TP50 / K=5 training + honest bucket diagnostic.

Run offline. Writes NEW artifact dirs. Does NOT touch the production symlink
(bot_artifacts_K7V) or the running bot.

Fixes vs tools/train_tp50_k5.py:
  * loads data/sophistication_k5.parquet (K=5), NOT sophistication_current.parquet (K=7)
  * LEFT-joins soph and leaves NaN (native HGB routing) instead of inner-join survivor bias
  * KEEPS the K-intersect-V join: that matches the live joint trigger
    (shadow_harness fires only on 'ready' = K=5 AND V=0.5). It is not survivor bias.
  * chronological OOS split by first_slot (not random) for an honest forward read
  * trains TWO entry heads and runs win-rate-by-score-bucket on OOS:
       NOSOPH : 22 K+V features (byte-parity train==live, proven 29181/29181)
       SOPHK5 : 31 features (22 K+V + 9 soph_k5, native NaN)  [offline-optimistic]
  * graveyard labeling: peak_ret NaN -> target 0 (no-op on K-int-V pop; kept explicit)
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/the-distribution-will-manifest")

CLASSIC_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
             "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
CLASSIC_V = [f"{c}_v" for c in CLASSIC_K]
CLASSIC = CLASSIC_K + CLASSIC_V
SOPH = ["soph_fee_p50_lam","soph_fee_p90_lam","soph_cu_p50","soph_cu_mean",
        "soph_jito_tip_rate","soph_jito_tip_p50_lam","soph_routed_rate",
        "soph_n_inner_ix_mean","soph_n_keys_mean"]
WIDE = CLASSIC + SOPH
PATH = ["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w","dts"]


def bucket_diag(score, peak, label):
    print(f"\n  [{label}] win-rate-by-score-bucket (OOS, win = peak_ret >= 0.5):")
    print(f"    {'bucket':>9s} {'n':>7s} {'win_rate':>9s} {'mean_peak':>10s}")
    edges = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m = (score>=lo)&(score<hi)
        n=int(m.sum())
        if not n: continue
        wr=float((peak[m]>=0.5).mean()); mp=float(np.nanmean(peak[m]))
        hh = 1.0 if hi>1 else hi
        print(f"    {lo:.1f}-{hh:.1f} {n:>7d} {wr*100:>8.1f}% {mp:>+10.2f}")


def prec_sweep(score, y, peak, label):
    print(f"\n  [{label}] OOS precision vs fire-rate sweep:")
    print(f"    {'fire %':>7s} {'cutoff':>9s} {'n':>7s} {'precision':>10s} {'mean_peak':>10s}")
    for pct in (0.5,1,2,3,5,10,15,20,25,30):
        cut=float(np.quantile(score,1-pct/100)); m=score>=cut; nf=int(m.sum())
        if not nf: continue
        print(f"    {pct:>6.1f}% {cut:>9.4f} {nf:>7d} {y[m].mean()*100:>9.1f}% {np.nanmean(peak[m]):>+10.2f}")


def train_entry(X, y, idx_tr, idx_te, peak_te, label):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
            max_depth=None, l2_regularization=1.0, random_state=42)
    clf.fit(X[idx_tr], y[idx_tr])
    s_tr=clf.predict_proba(X[idx_tr])[:,1]; s_te=clf.predict_proba(X[idx_te])[:,1]
    print(f"\n=== {label} ===")
    print(f"  features={X.shape[1]}  train AUC={roc_auc_score(y[idx_tr],s_tr):.4f}  "
          f"OOS AUC={roc_auc_score(y[idx_te],s_te):.4f}")
    bucket_diag(s_te, peak_te, label)
    prec_sweep(s_te, y[idx_te], peak_te, label)
    return clf, s_te


def main():
    print(f"=== train_tp50_k5_fixed @ {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} ===")
    k = pd.read_parquet(ROOT/"data/pumpfun_continuation_K7_k5_snap1/token_level.parquet").drop_duplicates("mint",keep="last")
    v = pd.read_parquet(ROOT/"data/pumpfun_continuation_V05_k5_snap1/token_level.parquet").drop_duplicates("mint",keep="last")
    v = v.rename(columns={c.removesuffix("_v"): c for c in CLASSIC_V if c.removesuffix("_v") in v.columns})
    df = k.merge(v[["mint"]+CLASSIC_V], on="mint", how="inner")   # K-int-V == live joint trigger
    print(f"  K=5={len(k):,}  V05={len(v):,}  K-int-V (joint-trigger pop)={len(df):,}")

    soph = pd.read_parquet(ROOT/"data/sophistication_k5.parquet").drop_duplicates("mint",keep="last")
    soph = soph[["mint"]+[c for c in SOPH if c in soph.columns]]
    wide = df.merge(soph, on="mint", how="left")               # LEFT join, keep NaN
    cov = wide[SOPH[0]].notna().mean()
    print(f"  soph_k5 LEFT-join: {len(wide):,} mints, soph fee present {cov:.3f}, all-NaN {1-cov:.3f}")

    peak = wide["peak_ret"].values.astype(float)
    y = (wide["peak_ret"]>=0.5).astype(int).to_numpy().copy()
    y[np.isnan(peak)] = 0   # graveyard: missing target = failed token
    print(f"  positive rate (peak>=+50%): {y.mean():.3f}")

    order = np.argsort(wide["first_slot"].values, kind="mergesort")   # chronological
    cut = int(len(order)*0.8)
    idx_tr, idx_te = order[:cut], order[cut:]
    print(f"  chrono split: train={len(idx_tr):,} (early)  OOS={len(idx_te):,} (late)  "
          f"train_pos={y[idx_tr].mean():.3f}  OOS_pos={y[idx_te].mean():.3f}")
    peak_te = peak[idx_te]

    Xc = wide[CLASSIC].values
    Xw = wide[WIDE].values
    clf_nosoph, _ = train_entry(Xc, y, idx_tr, idx_te, peak_te, "NOSOPH (22 K+V)")
    clf_soph,   _ = train_entry(Xw, y, idx_tr, idx_te, peak_te, "SOPHK5 (31 K+V+soph_k5, native NaN)")

    print("\n=== RECOVERY HEAD (shared; PATH + K, no soph) ===")
    sk = pd.read_parquet(ROOT/"data/pumpfun_continuation_K7_k5_snap1/path_snapshots.parquet")
    tk = wide[["mint"]+CLASSIC_K].copy()
    tk = tk.merge(k[["mint","terminal_ret"]].drop_duplicates("mint",keep="last"), on="mint", how="left")
    s2 = sk.sort_values(["mint","fwd"]).copy()
    tk_u = tk.drop_duplicates("mint",keep="last")
    s2["term"] = s2["mint"].map(tk_u.set_index("mint")["terminal_ret"])
    def suf(g):
        r=g["ret"].values; f=np.empty(len(r)); run=g["term"].iloc[0]
        for i in range(len(r)-1,-1,-1):
            f[i]=run
            if r[i]>run: run=r[i]
        return pd.Series(f,index=g.index)
    s2["fm"]=s2.groupby("mint",group_keys=False).apply(suf)
    dd=s2[s2.ret<0].merge(tk[["mint"]+CLASSIC_K],on="mint",how="left").dropna(subset=PATH+CLASSIC_K)
    Xr=dd[PATH+CLASSIC_K].values; yr=(dd["fm"]>=0).astype(int).values
    clf_r=HistGradientBoostingClassifier(max_depth=3,max_iter=250,learning_rate=0.05,
            l2_regularization=1.0,random_state=42).fit(Xr,yr)
    print(f"  recovery rows={len(dd):,}  train AUC={roc_auc_score(yr,clf_r.predict_proba(Xr)[:,1]):.4f}")

    def save(outname, clf_e, feats, oos_scores):
        OUT=ROOT/outname; OUT.mkdir(parents=True,exist_ok=True)
        pickle.dump(clf_e,open(OUT/"entry_model.pkl","wb"))
        pickle.dump(clf_r,open(OUT/"recovery_model.pkl","wb"))
        thr=float(np.quantile(oos_scores,0.98))
        spec={"sklearn_version":sklearn.__version__,
              "entry":{"features":feats,"target":"peak_ret>=0.5 (+50%, K=5)",
                       "fire_if":"predict_proba[:,1] >= entry_threshold",
                       "entry_threshold_top_decile":thr,"trigger":"K=5 AND V=0.5 joint",
                       "K_WINDOW":5},
              "recovery":{"features":PATH+CLASSIC_K,"death_cut_threshold":0.10},
              "exit_policy":"level_tp_50",
              "fix":"soph_k5 left-join + native NaN; K-int-V kept (matches live joint trigger); chrono split",
              "trained_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
        (OUT/"model_spec.json").write_text(json.dumps(spec,indent=2))
        print(f"  saved {OUT}  thr(top2%)={thr:.4f}")
    save("bot_artifacts_K7V_tp50_k5_nosoph", clf_nosoph, CLASSIC, clf_nosoph.predict_proba(Xc[idx_te])[:,1])
    save("bot_artifacts_K7V_tp50_k5_sophk5", clf_soph,   WIDE,    clf_soph.predict_proba(Xw[idx_te])[:,1])
    print("\nDONE. No production symlink or running service was modified.")


if __name__=="__main__":
    main()
