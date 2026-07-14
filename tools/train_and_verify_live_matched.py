"""train_and_verify_live_matched.py — train the 22 K+V entry head on the
live-matched training set and VERIFY that its live score distribution aligns
with training (the gate that the earlier models failed).

Inputs:
  data/live_matched_k5.parquet   (from extract_live_matched.py: 22 feats + peak_ret)
  bot_data/shadow_run.jsonl      (live decisions; current-era only for the check)

Outputs:
  bot_artifacts_K7V_tp50_k5_livematched/{entry_model.pkl, recovery_model.pkl, model_spec.json}
  (recovery_model.pkl copied from the _nosoph build; recovery head is not implicated.)

Does NOT touch the production symlink or the running bot.
"""
from __future__ import annotations
import json, pickle, time, datetime, shutil
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/the-distribution-will-manifest")
CK = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol","single_actor_share",
      "trades_per_sec","entry_sol","win_drawup","win_drawdown"]
FEATS = CK + [f"{c}_v" for c in CK]   # 22, matches feature_accum.ENTRY_FEATURE_NAMES order
RESTART = datetime.datetime(2026,6,9,10,52,20,tzinfo=datetime.timezone.utc).timestamp()  # K=5 era


def buckets(score, peak, label):
    print(f"\n  [{label}] win-rate-by-score-bucket (win = peak_ret >= 0.5):")
    print(f"    {'bucket':>9s} {'n':>7s} {'win_rate':>9s} {'mean_peak':>10s}")
    edges=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.01]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(score>=lo)&(score<hi); n=int(m.sum())
        if not n: continue
        print(f"    {lo:.1f}-{min(hi,1.0):.1f} {n:>7d} {100*(peak[m]>=0.5).mean():>8.1f}% {np.nanmean(peak[m]):>+10.2f}")


def load_live_features():
    X=[];
    for ln in open(ROOT/"bot_data/shadow_run.jsonl"):
        if '"entry_decision"' not in ln: continue
        try: e=json.loads(ln)
        except: continue
        if e.get("kind")!="entry_decision" or (e.get("t") or 0)<RESTART: continue
        f=e.get("features") or {}
        X.append([f.get(c) if f.get(c) is not None else np.nan for c in FEATS])
    return np.array(X,dtype=float)


def main():
    df=pd.read_parquet(ROOT/"data/live_matched_k5.parquet").drop_duplicates("mint",keep="last")
    peak=df["peak_ret"].values.astype(float)
    y=(peak>=0.5).astype(int)
    X=df[FEATS].values
    print(f"=== live-matched training set: {len(df):,} mints | peak>=0.5 base rate {y.mean():.3f} ===")
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    REG=dict(max_iter=150,max_depth=3,learning_rate=0.05,l2_regularization=5.0,random_state=42)
    cvauc=cross_val_score(HistGradientBoostingClassifier(**REG),X,y,
                          cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring="roc_auc")
    print(f"  5-fold shuffled CV AUC (regularized) = {cvauc.mean():.3f} +- {cvauc.std():.3f}")
    order=np.argsort(df["first_slot"].values,kind="mergesort")
    cut=int(len(order)*0.8); tr,te=order[:cut],order[cut:]
    clf=HistGradientBoostingClassifier(**REG).fit(X[tr],y[tr])
    s_tr=clf.predict_proba(X[tr])[:,1]; s_te=clf.predict_proba(X[te])[:,1]
    print(f"  chrono split train={len(tr):,} OOS={len(te):,} | train AUC={roc_auc_score(y[tr],s_tr):.4f} OOS AUC={roc_auc_score(y[te],s_te):.4f}")
    peak_te=peak[te]
    buckets(s_te, peak_te, "OOS (training holdout)")

    # ----- ALIGNMENT CHECK: live current-era decisions scored by this model -----
    Xlive=load_live_features()
    s_live=clf.predict_proba(Xlive)[:,1] if len(Xlive) else np.array([])
    print(f"\n=== ALIGNMENT: {len(Xlive)} live current-era decisions scored ===")
    def q(a): return " ".join(f"p{p}={np.percentile(a,p):.3f}" for p in (50,90,95,99,100))
    print(f"  training OOS scores: {q(s_te)}")
    print(f"  live scores:         {q(s_live)}")
    print("  high-band mass (frac of pop with score in band):")
    for lo in (0.5,0.7,0.8,0.9):
        print(f"    >= {lo}:  OOS {100*(s_te>=lo).mean():5.2f}%   live {100*(s_live>=lo).mean():5.2f}%")

    # ----- threshold selection: fire often at high precision (on OOS) + live fire-rate -----
    print("\n  threshold -> OOS precision / OOS fire-rate / LIVE fire-rate:")
    print(f"    {'thr':>7s} {'OOSprec':>8s} {'OOSfire%':>9s} {'LIVEfire%':>10s}")
    chosen=None
    for thr in np.round(np.arange(0.30,0.96,0.05),2):
        m=s_te>=thr
        if m.sum()<20: continue
        prec=(y[te][m]).mean(); oosfire=100*m.mean(); livefire=100*(s_live>=thr).mean() if len(s_live) else float("nan")
        print(f"    {thr:>7.2f} {prec*100:>7.1f}% {oosfire:>8.2f}% {livefire:>9.2f}%")
        # modest-edge target: precision clearly above base rate, firing often enough to learn
        if chosen is None and prec>=0.40 and livefire>=3.0:
            chosen=(thr,prec,oosfire,livefire)
    if chosen:
        print(f"\n  PICK: thr={chosen[0]:.2f}  OOS precision {chosen[1]*100:.1f}%  OOS fires {chosen[2]:.2f}%  LIVE fires {chosen[3]:.2f}%")
    else:
        print("\n  no thr clears prec>=40% & live>=3%; will fall back to OOS top-5% as a provisional shadow threshold")

    # save artifact (entry head); reuse recovery from the _nosoph build
    OUT=ROOT/"bot_artifacts_K7V_tp50_k5_livematched"; OUT.mkdir(parents=True,exist_ok=True)
    pickle.dump(clf,open(OUT/"entry_model.pkl","wb"))
    rec_src=ROOT/"bot_artifacts_K7V_tp50_k5_nosoph/recovery_model.pkl"
    if rec_src.exists(): shutil.copy(rec_src, OUT/"recovery_model.pkl")
    thr=float(chosen[0]) if chosen else float(np.quantile(s_te,0.95))
    spec={"sklearn_version":sklearn.__version__,
          "entry":{"features":FEATS,"target":"peak_ret>=0.5 (+50%, K=5 anchored)",
                   "fire_if":"predict_proba[:,1] >= entry_threshold",
                   "entry_threshold_top_decile":thr,"trigger":"K=5 AND V=0.5 joint","K_WINDOW":5},
          "recovery":{"features":["ret","run_max_ret","dd","fill_k","buy_frac_w","nsell_w","solo_sell_w","vel_w","dts"]+CK,
                      "death_cut_threshold":0.10},
          "exit_policy":"level_tp_50",
          "fix":"live-matched extraction (feature_accum over capture, classic+fresh gates), 22 K+V, no soph",
          "trained_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    (OUT/"model_spec.json").write_text(json.dumps(spec,indent=2))
    print(f"\n  saved {OUT} (entry thr={thr:.4f}). Symlink untouched.")


if __name__=="__main__":
    main()
