#!/usr/bin/env python3
"""Train + evaluate the AUGMENTED 2x-continuation model and ask the one question:
do the firehose-only features (per-mint failed-buy pressure + real-time congestion)
push the continuation selection BEYOND the 0.614 by-coin / 0.595 by-day wall, on a
metric that matters (incremental AUC AND net-selection PnL), without leakage.

Rigor:
  - temporal (by-day) split is the headline; a coin-random split is secondary.
  - leaky/realized columns (entry_slip, fill_mid, cross_mid, ret, dur_s) are NEVER
    model inputs (entry_slip/ret are only known after committing) -> diagnostics only.
  - nulls: shuffle-y (sanity) AND comp-shuffle (shuffle ONLY the new ff_/cong_ cols
    across rows, keep RICH intact) = proves the lift is real signal, not extra capacity.
  - executable metric: top-decile net-selection PnL (realistic fill via entry_slip,
    revert if slip>CAP, real costs) for baseline vs augmented.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np

RICH = ["dd","buy_frac","ntr","recent","tps","uniq","t_to_2x","log_t_to_2x","accel","last_gap",
        "mcap_sol","vol_sol","sol_per_trade","max_buy_sol","whale_frac","net_flow","n_buyers",
        "n_sellers","bs_ratio","signer_conc","up_frac","max_runup"]
FF   = ["ff_nfail","ff_nfail_signers","ff_fail_rate","ff_fail_prio_mean","ff_fail_prio_max","ff_fail_recent5s"]
CONG = ["cong_exec_prior10","cong_exec_last","cong_entries_prior10","cong_slot_partial",
        "cong_pumptx_1s","cong_pumptx_5s","cong_failtx_1s","cong_failtx_5s"]
SHRED = ["shred_nbuy","shred_nbuy_5slot","shred_uniq_signers","shred_prio_p90","shred_prio_max",
         "shred_tip_rate","shred_tip_max","shred_nslots","shred_maxperslot"]
REP  = ["rep_mean","rep_max","rep_nknown","rep_frac_known","rep_frachigh","rep_nsmart"]
REP_SHUF = ["rep_shuf_mean","rep_shuf_max","rep_shuf_nknown","rep_shuf_frac_known","rep_shuf_frachigh","rep_shuf_nsmart"]
NEW  = FF + CONG + SHRED + REP   # rep_shuf_* deliberately NOT in NEW (used only for the wallet-id null)
LEAKY = {"entry_slip","fill_mid","cross_mid","ret","dur_s","y","mint","cross_t","cross_slot"}

BET = 0.1; PUMP_RT = 0.02; FIXED_RT = 0.00161; CAP = 0.25   # 2% roundtrip, fixed, 2500bps slip cap
ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=f"{ROOT}/bot_data/cont_2x_aug2_panel.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/bot_data/cont_2x_report.json")
    ap.add_argument("--test-frac", type=float, default=0.30)
    return ap.parse_args()


def load(glob_pat):
    import pandas as pd
    rows = []
    for fn in sorted(glob.glob(glob_pat)):
        for ln in open(fn):
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return pd.DataFrame(rows)


def net_topk(te, score, k=0.10):
    n = len(te); order = np.argsort(-score)[: max(1, int(n * k))]
    sub = te.iloc[order]
    filled = sub["entry_slip"] <= CAP
    rr = sub["ret"].to_numpy()
    nf = BET * rr - PUMP_RT * BET - FIXED_RT
    nf = np.where(filled.to_numpy(), nf, 0.0)          # reverted fills: no position (~free)
    return float(nf.mean()), float(nf.sum()), float(filled.mean()), int(len(sub)), float(sub["y"].mean())


def main():
    a = parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance
    df = load(a.panels)
    sys.stderr.write(f"[cont-train] loaded {len(df)} crosses, win={df['y'].mean():.3f}\n")
    df = df.dropna(subset=["y"]).sort_values("cross_t").reset_index(drop=True)
    n = len(df); cut = int(n * (1 - a.test_frac))
    tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    feats_all = [c for c in (RICH + NEW) if c in df.columns]
    nzv = [c for c in feats_all if tr[c].std() > 0]
    rich = [c for c in RICH if c in nzv]; new = [c for c in NEW if c in nzv]
    dropped = [c for c in feats_all if c not in nzv]

    def fit(feats, label_shuffle=False, comp_shuffle=False):
        Xtr = tr[feats].to_numpy(float).copy(); Xte = te[feats].to_numpy(float).copy()
        ytr = tr["y"].to_numpy(int)
        rng = np.random.default_rng(0)
        if label_shuffle:
            ytr = rng.permutation(ytr)
        if comp_shuffle:  # shuffle ONLY the NEW columns (break their row-alignment), keep RICH
            for j, c in enumerate(feats):
                if c in NEW:
                    Xtr[:, j] = rng.permutation(Xtr[:, j]); Xte[:, j] = rng.permutation(Xte[:, j])
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=350, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0, early_stopping=True)
        clf.fit(Xtr, ytr)
        s = clf.predict_proba(Xte)[:, 1]
        return clf, s, roc_auc_score(te["y"], s)

    print("="*80)
    print(f"AUGMENTED 2x-CONTINUATION  n={n} train={len(tr)} test={len(te)} win={df['y'].mean():.3f}")
    print(f"RICH={len(rich)} NEW(ff+cong)={len(new)}  dropped_zerovar={dropped}")
    print(f"  baseline-to-beat: deployed lean OOS AUC 0.614 by-coin / 0.595 by-day")
    print("="*80)
    _, s_b, auc_b = fit(rich)
    clf_a, s_a, auc_a = fit(rich + new)
    _, s_n, auc_n = fit(new)
    _, _, auc_sh = fit(rich + new, label_shuffle=True)
    _, _, auc_cs = fit(rich + new, comp_shuffle=True)
    print(f"\nTEMPORAL (by-time) OOS AUC:")
    print(f"  RICH baseline      = {auc_b:.4f}")
    print(f"  RICH + ff + cong   = {auc_a:.4f}   (Δ vs RICH = {auc_a-auc_b:+.4f})")
    print(f"  ff + cong ONLY     = {auc_n:.4f}")
    print(f"  null: shuffle-y    = {auc_sh:.4f}")
    print(f"  null: comp-shuffle = {auc_cs:.4f}   (augmented should beat THIS to be real)")
    # executable net-selection
    print(f"\nTOP-DECILE NET-SELECTION (BET={BET} SOL, {PUMP_RT*100:.0f}% rt, cap {CAP*100:.0f}%):")
    for nm, s in [("RICH", s_b), ("RICH+ff+cong", s_a)]:
        mnet, tnet, fill, ns, wy = net_topk(te, s, 0.10)
        print(f"  {nm:14s}: net/sel {mnet:+.4f}  total {tnet:+.3f}  fill {fill:.0%}  win {wy:.0%}  n {ns}")

    # WALLET-IDENTITY NULL: swap real rep_* for permuted rep_shuf_* (same set sizes &
    # rep-value distribution; cross->its-own-wallets link broken). If this collapses toward
    # comp-shuffle while real rep gave auc_a, the reputation edge is REAL wallet skill.
    rep_in = [c for c in REP if c in nzv]
    repshuf_in = [c for c in REP_SHUF if c in df.columns and tr[c].std() > 0]
    new_rs = [c for c in new if c not in rep_in] + repshuf_in
    _, s_rs, auc_rs = fit(rich + new_rs)
    print(f"\nWALLET-IDENTITY NULL (real rep -> permuted rep, {len(repshuf_in)} shuf feats):")
    print(f"  RICH+shred+REP_SHUF = {auc_rs:.4f}   vs  real={auc_a:.4f}  comp-shuffle={auc_cs:.4f}  RICH={auc_b:.4f}")
    print(f"  net/sel permuted-rep {net_topk(te,s_rs,0.10)[0]:+.4f}  vs real {net_topk(te,s_a,0.10)[0]:+.4f}")

    # BLOCK-BOOTSTRAP: CI on (augmented AUC - RICH AUC) and augmented top-decile net/sel.
    yte = te["y"].to_numpy(int); slip = te["entry_slip"].to_numpy(); rety = te["ret"].to_numpy()
    nte = len(yte); bs = max(1, nte // 40); idxall = np.arange(nte)
    rng = np.random.default_rng(7); dA = []; nets = []
    for _ in range(400):
        starts = rng.integers(0, nte, size=40)
        ii = np.concatenate([idxall[st:st + bs] for st in starts]); ii = ii[ii < nte]
        ya = yte[ii]
        if ya.sum() == 0 or ya.sum() == len(ya):
            continue
        dA.append(roc_auc_score(ya, s_a[ii]) - roc_auc_score(ya, s_b[ii]))
        od = np.argsort(-s_a[ii])[:max(1, len(ii) // 10)]; k = ii[od]; fl = slip[k] <= CAP
        nf = BET * rety[k] - PUMP_RT * BET - FIXED_RT
        nets.append(float(np.where(fl, nf, 0.0).mean()))
    qa = np.percentile(dA, [5, 50, 95]); qn = np.percentile(nets, [5, 50, 95])
    print(f"\nBLOCK-BOOTSTRAP (400x, 40 time-blocks):")
    print(f"  AUC(aug)-AUC(RICH): p05={qa[0]:+.4f} p50={qa[1]:+.4f} p95={qa[2]:+.4f}  (p05>0 => robust lift)")
    print(f"  aug top-decile net/sel: p05={qn[0]:+.4f} p50={qn[1]:+.4f} p95={qn[2]:+.4f}  (p05>0 => robust net)")

    # REDUNDANCY ablation: marginal value of each family (drop-one-family-out).
    # Per user rule: a family is dropped ONLY if its marginal is ~0 AND others cover it.
    full = rich + new
    fams = {"RICH": rich,
            "FAILED": [c for c in FF if c in nzv],
            "CONG": [c for c in CONG if c in nzv],
            "SHRED": [c for c in SHRED if c in nzv],
            "REP": [c for c in REP if c in nzv]}
    fams = {k: v for k, v in fams.items() if v}
    print("\nREDUNDANCY (leave-one-family-out;  marginal = AUC_full - AUC_without):")
    abl = {}
    for k, cols in fams.items():
        sub = [c for c in full if c not in cols]
        if not sub:
            continue
        _, _, auc_wo = fit(sub)
        abl[k] = round(auc_a - auc_wo, 4)
        print(f"  drop {k:7s} ({len(cols):2d} feats) -> AUC {auc_wo:.4f}   {k} marginal {auc_a-auc_wo:+.4f}")
    # permutation importance (do the new features rank?)
    pi = permutation_importance(clf_a, te[rich+new].to_numpy(float), te["y"].to_numpy(int),
                                n_repeats=5, random_state=0, scoring="roc_auc", n_jobs=4)
    imp = sorted(zip(rich+new, pi.importances_mean), key=lambda x: -x[1])[:12]
    print(f"\nTOP-12 PERMUTATION IMPORTANCE (★=new firehose feature):")
    for f, v in imp:
        print(f"  {'★' if f in NEW else ' '} {f:22s} {v:+.4f}")
    rep = {"n": n, "auc_rich": round(auc_b,4), "auc_aug": round(auc_a,4),
           "auc_incremental": round(auc_a-auc_b,4), "auc_new_only": round(auc_n,4),
           "auc_shuffle": round(auc_sh,4), "auc_comp_shuffle": round(auc_cs,4),
           "top_importance": [(f, round(v,4)) for f,v in imp],
           "net_rich": net_topk(te, s_b, .10), "net_aug": net_topk(te, s_a, .10)}
    json.dump(rep, open(a.out,"w"), indent=2)
    print(f"\n[cont-train] wrote {a.out}")


if __name__ == "__main__":
    main()
