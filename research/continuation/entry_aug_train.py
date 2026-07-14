#!/usr/bin/env python3
"""Train + evaluate the augmented ENTRY-selection model from entry_aug_panel.jsonl.

Question (the user's): does execution-competition texture (priority-fee/tip/CU/
inner-ix on the early landed trades) add incremental entry-selection signal over
the proven 11-feature trade-economics baseline, on the most recent data?

Discipline (matches the offline thread): HGB depth-3, chronological mint-level
split (no mint overlap by construction = one row per mint), strictly-causal
window-only features, drop censored rows, shuffle-null sanity, top-decile lift +
decile monotonicity as the executable metric (AUC is diagnostic only), and
permutation importance so we can SEE which features carry the model.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

BASELINE = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
            "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
COMP = ["cmp_fee_mean","cmp_fee_max","cmp_prio_mean","cmp_prio_max","cmp_prio_escal",
        "cmp_cu_mean","cmp_cu_max","cmp_culimit_mean","cmp_culimit_max",
        "cmp_inner_mean","cmp_inner_max","cmp_nkeys_mean","cmp_nkeys_max",
        "cmp_tip_rate","cmp_tip_mean_lam","cmp_route_frac","cmp_distinct_buyers"]
CONG = ["cong_exec_mean","cong_exec_max","cong_slot_span"]
LABELS = ["y_peak50","y_peak100","y_term0","y_term25"]
ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=f"{ROOT}/bot_data/entry_aug_panel.jsonl")
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--out", default=f"{ROOT}/bot_data/entry_aug_report.json")
    ap.add_argument("--keep-censored", action="store_true")
    return ap.parse_args()


def load(panel):
    import pandas as pd
    rows = []
    with open(panel) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return pd.DataFrame(rows)


def decile_lift(y_true, score, k=10):
    n = len(score)
    order = np.argsort(-score)
    base = y_true.mean()
    top = y_true.iloc[order[: max(1, n // k)]].mean() if hasattr(y_true, "iloc") else y_true[order[: max(1, n // k)]].mean()
    # decile-by-decile positive rate (monotonicity check)
    deciles = []
    for d in range(k):
        idx = order[int(d * n / k): int((d + 1) * n / k)]
        yv = y_true.iloc[idx] if hasattr(y_true, "iloc") else y_true[idx]
        deciles.append(round(float(yv.mean()), 4))
    return float(top / base) if base > 0 else 0.0, float(base), float(top), deciles


def main():
    a = parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance
    df = load(a.panel)
    sys.stderr.write(f"[train] loaded {len(df)} rows\n")
    if not a.keep_censored and "censored" in df:
        df = df[df["censored"] == 0].copy()
    df = df.sort_values("trig_ts").reset_index(drop=True)
    n = len(df); cut = int(n * (1 - a.test_frac))
    tr, te = df.iloc[:cut], df.iloc[cut:]
    sys.stderr.write(f"[train] n={n} train={len(tr)} test={len(te)} "
                     f"(train end {tr['trig_ts'].max():.0f}, test start {te['trig_ts'].min():.0f})\n")

    # drop zero-variance features in train (e.g. cong_* are all-0 from capture)
    feat_all = [c for c in (BASELINE + COMP + CONG) if c in df.columns]
    nzv = [c for c in feat_all if tr[c].std() > 0]
    base_feats = [c for c in BASELINE if c in nzv]
    comp_feats = [c for c in (COMP + CONG) if c in nzv]
    all_feats = base_feats + comp_feats
    dropped = [c for c in feat_all if c not in nzv]
    sys.stderr.write(f"[train] dropped zero-variance: {dropped}\n")

    def fit_eval(feats, label, shuffle=False):
        Xtr = tr[feats].to_numpy(dtype=float); Xte = te[feats].to_numpy(dtype=float)
        ytr = tr[label].to_numpy(dtype=int); yte = te[label]
        if shuffle:
            rng = np.random.default_rng(0); ytr = rng.permutation(ytr)
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                             learning_rate=0.06, l2_regularization=1.0,
                                             random_state=0, early_stopping=True)
        clf.fit(Xtr, ytr)
        s = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, s) if yte.nunique() > 1 else float("nan")
        lift, base, top, deciles = decile_lift(yte, s)
        return clf, auc, lift, base, top, deciles

    report = {"n": n, "n_train": len(tr), "n_test": len(te),
              "base_feats": base_feats, "comp_feats": comp_feats, "dropped": dropped,
              "labels": {}}
    print("="*78)
    print(f"AUGMENTED ENTRY MODEL  |  n={n}  train={len(tr)}  test={len(te)}  K-window=10")
    print(f"competition features available: {len(comp_feats)}  -> {comp_feats}")
    print("="*78)
    for label in LABELS:
        if label not in df.columns or te[label].nunique() < 2:
            continue
        _, a_b, l_b, base, top_b, dec_b = fit_eval(base_feats, label)
        clf_all, a_a, l_a, _, top_a, dec_a = fit_eval(all_feats, label)
        _, a_c, l_c, _, top_c, dec_c = fit_eval(comp_feats, label) if comp_feats else (None, float("nan"),0,0,0,[])
        _, a_sh, l_sh, _, _, _ = fit_eval(all_feats, label, shuffle=True)
        # permutation importance of the ALL model (top movers)
        Xte = te[all_feats].to_numpy(dtype=float); yte = te[label].to_numpy(dtype=int)
        pi = permutation_importance(clf_all, Xte, yte, n_repeats=5, random_state=0,
                                    scoring="roc_auc", n_jobs=4)
        imp = sorted(zip(all_feats, pi.importances_mean), key=lambda x: -x[1])[:8]
        report["labels"][label] = {
            "base_rate": round(base, 4),
            "auc_baseline": round(a_b, 4), "auc_all": round(a_a, 4),
            "auc_comp_only": round(a_c, 4), "auc_shuffle_null": round(a_sh, 4),
            "auc_incremental": round(a_a - a_b, 4),
            "lift_baseline": round(l_b, 3), "lift_all": round(l_a, 3),
            "topdecile_base": round(top_b, 4), "topdecile_all": round(top_a, 4),
            "deciles_all": dec_a, "top_perm_importance": [(f, round(v, 4)) for f, v in imp],
        }
        print(f"\n[{label}]  base_rate={base:.3f}")
        print(f"  AUC: baseline={a_b:.4f}  +competition={a_a:.4f}  (Δ={a_a-a_b:+.4f})  "
              f"comp_only={a_c:.4f}  shuffle_null={a_sh:.4f}")
        print(f"  top-decile P(win): baseline={top_b:.3f} ({l_b:.2f}x)  +competition={top_a:.3f} ({l_a:.2f}x)")
        print(f"  deciles(+comp): {dec_a}")
        print(f"  top perm-importance: " + ", ".join(f"{f}={v:+.4f}" for f, v in imp))
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\n[train] wrote {a.out}")


if __name__ == "__main__":
    main()
