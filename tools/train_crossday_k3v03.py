#!/usr/bin/env python3
"""train_crossday_k3v03.py — cross-day validated 22-feature candidate at the live trigger.

Train Jun 7-8, test Jun 9 (a real day boundary, not an intraday split), on the
HONEST population (min_fwd=0: insta-dead ready mints kept with their ~0 peak).
Tail guard: rows whose first_slot falls in the last 10 min of capture are dropped
(their labels are truncated, not real).

Outputs bot_artifacts_k3v03_crossday/ (entry_model.pkl + model_spec.json).
Does NOT touch the live symlink.
"""
import calendar
import glob
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data/live_matched_k3v03_all.parquet"
OUT = ROOT / "bot_artifacts_k3v03_crossday"
REG = dict(max_iter=150, max_depth=3, learning_rate=0.05, l2_regularization=5.0, random_state=42)
JUN9 = calendar.timegm((2026, 6, 9, 0, 0, 0))
TAIL_GUARD_SEC = 600


def first_row(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
                if "slot" in r and "t" in r:
                    return float(r["slot"]), float(r["t"])
            except Exception:
                pass
    return None


def main():
    files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))
    a = first_row(files[0])
    b = first_row(files[-2])
    sps = (b[0] - a[0]) / (b[1] - a[1])

    def slot_to_t(s):
        return a[1] + (s - a[0]) / sps

    df = pd.read_parquet(PARQUET)
    df["ts"] = df.first_slot.map(slot_to_t)
    t_end = df.ts.max()
    n0 = len(df)
    df = df[df.ts < t_end - TAIL_GUARD_SEC].reset_index(drop=True)
    print(f"loaded {n0} rows, {len(df)} after tail guard")

    feats = [c for c in df.columns if c not in ("mint", "first_slot", "peak_ret", "n_fwd", "ts")]
    tr = (df.ts < JUN9).values
    te = ~tr
    print(f"train Jun7-8 n={tr.sum()}  test Jun9 n={te.sum()}")
    thin = (df.n_fwd < 5).values
    print(f"thin (n_fwd<5) share: train {thin[tr].mean():.3f}  test {thin[te].mean():.3f}")

    X = df[feats].values
    results = {}
    models = {}
    for tgt, tname in [(0.5, "peak_ge_50"), (2.0, "peak_ge_200")]:
        y = (df.peak_ret >= tgt).astype(int).values
        clf = HistGradientBoostingClassifier(**REG).fit(X[tr], y[tr])
        s_te = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], s_te)
        print(f"\n=== {tname}: train base {y[tr].mean():.3f}  test base {y[te].mean():.3f}  CROSS-DAY test AUC {auc:.4f} ===")
        d = pd.DataFrame({"s": s_te, "y": y[te], "peak": df.peak_ret[te].values, "thin": thin[te]})
        d["b"] = pd.cut(d.s, np.arange(0, 1.01, 0.1))
        tab = d.groupby("b", observed=True).agg(
            n=("y", "size"), win=("y", "mean"), mean_peak=("peak", "mean"), thin_frac=("thin", "mean")
        ).query("n>0")
        print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
        # where do thin mints score?
        print(f"thin score p50={np.median(d.s[d.thin]):.3f} p90={np.percentile(d.s[d.thin],90):.3f} | "
              f"nonthin p50={np.median(d.s[~d.thin]):.3f} p90={np.percentile(d.s[~d.thin],90):.3f}")
        bucket_table = [
            {"bucket": str(ix), "n": int(r.n), "win": float(r.win),
             "mean_peak": float(r.mean_peak), "thin_frac": float(r.thin_frac)}
            for ix, r in tab.iterrows()
        ]
        thr_table = []
        y_te = y[te]
        for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            m = s_te >= t
            thr_table.append({
                "thr": t,
                "oos_precision": float(y_te[m].mean()) if m.sum() else None,
                "oos_fire_rate": float(m.mean()),
                "n": int(m.sum()),
            })
        results[tname] = {"test_auc": float(auc), "train_base": float(y[tr].mean()),
                          "test_base": float(y[te].mean()), "buckets": bucket_table,
                          "thresholds": thr_table}
        models[tname] = clf

    # alignment vs live decisions of the current era
    cut = 1781027821.0
    live_rows = []
    with open(ROOT / "bot_data/shadow_run.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("t") or 0) < cut or r.get("kind") != "entry_decision":
                continue
            fd = r.get("features") or {}
            vec = []
            ok = True
            for c in feats:
                key = ("v_" + c[:-2]) if c.endswith("_v") else ("k_" + c)
                if key not in fd:
                    ok = False
                    break
                v = fd[key]
                vec.append(float("nan") if v is None else float(v))
            if ok:
                live_rows.append(vec)
    align = {}
    if live_rows:
        Xl = np.array(live_rows)
        clf = models["peak_ge_50"]
        sl = clf.predict_proba(Xl)[:, 1]
        ste = clf.predict_proba(X[te])[:, 1]
        print(f"\n=== ALIGNMENT (peak_ge_50 head): {len(sl)} live decisions ===")
        for name, arr in [("OOS Jun9", ste), ("live", sl)]:
            q = np.percentile(arr, [50, 90, 95, 99])
            print(f"  {name}: p50={q[0]:.3f} p90={q[1]:.3f} p95={q[2]:.3f} p99={q[3]:.3f}")
        align = {
            "n_live": len(sl),
            "oos_pct": {str(q): float(np.percentile(ste, q)) for q in (50, 90, 95, 99)},
            "live_pct": {str(q): float(np.percentile(sl, q)) for q in (50, 90, 95, 99)},
            "fire_rate_ratio_at": {
                str(t): (float((sl >= t).mean() / max((ste >= t).mean(), 1e-9)))
                for t in (0.35, 0.45, 0.55)
            },
        }
        print("  live/OOS fire-rate ratio:", {k: round(v, 2) for k, v in align["fire_rate_ratio_at"].items()})

    OUT.mkdir(exist_ok=True)
    with open(OUT / "entry_model.pkl", "wb") as f:
        pickle.dump({"model": models["peak_ge_50"], "features": feats}, f)
    with open(OUT / "entry_model_tp200.pkl", "wb") as f:
        pickle.dump({"model": models["peak_ge_200"], "features": feats}, f)
    spec = {
        "artifact_kind": "candidate_crossday_22f",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger": {"k": 3, "v_sol": 0.3, "joint": True},
        "population": "live-matched, min_fwd=0 (insta-dead ready mints INCLUDED), tail-guard 600s",
        "split": "train Jun7-8, test Jun9 (cross-day, single fold)",
        "features": feats,
        "results": results,
        "alignment": align,
        "NOT_DEPLOYED": "candidate only; needs exit-policy replay + recovery head decision + approval",
        "source": "tools/train_crossday_k3v03.py over data/live_matched_k3v03_all.parquet",
    }
    (OUT / "model_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nwrote {OUT}/ (entry_model.pkl [peak_ge_50], entry_model_tp200.pkl, model_spec.json)")


if __name__ == "__main__":
    main()
