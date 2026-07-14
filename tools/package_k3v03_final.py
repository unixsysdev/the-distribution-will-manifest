#!/usr/bin/env python3
"""package_k3v03_final.py — apply the pre-stated adoption rule to the final
replay results and package bot_artifacts_k3v03_final/ for the ModelServer
contract. Prints the decision trace. Does NOT touch the symlink or config.

Adoption rule (stated before the first replay was run):
  adopt (head, thr, tp) iff at lat=1: Jun9 mean_net>0 and bootstrap
  P(mean>0)>=0.90; the same (head,tp) at neighboring thresholds is positive
  at lat=1; and the lat=2 cell mean >= -0.01.
  Prefer the smallest TP among survivors; tiebreak on p25, then lat2 mean.
"""
import json
import pickle
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "bot_artifacts_k3v03_final"
RES = ROOT / "data/replay_exit_k3v03_final.json"
LOG = Path("/tmp/replay_final.log")

results = json.loads(RES.read_text())
log = LOG.read_text()
m = re.search(r"scored (\d+) \(tail-guarded\) \| Jun9 share ([0-9.]+)", log)
n_total = int(m.group(1))
jun9_share = float(m.group(2))
den9 = n_total * jun9_share
den78 = n_total * (1 - jun9_share)

cell = {(r["head"], r["thr"], r["tp"], r["lat"]): r for r in results}
thr_grid = {"tp50_head": [0.35, 0.45, 0.55, 0.65], "tp200_head": [0.20, 0.30, 0.40, 0.50]}

survivors = []
for r in results:
    if r["lat"] != 1 or r["mean_jun9"] <= 0 or r["p_gt0"] < 0.90:
        continue
    grid = thr_grid[r["head"]]
    i = grid.index(r["thr"])
    ok = True
    for j in (i - 1, i + 1):
        if 0 <= j < len(grid):
            nb = cell.get((r["head"], grid[j], r["tp"], 1))
            if nb is None or nb["mean_jun9"] <= 0:
                ok = False
    lat2 = cell.get((r["head"], r["thr"], r["tp"], 2))
    if lat2 is None or lat2["mean_jun9"] < -0.01:
        ok = False
    if ok:
        survivors.append((r, lat2))

print(f"survivors of the pre-stated rule: {len(survivors)}")
for r, l2 in survivors:
    print(f"  {r['head']} thr={r['thr']} tp={r['tp']}: lat1 mean {r['mean_jun9']:+.3f} "
          f"(n={r['n_jun9']}, p25 {r['p25_jun9']:+.3f}, win {r['win_jun9']:.1%}) lat2 {l2['mean_jun9']:+.3f}")

if not survivors:
    print("NO survivor — do not deploy.")
    raise SystemExit(1)

survivors.sort(key=lambda t: (t[0]["tp"], -t[0]["p25_jun9"], -t[1]["mean_jun9"]))
best, best_lat2 = survivors[0]
print(f"\nCHOSEN: {best['head']} thr={best['thr']} tp={best['tp']}")

head_file = "entry_model.pkl" if best["head"] == "tp50_head" else "entry_model_tp200.pkl"
src = ART / head_file
chosen = pickle.load(open(src, "rb"))
shutil.copy(ART / "entry_model.pkl", ART / "entry_model_tp50head.pkl")
shutil.copy(ART / "entry_model_tp200.pkl", ART / "entry_model_tp200head.pkl")
with open(ART / "entry_model.pkl", "wb") as f:
    pickle.dump(chosen, f)

note = json.loads((ART / "TRAIN_NOTE.json").read_text())
feats = note["features"]
exit_policy = {0.5: "level_tp_50", 1.0: "level_tp_100", 2.0: "level_tp_200"}[best["tp"]]
import sklearn

spec = {
    "sklearn_version": sklearn.__version__,
    "artifact_kind": "k3v03_final_crossera_22f",
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "entry": {
        "features": feats,
        "target": "peak_ret>=2.0 ranking head" if best["head"] == "tp200_head" else "peak_ret>=0.5",
        "fire_if": "predict_proba[:,1] >= entry_threshold",
        "entry_threshold_top_decile": best["thr"],
        "threshold": best["thr"],
        "trigger": "K=3 AND V=0.3 joint",
        "k": 3,
        "v_sol": 0.3,
        "rich_features": False,
    },
    "exit_policy": exit_policy,
    "recovery": {
        "disabled": True,
        "death_cut_threshold": -1.0,
        "features": [],
        "reason": "no recovery head trained at K=3 trigger yet; exits via level TP + stale watchdog (300s), matching the replay assumptions",
    },
    "training": {
        "data": "live-matched honest population (min_fwd=0): May coherent span Apr29-May5 (70,669) + June capture Jun7-8 (~16k)",
        "validation": {
            "month_gap_eval1": "train May -> test ALL June: peak_ge_50 AUC 0.784 (buckets 5.5%->95.4%), peak_ge_200 AUC 0.782",
            "final_eval2": "train May+Jun7-8 -> test Jun9: peak_ge_50 AUC 0.806, peak_ge_200 AUC 0.812",
        },
    },
    "holdout_result": {
        "kind": "exit-policy replay, Jun9 OOS bets, harness cost model (q=0.1, 250bps, 0.0015/tx, 300s)",
        "head": best["head"],
        "tp": best["tp"],
        "lat1_mean_net": best["mean_jun9"],
        "lat1_win_rate": best["win_jun9"],
        "lat1_p25": best["p25_jun9"],
        "lat2_mean_net": best_lat2["mean_jun9"],
        "p_gt0": best["p_gt0"],
        "test_n": best["n_jun9"],
        "test_fire_rate": best["n_jun9"] / den9,
        "val_n": best["n_jun78"],
        "val_fire_rate": best["n_jun78"] / den78,
        "expectation_note": "diagnostics, not promises; the honest test is live_bucket_diag at n>=50 fires",
    },
    "runtime_notes": {
        "requires_env": {"K_TRIGGER": 3, "V_TRIGGER": 0.3},
        "config_yaml": {"exit.policy": exit_policy},
    },
}
(ART / "model_spec.json").write_text(json.dumps(spec, indent=2))
print(f"wrote {ART}/model_spec.json (exit_policy {exit_policy}, thr {best['thr']})")

# parity / smoke: load through the real server and score live-matched rows
import sys
sys.path.insert(0, str(ROOT))
from model_serve import ModelServer  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

srv = ModelServer(ART)
assert srv.rich_entry is False, "rich path must be off for 22-feat artifact"
assert abs(srv.entry_threshold - best["thr"]) < 1e-9
df = pd.read_parquet(ROOT / "data/live_matched_k3v03_all.parquet").tail(2000)
direct = chosen.predict_proba(df[feats].values)[:, 1]
served = np.array([srv.score_entry({f: row[f] for f in feats})[0] for _, row in df.head(50).iterrows()])
assert np.allclose(served, direct[:50], atol=1e-12), "served != direct scores"
fire_frac = float((direct >= srv.entry_threshold).mean())
print(f"parity OK (50 rows byte-identical). recent-2000-mint fire rate at thr: {fire_frac:.2%}")
print(f"ModelServer: {srv}")
