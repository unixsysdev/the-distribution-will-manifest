#!/usr/bin/env python3
"""Train + serialize the GRADUATION (PumpSwap) RICH model into the bot's deploy format.

Clone of cont_2x_deploy_train.py but: RICH-only (rep/shred added ~0 on graduation), trained on
the MIRROR-trigger rows (first-2x-from-launch = what RichTracker(k=2.0) does live), target y =
reach +0.5x before -0.3x. Output matches what continuation_live_rep_bot loads:
  grad_deploy_model.pkl  (sklearn HGB; bot extracts _predictors/_baseline_prediction -> fast trees)
  grad_model_spec.json   {features, lean, rep, ...}
"""
import json, pickle, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import sklearn

ROOT = "/root/the-distribution-will-manifest"
RICH = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq", "t_to_2x", "log_t_to_2x", "accel", "last_gap",
        "mcap_sol", "vol_sol", "sol_per_trade", "max_buy_sol", "whale_frac", "net_flow", "n_buyers",
        "n_sellers", "bs_ratio", "signer_conc", "up_frac", "max_runup"]
MODE = "mirror"
PANEL = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/bot_data/grad_cont_panel_4d.jsonl"   # argv override; default = stable 4-day panel

rows = [json.loads(l) for l in open(PANEL)]
df = pd.DataFrame([r for r in rows if r.get("mode") == MODE and r.get("y") is not None])
df = df.sort_values("cross_t").reset_index(drop=True)
for c in RICH:
    if c not in df.columns:
        raise SystemExit(f"missing feature column {c} in panel")
n = len(df); cut = int(n * 0.7); tr, te = df.iloc[:cut], df.iloc[cut:]


def fit(data):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=350, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=0, early_stopping=True)
    m.fit(data[RICH].to_numpy(float), data["y"].to_numpy(int))
    return m


m = fit(tr)
s = m.predict_proba(te[RICH].to_numpy(float))[:, 1]
auc = roc_auc_score(te["y"].to_numpy(int), s)
base = te["y"].mean()
k = max(1, len(s) // 10); top = np.argsort(-s)[:k]
print(f"mode={MODE}  n={n}  test={len(te)}  base_rate={base:.3f}  OOS AUC(RICH)={auc:.4f}  top-decile prec={te['y'].to_numpy(int)[top].mean():.3f}")

mF = fit(df)   # final: fit on all rows
pickle.dump(mF, open(f"{ROOT}/bot_data/grad_deploy_model.pkl", "wb"))
json.dump({"features": RICH, "lean": RICH, "rep": [], "mode": MODE,
           "oos_auc": round(float(auc), 4), "n_train": int(n),
           "sklearn_version": sklearn.__version__,
           "target": "reach +0.5x before -0.3x (k=2.0 first-2x-from-launch / mirror)",
           "trigger": "RichTracker(k=2.0) on PumpSwap AMM mid (quote_res/base_res)"},
          open(f"{ROOT}/bot_data/grad_model_spec.json", "w"), indent=2)
print(f"saved grad_deploy_model.pkl + grad_model_spec.json  (sklearn {sklearn.__version__})")
