"""Train the deployable LEAN+REP continuation filter on the full rich+rep panel; save the model
artifact + the exact feature order for the live bot to load. Prints a by-coin OOS AUC sanity.
"""
import json, hashlib, os, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))
PANEL = f"{ROOT}/bot_data/cont_rich_rep_panel.jsonl"
MODEL = f"{ROOT}/bot_data/cont_leanrep_model.pkl"
SPEC = f"{ROOT}/bot_data/cont_leanrep_features.json"
BASE6 = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq"]
LEAN = BASE6 + ["vol_sol", "mcap_sol", "signer_conc", "max_buy_sol", "max_runup", "bs_ratio", "whale_frac"]
REP = ["cre_n_launch", "cre_n_2x_res", "cre_winrate", "buy_n", "buy_known_frac", "buy_rep_mean", "buy_rep_max"]
FE = LEAN + REP

cr = {}; oc = {}
for l in open(PANEL):
    r = json.loads(l)
    if r.get("kind") == "cross": cr[r["mint"]] = r
    elif r.get("kind") == "outcome": oc[r["mint"]] = r
keys = [m for m in oc if m in cr]
X = np.array([[cr[m].get(f, 0.0) for f in FE] for m in keys], float)
y = np.array([1 if oc[m]["y"] == 1 else 0 for m in keys], int)
print(f"training on {len(keys)} crosses, {len(FE)} features (LEAN {len(LEAN)} + REP {len(REP)}), base win {y.mean():.0%}")

bk = np.array([int(hashlib.md5(m.encode()).hexdigest(), 16) % 100 for m in keys])
tr = bk < 70; te = bk >= 70
oos = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4, l2_regularization=1.0).fit(X[tr], y[tr])
print(f"by-coin OOS AUC sanity (module-computed REP features): {roc_auc_score(y[te], oos.predict_proba(X[te])[:, 1]):.3f}")

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4, l2_regularization=1.0).fit(X, y)
with open(MODEL, "wb") as f:
    pickle.dump(clf, f)
json.dump({"features": FE, "lean": LEAN, "rep": REP, "base6": BASE6,
           "n_train": len(keys), "trained_win": float(y.mean())}, open(SPEC, "w"), indent=1)
print(f"saved model -> {MODEL}\nsaved feature spec -> {SPEC}")
