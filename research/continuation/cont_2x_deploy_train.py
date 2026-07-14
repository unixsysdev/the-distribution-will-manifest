#!/usr/bin/env python3
"""Parity retrain on the SHARED-MODULE panel: confirm AUC reproduces 0.5874, re-save artifact."""
import json, os, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))
RICH = ["dd","buy_frac","ntr","recent","tps","uniq","t_to_2x","log_t_to_2x","accel","last_gap","mcap_sol",
        "vol_sol","sol_per_trade","max_buy_sol","whale_frac","net_flow","n_buyers","n_sellers","bs_ratio",
        "signer_conc","up_frac","max_runup"]
SHRED = ["shred_nbuy","shred_nbuy_5slot","shred_uniq_signers","shred_prio_p90","shred_prio_max",
         "shred_tip_rate","shred_tip_max","shred_nslots","shred_maxperslot"]
REP = ["rep_mean","rep_max","rep_nknown","rep_frac_known","rep_frachigh","rep_nsmart"]
FE = RICH + SHRED + REP

df = pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/bot_data/cont_2x_deploy_panel.jsonl")]).sort_values("cross_t").reset_index(drop=True)
n = len(df); cut = int(n * 0.7); tr, te = df.iloc[:cut], df.iloc[cut:]

def fit(F):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=350, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=0, early_stopping=True)
    m.fit(tr[F].to_numpy(float), tr["y"].to_numpy(int))
    s = m.predict_proba(te[F].to_numpy(float))[:, 1]
    return m, s, roc_auc_score(te["y"], s)

_, sb, ab = fit(RICH)
_, sa, aa = fit(FE)
o = np.argsort(-sa)[:max(1, len(sa) // 10)]; sub = te.iloc[o]; fl = sub["entry_slip"] <= 0.25
nf = np.where(fl.to_numpy(), 0.1 * sub["ret"].to_numpy() - 0.02 * 0.1 - 0.00161, 0.0)
verdict = "PASS" if abs(aa - 0.5874) < 0.004 else "CHECK"
print(f"n={n} test={len(te)}  AUC RICH={ab:.4f}  RICH+SHRED+REP={aa:.4f}  (reference 0.5874)")
print(f"PARITY: {verdict}   net/sel={nf.mean():+.4f}  tot={nf.sum():+.2f}  win={sub['y'].mean():.0%}  fill={fl.mean():.0%}")

mF = HistGradientBoostingClassifier(max_depth=3, max_iter=350, learning_rate=0.05,
                                    l2_regularization=1.0, random_state=0, early_stopping=True)
mF.fit(df[FE].to_numpy(float), df["y"].to_numpy(int))
pickle.dump(mF, open(f"{ROOT}/bot_data/cont2x_deploy_model.pkl", "wb"))
json.dump({"features": FE, "rich": RICH, "shred": SHRED, "rep": REP, "oos_auc": round(aa, 4),
           "oos_auc_rich": round(ab, 4), "n_train": n, "built_by": "cont_aug_features shared module"},
          open(f"{ROOT}/bot_data/cont2x_deploy_features.json", "w"), indent=1)
print("re-saved cont2x_deploy_model.pkl + cont2x_deploy_features.json from shared-module panel")
