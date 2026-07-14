"""Bot-vs-offline FEATURE-LEVEL parity check.

The bot logs its 22 entry features in every entry_decision event under
`features`. The offline extractor produces the same features per mint in
token_level.parquet. For each mint in BOTH:
  - compare each feature directly (bot vs offline)
  - compute model score on each and compare

This conclusively rules out (or finds) any per-feature accumulator drift.
"""
from __future__ import annotations
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
ENTRY_V = [f"{c}_v" for c in ENTRY_K]
ALL_FEATS = ENTRY_K + ENTRY_V

# 1. Bot-logged features per mint
bot_feats = {}
bot_scores = {}
with open(ROOT / "bot_data/shadow_run.jsonl") as f:
    for ln in f:
        try: r = json.loads(ln)
        except: continue
        if r.get("kind") == "entry_decision":
            m = r.get("mint")
            if m and m not in bot_feats:
                ef = r.get("features")
                if isinstance(ef, dict):
                    bot_feats[m] = ef
                    bot_scores[m] = float(r["score"])
print(f"bot mints with logged features: {len(bot_feats)}")

# 2. Offline features
tk = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_capture_2d/token_level.parquet")
tv = pd.read_parquet(ROOT / "data/pumpfun_continuation_V05_capture_2d/token_level.parquet")
joined = tk.merge(tv[["mint"] + ENTRY_K].rename(columns={c: f"{c}_v" for c in ENTRY_K}),
                  on="mint", how="inner")
print(f"offline mints in 34h capture: {len(joined)}")

# 3. Per-feature comparison on matched mints
rows = []
for _, r in joined.iterrows():
    m = r["mint"]
    if m not in bot_feats: continue
    row = {"mint": m}
    for f in ALL_FEATS:
        bv = float(bot_feats[m].get(f, np.nan))
        ov = float(r[f])
        row[f"{f}__bot"] = bv
        row[f"{f}__off"] = ov
        row[f"{f}__diff"] = ov - bv
    rows.append(row)
df = pd.DataFrame(rows)
print(f"matched mints (bot ∩ offline): {len(df)}")
print()

# Per-feature diff summary
print(f"{'feature':25s} {'mean_diff':>11s} {'med_diff':>10s} {'p99_|d|':>10s} {'%|d|>1e-6':>10s}")
for f in ALL_FEATS:
    d = df[f"{f}__diff"].values
    ad = np.abs(d)
    print(f"{f:25s} {d.mean():>+11.4f} {np.median(d):>+10.4f} "
          f"{np.percentile(ad, 99):>10.4f} {100*(ad > 1e-6).mean():>9.1f}%")
print()

# Score comparison via model
clf = pickle.load(open(ROOT / "bot_artifacts_K7V/entry_model.pkl", "rb"))
# Score bot features
X_bot = df[[f"{f}__bot" for f in ALL_FEATS]].values
X_off = df[[f"{f}__off" for f in ALL_FEATS]].values
s_bot = clf.predict_proba(X_bot)[:, 1]
s_off = clf.predict_proba(X_off)[:, 1]
s_logged = df["mint"].map(bot_scores).values
print("score comparison:")
print(f"  model(bot_features)    p50={np.median(s_bot):.4f}  p90={np.percentile(s_bot,90):.4f}  max={s_bot.max():.4f}")
print(f"  model(offline_features) p50={np.median(s_off):.4f}  p90={np.percentile(s_off,90):.4f}  max={s_off.max():.4f}")
print(f"  bot_logged_score       p50={np.median(s_logged):.4f}  p90={np.percentile(s_logged,90):.4f}  max={s_logged.max():.4f}")
print()
print(f"  diff (model_on_bot - bot_logged):  mean={np.mean(s_bot - s_logged):+.6f}")
print(f"  diff (model_on_offline - model_on_bot): mean={np.mean(s_off - s_bot):+.6f}")
