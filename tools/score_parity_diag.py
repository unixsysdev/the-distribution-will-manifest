"""Score parity diagnostic — compare bot-logged entry_score vs offline-recomputed.

For each ready mint in shadow_run.jsonl, look up its features in the offline
capture extraction (token_level.parquet from K7_capture_2d + V05_capture_2d),
run them through the production entry_model, and compare to the bot's logged
score.

If gap is small (<0.05 mean): scores agree, drift is real market regime change.
If gap is large (>0.10 mean): bot's FeatureAccum produces different features
than offline extractor, even though midK/vsK/vtK match (per partial_history_diag).
"""
from __future__ import annotations
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]

# 1. Bot-logged scores per mint
bot_scores = {}
with open(ROOT / "bot_data/shadow_run.jsonl") as f:
    for ln in f:
        try: r = json.loads(ln)
        except: continue
        if r.get("kind") == "entry_decision":
            m = r.get("mint")
            if m and m not in bot_scores:
                bot_scores[m] = float(r["score"])

# 2. Offline features
tk = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_capture_2d/token_level.parquet")
tv = pd.read_parquet(ROOT / "data/pumpfun_continuation_V05_capture_2d/token_level.parquet")
joined = tk.merge(tv[["mint"] + ENTRY_K].rename(columns={c: f"{c}_v" for c in ENTRY_K}),
                  on="mint", how="inner")

# 3. Score with production model
clf = pickle.load(open(ROOT / "bot_artifacts_K7V/entry_model.pkl", "rb"))
X = joined[ENTRY_K + [f"{c}_v" for c in ENTRY_K]].values
joined["offline_score"] = clf.predict_proba(X)[:, 1]

# 4. Match to bot
matched = []
for _, row in joined.iterrows():
    m = row["mint"]
    if m in bot_scores:
        matched.append({"mint": m, "bot": bot_scores[m],
                        "offline": float(row["offline_score"]),
                        "diff": float(row["offline_score"]) - bot_scores[m]})

if not matched:
    print("no overlap between bot-logged mints and offline extraction")
    raise SystemExit

df = pd.DataFrame(matched)
print(f"matched {len(df)} mints (bot ready ∩ offline extraction)")
print()
print(f"bot     score: p50={df.bot.median():.4f}  p90={df.bot.quantile(0.9):.4f}  "
      f"p99={df.bot.quantile(0.99):.4f}  max={df.bot.max():.4f}")
print(f"offline score: p50={df.offline.median():.4f}  p90={df.offline.quantile(0.9):.4f}  "
      f"p99={df.offline.quantile(0.99):.4f}  max={df.offline.max():.4f}")
print()
print(f"diff (offline - bot):")
print(f"  mean    {df['diff'].mean():+.4f}")
print(f"  median  {df['diff'].median():+.4f}")
print(f"  p5      {df['diff'].quantile(0.05):+.4f}")
print(f"  p95     {df['diff'].quantile(0.95):+.4f}")
print(f"  |diff| > 0.05: {(df['diff'].abs() > 0.05).mean()*100:.1f}%")
print(f"  |diff| > 0.20: {(df['diff'].abs() > 0.20).mean()*100:.1f}%")
print()
# Top divergences
print("biggest offline > bot (offline says fire, bot didn't):")
top = df.nlargest(10, "diff")
for _, r in top.iterrows():
    print(f"  {r.mint[:24]}  bot={r.bot:.4f}  offline={r.offline:.4f}  diff={r['diff']:+.4f}")
