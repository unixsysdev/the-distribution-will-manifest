"""Coverage diagnostic — are the mints the bot is MISSING higher-scoring?

If yes, the bot's drift is selection bias: it sees only the slow mints; the fast
winners reach K=7+V=0.5 before the bot logs them. That would explain why
live score p90 = 0.24 while offline on all mints has p90 = 0.60.
"""
from __future__ import annotations
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]

# Bot-seen mints
bot_seen = set()
with open(ROOT / "bot_data/shadow_run.jsonl") as f:
    for ln in f:
        try: r = json.loads(ln)
        except: continue
        if r.get("kind") == "entry_decision":
            bot_seen.add(r["mint"])

# Offline mints
tk = pd.read_parquet(ROOT / "data/pumpfun_continuation_K7_capture_2d/token_level.parquet")
tv = pd.read_parquet(ROOT / "data/pumpfun_continuation_V05_capture_2d/token_level.parquet")
joined = tk.merge(tv[["mint"] + ENTRY_K].rename(columns={c: f"{c}_v" for c in ENTRY_K}),
                  on="mint", how="inner")

clf = pickle.load(open(ROOT / "bot_artifacts_K7V/entry_model.pkl", "rb"))
X = joined[ENTRY_K + [f"{c}_v" for c in ENTRY_K]].values
joined["score"] = clf.predict_proba(X)[:, 1]

joined["in_bot"] = joined["mint"].isin(bot_seen)
in_bot = joined[joined.in_bot]
not_in_bot = joined[~joined.in_bot]
print(f"offline: {len(joined)} mints")
print(f"  in bot:    {len(in_bot)} ({100*len(in_bot)/len(joined):.1f}%)")
print(f"  NOT in bot: {len(not_in_bot)} ({100*len(not_in_bot)/len(joined):.1f}%)")
print()
print("score distribution:")
def show(df, label):
    s = df["score"].values
    print(f"  {label:20s} n={len(s):>5} p50={np.percentile(s,50):.4f}  p75={np.percentile(s,75):.4f}  "
          f"p90={np.percentile(s,90):.4f}  p95={np.percentile(s,95):.4f}  p99={np.percentile(s,99):.4f}  max={s.max():.4f}")
show(joined, "all offline")
show(in_bot, "bot saw")
show(not_in_bot, "bot MISSED")
print()
# Higher-score regions where the bot is missing
print("How many mints with score >= X are MISSED by the bot:")
for cut in [0.20, 0.30, 0.40, 0.50, 0.5108, 0.60, 0.70, 0.80]:
    nm = (not_in_bot.score >= cut).sum()
    nb = (in_bot.score >= cut).sum()
    tot = nm + nb
    if tot == 0: continue
    print(f"  score>={cut:.4f}: bot_saw={nb:>4}  bot_missed={nm:>4}  miss_rate={100*nm/tot:.1f}%")
print()
# Peak ret check — are the misses winners or losers?
print("Among mints with score >= 0.30:")
for label, df in [("bot saw", in_bot), ("bot MISSED", not_in_bot)]:
    s = df[df.score >= 0.30]
    if len(s):
        peak = (s.peak_ret >= 1).mean()  # peak >= 2x
        term = (s.terminal_ret >= 0).mean()
        print(f"  {label:14s}: n={len(s)}  peak>=2x: {100*peak:.1f}%  terminal>=0: {100*term:.1f}%")
