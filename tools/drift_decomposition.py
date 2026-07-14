"""Rigorous drift decomposition: where does live p90 = 0.225 come from?

Apply the IDENTICAL production entry_model to three populations and compare:

  A. Training data (May fresh + OOS, what the model was trained on)
  B. Full offline June capture (every mint that reaches K=7 + V=0.5, no bot)
  C. Bot's logged scores on the subset of June it actually observed

If A ~ B and B != C: the gap is selection bias. The bot sees a biased subset
  of the June population, and the subset systematically scores lower.

If A != B and B ~ C: regime drift between May and June. Retraining helps.

If A != B != C: both effects compound.

If A == B == C: there's no drift; the alert is wrong.

This is the diagnostic the drift monitor SHOULD be doing instead of
comparing live to a stale hardcoded value.
"""
from __future__ import annotations
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
           "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
ENTRY_V = [f"{c}_v" for c in ENTRY_K]
K_TO_V = dict(zip(ENTRY_K, ENTRY_V))


def _load_joined(prefix, suffix):
    tk_p = ROOT / f"data/pumpfun_continuation_{prefix}K7{suffix}/token_level.parquet"
    tv_p = ROOT / f"data/pumpfun_continuation_{prefix}V05{suffix}/token_level.parquet"
    if not (tk_p.exists() and tv_p.exists()): return None
    tk = pd.read_parquet(tk_p); tv = pd.read_parquet(tv_p)
    return tk.merge(tv[["mint"] + ENTRY_K].rename(columns=K_TO_V),
                    on="mint", how="inner")


def _qs(p, label, full=False):
    if len(p) == 0:
        print(f"  {label:40s} n=0 (empty)"); return
    q = np.percentile(p, [10, 25, 50, 75, 90, 95, 99])
    line = (f"  {label:40s} n={len(p):>6}  "
            f"p10={q[0]:.4f}  p50={q[2]:.4f}  "
            f"p75={q[3]:.4f}  **p90={q[4]:.4f}**  "
            f"p99={q[6]:.4f}  max={p.max():.4f}")
    print(line)
    if full:
        print(f"    score >= 0.5108 (production thr): "
              f"{(p >= 0.5108).sum()} ({100*(p >= 0.5108).mean():.2f}%)")
        print(f"    score >= 0.4453 (drift monitor's train_p90 ref): "
              f"{(p >= 0.4453).sum()} ({100*(p >= 0.4453).mean():.2f}%)")


def main():
    print("=" * 86)
    print(" RIGOROUS DRIFT DECOMPOSITION")
    print(" same production entry_model.pkl, three populations")
    print("=" * 86)
    clf = pickle.load(open(ROOT / "bot_artifacts_K7V/entry_model.pkl", "rb"))
    print(f"\nmodel: bot_artifacts_K7V/entry_model.pkl  "
          f"(production, trained on May fresh + OOS)\n")

    # ---- A. Training data (May fresh + OOS) ----
    print("A. TRAINING DATA (what the model was trained on)")
    tr_fresh = _load_joined("", "_fresh")
    tr_oos   = _load_joined("oos_", "_fresh")
    if tr_fresh is not None and tr_oos is not None:
        tr = pd.concat([tr_fresh, tr_oos], ignore_index=True).drop_duplicates("mint")
        X = tr[ENTRY_K + ENTRY_V].values
        p_train = clf.predict_proba(X)[:, 1]
        _qs(p_train, "training (fresh + OOS)", full=True)
    else:
        p_train = np.array([])
        print("  training parquets not found")

    # ---- B. Full offline June capture (no bot bias) ----
    print("\nB. FULL OFFLINE JUNE CAPTURE (every mint that reaches K=7+V=0.5)")
    jb = _load_joined("", "_capture_2d")
    if jb is not None:
        X = jb[ENTRY_K + ENTRY_V].values
        p_jun = clf.predict_proba(X)[:, 1]
        _qs(p_jun, "offline _capture_2d (full pop)", full=True)
    else:
        p_jun = np.array([])
        print("  capture_2d not found")

    # ---- C. Bot's logged scores on the subset it observed ----
    print("\nC. BOT-OBSERVED SUBSET (bot's logged scores from shadow_run.jsonl)")
    bot_scores = []
    with open(ROOT / "bot_data/shadow_run.jsonl") as f:
        for ln in f:
            try: r = json.loads(ln)
            except: continue
            if r.get("kind") == "entry_decision" and "score" in r:
                bot_scores.append(float(r["score"]))
    p_bot = np.array(bot_scores)
    _qs(p_bot, "bot logged (all entry_decisions)", full=True)

    # Also: bot's observable subset re-scored OFFLINE (eliminates any bot-side
    # FeatureAccum suspicion). We use the bot's mint set.
    print("\nD. BOT-OBSERVED SUBSET, but offline-rescored "
          "(removes any bot-side FeatureAccum suspicion)")
    if jb is not None:
        bot_mints = set()
        with open(ROOT / "bot_data/shadow_run.jsonl") as f:
            for ln in f:
                try: r = json.loads(ln)
                except: continue
                if r.get("kind") == "entry_decision":
                    bot_mints.add(r["mint"])
        jb_bot = jb[jb.mint.isin(bot_mints)]
        if len(jb_bot):
            X = jb_bot[ENTRY_K + ENTRY_V].values
            p_jun_bot = clf.predict_proba(X)[:, 1]
            _qs(p_jun_bot, f"offline of bot's {len(jb_bot)}-mint subset", full=True)

    # ---- Verdict ----
    print("\n" + "=" * 86)
    print(" VERDICT")
    print("=" * 86)
    if len(p_train) and len(p_jun) and len(p_bot):
        train_p90 = np.percentile(p_train, 90)
        jun_p90   = np.percentile(p_jun, 90)
        bot_p90   = np.percentile(p_bot, 90)
        print(f"\n  train p90:           {train_p90:.4f}")
        print(f"  offline June p90:    {jun_p90:.4f}  (gap to train: {jun_p90-train_p90:+.4f})")
        print(f"  bot-observed p90:    {bot_p90:.4f}  (gap to offline: {bot_p90-jun_p90:+.4f})")
        # Diagnosis
        if abs(train_p90 - jun_p90) < 0.05 and abs(jun_p90 - bot_p90) > 0.10:
            print("\n  -> SELECTION BIAS confirmed.")
            print("     Training and offline-June are aligned; bot-observed is the outlier.")
            print("     The bot sees a biased subset of mints; the FIX (capture-replay")
            print("     bootstrap) addresses this. Drift signal should narrow over the")
            print("     next 24h as the rolling window flushes pre-fix data.")
        elif abs(train_p90 - jun_p90) > 0.10 and abs(jun_p90 - bot_p90) < 0.05:
            print("\n  -> REGIME DRIFT confirmed.")
            print("     Bot and offline-June agree; training is the outlier.")
            print("     June pump.fun activity has genuinely shifted from May. Retraining")
            print("     on June data (bot_artifacts_K7V_jun_only) would close the gap.")
        elif abs(train_p90 - jun_p90) > 0.10 and abs(jun_p90 - bot_p90) > 0.10:
            print("\n  -> BOTH selection bias AND regime drift.")
            print("     Need both: fix selection (done via bootstrap) + retrain on June.")
        else:
            print("\n  -> No meaningful drift detected; alert is a false positive.")


if __name__ == "__main__":
    main()
