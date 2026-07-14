# STATE OF PLAY — pump.fun bot, 2026-06-10 (midday)

This document supersedes the status sections of all earlier MD writeups (each
now carries a pointer note). The living decision record remains
`SHADOW_HARNESS_LOG.md` (GOTCHAS at top = canonical failure rules); the
migration contract is `ARCHITECTURE.md`. This file is the synthesis: what is
deployed, exactly how it has performed, what features exist vs are pending,
what stands between paper and real capital, and the arming recommendation.

Copies: sol:/root/the-distribution-will-manifest/STATE_OF_PLAY_2026-06-10.md
and the local mirror root. Written by the session of 2026-06-09/10.

---

## 1. What is deployed right now

- Host: sol, `pumpfun-bot.service`, paper/dry-run (`--live` + `PUMPFUN_LIVE_OK=1`
  armed, `JITO_DRY_RUN=1` + 0-SOL wallet block real submission).
- Model: `bot_artifacts_K7V -> bot_artifacts_k3v03_final` (deployed 01:18 CEST,
  instrumentation restarts 01:18/02:03, model unchanged; era cut for all
  verdicts: `--since 1781047096`).
- 22 K+V features (non-rich serve path), tp200-ranking head (target
  peak>=+200%) used as entry ranker, threshold 0.50, trigger K=3 AND V=0.3
  (env-pinned), exit `level_tp_50` + 300s stale watchdog. No recovery /
  death-cut head (known gap, see section 5).
- Trained on the honest live-matched population (min_fwd=0: insta-dead ready
  mints labeled, not dropped): May coherent span (Apr 29-May 5, 70,669 mints)
  + June capture Jun 7-8 (16,052), Jun 9 as selection holdout.

## 2. Exact performance to date

All "diagnostics" below are offline expectations; "realized" is the live paper
record. Per the GOTCHAS rules, diagnostics are never promises.

**Validation chain (diagnostics):**
| test | result |
|---|---|
| Month-gap (train May -> test ALL June) | AUC 0.784 (peak50) / 0.782 (peak200), buckets monotonic 5.5% -> 95.4% |
| Final-model holdout (May+Jun7-8 -> Jun 9) | AUC 0.806 / 0.812 |
| Exit replay, Jun 9 OOS bets (n=40, cost model 250bps + 2x0.0015 fee, 300s) | lat0 +0.727, lat1 +0.488 (win 95%, p25 +0.444), lat2 +0.056 per 1.0 SOL notional |
| Shuffle-null / selection | pre-stated adoption rule, single surviving cell; no test-set shopping |

**Realized live era (deploy 01:18 -> ~13:00 CEST, ~12h):**
- 1,882 decisions, 14 fires = 0.74% fire rate (spec expectation 0.61%, ratio 1.2, healthy)
- 12W / 2L, policy accounting (level_tp_50): **+0.218/fire** (book accounting +0.192)
- Pattern dedup: 14 fires = only **4 independent patterns**. One launch farm
  (identical script: cum_buy exactly 16.245 SOL, score 0.513838) accounts for
  11 fires at +0.252 mean; **deduped: +0.135/pattern, n_pat=4** (too small to call)
- 0 feature errors since the 02:03 restart

**Offline forward test on unseen Jun 10 data (the strongest evidence so far):**
- n=1,558 ready mints, base rates stable (0.178 / 0.054 vs 0.176 / 0.053 on Jun 9)
- AUC 0.768 / 0.784 (mild healthy decay from Jun 9's 0.791 / 0.809)
- Threshold band: 11 fires (0.71%, == spec), **11/11 peaked >= +50%**, 72.7% >= +200%,
  mean peak +3.45; band >=0.40: n=35 at 94.3% hit50; deduped 4/4 patterns clean
- Caveat: peak-based (entry-side); realized net depends on exit + execution

**Previous model for contrast (june_causal K3/V0.3/TP200, 160-cell intraday
sweep):** final record 46 fires, +0.124/fire book, win 52%: positive but ~7x
under its sweep promise. Winner's-curse deflation, as predicted at deploy.

**Economic scale honesty:** at 0.1 SOL bets the realized paper edge is
~+0.02 SOL/fire so far (replay suggests up to ~+0.05), at 15-60 fires/day.
Real capital at current size is a calibration instrument, not an income.

## 3. Features: have vs don't have

**Serving live (22):** K-window + V-window pairs of win_ret, dir_eff, buy_frac,
uniq, net_sol, tot_sol, single_actor_share, trades_per_sec, entry_sol,
win_drawup, win_drawdown. Trigger semantics env-pinned (K=3/V=0.3).

**Validated offline, NOT yet deployed (the big upside):** the rich set
(165 features: K/V/decision-anchored windows, inter-arrival-time and trade-size
percentiles, entry reserve anchors, cu_limit / priority-fee execution extras)
plus intent (33: 3 lookback windows x shred-pending-flow stats, now with
`_present` flags). HONEST cross-day result (CORRECTED 2026-06-11): the earlier
**0.97/0.98 AUC was LEAK-INFLATED** by 6 features incl `n_total_trades_seen`
(full-lifespan trade count = lookahead) + decision_idx/k_idx/v_idx/decision_slot/
first_ts. Clean live-reproducible rich AUC is **0.79 vs 22feat 0.78** (marginal on
AUC). But execution-adjusted (deduped, same population), rich beats 22feat by
~+0.09/fire at every latency, lifting realistic-latency net from -0.089 to ~0
(breakeven), not profit. So rich is a MODEST real edge, not transformational. The
LIVE rich shadow scorer already uses the clean 192-feature set (no contamination).
Gauntlet before deploy: more daily folds (free, accumulating), serve-path
parity (`build_entry_features` vs offline builder), fix the "rich trigger not
ready" race (was ~3% of decisions), alignment check, deploy checklist.

**Collected but unused by the live model:** shred signal (drives only the
adaptive Jito tip), sophistication soph_* aggregates, block_meta timing.

**Known feature gaps / still gathering:**
- Shred intent coverage is structurally **49%** of executed fresh buys (single
  region, fra6). Median lead over gRPC only 12ms (p90 107ms): shred value is
  metadata, not trigger speed. Second-region endpoint is the coverage play if
  intent keeps earning its increment.
- Creator/metadata features: REJECTED earlier on OOS transfer (era-specific);
  possible revisit as a hard pre-trade exclude only.
- Post-graduation (pumpswap) regime: not modeled at all; bot exits before.
- Recovery/death-cut head at K=3: TRAINED and validated same day (test
  Jun 10 AUC 0.842, monotone calibration; data: 1.04M live-matched forward
  snaps, tools/extract_recovery_k3v03.py + train_recovery_k3v03.py). Cut
  book-value inconclusive at thr-0.50 entries (in-train cuts halve the tail;
  the one Jun-10 OOS cut was false), so it runs as a LOG-ONLY shadow
  (`shadow_death_cut` events at P<0.20); the would-cut vs realized record
  decides the policy at arming prep. Exits still TP + stale until then.

## 4. Pending tests and gates

1. **Tonight 21:03:** scheduled deduped live verdict at n>=50 raw fires
   (judge the deduped pattern buckets, not raw; cron armed in-session).
2. Multi-day deduped accumulation: want n_pat >= 15-20 distinct positive
   patterns across >= 3 forward days before arming.
3. Rich-model gauntlet (above). Re-evaluate after each new capture day.
4. Downside policy: K=3 recovery head (preferred) or a fixed stop;
   plus a per-pattern exposure cap (farm-following risk: one actor is 75%+
   of current fires).
5. Capacity/impact study before any bet-size increase (0.1 SOL is ~0.2% of a
   46-SOL virtual curve; 1 SOL is ~2%: the cost model changes with size).
6. Standing: collector freeze guard (tests fail any closure change), archive
   sync hourly to storagebox, auto-retrain/auto-policy services stay OFF.

## 5. The execution gap (fills, latency, what paper cannot see)

Everything above assumes the paper cost model: fill at observed reserves with
1 snap of latency, 250bps roundtrip cost, 0.0015 SOL/tx. Real execution risk
lives in four places, none measurable in dry-run:

1. **Land rate**: do our bundles land in the next block, and at what tip?
   The broker already records tip-vs-landing (`recent_outcomes`); the
   tip-vs-land curve is the FIRST deliverable of the armed phase.
2. **Latency to leader**: replay says the edge is latency-fragile
   (lat1 +0.488 -> lat2 +0.056). Local path is clean (assembly p50 80us,
   blockhash p50 121ms old, 1.5s stall guard); the unknown is network/auction.
3. **Adverse selection**: fills we win at the worst moments. Bounded by the
   slippage cap, which is why the slippage review below matters.
4. **Real slippage vs modeled 250bps.**

**Code review verdicts (2026-06-10, both user hypotheses checked):**

- **Tips: yes, we over-tip on exits, but it is hygiene, not P&L.** Every sell
  (TP, stale, slices) ships as a Jito bundle with the full base tip (100k lam).
  A TP50 sell into strength and a stale exit have no same-slot urgency.
  Magnitude: ~0.0002 SOL per roundtrip = under 0.5% of expected edge/bet at
  0.1 SOL. Fix at arming prep: route non-urgent sells as regular transactions
  with a small priority fee; keep bundle+tip only for urgent exits (death-cut
  class). Entry-side tipping is NOT over-tipping: base 100k = the tipped
  median; the adaptive bump (outbid visible p90, capped 1M/5M by tier) is
  upside-capture. First live datapoint: a tier-2 cluster showed visible p90
  12M lam, above our 5M cap: if anything the entry cap may be too LOW when
  contested (cap is offline-simulatable from the bump logs before arming).
- **Slippage: REVISED 2026-06-10 by tools/exec_sim.py (earlier "tighten entry"
  advice RETRACTED).** Measured realized ENTRY slippage on 164 fires is large
  (p50 +84%): K=3 decides at a very low early price, realistic landing is a slot
  later post-pump. Counterintuitively high entry slip correlates with WINNERS
  (momentum rockets) and low slip with LOSERS (duds); the cap sweep shows total
  net highest at NO cap and negative at 10-25% caps. So a tight ENTRY cap is
  backwards for this momentum strategy -- keep entry loose. Atomic-bundle reverts
  still cost ~nothing, but reverting on slip would reject the rockets. SELL-side
  slippage is the opposite (still wants protection) and is not yet modeled; that
  is the real arming-prep slippage task, not the entry cap.

**Arming recommendation (the "when"):** arm when ALL of:
(a) tonight's deduped verdict is positive and the deduped pattern count
reaches ~15-20 cumulative positives over >=3 forward days (earliest realistic:
**~Jun 13** if nothing degrades); (b) a downside policy is shipped and
replay-checked; (c) the arming-prep changeset lands in one reviewed window
(slippage split, sells-as-regular-tx, entry tip cap revisit, risk limits:
daily_loss ~-0.5 SOL, max 4 fires/min). Then fund 2-3 SOL, set bet to
**0.05 SOL** (half the user's 0.1 suggestion: same information, half the blast
radius), flip `JITO_DRY_RUN=0` (explicit per-instance approval required), and
run a 3-5 day **execution-calibration phase whose goal is measurement, not
profit**: land-in-decision-slot rate measured (fill quality is dominated by
winning the launch slot, NOT a low-slip target -- high entry slip = momentum
here), paper-vs-real net gap <=30% per fire. Reconcile, then scale
stepwise 0.05 -> 0.1 -> 0.25 SOL, re-verifying the gap at each step.

## 6. Data custody (fixed 2026-06-10)

Storagebox now holds: raw firehose (87G) + raw shreds (82G) via the shipper,
PLUS (new `pumpfun-archive-sync.timer`, hourly, copy-only): filtered capture
gz, intent capture gz, training parquets, model artifacts, bot_data decision
logs (non-regenerable), git bundle, and the desktop's May coherent CSVs
(`archive/may_coherent_417849958/`). Sol is RAID1 on top. The desktop mirror
(`Investigatory AI/pumpbot/`) intentionally carries code + docs only.

## 7. Superseded documents

The status/plan content of these is superseded by this file (each has a
pointer note appended): `README.md` (mirror snapshot status),
`local/pumpfun_runbook.md`, `local/future_oot_pipeline_readme.md`,
`local/oot_v2_exact_readiness_readme.md`,
`local/PUMPFUN_LAUNCH_SNIPER_OOT_TEST_PENDING.md`,
`local/PUMPFUN_GAP_BRIDGE_MERGE_NOTES.md`,
`local/PUMPFUN_GAP_BRIDGE_FINALIZE_NOTES.md`.
Still authoritative for their own subject matter: `SHADOW_HARNESS_LOG.md`
(decisions + GOTCHAS), `ARCHITECTURE.md` (migration),
`local/PUMPFUN_DATA_INVENTORY_AND_RECOVERY.md` (May raw data locations; its
operational "current status" is superseded), `grpc_firehose_SCHEMA.md` and
`shred_bot/intent_capture/SCHEMA.md` (schemas).
