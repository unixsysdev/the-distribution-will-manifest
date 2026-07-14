# RUNNING & ROADMAP — pump.fun bot

Last written: 2026-06-10 ~17:00 CEST. Read this to answer, cold: what is
running, what are we waiting for, and what the finished system looks like.

Companions: `STATE_OF_PLAY_2026-06-10.md` (current synthesis + numbers),
`SHADOW_HARNESS_LOG.md` (dated decisions + GOTCHAS rules), `ARCHITECTURE.md`
(package migration). This file is the watchlist + the destination.

Today is 2026-06-10. Deployed model era cut for all live verdicts:
`--since 1781047096` (k3v03_final, paper/dry-run).

---

## 1. What is running right now

**Always-on systemd services on the research host:**

| service | role | notes |
|---|---|---|
| pumpfun-bot | the trader (paper/dry-run) | 22-feat model, level_tp_50 + 300s stale, shadow death-cut LOG-ONLY |
| pumpfun-grpc-capture | filtered TradeEvent capture | training source; rotate-and-reopen hourly |
| pumpfun-grpc-firehose | raw gRPC archive | buffer/ -> shipper |
| pumpfun-shred-firehose | raw shred archive | buffer/ -> shipper |
| pumpfun-shred-intents | shred Buy-intent recorder | writes the SHM ring + intent_capture/ |
| pumpfun-storagebox-shipper | moves buffers to Hetzner | CIFS mount |
| fail2ban | ssh brute-force bans | added 2026-06-10 |

These five collectors are a FROZEN SET (tests/test_collector_freeze.py fails
any import-closure change). Do not restart casually; they are Restart=always
and a broken closure becomes a capture-losing crashloop.

**Timers:**
- `pumpfun-archive-sync.timer` hourly at :24 — copies capture/intents/parquets/
  artifacts/decision-logs/git-bundle to storagebox (copy-only).
- `pumpfun-drift-monitor.timer` nightly ~00:05 — era-NAIVE, verify its window
  before trusting any alert (it produced the stale dashboard ALERT today).
- `pumpfun-verdict-20260610.timer` ONE-SHOT 21:03 tonight — writes the deduped
  live verdict + health to `logs/verdict_2026-06-10_2103.txt`, then auto-clears.

**Inside the running bot (not separate processes):**
- shadow death-cut: scores drawdown snaps of open positions, logs
  `shadow_death_cut` (P(recover)<0.20) to shadow_run.jsonl, NEVER acts.
- shadow RICH scorer (added 2026-06-10): scores every decision with the rich
  192-feat model, logs `shadow_rich_score` (rich score + live intent context),
  NEVER acts. Live n_missing_feats=0 (full vector incl intent reproduced live).
  A few `shadow_rich_error` "trigger not ready" are the restart-boundary race.
- shadow EXIT accounting (added 2026-06-11): every close logs `policy_nets`,
  the counterfactual {tp_50, tp_100, tp_200} nets on the same snap timeline.
  exit_lab's offline tournament put tp_100 ahead (deduped +0.116 vs -0.015);
  the live deduped race decides the swap. LSM optimal-stopping overfit train
  and is rejected; trailing/ladders lose to gap-downs + per-slice fees.
- adaptive Jito tip: logs `front_run_tip_bump` with tier + visible-p90; in
  dry-run it only changes what gets logged.

**Recent one-off result:**
- `tools/parity_rich.py`: rich-feature serve-path parity. RESULT 2026-06-10:
  PASS. 159/159 non-intent features byte-identical, decision-index 200/200,
  0 trigger failures. Re-runnable anytime; ~13min (replays capture).

**Check it all in one shot:**
```
ssh <research-host> 'cd /root/the-distribution-will-manifest && ./venv/bin/python -m pumpbot health'
ssh <research-host> 'cat /root/the-distribution-will-manifest/logs/verdict_2026-06-10_2103.txt'   # after 21:03
ssh <research-host> 'cd /root/the-distribution-will-manifest && ./venv/bin/python -m pumpbot diag --since 1781047096'
```

---

## 2. What we are waiting for (the gates, in order)

The binding constraint is calendar days, not data volume (we collect ~7,700
ready mints/day, already well-powered offline). Each gate below unblocks the
next; none can be rushed by capturing more per day.

1. **Tonight 21:03 — deduped live verdict.** ~50 raw fires. Judge the DEDUPED
   pattern buckets (one launch farm dominates raw counts). Decides whether the
   22-feat model keeps its slot. Kill criteria: fire-rate collapse or
   non-monotone deduped buckets = population mismatch, do NOT recalibrate.
2. **~Jun 13 — multi-day pattern accumulation.** Want 15-20 distinct POSITIVE
   patterns over >=3 forward days before trusting the live edge. ~12-15 new
   distinct patterns/day.
3. **~Jun 14-16 — rich model deployable.** Serve-path parity DONE: non-intent
   159/159 AND intent structural 33/33 byte-identical (2026-06-10). A rich
   SHADOW scorer now runs live (shadow_rich_score) gathering score-alignment +
   the intent-timing gap. Still needs: trigger-not-ready race fix, a few more
   cross-day folds (1/day). NOTE: the morning 0.97 AUC was LEAK-INFLATED
   (n_total_trades_seen lookahead); clean rich AUC is 0.79 (~+0.01 over 22feat),
   a MODEST +0.09/fire execution-adjusted edge, not transformational.
4. **A few days — shadow death-cut verdict.** It must observe real rug events
   (it went live AFTER today's two -0.8 farm bleeds). Join `shadow_death_cut`
   events to realized closes; if would-cuts beat realized losses, promote the
   cut from shadow to active. Else a fixed stop.
5. **Only when armed — the execution gap.** Land rate, latency to leader, real
   slippage, adverse selection. UNMEASURABLE in dry-run. This is what
   "competitive" actually means, not AUC.

---

## 3. Arming sequence (the path to real capital)

Arm only when ALL hold: (a) verdict positive + ~15-20 positive patterns over
>=3 days; (b) a downside exit shipped (death-cut or stop) and replay-checked;
(c) the arming-prep changeset landed in one reviewed window (see section 4
PENDING items). Then:

- Fund 2-3 SOL, set bet to **0.05 SOL** (half of 0.1: same info, half blast).
- Flip `JITO_DRY_RUN=0` — requires explicit per-instance human approval.
- Run a 3-5 day EXECUTION-CALIBRATION phase whose goal is MEASUREMENT, not
  profit: land-in-decision-slot rate measured (fill quality is dominated by winning
  the launch slot, NOT a low-slip target -- high entry slip = momentum here),
  paper-vs-real net gap <=30% per fire.
- Reconcile, then scale stepwise 0.05 -> 0.1 -> 0.25 SOL, re-verifying the gap
  at each step. Capacity/impact matters: 0.1 SOL ~0.2% of a 46-SOL curve.

---

## 4. End state — the finished system (what all of today's features build toward)

DONE = shipped today (paper/dry-run safe). PENDING = built/specified, gated.
FUTURE = agreed direction, not started.

**Entry model**
- NOW: 22 K+V features, tp200-ranking head, thr 0.50, cross-era validated
  (month-gap AUC 0.78, monotone buckets). [DONE, live]
- END: rich 165 + intent 33 model (clean cross-day AUC 0.79, ~+0.01 over 22feat;
  the 0.97 was leak-inflated, corrected 2026-06-11). Modest +0.09/fire exec edge.
  Replace the 22-feat head once folds + checklist pass. [PENDING gate 3]
- Serve-path parity (non-intent): live builder reproduces training features
  byte-identical, 159/159. [DONE 2026-06-10]
- Intent features carry _present flags + NaN passthrough (no zero-fill). [DONE]
- Intent serve parity (ring vs jsonl) + live trigger-not-ready race. [PENDING gate 3]

**Exit policy (two-sided)**
- NOW: level_tp_50 (upside: sell all at +50%) + 300s stale watchdog. Upside
  only; no fast downside stop. [DONE, live]
- END: level_tp_50 + K=3 recovery/death-cut head (downside: cut when
  P(recover) low). Head trained, test AUC 0.842, running as shadow. [PENDING gate 4]

**Risk circuits**
- Era-scoped, SOL-unit daily-loss circuit (was cross-era + fraction/SOL bug). [DONE]
- Concurrent / rate / failure-rate caps. [DONE, live]
- END: per-pattern exposure cap (one launch farm was 75%+ of fires; cap repeat
  exposure to a single scripted actor). [PENDING]
- END: proper rolling-24h era-aware daily-loss window (current is era-cumulative). [PENDING, arming prep]

**Execution (Jito) path**
- Adaptive tip: outbid visible competitor p90, capped 1M/5M lam by cluster
  tier. [DONE, dry-run-logged]
- Blockhash fetch 1.5s timeout (kills the assembly stall tail). [DONE]
- SELL-side slippage protection retune (TP/urgent sells). ENTRY cap stays LOOSE:
  exec_sim (2026-06-10) showed high entry slip = momentum winners, a tight entry
  cap reverts the rockets (earlier "tighten entry" advice retracted). [PENDING, arming prep]
- END: route non-urgent sells (TP, stale) as regular tx with priority fee, keep
  bundle+tip only for urgent exits (saves ~0.5% of edge/bet). [PENDING, arming prep]
- END: tip-vs-land curve from recon outcomes — FIRST deliverable once armed. [FUTURE]

**Data / coverage**
- Live-matched extraction (train==live by construction), honest population
  (min_fwd=0). [DONE]
- Archive sync to storagebox (custody for the non-regenerable decision logs +
  parquets + artifacts). [DONE]
- END: second shred region endpoint — intent coverage is structurally 49% from
  one region; a second merged into the ring raises it, IF intent earns its
  keep in the rich model. [FUTURE, conditional]

**Data maturity (rare events + regime robustness)**
- The tail-dependent pieces need WEEKS, not days: the death-cut policy must
  see enough real rug events, the peak>=200% precision rests on rare winners,
  and cross-day folds must span weekend/weekday + volatility + SOL-price
  regimes before the edge is proven non-regime-specific. Target: >=1-2 weeks of
  continuous capture (we hold ~7,700 ready mints/day; rare events accumulate on
  the calendar, not by capturing more per day). This runs IN PARALLEL with being
  live at small size, so it is not a pre-arming blocker, it is what graduates
  small-bet -> meaningful size. [FUTURE, calendar-bound]

**Infrastructure / safety**
- Collector freeze guard + health tool (`pumpbot health`). [DONE]
- ssh key-only + fail2ban + needrestart list-only (apt can't bounce services). [DONE]
- pumpbot package (lazy surface) + CLI + 42 tests + ruff. [DONE, stage 2a]
- END: stage 2b package move (file moves + import shims + per-service restart
  drill), then CLI consolidation, typed config that refuses env/spec trigger
  mismatch, CI on push. [PENDING, planned window]
- FUTURE (optional belt-and-braces): non-root deploy user, UFW.

**Definition of "competitive" (the honest one)**
Not a higher AUC. A model whose edge survives REAL execution: bundles land in
the next block at a tip that prices the visible competition, realized slippage
matches the model's cost assumption, and paper P&L reconciles with live P&L
within tolerance. That is only provable in the armed calibration phase, which
is why arming small is a measurement step, not an income step.

---

## 5. Honest caveats (do not let these get lost)

- Diagnostics (AUC, replay net) are NEVER promises. The only honest tests are
  the live-vs-training score ALIGNMENT and realized win-rate-by-score-bucket.
- The live edge today is real but small (~+0.02-0.05 SOL/fire at 0.1 bets) and
  currently farm-carried; deduped per-pattern it hovers near breakeven on tiny n.
- The rich 0.97 was LEAK-INFLATED (corrected 2026-06-11); clean edge is modest. Treat as a hypothesis
  until multiple folds + a live verdict confirm it.
- Every "current/daily" stat must name its era; cross-era blending caused two
  real bugs today (the dashboard P&L and the risk circuit).
