# Shadow Harness — Decision & Progress Log

Live paper-trading verification of the validated pump.fun book (scale-out-into-strength +
precision death-cut). Measurement only: NO execution, NO funds. Test wallet has no capital.

Goal: measure the one thing backtest cannot — real fills, latency, live eligible deal-flow —
and reconcile live paper P&L against the OOS expectation (Findings 30/31).

---

## GOTCHAS — pipeline failures that cost real PnL, DO NOT REPEAT

### 2026-06-09 — TP50/K=5 catastrophe: the real cause was TRAINING POPULATION != LIVE POPULATION

**Symptom:** TP50/K=5 model trained with a reported 99% precision at top-10% OOS. Live, win rate was ~34% and the bot lost paper money for hours.

**First (INCOMPLETE) diagnosis. DO NOT stop here. The original version of this note did, and it was wrong.** The model added 9 sophistication features inner-joined from `sophistication_current.parquet` (extracted `--k 7`) while the live trigger is K=5. That K-window soph mismatch plus the inner-join survivor bias are real, and re-extracting soph at `--k 5` with a left-join does clean up the in-sample high-score band. BUT: (a) soph adds only +0.005 OOS AUC over the 22 K+V features, so it was DROPPED, not adopted; and (b) with soph corrected the model STILL did not transfer to live (live scores compressed, the training-top threshold fired ~0% live). Soph was a red herring, not the cause.

**Actual root cause:** the offline extractor built a DIFFERENT population than the bot trades. The live harness fires ONLY on the JOINT trigger (K=5 AND V=0.5; `shadow_harness.on_trade` fires on `result=="ready"`) and ONLY on CLASSIC-curve, fresh-launch tokens (`is_classic_curve` = `abs(vsol-30e9-rsol)<50e6`, plus first-seen rsol<3 SOL). The extractors applied neither the classic-curve gate nor the joint-trigger semantics. So:
- Training peak>=0.5 base rate was 0.47; the live firing population's is ~0.23. Training was a fundamentally easier, different population.
- The model's high-confidence region was populated by tokens the bot never trades. Live scores compressed, so the training-top threshold (0.9945, then "recalibrated" to 0.3741) was either unreachable or fired into the low-precision band. Recalibrating to a live percentile is a TRAP, not a fix.

**Fix that works (train==live by construction):** `tools/extract_live_matched.py` runs the ACTUAL live accumulator (`feature_accum.TokenState`) over `grpc_capture` with the EXACT harness gates (classic + fresh-launch), the joint K=5/V=0.5 trigger, and `peak_ret = run_max_ret`. Retrain the 22 K+V head with `tools/train_and_verify_live_matched.py` (regularized HGB). Results on 12,815 matched mints:
- live and training score distributions ALIGN (>=0.5 band: OOS 6.8% vs live 4.6%; the old broken model was 18% vs 0%); `uniq` caps at 5; base rate 0.23 matches live.
- the edge is REAL but MODEST: 5-fold CV AUC 0.70, chronological OOS AUC 0.71, monotonic win-rate-by-score-bucket (2.6% below 0.1 rising to 87% at 0.7-0.8). The "99% precision / 0.83 AUC" was a population artifact, not alpha.
- a deployable threshold exists, e.g. 0.35 -> ~11% live fire at ~43% precision (1.85x the 0.23 base rate), or 0.55 -> ~3.4% live fire at ~54% precision. Net profitability still depends on the exit-policy book economics, not yet confirmed.

**Data gotcha found along the way:** the capture has two schema eras. Newer rows tag trades `"event":"TradeEvent"` and also log non-trade rows; older rows (before ~2026-06-09) have no `event` key but carry the same trade fields. An over-strict prefilter on the `"TradeEvent"` string silently dropped ~60 older files, leaving only 1,598 matched mints and a misleading "no edge, OOS AUC 0.56" read. Keying the prefilter on the trade fields (`vsol`/`mint`, like `extract_from_capture`) recovered ~5.4M trades -> 12,815 mints and the stable 0.71 OOS.

**Rules to not repeat this:**

1. A training set is only valid if its POPULATION matches what the bot actually trades. Replicate EVERY live gate in extraction: trigger semantics (joint K+V), curve type (classic), fresh-launch (rsol<3). Check the base rate and key feature distributions (uniq, tot_sol) against live decisions BEFORE trusting any AUC.
2. Prefer reusing the live code path (`feature_accum.TokenState`) over the capture rather than a parallel reimplementation, and verify it reproduces the live score distribution (the ALIGNMENT check), not just in-sample AUC.
3. **If a training-top-X% threshold fires ~0 times live, STOP.** Do NOT recalibrate to a live percentile (that moves firing into the low-precision band and loses money). It is a population-mismatch signal; find and remove the cause.
4. Training AUC and precision sweeps are diagnostics, not promises. The honest tests are (a) the live-vs-training score-distribution ALIGNMENT and (b) realized win-rate-by-score-bucket on live decisions.
5. Whenever `K_TRIGGER` changes, re-extract EVERY parquet aligned to the new K; never reuse old K-anchored data. Inner-joins are silent filters; default to left-join + explicit NaN.
6. Append-only logs (`shadow_run.jsonl`) span model eras across restarts. Filter to the current process start time before diagnosing live behavior, or you will mix K=7-era and K=5-era decisions (this caused a false "uniq=7 / score-compression" alarm this session).

7. Forward-trade minimums are survivor filters. `extract_live_matched` MIN_FWD=5 silently dropped
   27% of ready mints (insta-dead at the trigger, median peak_ret 0.000). The live bot CANNOT apply
   that filter (it fires before forward trades exist), so training base rate was inflated
   (0.188 -> 0.254) and live score/fire-rate compression followed. Label the deaths (--min-fwd 0,
   added 2026-06-09); only guard against capture-tail truncation.


### Earlier gotchas in this codebase, for context

- `snap_every=1` modulo bug: `n % 1 == 1` is impossible (anything mod 1 = 0). 7027 fires produced 0 snaps and every exit_ret was 0. See `shadow_harness.py` around line 748 for the guard.
- V2 ABI account-layout bug: pump.fun v2 buy/sell instructions have mint at `accs[1]` not `accs[2]`, user at `accs[13]` not `accs[6]`. Before the fix, 26.4% of records had wrong mint. See `shred_bot/intent_extractor.py` for the version-aware dispatch.
- Initial TP50/K=5 deploy: threshold set to 0.9945 (training top-10%) caused 0 fires. (See the false trail above — recalibrating was the wrong fix.)
- ModelServer rich-path misdetection: the heuristic `feature.startswith("entry_")` matches the LEGACY 22-feat name `entry_sol`, silently routing 22-feat artifacts onto the rich feature builder. Caught by the packaging parity check 2026-06-10; fixed by honoring an explicit `rich_features` flag in model_spec (heuristic only as fallback). Pre-deploy parity checks work.
---

## Key decisions (newest first)

- **2026-06-11 — ARM-READINESS SAFETY FIXES (audit + fix all, validated). Was NOT arm-ready; now blockers closed.**
  Skeptical audit of the live path found gaps that bite only with real money:
    (1) CRITICAL: failed sell_all was NOT retried. The harness closes the position + open_paper.discard
        BEFORE the sell confirms (fire-and-forget), and _on_broker_failure couldn't retry sell_all (mint gone
        from open_paper). => a dump reverting our 15% slippage left us holding the full bag; the -30% stop was
        illusory exactly when needed. FIX: broker-side panic-retry — on a failed sell_all, re-fetch ACTUAL chain
        balance + FRESH reserves, escalate slippage (3000/5000/8000 bps), final attempt = MARKET sell (min_sol=1);
        aborts safely if curve migrated. **VALIDATED LIVE** (tools/force_sell_fail_test.py: forced all attempts to
        revert -> escalated -> attempt 4 market sell LANDED -> recovered to 0 tokens in ~36s). MAX_SELL_RETRIES=4.
    (2) Daily-loss circuit used PAPER P&L (era_book_nets) + had an n>=5 cold-start hole. FIX: when LIVE
        (broker not dry_run) gate on broker.realized_net_sol() = sum of CLOSED round-trip wallet lamport deltas
        (paired buy+sell; open positions don't drag it), active from the FIRST fire; DRY_RUN keeps paper accounting.
    (3) priority_fee_micro(2M)+cu_limit(200k) pinned in config.yaml (were code defaults).
    (4) BET FOOTGUN: --bet-sol argparse default was 1.0; create(bet_sol=args.bet_sol) meant config bet_sol was
        IGNORED and a missing --bet-sol would bet 1.0 SOL. FIX: default -> None (absent falls back to config). bet
        stays 0.1 via ExecStart. ARMING = ONE systemd edit (JITO_DRY_RUN=0 + --bet-sol <bet>) + daemon-reload.
  45 tests pass, bot restarted on safety code (dry_run=True, bet=0.1, 0 tracebacks), collectors up. Backups
  jito_broker.py.bak_pre_safety / shadow_harness.py.bak_pre_safety. Note: real-P&L circuit logic-validated (tests
  + live fill pairing) but not yet trip-tested by a live losing run. NOT armed (gated). Tools: force_sell_fail_test.py.

- **2026-06-11 — LIVE EXECUTION WORKS END-TO-END (buy + sell, real on-chain). Landing problem SOLVED.**
  Per the Jito docs (user-supplied): no auth needed; the real landing check is getInflightBundleStatuses by
  bundle_id; live tip_floor endpoint gives the TRUE landed tips. Findings:
    - LANDED tip_floor: p50 2k / p75 7k / p95 91k / p99 241k lamports (TINY). My intent-data "p90 1.5M" was
      whale-skewed (only 10% tip); tip was NEVER the blocker. 100k tip lands fine.
    - sendBundle on the public endpoint ACCEPTS (success+UUID) but bundle goes Invalid immediately (never enters
      auction) -> 1-tx bundles don't land. **FIX: switched broker to the sendTransaction proxy** (/api/v1/transactions,
      forwards directly to validator w/ MEV protection) -> jito_exec.send_transaction_b64; both _do_buy/_do_sell use it.
    - SELL layout was wrong (had 17 accts w/ user_vol -> threw 6057 BuybackFeeRecipientNotAuthorized because the
      recipient slot was shifted). Real sell = 16 accts: ...fee_config, fee_program, bonding_curve_v2@14,
      breaking_fee_recipient@15 (NO user_vol). Fixed build_sell_ix; all 8 breaking recipients then AUTHORIZED.
  VALIDATED LIVE (wallet supplied through local configuration): BUY via sendTransaction Confirmed in ~2s, filled 71500 tok (Token-2022 ATA
  created, priority fee+tip paid); SELL Confirmed, 0 tok left, SOL recovered. Also via the BROKER path: buy landed
  5.75s. Round-trip execution cost ~0.0035 SOL on a non-moving 0.002 bet (2 tips + 2 priority fees + ATA rent + fees).
  Bot RESTARTED on full working code (dry-run/gated), 45 tests pass, collectors up. The execution layer is now
  FUNCTIONAL on current Token-2022 tokens. NOT armed (still dry-run; arming = fund + flip JITO_DRY_RUN=0 on go).
  Tools: tools/sendtx_test.py, sell_test.py, landing_test.py, decode_sell_full.py, sim_buy4.py.

- **2026-06-11 — SHIPPED (dry-run): priority-fee ix + data-driven contention tip model.** Acting on tip_model.py:
  (1) jito_broker._compute_budget_ixs() prepends ComputeBudget set_compute_unit_limit(cu_limit, default 200k) +
  set_compute_unit_price(priority_fee_micro, default 2M = data p75) to BOTH buy and sell bundles. 90% of the
  competing field uses a priority fee; our bundles previously set NONE (likely a landing handicap). Configurable
  via cfg.broker.priority_fee_micro / cu_limit (getattr defaults). Buy WITH priority fee simulates CLEAN
  (tools/sim_buy4.py, SIM ERR None). (2) shadow_harness adaptive tip: REPLACED base*2/base*4 multipliers with
  contention-mapped floors from the competing jito-tip distribution — tier2 (hot, shred_buy_500>=4 or 2k>=8) floor
  1.5M (~competing p90); tier1 (mild) floor 400k (~p75-p90); tip = min(5M, max(base, floor, visible_p90*1.5)).
  45 tests pass, restart clean, collectors up. NOTE: still can't LAND until Jito AUTH is set up (the remaining
  infra blocker); these make us competitive once bundles are being included. NOT armed (dry-run).

- **2026-06-11 — TOKEN-2022 REWORK COMPLETE (sim-validated, deployed); live LANDING blocked by public Jito endpoint.**
  Cracked the full current pump.fun ABI via the canonical IDL + real on-chain buys + the chainstacklabs working bot:
  buy = 18 accounts (token-2022 ATAs, creator_vault, global/user volume accumulators, fee_config, fee_program,
  bonding_curve_v2 = PDA(["bonding-curve-v2", mint]), breaking_fee_recipient = one of 8 hardcoded fee accounts)
  + args end with track_volume=Some(true) ([1,1]). SELL = 17 accts (creator_vault before token_program, no
  global_vol_accum, breaking last). **simulateTransaction is CLEAN** (tools/sim_buy2.py: Buy -> fee GetFees ->
  Token-2022 TransferChecked -> success). Rewrote pump_fun_ix.py; wired jito_broker._get_mint_meta (fetches mint
  token-program owner + bonding-curve creator@offset49, cached) into _do_buy/_do_sell. 45 tests pass; bot RESTARTED
  on the corrected code (still dry-run/gated). Backups: pump_fun_ix.py.bak_pre_t22.
  LIVE LANDING BLOCKER (not code, not tip): canary buys at 100k/500k/1.5M tips on a DEAD uncontested token all
  EXPIRED — Jito ACCEPTS the bundle (success:true + UUID) but it never lands (status Invalid). Uncontested token =
  tip competitiveness isn't it; sim proves the tx is valid. => the public UNAUTHENTICATED Jito Frankfurt endpoint
  isn't reliably including our bundles. FIX = Jito auth (register keypair -> UUID) and/or multi-region submit /
  dual RPC submit. No funds spent (balance 11.6225 unchanged throughout).
  TIP ANALYSIS (tools/tip_model.py, 21k intents): **only 10% of competing buys use a Jito tip; 90% compete via
  PRIORITY FEE.** Competing jito tips p50=100k/p75=200k/p90=1.5M; priority_fee p75=2M/p90=4M micro. Contention low
  (mean 1.47, p90=4). MODEL to replace base*2/base*4: tip = competing-jito-p90 by contention (~1.5M contested,
  lower calm), AND our bundle currently sets NO priority fee while 90% of the field does -> add a ComputeBudget
  priority-fee ix (~p75 2M micro) as the bigger landing lever. Caveat: jito-tip coverage ~10% -> lower bound.

- **2026-06-11 — TOKEN-2022 REWORK: core done + validated to the Buy handler; blocked on buyback fee accounts.**
  Rewrote pump_fun_ix.py for the current pump.fun ABI: token-program-aware ATAs (Token-2022), creator_vault,
  global/user volume accumulators, fee_config (PDA(['fee_config', pump_program], fee_program)), fee_program.
  Verified vs canonical IDL (pump-fun/pump-public-docs idl/pump.json) AND real on-chain buys. SIMULATION PROOF
  (tools/sim_buy2.py): the Token-2022 ATA create now SUCCEEDS (was the original instruction-0 failure), and the
  buy reaches the pump.fun Buy handler. Accounts 0-15 CONFIRMED. NOTE buy/sell differ: buy=...8 token_program,
  9 creator_vault; SELL SWAPS (8 creator_vault, 9 token_program), no vol accumulators.
  REMAINING BLOCKER: buy needs 2 trailing 'buyback fee recipient' accounts -> AnchorError 6062
  BuybackFeeRecipientMissing (pump.fun fee-sharing-v2). idx16 uninitialized (created on first use), idx17
  fee-program-owned 208B holding ~8 pubkeys (a fee-distribution config), disc 99a64790b3bd89fb. Derivation is
  NON-OBVIOUS: tried IDL (docs truncate these), brute-force standard PDA seeds, discriminator match vs
  pump_fees.json, and account-data dump — none cracked it. Under-documented. NEXT: get exact derivation from a
  current working SDK (handles 'creator fee sharing') OR decode the fee-program CPI inner instruction of a real
  buy. Then add via build_buy_ix(extra_accounts=...), wire jito_broker to fetch token_program(mint owner)+creator
  (bonding curve offset 49), sim clean, re-run canary. NO funds spent (balance 11.6225 unchanged throughout).
  STAGING: rework staged at pump_fun_ix.py.t22_wip; ACTIVE pump_fun_ix.py REVERTED to legacy (consistent with
  un-wired jito_broker so a bot restart can't error). jito_broker tip-cache + base64 fixes ALREADY committed.
  COLLECTOR SERVICE NAMES are prefixed pumpfun- (pumpfun-grpc-capture, pumpfun-grpc-firehose, pumpfun-shred-firehose,
  pumpfun-shred-intents, pumpfun-storagebox-shipper) — all active; do NOT query bare names.

- **2026-06-11 — LIVE CANARY caught the live submit path is fundamentally broken; NO money spent.** Ran
  tools/tiny_live_roundtrip.py --yes (0.002 SOL). The live submission path had NEVER run (always dry-run) and
  the canary surfaced a cascade of real bugs, each fixed/diagnosed in turn, balance unchanged at 11.6225 SOL
  throughout (every failure was pre-landing, no tip paid since nothing landed):
    1. FIXED: get_random_tip_account() called Jito getTipAccounts PER BUNDLE -> public endpoint 429 -> SDK
       returns None -> Pubkey.from_string(None) crash. Now cached (jito_exec.get_cached_tip_account, fetch once
       + fallback set). Would have hit the bot on every fire.
    2. FIXED: bundle was base58-encoded but the SDK sends {'encoding':'base64'} -> Jito 400 Bad Request. Now
       base64 at all 3 send_bundle sites. After this, Jito ACCEPTS bundles (success:true + UUID).
    3. BLOCKER (NOT fixed): bundles accepted but go Invalid/expired at any tip (100k, 500k). simulateTransaction
       (tools/sim_buy.py) shows the BUY fails at instruction 0 (ATA create): InstructionError(0, IncorrectProgramId).
       Root cause: **current pump.fun mints are 100% TOKEN-2022** (tools/token_prog_survey.py: 12/12 recent mints
       owned by TokenzQd...), but pump_fun_ix derive_ata / build_ata_create_idempotent_ix / build_buy_ix /
       build_sell_ix all hardcode LEGACY SPL Token (TokenkegQ...). So every live buy fails sim -> Jito drops it.
  IMPLICATION: live execution is BLOCKED. The bot could never have executed on current tokens; the "ready to go
  live at 0.05" framing was predicated on execution working, which it does not. Paper/shadow P&L is a STRATEGY
  signal only; the execution layer needs real work first. NOT close to live.
  FIX NEEDED (substantial): make pump_fun_ix token-program-aware — derive_ata with the mint's actual token program
  (Token-2022), pass it in the ATA-create + buy/sell account lists; then re-sim build_buy_ix (the docstring says
  "WITHOUT creator" — the current pump.fun ABI likely also needs a creator_vault account, untested since we never
  got past the ATA). Validate iteratively with tools/sim_buy.py until the buy simulates clean, THEN re-run canary.
  WALLET: 11.6 SOL sits in a wallet whose key was pasted in chat (exposed). Rotate + move funds regardless.

- **2026-06-11 — SUCCESS-path canary built (tools/tiny_live_roundtrip.py) to test a real fire before arming.**
  Existing tools/tiny_tip_test.py only proves the FAILURE path (tip=1, never lands -> recon classifies failed
  + rollback). The new canary does ONE tiny REAL buy + immediate sell (~0.002 SOL) on a live curve with a real
  tip so it LANDS, validating the full path: blockhash -> bundle -> Jito -> land -> recon -> actual-fill log.
  Reads on-chain reserves via a validated BondingCurve parser (tools/curve_reader_validate.py: vtok@8/vsol@16/
  complete@48; 9/10 live curves sane, migrated ones show complete=True). Gated behind --yes; --dry rehearses
  with no submission (rehearsal PASSED: reserve fetch + broker.create + buy path + report all clean). Safety:
  tiny default, asserts complete=False + sane reserves, sells exactly the on-chain balance, isolated process.
  KEY SEQUENCING INSIGHT: placing the funded key in .env does NOT arm the bot (bot stays dry via its systemd
  JITO_DRY_RUN=1); the canary opts into live in ITS OWN process, so we test REAL execution in isolation, then
  arm the bot only after the canary passes. Does not touch collectors (separate process, external RPC for bh).
  RUN ORDER (after key placed, on explicit go): dry rehearse -> --yes one 0.002 round-trip (watch land+fill) ->
  if clean, flip bot service to JITO_DRY_RUN=0 + bet 0.05. NOT run yet (no key, spends real SOL).

- **2026-06-11 — VERIFIED dashboard + CLOSED the actual-fill recon TODO (pre-live plumbing).** User asked to
  vet the dashboard before real money. Findings: (1) CONFIG!=SPEC(level_tp_50) and fire-rate 1.67% vs 0.61%
  are BOTH real+expected, the spec records exit=level_tp_50 / test_fire_rate 0.0061 and we deliberately
  switched the live exit to stop30_cap120; the 4 fires were 2 distinct patterns (farm replay), deduped ~0.84%
  ~ spec. (2) ACCOUNTING VERDICT (pinned on real 35Fwy close): TRUST net(pol)/live_policy_net (~+0.0214) and
  GREEN net_return (~+0.0233), NOT the broker headline (~+0.0738). The broker books entry at the zero-latency
  DECISION price (lat0) capturing the full +74.7% mark; ReplayContext enters at snaps[0] (lat1) + own-impact +
  fees = realistic +21%. So era P&L +0.3053 / 24h +4.79 / all-time +11.59 'actual SOL' are OPTIMISTIC and
  dry-run estimates (recon=0, nothing submitted); the dashboard's 'GROUND TRUTH (broker)' label is backwards.
  Block-timing is the SAME phenomenon: we can't fill in the trigger block, realistic is next-slot fill (= lat1).
  (3) Block extraction verified correct: get_fresh_blockhash(cap) + _bh_or_none hard 1.5s stale-guard (skips
  rather than sends stale); bh_age p50 59ms; buy logs target slot + slot_gap; recon records real landed_slot
  (no live data yet). (4) Implemented _log_actual_fill in jito_broker.py (reconciler landed branch -> fetch tx
  meta -> log actual token+lamport delta vs expected = REAL slippage per fill); closes the 'reconcile vs chain
  after each fill' TODO. Best-effort, never raises into reconciler, LIVE-only. 3 pytest guards added (45 pass),
  dry-run restart clean. DECISION: keep risk caps as-is for the 0.05 plumbing phase (peak exposure 16*0.05=0.8
  SOL, never approach 16; -1.0 daily = ~20-loss headroom); revisit max_concurrent only at the 0.5 phase.
  PLAN: 0.05 plumbing test; key placed on host by user (never pasted); flip JITO_DRY_RUN=0 only on explicit go.

- **2026-06-11 — DECISION: KEEP stale_sec=300, no change** (supersedes the "lower to 150" recommendation above).
  On reflection the measurement undercuts the change: 150 vs 300 is P&L-identical (not better), we're not
  slot-bound (2-3 fires/hr into 16 slots) so faster recycling is worth ~0, and 300 is marginally SAFER for
  the live phase (more feed-gap tolerance before force-closing a live runner at a stale mark). stale_sec is
  P&L-irrelevant by construction (silent-token marks are frozen). The -30% hard STOP was the actual fix for
  the "300s eats profits" concern; stale_sec was a red herring. No config churn before a possible live arm.
  Revisit ONLY if a higher live fire rate makes us slot-bound.

- **2026-06-11 — MEASURED: the 300s stale watchdog earns ~nothing vs a flat 120s cap (tools/revival_value.py).**
  The only thing the 300s/event-driven design buys over a wall-clock 120s cap is the option to catch a late
  revival on a token that went quiet then traded again. Measured on OOS Jun10-11 (37 fires, deduped):
  CURRENT (event-cap+stale) +0.065 vs WALLCLOCK120 (frozen pre-120) +0.067 — identical (wall-clock a hair
  higher = noise). Only 3/37 fires diverge and their "silence gaps" straddling 120s are 4-15s (normal trade
  cadence, NOT real silence). Revival value = -0.0025 SOL total (1 pumped post-120, 2 dumped). Conclusion:
  no quiet-then-revive population exists in our data; an early-quiet token is a dud, not a sleeper. A flat
  120s exit is P&L-neutral and recycles the slot ~2x faster. RECOMMENDED minimal change: lower stale_sec
  300 -> 150 (keeps the event-cap's fresh-price exit for still-trading tokens; only quiet duds close sooner).
  NOT yet applied — offered, awaiting go.

- **2026-06-11 — ACTIVATED (paper): switched active exit level_tp_50 -> level_tp_50_stop30_cap120** (user approved).
  The bot now SELLS on +50% TP / -30% hard stop / 120s cap (was: +50% TP + naked 300s stale). tp_50 still
  logged as a shadow counterfactual so we keep the before/after comparison on identical fills. Restart clean,
  runtime confirms `exit policy = 'level_tp_50_stop30_cap120'`. config.yaml.bak_pre_stopcap kept.
  RESIDUAL (flagged to user): harness stale_sec is STILL 300. The policy's cap/stop only fire on an arriving
  forward snap, so SILENT tokens (trading stops, no further snap) won't hit the 120s cap and will sit until the
  300s stale watchdog. The -30% stop DOES catch violent dumps (rug = heavy sell flow = snaps crossing -30%);
  silent fades are the gentler residual. To fully close the "300s eats profits" concern, lower stale_sec ~150
  (>120 so it only affects truly-silent tokens) — OFFERED, not yet done. Note this is paper; not a proven alpha
  bump (n_pat=12 CIs overlap) but a mechanically-sound downside rule that forward-tests the armed-phase exit.

- **2026-06-11 — SHIPPED (shadow): hard-stop+cap exit policies, fixing the 300s loss-bleed.** loss_control.py
  (OOS Jun10-11, 37 fires, 9 non-winners) showed the live 300s stale watchdog lets bleeders decay
  (non-winner mean -0.715) and a TIMEOUT alone barely helps (cap90 -0.554, bleeders trade through it).
  A -30% HARD STOP cuts loser bleed to -0.486 and lifts elite all-fires deduped net -0.008 -> +0.090
  (best of all variants; stop30+cap120 +0.065/-0.487); hazard cut didn't help at elite (too few clean
  signals, n=9). Registered level_tp_50_stop30_cap120 and level_tp_100_stop30_cap120 (TP at +50/+100,
  hard stop -30%, 120s cap; uses pf ret+dts), added BOTH to the live shadow counterfactual race
  (now logging policy_nets for 6 exits). Decision logic verified (tp/stop/cap/hold all correct).
  ACTIVE exit UNCHANGED (level_tp_50). This is the proposed ARMED-PHASE downside rule: replace the
  naked 300s with stop30+cap120. Caveat: n_pat=12, CIs still overlap zero (directional). Tests 42 pass,
  restart clean, both in registry. NOTE for arming pre-flight: wire one of these as the ACTIVE exit
  (not just shadow) before real capital, the hard stop is the immediate robust loss-control; the 0.896
  hazard death-cut layers on later once its live shadow confirms.

- **2026-06-11 — Farm-independence test on RICH model (tools/farm_test_rich.py), head-to-head w/ 22feat, same novel split.**
  On the common (candidates+paths) set, train Jun7-9 / test Jun10-11: RICH generalizes too and is
  marginally stronger. Novel-pattern AUC(reach+100%): RICH 0.842 [0.805,0.877] vs 22feat 0.828
  [0.787,0.868]; rich leads on ALL/NOVEL/SEEN (+0.014 each) but CIs OVERLAP = directional not
  significant (consistent w/ rich = modest real edge, not transformational). Organic-only tp_100_t120
  net (top-3%): RICH +0.474 [+0.282,+0.605... +0.666] win67% (n=15) vs 22feat +0.389 [+0.163,+0.605]
  win63% (n=19) -- both CIs CLEAR ZERO, rich higher. IMPORTANT WRINKLE: on THIS window farm fires were
  NET-NEGATIVE (toxic 0.5091 rug dominated) so organic-only > all-fires (+0.02/-0.02), opposite of the
  full-set where the 0.5138 good farm made farms +0.240. => farms are a VOLATILE +/- component (good
  farm vs rug farm), the ORGANIC core is the stable positive in every cut (+0.097 full, +0.39 subset).
  Cleanest future strategy may DOWN-WEIGHT farm-pattern fires, not lean on them. Caveats: organic n
  small (15-19), magnitude unstable across populations (trust the SIGN not the level); rich>22feat not
  significant. CONFIRMS durable organic core for BOTH feature sets; rich marginally better.

- **2026-06-11 — DECISIVE: is there an edge WITHOUT the farms? YES on detection, PROBABLY-YES on P&L (tools/farm_test.py).**
  Farm = exact-repeat feature vector (scripted replay). Prevalence: 40% of fires are farm-cluster,
  **60% are unique organic launches**; ready universe 68% unique. DETECTION GENERALIZES (the key
  result): train Jun7-9 / test Jun10-11, split by pattern-seen-in-train. NOVEL patterns (4,145 mints,
  vectors never in train): **AUC 0.853 [0.838,0.870]**; SEEN/repeat patterns 0.702; ALL 0.828. The
  model ranks BRAND-NEW organic winners BETTER than the farms -> NOT farm-memorization, the signal is
  real and generalizing (farm-memorization would give ~0.5 on novel). ORGANIC-ONLY P&L: 108 unique
  fires, mean net **+0.097 [-0.011,+0.200], win 65%** -> positive point est, CI lower bound just kisses
  0 (very-likely-real, not yet 90%-sig). FARM fires +0.240 (scripted pumps, more profitable) = a
  TEMPORARY bonus on top, not the core. IMPLICATION if all farms vanish: lose ~40% of fires AND
  per-fire net drops +0.156->~+0.097; business thins but SURVIVES. Also corrects my sample-starve
  worry: organic fires are ALREADY distinct (n=108 independent), far more power than the
  farm-collapsed n_pat=17; the live window just happened to be farm-heavy (~50%). DURABLE CORE EXISTS:
  detection AUC 0.85 on novel + organic net +0.097/65%-win. The farms flatter it, they aren't it.

- **2026-06-11 (morning) — VERIFIED + COMMITTED: split the proven from the unproven; power/ETA estimated.**
  Re-checked tonight's claims with bootstrap CIs over patterns (tools/recheck.py, verify_entry_claims.py,
  power_estimate.py); separates two questions I had wrongly blurred into "everything is noise":
  * ENTRY EDGE = PROVEN, well-powered. Cross-day AUC (reach +100%) = **0.828, 90% CI [0.813, 0.842]**
    on 5,306 test mints / 559 pos; win-rate by score decile MONOTONE 0%->44%. The model genuinely
    detects winners. (Earlier rich-0.97 was a leak; this 0.83 on the 22-feat is the real, clean edge.)
  * MONETIZATION = sample-starved. At the elite thr, deduped per-fire net: tp_50 +0.088 [-0.19,+0.34],
    tp_100_t120 +0.077 [-0.25,+0.40], tp_100 -0.030 [-0.41,+0.33] (n_pat=14). All CIs span zero and
    overlap -> magnitude and best-exit are NOT yet decidable. peak>=1x entry "improvement" RETRACTED
    (flipped with label def, all CIs overlap) = was noise.
  * RETRACTIONS this session (all offline-analysis only; live/deployed path was clean each time):
    slippage cap advice, rich-0.97 (leak), peak>=1x entry, and the overstated profit certainty.
  POWER / ETA (live era pattern stats: n_pat=17, mean +0.082, std 0.274, ~13 NEW patterns/day):
    - strategy>0 at 90% conf: need ~31 patterns -> **~Jun 12**; at 95%: ~43 -> **~Jun 13**.
    - exit-policy delta of 0.10: paired, needs only ~6-8 CROSSING patterns (fast once crossers accrue).
  LIVE era now: 9,327 decisions, 63 fires (0.68% == spec), active +0.176/fire raw, win 53/63,
  deduped +0.086 over 17 patterns. Exit-race shadow VERIFIED correct (policies differentiate on
  crossing tokens; identical only on non-crossers, as designed; needs more post-02:33 crossers).
  ACTION: accumulate; meaningful strategy verdict ~Jun 12-13, policy verdict shortly after.

- **2026-06-11 — TAIL / MOONBAG / TIP study (tools/tail_and_tip.py). Three strategic answers, all directional (test n_pat 12-14).**
  TAIL DETECTION: +5x base 0.6% of ready (2.6% of elite fires), +10x 0.04%; entry-model cross-day AUC
  for >=2x 0.82, >=5x 0.65, >=10x unscoreable (3 test pos). We rank doublers well, moonshots barely,
  10x not at all -> the 1000x is real aggregate money but a lottery we CANNOT pick tickets for.
  MOONBAG REJECTED: runner sleeve is monotonically WORSE (w=0 pure tp100_t120 +0.170; w=0.10 +0.140;
  w=0.33 +0.104). Because we can't detect moonshots, the runner rides ALL fires; collapses outweigh the
  rare flyer even with the 0.896 hazard cut. The mu>=h*loss rule is right but mu on the runner
  population isn't high enough; a moonbag only pays if ROUTED to high-rocket-prob tokens we can't yet
  detect. TIP OPTIMUM ~ZERO: net(tip) is MONOTONICALLY DECREASING (0k -0.015 -> 1M -0.025 -> 5M -0.065
  -> 10M -0.115); every lamport is cost that landing gain doesn't recoup, BECAUSE we're 0.2ms from the
  Frankfurt node = already winning the slot on SPEED, so the tip is a pure tax. Keep base tip minimal;
  adaptive bump only in rare contested clusters (already the case). At 0.1 SOL a 1M tip=1% / 5M=5% of
  position. Caveat: 21% shred tip coverage understates contention so true T* is >0 but the SHAPE (more
  tip = less profit at our latency) is robust. ACTION: do NOT raise the base tip; do NOT add a moonbag
  until/unless a rocket-detector with real >=5x AUC exists. tp_100_t120 stays the exit.

- **2026-06-11 — EXIT FOLLOW-UPS (tools/exit_followup.py, hazard_shred_extract/auc.py): all answered.**
  (A) collapse-hazard AUC: path9 alone 0.896; +11 entry-K features 0.892 (slightly WORSE, constant-per-mint
  noise); so hazard is near-maxed on path features. (B/Q1) 50%->100%: of fires hitting +50%, only 48% reach
  +100%; conditional 50->100 time p50=42s p75=78s p90=157s -> validated the ~120s cap and the "grab 50 if
  100 unlikely" logic. (C) policy comparison TEST deduped (n_pat=14, NOISE-LIMITED, all tied): tp100_t120
  +0.138, tp100_grab50 +0.136, tp100_hazard +0.127, tp100_hazard_grab50 +0.129. => grab-50 floor and
  hazard-GATING do NOT beat the simple time cap on the elite winner population; the hazard model's payoff
  is as a DEATH-CUT on LOSERS (a population the elite threshold barely contains), not a winner-exit refinement.
  (SHRED test / user's jito instinct) path9 0.896 vs path9+shred 0.891 (worse) vs shred-only 0.547: shred
  sell-intent does NOT improve collapse prediction -- only 17% of snaps have shred intent in the 2s window
  (single fra6 region too sparse per-snap), and it's redundant with executed nsell_w/solo_sell_w/vel_w (on
  this asset the collapse IS the landed selling). Clean negative; would need the 2nd-region endpoint + still
  competes with an already-strong 0.896. SETTLED: tp_100+120s cap is the winner exit (shipped, shadow-racing);
  the 0.896 path-only hazard is the death-cut candidate for the LOSER population (stronger than recovery_candidate).

- **2026-06-11 — SHIPPED (shadow): level_tp_100_t120 exit policy.** From optimal_exit P2: sell at +100%
  else liquidate at a 120s time cap (median time-to-+100% was 99s; the cap beat uncapped tp_100 on the
  OOS fold, deduped +0.138 vs +0.115). Registered in exit_policies/level_tp.py (uses pf["dts"]); added
  to the live shadow-accounting set, so every position_close now logs counterfactual nets for
  {tp_50, tp_100, tp_200, tp_100_t120} and the diag races all four deduped. ACTIVE exit still
  level_tp_50 (unchanged); swap to whichever leads on accumulating live deduped patterns. Restart 02:33,
  clean, all shadows load, era cut 1781047096 intact. Tests 42 pass.

- **2026-06-11 — OPTIMAL-EXIT STUDY (tools/optimal_exit.py): optimal-stopping math, 4 parts, design Jun7-9 / one test look Jun10-11, deduped.**
  Framed exit as Snell-envelope optimal stopping with a COLLAPSE-HAZARD jump term: hold iff
  mu(X) >= h(X)*E[loss|collapse]; decomposes into a winner-drift estimator and a cut-losers hazard
  estimator (validates the user's Q3 that cut-losers is a separate, possibly stronger model).
  RESULTS: (P1 ceiling) a clairvoyant peak-sell gets DEDUP +0.65/fire; tp_100 captures 18%, tp_50 ~0%.
  5x theoretical headroom but the ceiling needs hindsight; the real question is what a CAUSAL rule
  closes. (P2/Q1) time-to-+100% median 99s (p25 70, p90 200); a TIME CAP improves the exit: cap 120s
  DEDUP +0.138 > uncapped tp_100 +0.115, generalized to test => SHIPPABLE refinement "tp_100 else take
  it by ~120s". (P3/Q3) **collapse-hazard head TEST AUC 0.896 vs winner head 0.792** -- cut-losers
  separates materially better, the ~0.9 the user predicted is REAL (collapse has loud microstructure:
  sell-pressure, dd, velocity). This is the rigorous death-cut. (P4/Q2) fitted backward-induction
  continuation value OVERFIT: train DEDUP +0.297 -> test +0.104, LOST to tp_100 +0.116; same trap as
  the afternoon LSM head; 153 train fires too few to estimate a continuation surface. Do NOT ship the
  fitted stopper. CONCLUSION: the robust generalizing winners are (a) tp_100 + ~120s time cap and
  (b) the hazard head as the downside cut -- NOT the elaborate boundary (correct in theory, premature
  on data). Next: the hazard head (collapse>=40% within rest-of-path) is a stronger death-cut than the
  current recovery candidate; candidate to replace it once the shadow comparison + more folds confirm.

- **2026-06-11 — EXIT-POLICY TOURNAMENT (tools/exit_lab.py) + tp_100 shadow accounting live.**
  Lab on the deployed pkl's fires (train Jun7-9 n=153 for design, ONE look at test Jun10-11 n=39;
  deduped judge, per-tranche fees charged, lat1 entries). PATH ANATOMY: P(cross+50)=65%,
  P(+100|crossed50)=48%, P(+200|crossed100)=10%; give-back after crossing 50 p50=0.28; worst
  single-snap gap p50=-23%/p10=-45% (collapses gap THROUGH trailing stops -> Finding 7 re-confirmed
  on elite fires); time-to-peak p50=100s. TOURNAMENT (11 policies: level TPs, time, ladders,
  bank+trail, trail-only, LSM optimal-stopping x2, score-conditional): **TEST winner = tp_100**
  (DEDUPED +0.116, med +0.748) vs incumbent tp_50 (DEDUPED -0.015). The 48% continuation past +50
  makes banking at +50 too early. TRAP AVOIDED: LSM continuation head WON TRAIN decisively (+0.30
  deduped) then collapsed to ~0 on test = overfit; the one-look discipline caught it; do NOT ship LSM.
  Trailing + ladders lose (gap severity + 1.5%/slice fee tax at 0.1 SOL). Caveats: test = 14 deduped
  patterns (small); tp_100 wins mean/median not tail (p25 ~equal); tp_100 was the pre-Jun9 configured
  exit (rediscovery with better method). DECISION: no live exit change; wired SHADOW EXIT ACCOUNTING
  instead: every position_close now logs policy_nets = counterfactual {tp_50, tp_100, tp_200} nets on
  the same snap timeline (harness _compute_live_policy_net extended; diag shows the deduped live race).
  Swap to tp_100 only if it holds its lead on accumulating live deduped patterns. Restart 02:06
  (model + active exit unchanged; era cut 1781047096 still valid for entries).

- **2026-06-11 — SETTLED reconciliation (3rd pass, strictest: EXACT deployed pkl + DEDUPED). I oscillated;
  this is the number to trust.** Deployed bot_artifacts_k3v03_final pkl, thr 0.50, cross-day test
  Jun10-11, 39 fires / 14 distinct patterns: lat0 raw +0.383 / DEDUPED +0.314 (87% win); lat1 raw
  +0.172 / **DEDUPED -0.015**; slot-aware raw +0.152 / **DEDUPED +0.005**. HONEST READ: at realistic
  latency the deployed elite point is ~BREAKEVEN on a fair deduped sample (+0.005 to -0.015), clearly
  positive ONLY at instant fill (+0.31 deduped). My two prior claims were both off: "clearly negative
  at realistic latency" (was the broad top-3% case) and "solidly positive +0.16" (was the RAW,
  farm-inflated number). Truth is between = deduped realistic-latency breakeven. LIVE puzzle: bot
  realized +0.20/fire (recomputed raw from position_close, n=51, policy & book agree) -- ABOVE offline
  deduped because live is RAW + small-n + farm-carried (paper_book.entry_lat_snaps=1 == offline lat1
  by construction, so the comparison is apples-to-apples on convention). Live +0.20 ~ offline RAW lat1
  +0.17 -> consistent once you compare raw-to-raw; the deduped breakeven is the conservative truth and
  the farm repetition is the upside that may or may not persist. NET unchanged in spirit: edge is real
  but thin, concentrated in instant fills + elite threshold + farm repetition; arming still hinges on
  land-rate. Leak conclusion (rich 0.97 = n_total_trades_seen lookahead, clean 0.80) UNAFFECTED, that
  was triple-confirmed. Methodological note: judge on DEDUPED + the exact deployed pkl, not retrained
  proxies or raw means -- both shortcuts moved my numbers this session.

- **2026-06-11 — RE-VERIFICATION (user asked to double-check across a model change): leak CONFIRMED, but my
  "latency kills it" conclusion CORRECTED (it was overstated).** Three independent confirmations of the
  rich leak: n_total_trades_seen univariate AUC 0.929 / corr 0.561 with peak; ablation clean 0.805 ->
  +n_total_trades_seen 0.978 -> +all6 0.980 (reproduces the morning 0.98). Clean rich ~0.80 vs 22feat
  ~0.78 stands. CORRECTION TO MY OWN PRIOR CLAIM: "positive only at lat0, negative at realistic latency"
  was true for the BROAD top-3% only, NOT the deployed ELITE threshold. At thr 0.50, cross-day test
  Jun10-11 (53 fires): lat0 +0.597, **lat1 +0.160, slot-aware +0.224** -- POSITIVE at realistic latency.
  And the LIVE bot's realized book +0.151/fire reconciles with offline ELITE lat1 +0.160 (NOT lat0), so
  the live dry-run is already a ~realistic-latency result and it is positive. Net: deployed elite
  operating point is net-positive at realistic latency and live-confirmed; the negative-latency warning
  applies to LOWERING the threshold (broad top-3%), which we must not do. Rich edge modest (+0.09/fire)
  and real. This re-check IMPROVED the outlook vs the earlier over-pessimistic read -- surface it as
  loudly as the pessimism. (Conclusions reproduced from scratch; independent of which model ran them.)

- **2026-06-11 — LEAK CAUGHT: this morning's rich AUC 0.97-0.98 was LEAK-INFLATED; clean rich AUC is 0.79.**
  tools/rich_exec_compare.py (join rich feats from candidates.parquet + forward paths from
  sweep_k3v03.pkl, same population, train Jun9 / test Jun10-11, top-3%). The morning train_rich_crossday
  feature filter included 6 columns the clean live-reproducible set (bot_artifacts_rich_shadow) excludes:
  **n_total_trades_seen (full-lifespan trade count = direct lookahead)**, decision_idx, k_idx, v_idx,
  decision_slot, first_ts. Those leaked survival/era info -> inflated AUC to 0.97. CLEAN result on the
  identical population: rich AUC 0.790 vs 22feat 0.781 (marginal on AUC). EXECUTION-ADJUSTED (clean,
  deduped): rich beats 22feat by ~+0.09/fire at EVERY latency (lat0 +0.375 vs +0.258; lat1 -0.003 vs
  -0.089; slot -0.061 vs -0.167). So rich's selection (higher-headroom rockets) is REAL and helps, but
  it lifts realistic-latency net only to ~BREAKEVEN, not profit. CRITICAL: the LIVE rich shadow scorer
  uses the clean 192-feature set (leaky 6 excluded), so NO live contamination -- only the morning's
  offline eval was wrong. CORRECTION: retract the rich-0.97 headline everywhere; honest rich edge is
  MODEST (+0.01 AUC, +0.09/fire exec-adjusted), not transformational. Binding constraint unchanged:
  launch-slot latency + elite threshold. (Second inflation catch of the session after the slippage
  retraction; both surfaced by deeper testing. The deployed/live paths were clean in both cases.)

- **2026-06-11 — K/V SWEEP (pre-registered, execution-adjusted): hypothesis REJECTED + a sobering OOS finding.**
  tools/sweep_extract.py + sweep_compare.py, both cells through the IDENTICAL pipeline, cross-day
  train Jun7-9 / test Jun10-11, judged on DEDUPED execution-adjusted net at matched top-3% selectivity
  (n_pat ~47, a real sample). RESULT: K=7/V=0.5 is WORSE than the deployed K=3/V=0.3 on every metric
  (AUC 0.70 vs 0.82; all exec cells negative). The "later trigger executes better" hypothesis is
  FALSIFIED; do not switch triggers. BIGGER FINDING (surface, do not bury): at top-3% OOS, K=3/V=0.3
  is positive ONLY at lat0 (instant fill, +0.110 deduped) and NEGATIVE at every realistic latency
  (lat1 -0.077, lat2 -0.102, slot-aware -0.083). The lat0->lat1 gap (~+0.19/fire) IS the whole edge =
  the launch-slot race. Implications: (a) trigger is not the lever, THRESHOLD and LATENCY are; (b) the
  edge concentrates in the elite top scores -- live verdict's +0.097 deduped was at thr 0.50 (top
  ~0.5%); by top-3% the OOS edge is already negative, so LOWERING the threshold goes negative and the
  deployed high thr is load-bearing; (c) the dry-run book (+0.15/fire) IS the lat0 optimistic case;
  reality at lat1+ is negative on the broad population. ARMING CONSEQUENCE: net profitability hinges
  entirely on near-lat0 landing (winning the launch slot) AND/OR firing only the elite threshold; the
  live land-rate is now THE determinant, not one of several. Data: data/sweep_k3v03.pkl,
  data/sweep_k7v05.pkl. Caveat: top-3% is broader than the deployed thr-0.50 elite; the elite slice
  (live n_pat=12 +0.097) is the only positive realistic-latency evidence and is thin.

- **2026-06-10 21:03 — FIRST SCHEDULED VERDICT fired (interim, healthy, NOT a kill).** k3v03_final era
  (since 1781047096): 5,567 decisions, 30 fires, rate 0.54% vs spec 0.61% (IN BAND, no collapse/no
  population-mismatch signal). Raw +0.152/fire, 83% win. DEDUPED +0.0966/pattern across 12 distinct
  patterns (the honest read; positive but n_pat=12 < the 15-20 target). Launch farm 0.513838 = 17/30
  fires (+0.244 ea, ~57% concentration); toxic rug pattern 0.509139 = 2 fires at -0.78 (main drag).
  Buckets monotone within the single 0.5-0.6 score cluster. Verdict mechanism note: the transient
  systemd-run timer FIRED at 21:03 then auto-removed (looked "vanished" on inspection at 21:29; it had
  simply completed). Output at logs/verdict_2026-06-10_2103.txt. Read: continue accumulating toward
  ~50 raw fires / 15-20 deduped positive patterns (lands ~Jun 11 midday); model is tracking its
  validated behavior. All collectors + both shadows healthy and gathering throughout.

- **2026-06-10 (evening) — EXECUTION + SLIPPAGE simulator: a prior recommendation OVERTURNED by data.**
  tools/exec_sim_extract.py (forward trades w/ slot[full cov] + jito tip[shred-sig join, 21% cov]) +
  tools/exec_sim.py compute, per fire (deployed 22-feat, score>=0.50, n=164): Model A fixed-latency
  {lat0 +0.426, lat1 +0.083, lat2 +0.076 mean net} vs Model B slot-aware+tip-rank landing {+0.040,
  identical across 100k/1M/5M tips because tip coverage too thin to rank}. KEY FINDINGS:
  (1) realized ENTRY slippage is huge: p50 +84.6%, p90 +103% (K=3 decides at a very low early price;
  realistic landing is a slot later, post-launch-pump).
  (2) COUNTERINTUITIVE + ACTIONABLE: high entry slippage correlates with WINNERS (momentum rockets),
  low slippage with LOSERS (duds): slip<10% bucket mean -0.44, slip>25% bucket +0.05/69% win. Cap
  sweep: TOTAL net highest at NO cap (+6.24), NEGATIVE at 10-25% caps. => a tight entry slippage cap
  is BACKWARDS for this momentum-launch strategy; it reverts rockets, keeps duds. **RETRACTION:** the
  earlier "tighten entry to 400-500bps" advice (STATE_OF_PLAY exec-gap section) is WRONG for entry;
  keep the entry cap loose. Sell-side slippage is opposite (still wants protection) and not yet modeled.
  (3) execution bracket is enormous and dominated by winning the LAUNCH SLOT (lat0 +0.43 vs slot-late
  +0.04), not the tip ladder (unmeasurable at 21% tip coverage) nor the slippage cap. land-rate is the
  live-only unknown. (4) at n=164 the realistic-latency MEAN (+0.08 lat1) is far below the 40-fire
  replay's +0.49; median stays +0.44. Median-positive, mean-fragile, farm-influenced => dedup matters.
  Saved data/exec_sim_result.json for accumulation + post-arming compare vs realized fills. Limits:
  21% tip cov (needs 2nd shred region or live), Model B "full-slot-late" is the pessimistic bound,
  lat0 the optimistic; truth between, set by launch-slot win rate.

- **2026-06-10 (evening) — INTENT structural parity PASS + RICH SHADOW SCORER live.**
  Intent parity (tools/parity_intent.py): jsonl-hindsight vs ring-faithful, 400 decisions, all 33
  intent features match to float epsilon (worst 7e-9 percentile rounding; net_limit_sol 7e-15). Proves
  units (SOL vs lamport), signer-vs-user identity, the 200-deque cap, and formula are all equivalent.
  The ONLY residual intent gap is live TIMING (intents in-flight at the decision instant), which is
  unmeasurable offline. KEY FACT: intent_recorder writes BOTH the jsonl and the ring from one stream,
  and both filter recv_ns<=decision_ts, so training already uses only intents available by decision
  time (not future) -- the gap is bounded to drain latency, which drain_now() at decision minimizes.
  RICH SHADOW SCORER (bot_artifacts_rich_shadow, 192 live-reproducible feats incl 33 intent, peak_ge_200
  head): wired into the harness LOG-ONLY (shadow_rich_score events), never acts. Accumulates
  rich_trade_hist for all fresh mints (same profile as the june_causal rich era) only because a shadow
  model is loaded; gated by self.rich_hist_needed. First minutes live: scores emit with n_missing_feats=0
  (full 192-vector computed live incl intent), intent_present=true; ~3 shadow_rich_error "rich trigger
  not ready" are the expected restart-boundary race (TokenState restored but rich_trade_hist post-restart
  only), self-clearing. This turns the live stream into rich-model evidence (live score alignment + the
  intent-timing gap via shadow_rich_score vs jsonl backfill) at ZERO deploy risk. Bot restart 19:01;
  22-feat model still the only ACTOR; verdict era cut UNCHANGED (1781047096). New tools: parity_intent.py,
  train_rich_shadow_artifact.py. Tests 42 pass.

- **2026-06-10 (evening) — SERVE-PATH PARITY for the rich feature set: PASS (non-intent).** tools/parity_rich.py
  reuses the offline load_trades (so the trade input is the EXACT sequence that built candidates.parquet),
  converts each mint's arrays to the live row schema, runs the live build_entry_features, and diffs every
  shared non-intent feature. Result on 200 sampled k3/v03 mints: decision-index match 200/200,
  trigger-not-ready 0, and **159/159 non-intent features byte-identical** (max abs diff 0 within 1e-6).
  This closes the train!=live feature-distribution risk (the TP50/K=5 failure mode) for the rich head's
  non-intent features: given identical trades the live builder reproduces training features exactly.
  STILL OPEN before a rich deploy: (1) intent parity is structurally different (live SHM ring, real-time,
  49% coverage vs offline jsonl-by-ts) and was excluded by design; measure how load-bearing intent is and
  either train ring-faithfully or treat as best-effort + presence-flag. (2) the live "rich trigger not
  ready" race is a live-TIMING artifact (decisions before the trade-history buffer fills), NOT a builder
  bug (0 failures here on complete histories); separate live-path fix. Gate 3 (rich deployable) now needs
  only: intent decision + race fix + a few more cross-day folds + alignment + checklist.

- **2026-06-10 — SSH hardened (user-approved): key-only auth + fail2ban.** PasswordAuthentication no
  + KbdInteractiveAuthentication no + PermitRootLogin prohibit-password via
  /etc/ssh/sshd_config.d/00-pumpfun-hardening.conf (00- prefix wins first-match; validated sshd -t,
  key login confirmed working before AND after reload = no lockout). fail2ban was already INSTALLED
  but never enabled/configured (hence "thought we had it"); now enabled with an sshd jail (maxretry 5,
  findtime 10m, escalating bantime 2h->1w). It banned 11 IPs in the first 4s from the journal backlog
  (86 total). Collectors + bot unaffected (health HEALTHY throughout). Rationale: the box faces
  constant internet brute-force and will hold a wallet key when armed; this closes the password
  surface. Remaining belt-and-braces (optional): a non-root deploy user, UFW to drop all but SSH +
  egress. Pre-arming security posture now adequate.

- **2026-06-10 (afternoon) — Collector health sweep: all green; found + neutralized the silent collector-bouncer (apt needrestart).**
  All five collectors active and writing (+3.8-6.4MB/3s per stream), ring advancing, disk 83% free,
  capture parse_fail=0. Investigated grpc-capture's unexplained 06:58 restart: apt-daily-upgrade ran
  06:57:42 and needrestart auto-restarted the service after a lib upgrade (graceful + gapless thanks
  to rotate-and-reopen, and the uninterrupted raw firehose covers the handover seconds; recoverable
  by design). But an auto-updater that can bounce collectors or the bot mid-position violates the
  do-not-disturb directive: needrestart set to LIST-ONLY mode (/etc/needrestart/conf.d/
  50-pumpfun-no-auto-restart.conf): security patches still install nightly, service restarts are now
  exclusively manual/planned. SSH posture noted while investigating: constant internet brute-force on
  root/guessed users; root is already key-only (permitrootlogin without-password) and root is the
  only login user, so the practical surface is closed; PasswordAuthentication=no + fail2ban
  RECOMMENDED as belt-and-braces before arming (sshd change left to the user: remote lockout risk).

- **2026-06-10 (afternoon) — K=3 RECOVERY HEAD built and validated; death-cut deployed as LOG-ONLY shadow.**
  Data: tools/extract_recovery_k3v03.py walks every joint-trigger mint's 300s forward window calling
  the LIVE TokenState.path_features per snap (train==live by construction): 1.04M snaps / 22,531
  mints / 187,868 drawdown snaps, recover-to-breakeven base 0.21. Head (validated convention: 9 path
  + 11 K feats, drawdown snaps, fm>=0 label): train Jun7-9 AUC 0.848, **test Jun10 AUC 0.842**,
  calibration monotone (P<0.05 bucket recovers 2.4%, P>0.4 recovers 62%). The HEAD is real.
  The CUT's book value at thr-0.50 entries is INCONCLUSIVE: replaying level_tp_50 +- death-cut on the
  deployed model's fired bets: Jun7-9 (n=153, in-train) cuts fire 9x and HALVE the tail (es10 -0.292
  -> -0.138, mean +0.421 -> +0.436); Jun10 OOS (n=11) the single cut was FALSE (clipped a winner,
  mean +0.532 -> +0.460). n=1 OOS decides nothing, and the current peaceful sample barely contains
  the tail events the cut exists for (farm flip / rug wave). DECISION: per the no-behavior-change-on-
  tiny-n rule, the head ships as a SHADOW: harness scores drawdown snaps of open positions and logs
  `shadow_death_cut` (p_rec<0.20, once per mint), never acts. The accumulated would-cut vs realized-
  outcome record decides the cut at arming prep. Artifact: bot_artifacts_k3v03_final/
  recovery_candidate.pkl (gitignored, storagebox-archived). Restart 4 (model-spec unchanged; verdict
  era cut still 1781047096). Tests 42 pass.

- **2026-06-10 (afternoon) — Bug-family audit after the risk-circuit find: 3 more fixed, 2 labeled, rest of live path clean.**
  Audited the live path for the same classes (era-blending, unit mismatch, accounting-basis mixing,
  restart semantics). FIXED: (1) zombie restored-open positions: after a restart, restored open mints
  had no last_trade_ts, and the stale watchdog's get(m, now) default ages them from "now" forever:
  a dead token restored open would NEVER close. Never bit (recent restarts restored 0 open), now
  seeded at restore. (2) era_policy_nets silently absorbed BOOK nets when a close lacked
  live_policy_net (shutdown force-closes): policy series is now pure-policy. (3) Dashboard open-
  positions panel derived opens from a 5000-row tail that snaps dominate ~95%: a live position older
  than the window was invisible; now sourced from the throttled full-scan cache. LABELED (dormant,
  stay OFF until era-aware): tools/drift_monitor.py and tools/auto_policy.py both use raw 24h/last-N
  lookbacks across model eras; auto_policy additionally ACTS on that blended window (policy swap).
  CHECKED CLEAN: stale watchdog for normally-fired mints (ts set on the firing trade), risk
  max-concurrent/rate/failure circuits (process-scoped, unit-consistent), paper book q_sol=bet_sol,
  broker holdings reservation/rollback, recon summary (live-only deque), exit-policy threshold units
  (fractions throughout), closed_mints re-entry block (intentional cross-era), diag joins.
  42 tests pass; restart 3 of the day (model unchanged; verdict era cut still 1781047096).

- **2026-06-10 (afternoon) — RISK CIRCUIT double bug fixed (era-blending + SOL/fraction unit mismatch); dashboard P&L made era-honest.**
  User spotted the dashboard "daily P&L -4.00" against the -1.0 limit. Decomposition: the -4.00 was
  arithmetically right but blended THREE model eras in the 24h window: broken tp50_k5 **-7.13 SOL**
  (108 roundtrips), june_causal +2.00 (46), current k3v03_final **+1.14** (14). Not a strategy signal.
  Worse, the BOT's internal daily-loss circuit had two real bugs: it summed book.returns() INCLUDING
  restored cross-era positions (a prior model's record gates the current one), and compared summed
  FRACTIONAL returns against a limit configured in SOL (10x mismatch at bet=0.1). Fixed: era-scoped
  (era_book_nets) and SOL units. Dashboard: risk panel now shows "era P&L (broker roundtrips)" vs the
  limit and the 24h all-era number dim + uncolored with an explicit not-a-signal label; drift panel
  flags PRE-DEPLOY-ERA checks as ignore instead of red-alerting on the previous model's distribution
  (the location_shift alert on screen was entirely june_causal-era data); recon failures filtered to
  the current era. Accounting note: era broker roundtrips (+1.14) > policy counterfactual (+0.32)
  mostly because real TP exits fill PAST the exact +50% trigger at snap granularity, while the
  policy/replay convention sells at tp-exact minus costs: the replay numbers are conservative on
  winners. Tests 42 pass; bot restarted 12:00 CEST (model unchanged; verdict era cut still 1781047096).

- **2026-06-10 (midday) — Forward-day verdict STRONG; rich features VINDICATED cross-day; data custody gap closed.**
  OFFLINE FORWARD TEST of the deployed k3v03_final on Jun 10 data it never saw (n=1,558 ready mints,
  honest population): AUC 0.768/0.784 (peak50/peak200), fire rate 0.71% (== spec), **11/11 fires peaked
  >=+50%** (72.7% >=+200%, mean peak +3.45), 4/4 deduped patterns clean; >=0.40 band n=35 at 94.3%.
  Jun 9 ref: AUC 0.791/0.809, deduped 87.5% over 8 patterns. Mild healthy decay, stable base rates.
  Live era meanwhile: 14 fires, +0.205/fire policy net, win 82% raw / +0.135 per deduped pattern.
  RICH/INTENT CROSS-DAY (the queued honest re-run of the june_causal idea; tools/train_rich_crossday.py
  over data/rich_crossday_20260610 candidates, day-boundary split Jun9->Jun10, NO selection, NaN
  passthrough + new intent _present flags): 22f-equiv test AUC 0.845/0.801 vs **rich 0.971/0.980**;
  top-5% band precision 74.4% -> **97.5%** (tp50, net/bet +0.378 -> +0.537) and 29.1% -> **59.5%**
  (tp200, net/bet +0.550 -> +1.494). The june_causal FEATURES were right; its intraday 20-fire
  selection methodology was what failed. Intent adds a small clean increment once presence-flagged.
  Next gauntlet for a rich candidate: more cross-day folds as rich capture accumulates, serve-path
  parity (build_entry_features vs offline builder), alignment, then the deploy checklist. NOT deployed.
  DATA CUSTODY audit + fix: storagebox previously held ONLY raw firehose+shreds (87G+82G); the filtered
  capture, intent capture, parquets, artifacts, and the NON-REGENERABLE bot_data decision logs were
  single-site on sol (RAID1 != backup). New additive unit pumpfun-archive-sync.timer (hourly :24,
  copy-only, bwlimit 25MB/s, frozen set untouched) -> /mnt/storagebox/backup/archive/{grpc_capture,
  intent_capture,data,artifacts,bot_data,repo(git bundle)}. Desktop-only May coherent CSVs (the
  deployed model's training source) uploading to archive/may_coherent_417849958/.

- **2026-06-10 — LAUNCH-FARM PATTERN CONCENTRATION found in first live fires; diag gained pattern dedup.**
  The k3v03_final era's first 4 fires (23:36-00:27) were four different mints with BYTE-IDENTICAL
  trigger states: score 0.513838, cum_buy exactly 16.245 SOL in 3 trades, identical vsK reserves.
  One scripted launch farm replaying a 16-SOL bundle playbook; the model keys on it just above thr.
  All four ran to TP50 (+0.266 net each) so it is currently a PROFITABLE pattern, but 4/4 wins = ~1
  independent observation, and a farm that flips behavior takes the model with it (concentration /
  bait risk). tools/live_bucket_diag.py now groups fires by exact score (identical score == identical
  vector for HGB) and reports deduped per-pattern stats; the n>=50 verdict must be judged on the
  DEDUPED buckets. Also: first adaptive tip-bump datapoint: tier-2 cluster with visible p90 tip 12M
  lam (vs 4M global p99 measured earlier); our 5M cap would have UNDERBID it: hot-cluster tip
  competition is heavier than the background distribution suggested; the bump log records
  p90_visible so any cap can be simulated offline before arming. Era instrumentation clean:
  0 feature errors since the 02:03 restart.

- **2026-06-10 — Stage 2a: additive package surface + COLLECTOR FREEZE GUARD; collectors verified healthy, untouched.**
  Constraint honored: the five collector services (grpc-capture, grpc-firehose, shred-firehose,
  shred-intents, storagebox-shipper) were not restarted, modified, or import-coupled to new code.
  Their import closure was mapped statically (15 files), checked into
  tests/collector_frozen_manifest.txt, and tests/test_collector_freeze.py now FAILS any commit that
  changes it (Restart=always means a broken closure turns the next incidental crash into a
  capture-losing crashloop; manifest changes only in a planned migration window with restart drills).
  Tonight's earlier instrumentation patches touched ZERO frozen files (verified). Health baseline at
  02:18-02:25: all five active, all outputs growing live (+5.5MB/3s capture, +9.3MB/3s firehose,
  +4.8MB/3s shreds, +53KB/3s intents), ring write_seq advancing, 86% disk free, storagebox mounted.
  New: `pumpbot/` package with PEP 562 LAZY re-export submodules (zero import side effects, zero file
  moves), `python -m pumpbot <bot|dashboard|diag|health|probe-shreds|extract|...>` CLI (runpy dispatch,
  byte-identical; collector entries deliberately NOT exposed), tools/collector_health.py read-only
  health check (exit-coded for cron). Tests 42 passing; ruff clean on all new code. Stage 2b (real
  file moves with import shims + per-service restart drills) stays gated on a planned window.

- **2026-06-10 — Instrumentation + execution wave; dashboard reporting corrected; project stage 1 (tests/tooling).**
  Measured first (tools/shred_coverage_probe.py, 2h sig-exact join): the shred intent feed sees only
  **49%** of executed fresh-classic buys pre-block (structural coverage, not recorder uptime), and its
  median lead over the gRPC processed feed is only **12ms** (p90 107ms; 5.7% a full slot). So the shred
  path's value is METADATA (pending tips/priority/spoof) rather than trigger latency. Tips observed:
  22% of buy intents tipped, tipped p90 1.04M lam; inside real 500ms fresh-mint clusters only 5.4%
  tip at all, p99 ~4M. The replay latency gradient prices ONE trade of latency at ~24M lam on 0.1 SOL
  bets, so the old fixed 2x/4x bump (200k/400k) was ~50x under marginal value when contested.
  CHANGES (dry-run-safe; bot restarted 02:03 CEST, model UNCHANGED): adaptive tip = outbid the visible
  p90 (1.25x tier-1 / 2x tier-2, caps 1M/5M lam); shred_window.drain_now() at decision time (kills the
  0-50ms background-tick staleness); intent feature presence flags (coverage-miss != quiet);
  jito_broker blockhash fetch 1.5s timeout (the 2.15s asm tail); status.json now carries model
  identity + a this-process era block; model_loaded event on startup.
  DASHBOARD corrections: the headline paper stats previously MIXED model eras (PaperBook restores
  prior-era positions at restart: closed=49 blended june_causal's 46 with the new model's 3) and used
  book accounting while the fires table used policy accounting; the risk panel's broker P&L
  multiplied actual-SOL logs by bet_sol AGAIN (10x understated at bet=0.1) and counted open buys as
  losses. Fixed: this-run vs all-time split, policy-accounting labels, roundtrips-only P&L, live
  fire-rate vs spec with mismatch coloring, config-vs-spec exit-policy warning, era column in the
  fires table, full-file rescans throttled to every 5th tick. Headless render test passes.
  PROJECT stage 1 (ARCHITECTURE.md has the staged plan): pyproject.toml (sklearn pinned 1.8.0 to
  match pickles), tests/ with 23 passing (ring roundtrip, rich-flag regression, exit registry,
  trigger/feature contract, shred-window semantics), ruff baseline: 1,276 legacy findings, live path
  clean of undefined-name classes. Stage 2 = package move with import shims + per-service restarts.
  Verdict-era note: the model deployed 01:18 is unchanged across the 01:18/02:03 restarts; judge it
  with `tools/live_bucket_diag.py --since 1781047096`.

- **2026-06-10 — CROSS-ERA VALIDATED K=3/V=0.3 model DEPLOYED (bot_artifacts_k3v03_final): tp200-ranking head, thr 0.50, level_tp_50 exit.**
  The month-gap test PASSED: trained on the May coherent span (Apr 29 - May 5, 70,669 live-matched ready
  mints, honest min_fwd=0 population, extracted locally from data_pull/coherent_417849958) and tested on
  ALL of June (21,567): peak_ge_50 AUC **0.784** (buckets monotonic 5.5% -> 95.4%, top bucket n=237),
  peak_ge_200 AUC **0.782**. Base rates stationary across the month (0.199 May / 0.188 June; tp200
  0.058/0.057). Final model trained May+Jun7-8 (n=86,725), Jun 9 holdout: AUC 0.806/0.812.
  EXIT-POLICY REPLAY (harness cost model: q=0.1, 250bps, 0.0015/tx, 300s stale; full capture paths,
  cached at data/replay_paths_k3v03.pkl): pre-stated adoption rule (lat=1 mean>0 with bootstrap
  P>=0.90, neighbor-threshold robustness, lat=2 floor >= -0.01, prefer smallest TP, tiebreak p25)
  left exactly one head/threshold: **tp200_head @ 0.50** with **level_tp_50** (Jun 9 OOS: n=40,
  lat1 **+0.488**/bet, win 95.0%, p25 +0.444, lat2 +0.056; Jun 7-8: n=104, +0.380). Expected live
  fire rate ~0.5-0.65% of decisions (~2-3 fires/hr). Deployed 01:18 CEST: symlink swap (backup
  bot_artifacts_K7V_pre_k3v03final_swap_20260609T231816Z), config.yaml exit.policy -> level_tp_50,
  restart clean (dry_run=True, checkpoint restore 17.8s downtime). These replay numbers are
  diagnostics, NOT promises: the honest verdict is tools/live_bucket_diag.py at n>=50 fires.
  june_causal model's final live record (5.4h era): 46 fires (2.3% rate vs spec 5.1%), book
  +0.124/fire, win 52% — positive but ~7x under its sweep promise, the expected winner's-curse
  deflation. Also fixed pre-deploy (parity check caught it): ModelServer rich-path misdetection
  on the legacy `entry_sol` feature name (see GOTCHAS).

- **2026-06-09 — june_causal K=3/V=0.3/TP200 deployed; selection-methodology risk + first live readout; MIN_FWD survivor bias found, fixed, cross-day candidate built.**
  Deployed 19:49 CEST: `bot_artifacts_june_causal_k3v03_tp200` (189 rich+intent features, thr 0.3223,
  exit level_tp_200, recovery head OFF). Selection risk flagged: argmax over 160 sweep cells x 10
  threshold candidates on ~20 val fires, all data from a single day (Jun 9, intraday chrono split),
  train AUC 0.998. Expect winner's-curse deflation. First 2h10m live: 968 decisions, 19 fires
  (1.96% vs spec val 5.1%/test 3.3%), 18 closed, policy net **+0.19/fire** vs promised +1.13, win 6/18.
  Also: training kept only intent-present mints (66%) while live only ~47% of decisions have nonzero
  intent windows (population mismatch on the intent axis); 12x `entry_feature_error` "rich trigger not
  ready" (~3% of decisions lost).
  **NEW GOTCHA (rule 7): MIN_FWD=5 survivor bias.** 27% of joint-trigger-ready mints have <5 forward
  trades (median peak 0.000 — they die at the trigger). Training excluded them; live fires on them.
  This inflated training base 0.188->0.254 and is the mechanism behind the chronic live fire-rate
  compression. Fixed with `--min-fwd 0` re-extraction: `data/live_matched_k3v03_all.parquet`
  (21,647 ready mints, Jun 7-9).
  **Cross-day candidate (the first real day-boundary validation in the project):** 22 K+V features,
  train Jun 7-8 (n=16,052), test Jun 9 (n=5,515): peak_ge_50 test AUC **0.807**, buckets monotonic
  3.4% -> 94.0% (0.9-1.0, n=67); peak_ge_200 test AUC **0.825**, top band 59.6% @ n=57. ALIGNMENT vs
  the 979 live decisions of the current era: live/OOS fire-rate ratio ~1.0 at thr 0.35/0.55 (the
  thin-inclusion removed the compression). Artifact: `bot_artifacts_k3v03_crossday/` — **NOT deployed**,
  single cross-day fold, entry-side only. Next: exit-policy replay (level_tp vs scale-out + recovery
  head) on the same OOS bets, then deploy decision. Rich/intent features become cross-day testable
  Jun 10+ (rich capture schema only exists since Jun 9). New tools: `tools/live_bucket_diag.py`
  (one-command live bucket diagnostic), `tools/extract_live_matched.py --min-fwd`,
  `tools/train_crossday_k3v03.py`.


- **2026-06-07 — SKEPTIC SWEEP 4/4 PASS — V+K7 at K=7 trigger is REAL, beats the leaky score.**
  Multi-seed shuffle null (n=10): mean **0.4902**, std 0.0445, range [0.425, 0.561]. Real V+K7
  OOS AUC 0.7734 is **z=6.4 std above null mean** — single-seed 0.5429 was just one draw from
  a wide null distribution (95th-pctile 0.5535). Shuffle-null PASSES decisively.
  Full skeptic results: (1) shuffle z=6.4 [PASS]; (2) K7+V orthogonality rho=0.77, stack lift
  +0.020 over best single head [PASS]; (3) latency margin survives entry_lat=4 (mean +0.090,
  P>0=99%) [PASS strongly]; (4) profit flat across all 4 OOS quartiles +0.40 to +0.45 each
  [PASS strongly]. Latency table at fee=0.0015:
    el=0: +0.495 (P>0=100%); el=1: +0.430; el=2: +0.254; el=3: +0.167; el=4: +0.090 (P=99%)
  WE BEAT THE LEAKY SCORE — apples-to-apples on the SAME OOS slice:
    metric          leaky K=10 (Finding 31, contaminated)   corrected V+K7 (validated)
    baseline                +0.234/bet                          **+0.507/bet** (2.2x)
    PLAUSIBLE               +0.094/bet                          **+0.430/bet** (4.6x)
    PESSIMISTIC             (tight/neg)                         **+0.243/bet** (flipped to positive)
  All corrected numbers are P(profit>0) = 100% bootstrap. Found and fixed a leak that was
  showing +0.094 PLAUSIBLE; built a better causal feature pipeline (V→K7+V) and ended up at
  4.6x the leaky number, this time real. Safe to swap production V=0.5 → K=7 with V+K7 head.

- **2026-06-07 — Live shadow on sol: pipeline runs but zero entries fire in 5 min.** 9,633 TradeEvents
  parsed in 5 min, 2,683 fresh classic-curve launches, 18 V=0.5 windows triggered, **0 entry decisions
  fired** (all scores below 0.3771 threshold). Could be small-sample noise OR a live-vs-training
  distribution shift in the entry score. Sample of logged decisions: scores 0.13-0.22, n_at_trigger=3,
  cum_buy_sol=1.8-3.1 SOL (significantly above the 0.5 trigger floor — typical sniper-bursty). Logged
  to `shadow_run.jsonl` on sol. Queued as Task 11: diagnose live-vs-training score distribution.

- **2026-06-07 — SILENT BUG in `pumpfun_parse.py` caught and fixed.** First live shadow run on sol
  parsed ZERO TradeEvents out of 8,002 log notifications. Root cause: `parse_program_data_line`
  calls `base64.b64decode` but the module only imported `base58` — `NameError` was silently
  caught by `except Exception` and returned None. Fixed by adding `import base64`. After fix:
  60/60 bddb payloads parsed cleanly on live stream. Classic catch-all-exception trap. Lesson:
  silently-swallowed exceptions hide everything — make the parser exceptions specific or log.

- **2026-06-07 — V+K7 stacking at K=7 trigger is the new production target (subject to one more rigor check).**
  Corrected V+K7 OOS book (entry at K=7 reserves, K7-anchored path snapshots, 2,969 top-decile bets):
    baseline: **+0.507**/bet (V-only +0.124), p5=+1154, P>0=100%
    PLAUSIBLE: **+0.430**/bet (V-only +0.051), p5=+936, P>0=100%
    PESSIMISTIC: **+0.243**/bet (V-only −0.008), p5=+448, P>0=100%
  First positive PESSIMISTIC case in the entire project. The K=7 trigger has natural survivor-
  selection (tokens that die before n=7 never enter the universe), and the K7 features stacked
  with V features give OOS peak AUC 0.7734 (V-only 0.694, +0.079). Trade-off: decision
  latency goes from V's median n=3 to K=7's n=7 (~4 more trades / few seconds of wait).
  For context, the OPTIMISTIC version (V-anchored entry reserves with K7 features added, i.e.
  faster-than-physically-possible entry) gave +0.408 / +0.277 / -0.014. Correcting to K=7 entry
  reserves should have DROPPED the number (higher cost basis) — instead it ROSE. Why: K=7
  trigger's survivor-filter benefit (~6,800 tokens drop out vs V) outweighs the higher entry cost.
  NEXT: build the K=7-trigger production stack (replace V trigger in feature_accum, retrain
  + repickle, parity-test, redeploy to sol). Caveat: still want to look harder for any subtle
  leak before committing — claim is "PESSIMISTIC +0.243 with 100% bootstrap confidence" is
  unusually strong. SKEPTIC SWEEP queued (task 12): shuffle-null, K7-vs-V score orthogonality,
  latency sensitivity entry_lat=3,4, time-of-day / slot-quartile bias check.


- **2026-06-07 — Closed-loop offline replay MATCHES analytical reference exactly.** Built
  PaperBook (scale-out + death-cut, AMM impact compounded across slices, per-tx fee) +
  offline orchestrator that walks OOS V snapshots through ModelServer + PaperBook. Using
  OLD-only-trained models (apples-to-apples vs analytical): mean **+0.0506** (analytical
  +0.051, delta −0.0004), bootstrap p5 **−30 SOL** (matches), P(profit>0) **90%** (matches),
  policy mix rider/cut/hold 2464/573/250. Live-pipeline LOGIC validated.
  Using PRODUCTION combined-train models (bot_artifacts_V05, for live deployment): mean
  +0.068/bet — slightly better due to more training data, not a real leak for live (OOS
  becomes past data when we go forward). Live expectation: ~+0.06-0.07/bet PLAUSIBLE.


- **2026-06-07 — V=0.5 LOCKED as production via OOS-book sweep.** AUC sweep alone left V=0.5
  vs V=1.0 essentially tied (V=0.5 best on peak 0.707, V=1.0 best on rocket by +0.001).
  OOS BOOK sweep at V∈{0.25,0.5,1.0,2.0} breaks the tie decisively:
    V=0.25: baseline +0.059 / PLAUSIBLE −0.010 / PESS −0.064 (P=34%)
    V=0.50: baseline **+0.124** / PLAUSIBLE **+0.051** / PESS −0.008 (P=90%)  <-- BEST
    V=1.00: baseline +0.098 / PLAUSIBLE +0.031 / PESS −0.037 (P=89%)
    V=2.00: baseline +0.056 / PLAUSIBLE −0.013 / PESS −0.074
  V=0.5 wins on both baseline (+0.026 over V=1.0) and PLAUSIBLE (+0.020). The slight in-
  sample rocket-AUC edge at V=1.0 (+0.001) does NOT translate. V=0.5 also fires sooner =
  more execution runway, and produces ~8% more bets. Production V=0.5 stays.


- **2026-06-07 — Metadata stack on V=0.5: REJECTED on OOS.** In-sample CV on OLD showed
  +0.015/+0.021/+0.018 incremental (peak/rocket/term0) — promising. But OOS book gets WORSE
  across all scenarios: baseline +0.124 → +0.097, PLAUSIBLE +0.051 → +0.030, PESSIMISTIC
  -0.008 → -0.027. OOS AUC essentially flat on peak/rocket; term0 actually dropped. The
  metadata features (creator history, URI hosts, jito tip norms) likely shift between OLD
  training and OOS forward window — combined model picked up era-specific patterns that don't
  transfer. PRODUCTION TARGET STAYS V=0.5 ONLY. The metadata work isn't useless (standalone
  disaster AUC 0.81 might still help as a hard pre-trade exclude filter, untested) but the
  joint-head stack is out. Rigor working: in-sample CV alone would have led us to ship a
  worse model.


- **2026-06-07 — V=0.5 BOOK BEATS K=10 ON OOS UNDER REALISTIC EXECUTION.** Apples-to-apples
  re-run of the OOS book + stress with V=0.5 entry trigger (instead of K=10 trade window),
  trained on causal OLD V=0.5, applied to causal OOS V=0.5 forward slice (32,877 unseen
  tokens, 3,287 top-decile bets). Same scale-out + death-cut policy, same stress definitions.
  RESULTS:
    baseline: K=10 +0.060/bet (P>0=96%) → V=0.5 **+0.124/bet (P>0=100%)** — 2x
    PLAUSIBLE: K=10 +0.007/bet (P>0=58%) → V=0.5 **+0.051/bet (P>0=90%)** — 7x, breakeven→edge
    PESSIMISTIC: K=10 −0.055 (P>0=4%) → V=0.5 −0.008 (P>0=26%) — still negative but near break
  **PLAUSIBLE is the realistic operating point** (~1 block of fill delay, which Confirmed RPC
  already provides). At that point, V=0.5 delivers +0.051/bet with 90% bootstrap probability
  of profit, 6/8 chunks positive, boot p5 −30 SOL. First time in the project that a realistic-
  execution causal book has a meaningful positive edge with high statistical confidence.
  CAVEATS: OOS AUC degrades from in-sample (peak 0.707→0.694, held5x 0.689→0.628 OOS) but the
  book still flips positive because V-windowed selection concentrates rocket density better in
  the top decile. Book is still tail-carried (top 2% = 134-331% of profit), same lottery
  shape, just with a fatter tail. Latency discipline matters: PESSIMISTIC (2-trade lat) is
  still negative, so the entry race must land in the next block to capture the edge.
  Metadata head not yet stacked — pure upside if applied.

- **2026-06-07 — Predicting tps causally does NOT recover the leak's value.** Quick test: train
  HGR(causal 10 features → leaky tps), use OOF prediction as a 12th feature. Spearman(pred,
  leaky)=0.36 — moderate. Adding pred_tps to entry model gives ~zero AUC change (peak 0.7465
  vs 0.7452 without tps). Confirms: the leak's value was future info (lifespan), not a feature
  engineering trick. Causal prediction is bounded by what causal features know, which the main
  HGB already extracts when trained directly. The way to recover lost signal is a DIFFERENT
  feature construction (like the V-window which conditions on volume-survival), not a
  predicted-tps surrogate.


- **2026-06-07 — VOLUME-WINDOWED entry beats trade-count windows materially.** V-sweep on
  causal features (cumulative-buy-SOL trigger replacing trade-count K). Best operating points
  vs best K (K=7): peak≥2x **0.707 @V=0.5 vs 0.691** (+0.016); held≥5x rocket **0.689 @V=1.0
  vs 0.660** (+0.029); term>0 **0.818 @V=2.0 vs 0.782** (+0.036). V=0.5 has 14% MORE tokens
  than K=7 (less restrictive trigger), yet higher AUC → genuine separation power, not a
  population trim. The hypothesis (volume separates organic burst from sniper noise) was
  right. Biggest win on the ROCKET target which carries the book's tail. NEXT: re-do OOS
  book + stress with V=0.5 entry vs K=10 causal baseline to see if realistic-execution case
  moves from breakeven into positive territory.

- **2026-06-07 — Launch metadata: small positive incremental, useful as pre-trade negative
  filter.** Causal features only (creator_prior_count, URI host class, name/sym gibberish,
  reuse counts, jito tip). METADATA standalone disaster AUC = 0.81 → strong pre-trade negative
  filter (exclude predicted disasters before entry model runs). Incremental over K=10 entry:
  +0.005 (peak), +0.008 (rocket), +0.012 (term>0), +0.010 (disaster). Not a transform but
  consistently positive, especially on rocket.

- **2026-06-07 — Manifold rung-gates on CAUSAL stack: 0.59 wall is REAL, not the leak.**
  Re-ran Finding 9 on causal features. AUCs land within ±0.05 of leaky-stack numbers (1.5x
  0.586, 2x 0.602, 3x 0.596, 5x 0.543, 10x 0.558). Confirms the rung-continuation wall is in
  the data, not in methodology. Stops the "ride past 2x with model-gated tranches" line
  cleanly — that frontier is bounded by the asset, not by feature engineering.

- **2026-06-07 — K-sweep: no big hidden edge at a different trade-count window.** Tested K in
  {3,5,7,10,15,20}. AUC band narrow (0.66-0.69 peak, 0.66-0.68 rocket). **K=7 marginally
  best** for peak (0.691); decision fires ~3 trades sooner than K=10 (more execution runway).
  Small win, not a paradigm shift. Most of the entry signal is robust to window size.


- **2026-06-07 — DECISIVE: causal book collapses to breakeven-under-execution. DO NOT DEPLOY.**
  Re-ran OOS book + stress on causal data (leak fixed). Causal OOS entry AUC peak>=2x **0.753**
  (leaky 0.838), terminal>0 0.693. Causal book (2915 OOS bets, 1 SOL, scale-out+death-cut):
  baseline (0 latency, 0 fee) **+0.060/bet** (boot P>0 96%); PLAUSIBLE (1-trade entry lat +
  small fee) **+0.007/bet** (boot P>0 only 58%, p5 −140) = BREAKEVEN; PESSIMISTIC (2-trade lat
  + fee.003) **−0.055/bet** (boot P>0 4%) = LOSES. vs leaky Finding 31 (+0.234 baseline / +0.094
  pessimistic, both artifacts). The leak fix specifically gutted the ROCKET-target AUC (0.77→0.66),
  which carried the book's tail → tail-mean shrinks to +0.06 → realistic latency+fees erase it.
  CONCLUSION: once the lookahead is removed, the edge sits AT the execution cost floor — the same
  place every prior pump.fun candidate died (Finding 23 / original landing). The market is near-
  efficient; the deployable-book result was a leak artifact. RECOMMENDATION: do NOT deploy real
  capital. Shadow harness still valuable for LEARNING + hunting a non-leaky edge, not as a business.
  Deal-flow (measured live on sol): 35 fresh-classic launches/min (~3.5 top-decile bets/min) — flow
  is fine; the EDGE is the problem, not capacity.


- **2026-06-07 — CAUSAL AUC quantified (re-extracted with the fix).** Causal tps median=2.0
  (10 trades / ~5s window) adds ~nothing vs dropping it (peak 0.7467 vs 0.7452) → ALL of tps's
  power was the leak. TRUE causal entry AUC: peak>=2x **0.747** (was 0.823), held>=5x rocket
  **0.664** (was 0.773), terminal>0 **0.767** (was 0.782). Entry signal is REAL but weaker;
  the ROCKET target (drives the book tail) is the weakest at 0.66. NEXT: re-extract OOS (running),
  retrain production models on causal features, re-run OOS book + stress → TRUE book economics.
  The +0.255/bet book result (Finding 31) used the leaky score and is NOT yet confirmed causally.

- **2026-06-07 — CRITICAL: lookahead leak found in #1 feature `trades_per_sec`.** The live-
  parity test (live accumulator vs offline parquet) caught it: offline `entry_feats` computed
  `trades_per_sec = n / (last_ts - first_ts)` where `last_ts` kept updating through FORWARD
  trades (entry_feats was called after the full stream), so it = `10 / token's entire lifespan`
  = LOOKAHEAD (encodes how long the token survives = the outcome). Proof: parquet tps 0.0003 ==
  10/32010s lifespan. SEVERITY (5-fold CV, drop the feature): peak>=2x AUC 0.823→0.745 (−0.078),
  held>=5x rocket 0.773→0.663 (−0.110), terminal>0 0.782→0.764 (−0.018). The leak inflated the
  entry + rocket AUCs substantially; terminal less so. The causal value (n/window-span) keeps
  some signal so true causal AUC is between 0.745 and 0.823. **ALL prior results that used the
  entry score (OOS Finding 30, book Findings 27-29, stress 31) are leak-contaminated and need a
  CAUSAL re-run.** FIX: extractor freezes window span at trade K (`window_last_ts`); accumulator
  already causal (that's why it caught it). ACTION: re-extract → retrain → re-OOS → re-stress →
  report TRUE causal edge before any further build. Caught pre-capital = shadow-first working.


- **2026-06-07 — Serve on Confirmed, log Processed too.** Current model trained on committed
  data → serve on Confirmed to keep train/serve matched. Skip rate is low (~0.1-0.3%) so
  rollback risk is small, but the binding risk is feature-distribution skew (ordering/
  completeness near head), which skip rate does NOT bound. Decision: serve Confirmed now;
  LOG the Processed stream in parallel to (a) measure the skew, (b) accumulate a Processed-
  native training set for a future latency-edge model (train==serve on not-yet-validated data).

- **2026-06-07 — Gate live to CLASSIC curve.** Live `bddb` TradeEvent stream is mixed-
  population (~42% classic in first sample). Classic pre-graduation curve: `vsol = 30 + rsol`.
  Training was 82% classic. Non-classic (pumpswap/variant) makes `fill_k` garbage → MUST gate
  to `is_classic_curve` AND only feed fresh near-launch tokens to the entry window.
  Implication: eligible deal-flow << raw firehose → live capacity lower than Finding-29/30
  concurrency suggested. Must re-measure eligible flow before trusting live throughput.

- **2026-06-07 — Models are pickled sklearn HGB.** `bot_artifacts/{entry,recovery}_model.pkl`
  trained on all 114,439 tokens. Entry AUC 0.83, top-decile threshold 0.5195. Recovery AUC
  0.84, death-cut 0.10. Serve = load pkl + predict_proba + threshold. Pin sklearn version on sol.

- **2026-06-07 — Replace slots.py naive policy.** Scaffold's fixed −10% stop + 3x/5x/8x
  tranches = the policy Findings 3/4/7 disproved. Paper book uses scale-out-into-strength +
  model death-cut (the validated policy) instead.

- **2026-06-07 — Execution architecture (spotter/sniper) is sound, stays OFF.** blockhash_cache
  = spotter, jito_exec = Frankfurt sniper, queue+slots = decoupled read/write. When eventually
  armed: pre-build/pre-sign before trigger, adaptive Jito tip. Not now (shadow only).

---

## Build status (V=0.5 production pivot)

- **2026-06-07 — V=0.5 production artifacts + accumulator + serving module READY.**
  Trained on combined OLD V + OOS V (128,749 tokens). Entry AUC train 0.704, top-decile
  threshold 0.3771. Recovery AUC train 0.805, death-cut 0.10. Saved to `bot_artifacts_V05/`.
  Updated `bot_shadow/feature_accum.py` to V-trigger (cumulative_buy_sol >= 0.5, MIN_N=3,
  window_last_ts frozen at trigger for causal tps). PARITY TEST: 11/11 features byte-identical
  to offline V parquet (max diff 0.00e+00 on all). `bot_shadow/model_serve.py` smoke-tested
  end-to-end (loads pickles, scores entry, scores recovery, applies thresholds correctly).
  Remaining: paper_book.py (scale-out + death-cut, mark-to-live, JSONL logger), shadow
  orchestrator (asyncio: listener + parser + classic-curve gate + fresh-launch detection +
  accumulator + model_serve + paper_book), then closed-loop offline replay validation, then
  deploy to sol.

## Build status

| component | status | notes |
|---|---|---|
| pump.fun TradeEvent parser | **DONE+validated+bug-fixed** | `pumpfun_parse.py`; missing `import base64` caught after silent failure on sol |
| curve-type gate | **DONE** | `TradeEvent.is_classic_curve` (vsol-rsol==30) |
| V=0.5 feature accumulator | DONE (SUPERSEDED) | replaced by V+K7 dual-trigger accumulator |
| V=0.5 model_serve | DONE (SUPERSEDED) | replaced by bot_artifacts_K7V loader |
| paper book (scale-out+death-cut) | **DONE+validated** | `paper_book.py`; closed-loop replay matches analytical to 0.0004 (V=0.5) and -0.0006 (V+K7) |
| V=0.5 shadow harness | **RETIRED** | replaced by systemd-managed bot |
| **pumpfun-bot.service** | **LIVE under systemd (paper mode)** | persistence + restart recovery; double-gated live |
| **pumpfun-grpc-capture.service** | **LIVE under systemd** | gRPC training-data recorder; independent of bot |
| eligible deal-flow capture | DONE | 35 fresh classic launches/min (~3.5 top-decile/min) |
| K=7-anchored extractor | **DONE** | `pumpfun_continuation_value_K7.py`; OLD+OOS K7 parquets extracted |
| corrected V+K7 OOS book | **DONE** | PLAUSIBLE +0.430/bet (P>0=100%); PESSIMISTIC +0.243 (P>0=100%) |
| **V+K7 dual-trigger feature_accum** | **DONE+parity 29181/29181** | `feature_accum.py`; byte-identical to offline K7+V parquets |
| **V+K7 model_serve (bot_artifacts_K7V)** | **DONE+smoke-tested** | 22-feat entry / 20-feat recovery |
| **V+K7 shadow harness orchestrator** | **DONE (not yet deployed)** | `shadow_harness.py`; dual-trigger logic; awaits diag-gate |
| **V+K7 closed-loop offline replay** | **DONE+wiring-OK** | `pumpfun_offline_shadow_replay_K7V.py`; -0.0006 SOL delta vs analytical |

## Open questions / to verify (next-up)
- ~~Skeptic sweep on V+K7~~ — **DONE 4/4 PASS (Task 12)**: shuffle z=6.4, orthogonality
  lift +0.020, latency robust through entry_lat=4 (mean +0.090, P>0=99%), profit flat
  across all 4 quartiles of the OOS window.
- ~~V+K7 bot script wiring~~ — **DONE (Task 13, this session)**: parity 29,181/29,181
  byte-identical against offline K7+V parquets; closed-loop replay matches analytical to
  -0.0006/bet.
- ~~Live score distribution diagnostic for V=0.5~~ — **DONE (Task 11)**: see "Calibration
  drift" section below. KS p=5.7e-11; live distribution shifted lower than training; V=0.5
  threshold 0.3771 fires ~1% live vs ~10% expected. Implication for V+K7 not yet measured.
- **Live score distribution for V+K7** (NEW, gated on this session's deploy): same diag
  on V+K7 JSONL after a fresh shadow run. Decides whether the 0.4481 threshold needs a
  live-recalibrated counterpart.
- **Training-population audit** — was the K7+V training set extracted with the same
  fresh_rsol<3 SOL filter the live harness applies? If not, the live distribution shift
  is a population mismatch and the right fix is a re-extraction + retrain, not a threshold
  patch. Open.
- Processed-vs-Confirmed feature/score divergence (dual-subscribe) — open from before.
- CreateEvent discriminator for clean launch detection (vs first-seen heuristic) — open.

## Calibration drift — V=0.5 live vs training (Task 11 finding)
The 15-min sol shadow on V=0.5 (Jun 7) ran 885s before crashing on a shutdown bug (see
"Shutdown bug" below), accumulated **76 windowed entries** of which **1 fired** (vs ~7-8
expected if training and live distributions matched at threshold 0.3771). Combined with
the preserved jsonl this gave **100 live entry_decisions** to compare against a 20k-token
training-set sample (`data/pumpfun_continuation_V05/live_score_diag.json`).

| stat | live (n=100) | training (n=20k) | delta |
|---|---|---|---|
| min | 0.0522 | 0.0236 | +0.029 |
| p25 | 0.1115 | 0.1518 | -0.040 |
| median | 0.1476 | 0.1908 | -0.043 |
| p75 | 0.1812 | 0.3126 | -0.131 |
| p90 | 0.2275 | 0.3793 | -0.152 |
| p95 | 0.2912 | 0.4120 | -0.121 |
| max | 0.7470 | 0.9981 | -0.251 |
| mean | 0.1577 | 0.2384 | -0.081 |

Kolmogorov-Smirnov 2-sample D=0.342, p=5.7e-11 -> distributions DIFFER significantly.
Live distribution is **shifted lower across the whole quantile curve** (not a tail
truncation — even the median is -0.043 below training).

`live scores >= threshold (0.3771)`: 1/100 = 1.0% (training: 10.5%).
Live top-decile threshold would be **0.2275** (drop of -0.150 from 0.3771).

Likely cause: **population mismatch**. The live harness drops tokens whose `real_sol_reserves`
on first observed trade exceeds 3 SOL (the FRESH_RSOL filter). The K7+V training extractors
do NOT apply this filter — they accept any token with valid reserves. So training includes
some mid-curve joins that score higher (more cumulative SOL, more uniq actors by V trigger
trade); live excludes them entirely.

Implication for V+K7: same FRESH_RSOL filter applies in live but not training. The V+K7
22-feature model will likely show the same downward shift relative to its training
distribution. The 0.4481 production threshold may fire less than 10% in live. We need to
MEASURE this directly with a V+K7 shadow before patching.

Decision: deploy V+K7 stack to sol as-is (FRESH_RSOL=3, threshold=0.4481), run a longer
shadow (60+ min), capture distribution, then decide between:
- (a) live-calibrated threshold (quick patch, weaker per-bet quality)
- (b) re-extract OLD+OOS K7+V parquets with the FRESH_RSOL filter applied + retrain
  (principled fix, day of compute)
- (c) RELAX the FRESH_RSOL filter in the live harness to match training (also quick,
  but admits more variable-quality tokens)

## Shutdown bug in shadow_harness (caught + fixed this session)
The V=0.5 shadow crashed at t=885s when the stale watchdog tried to close the 1 open
position via `self.log_event("position_close", mint=mint, kind=pos.kind, ...)`. The
`log_event(self, kind, **kw)` signature treats `"position_close"` as the positional
`kind` arg AND `kind=pos.kind` is a kwarg with the same name -> `TypeError: log_event()
got multiple values for argument 'kind'`. Fixed in `shadow_harness.py` by renaming the
kwarg `kind=pos.kind` to `exit_kind=pos.kind` in both the stale-watchdog and
shutdown-finalize paths. The V+K7 harness inherited the same bug from the V=0.5 harness
copy and is now also fixed.

## Production decision matrix (current)
| trigger | model | OOS PLAUSIBLE | OOS PESSIMISTIC | P>0 (PESS) | live latency |
|---|---|---|---|---|---|
| K=10 LEAKY (original Finding 31, contaminated) | leaky K=10 | +0.094 | (~0/neg) | — | — |
| K=10 causal (Finding 33, post-leak-fix) | K=10 entry | +0.007 | −0.055 | 4% | n=10 |
| V=0.5 (Finding 34, prior production target) | V-only entry | +0.051 | −0.008 | 26% | median n=3 (~few sec) |
| V=0.5 + metadata | V+META entry | +0.030 (rejected on OOS) | −0.027 | — | same |
| **K=7 (Finding 35, NEW production after 4/4 skeptic PASS)** | **V+K7 entry** | **+0.430** | **+0.243** | **100%** | median n=7 (~few sec slower) |

SKEPTIC SWEEP CLEARED V+K7 (4/4 PASS): shuffle z=6.4, orthogonality lift +0.020, latency
margin holds through entry_lat=4 (mean +0.090, P>0=99%), profit FLAT across all 4 quartiles
of the 39h OOS window (no single-burst bias). PESSIMISTIC +0.243 is the first meaningfully
positive pessimistic case in the entire project — and **4.6x the leaky-Finding-31 PLAUSIBLE
number on the same OOS slice, this time causal and validated**. Production swap from V=0.5
to K=7 with V+K7 stacked entry head is justified. Remaining gate: live-score-distribution
diagnostic (Task 11, in progress — 15-min sol run) before committing the swap, because the
current V=0.5 production threshold 0.3771 is firing ZERO entries in live (24+18=42 windows
over 10 min). Probably a recalibration job rather than a deeper issue, but verify first.

## Artifacts
- `bot_shadow/`: pumpfun_parse.py (bug-fixed), feature_accum.py (**V+K7 dual-trigger now**),
  model_serve.py (**bot_artifacts_K7V**), paper_book.py, shadow_harness.py (**V+K7**),
  parity_test_v_k7.py (**new**), tdwm_capture_sample.py, deal_flow_capture.py,
  SHADOW_HARNESS_LOG.md, capture_sample.json
- `bot_artifacts/`: K=10 production pickles (LEGACY, leaky-tps)
- `bot_artifacts_V05/`: V=0.5 production pickles (SUPERSEDED by K7V)
- `bot_artifacts_K7V/`: **V+K7 stacked production pickles (NEW CURRENT)** — entry 22-feat
  (AUC train 0.7664, top-decile threshold 0.4481), recovery 20-feat (AUC train 0.8163,
  death-cut 0.10); trained on 116,387 OLD+OOS joined tokens / 2.71M K7-anchored snaps
- `data/pumpfun_continuation_V05/`, `_oos_V05/`: V=0.5 parquets
- `data/pumpfun_continuation_K7/`, `_oos_K7/`: K=7-anchored parquets
- `data/pumpfun_continuation_oos_K7/V_plus_K7_corrected_book.json`: corrected V+K7 result
- `data/pumpfun_continuation_oos_K7/offline_shadow_replay_K7V.json`: **V+K7 closed-loop
  paper-book replay (this session)**: prod-model on OOS = +0.454/bet; strict OLD-only refit
  matches analytical reference to -0.0006/bet (wiring OK to floating-point)
- `pumpfun_offline_shadow_replay_K7V.py`: closed-loop replay script for V+K7
- host sol: /root/the-distribution-will-manifest
  - V=0.5 stack archived to `*.V05_bak`; old jsonl preserved as `shadow_run_V05_archived.jsonl`
  - **V+K7 stack DEPLOYED and RUNNING** (Jun 7, this session): `feature_accum.py`,
    `model_serve.py`, `shadow_harness.py`, `paper_book.py`, `pumpfun_parse.py`, and
    `bot_artifacts_K7V/{entry_model,recovery_model}.pkl + model_spec.json` rsynced to the research host
  - First V+K7 shadow run launched at t=1780801887 (PID 27703); stats line confirms the
    dual-trigger telemetry works (`k=0 v=1 ready=0 fired=0` after 15s); first
    `entry_decision` jsonl row at t=1780801921 (V at n=3 + 1.22 SOL cum_buy, K at n=7 4s
    later, score 0.136 -> no fire, threshold 0.4481)
  - At t+120s: 180 fresh launches, k_only=9, v_only=8, ready=6, entry_decisions=7
    (min=0.078, median=0.136, p90=0.213, max=0.213). **0/7 above threshold 0.4481.**
    Same calibration-drift pattern as the V=0.5 diag, this early sample just suggests
    V+K7 will show the same shift; rerun the V+K7 diag after >=100 decisions to confirm.
  - sklearn pinned 1.8.0 == training
  - Runbook: pull JSONL with `scp <research-host>:/root/the-distribution-will-manifest/shadow_run.jsonl
    shadow_logs/shadow_run_K7V.jsonl`, then `python pumpfun_live_score_diag_K7V.py`

## V+K7 bot-script wiring (Task 13, this session)
The shadow harness has been updated to use the new V+K7 stacked entry head end-to-end.

What changed in code:
- `feature_accum.py` rewritten as a DUAL-TRIGGER accumulator. Each `TokenState` tracks
  both the K=7 trade-count window and the V=0.5 cumulative-buy-SOL window from the same
  running state. Two independent 11-feature snapshots are frozen: K-features at trade #7
  with K-reserves (vsK, vtK), V-features at the first trade with cum_buy_sol>=0.5 AND n>=3
  with V-reserves (vsV, vtV). Decision happens at the later of the two trigger times.
  run_max_ret starts tracking from the K=7 trigger moment (K-anchored) even while waiting
  on V to fire, so by the time both have fired the path snapshots match offline K7 exactly.
  `update()` returns one of `skip|window|k_only|v_only|ready|fwd`.
- `model_serve.py` updated. Default artifact dir is `bot_artifacts_K7V`. `score_entry`
  takes a 22-feature dict (11 K + 11 V) and applies the new threshold 0.4481.
  `score_recovery` takes 11 K-features + 9 path features (V-features are NOT used by the
  recovery head because the recovery model was trained on K-anchored path snapshots only).
- `shadow_harness.py` updated. On `ready` the harness assembles the 22-feature dict by
  zipping `state.k_feats` against `ENTRY_FEATURE_NAMES_K` and `state.v_feats` against
  `ENTRY_FEATURE_NAMES_V` (`_v` suffix). On fire the paper book opens at K=7-anchored
  reserves (`vsK`, `vtK`). Stats now track k_fires / v_fires / both_ready separately.
- `parity_test_v_k7.py` (new). Streams a slice of the staged trades.csv through the new
  TokenState and verifies, for each token that fires BOTH triggers, that the K-features
  match `pumpfun_continuation_K7/token_level.parquet` byte-identically AND the V-features
  match `pumpfun_continuation_V05/token_level.parquet` byte-identically AND the reserves
  match both parquets. Run output on a 4M-row slice: **29,181/29,181 tokens PASS** on all
  four checks (K-features, V-features, K-reserves, V-reserves). The dual-trigger
  accumulator is byte-identical to running two independent offline extractors and joining.
- `pumpfun_offline_shadow_replay_K7V.py` (new). End-to-end closed-loop replay using the
  ModelServer (V+K7 pickles) and PaperBook (PLAUSIBLE settings: entry_lat=1 snap,
  fee=0.0015 SOL/tx, cost=250bps, max_slices=8, c_death=0.10). Prod-model on OOS gives
  +0.454/bet (mildly optimistic vs the OLD-only analytical reference because the prod
  model has seen the OOS tokens). The strict wiring check refits clf_e on OLD-only and
  routes through the same PaperBook: result **+0.4294/bet, delta vs analytical reference
  -0.0006/bet**. Wiring is bit-exact.

Next gates before pushing to sol:
- Wait for the in-progress live score distribution diagnostic on sol
  (`pumpfun_live_score_diag.py`, 15-min run; the diag is V=0.5 specific and is being kept
  until done so we don't disturb the running shadow). The diag answers whether the V=0.5
  entry threshold needs a live recalibration. If V=0.5 production is firing 0% live where
  training expects ~10%, the same issue may bite V+K7 production; we want to see the
  histogram before the swap.
- After the diag finishes, rsync the four updated bot_shadow modules + bot_artifacts_K7V
  to host sol, stop the running V=0.5 shadow, start the V+K7 shadow with a 60-min run.
- Re-baseline expected k_fires / v_fires / both_ready / entry_fire rates per minute on
  the new triggers (K=7 needs only 7 trades; V=0.5 still needs 0.5 SOL of buys). The
  JOINT trigger fires at the LATER of the two — typically V (cum_buy_sol >= 0.5 normally
  takes more than 7 trades on a healthy launch).

## On gRPC (user asked "do we even need grpc?")
**Short answer: no, V+K7 does not require gRPC.** The +0.430/bet PLAUSIBLE result is
already validated at `entry_lat_snaps=1` (one path-snap of latency, roughly 1-2 seconds of
trade-time after the K=7 trigger). PESSIMISTIC (`entry_lat_snaps=2`, roughly 3-5 seconds
of trade-time) is **+0.243/bet, P(profit>0)=100%** — still profitable, still extremely
robust. The skeptic sweep also tested entry_lat=3 and entry_lat=4 directly and both
remained mean-positive (entry_lat=4: mean +0.090, P>0=99%). So the system is alpha at
realistic websocket-Confirmed latencies on a vanilla RPC endpoint.

**Long answer: gRPC is a paid uplift, not a prerequisite.** Going from PESSIMISTIC ->
PLAUSIBLE buys roughly +0.187/bet of upside per filled bet. At roughly 3.5 top-decile
candidates per minute on the current deal-flow capture sample, that's ~5 SOL/hour
incremental if we fired on every top-decile candidate (unrealistic — we'll size down, but
the order of magnitude is meaningful). Whether that pays for a Jito/Triton/Helius gRPC
subscription depends on actual subscription pricing and bet sizing, both of which are out
of scope for this engineering log. The TECHNICAL conclusion is that the system is alpha
on plain websocket Confirmed and gRPC moves it from "good" to "very good", not "broken"
to "alpha."

## Live V+K7 distribution diag (Task 16, FINAL CLEAN RUN)
Pulled 69 V+K7 entry_decisions from the 26-min calibration shadow on sol; re-ran the diag
with `bot_artifacts_K7V` ModelServer against a 20k-row K7+V joined training sample (after
fixing the earlier contamination bug where the diag was reading old V=0.5 jsonl that had
11-feature scores from a different model). Result:

| stat | live (n=69) | training (n=20k) | delta |
|---|---|---|---|
| min | 0.0717 | 0.0272 | +0.045 |
| p25 | 0.1038 | 0.1339 | -0.030 |
| median | 0.1361 | 0.1889 | -0.053 |
| p75 | 0.1654 | 0.3275 | -0.162 |
| p90 | 0.2248 | 0.4453 | -0.221 |
| p95 | 0.2252 | 0.6547 | -0.430 |
| max | 0.3649 | 0.9993 | -0.635 |
| mean | 0.1419 | 0.2568 | -0.115 |
| std  | 0.0538 | 0.1900 | -0.136 |

KS D=0.351, p=4.7e-19 -> distributions DIFFER significantly.

**`live scores >= production threshold (0.4481)`: 0 of 69 = 0.0%** (training fires 9.9%).
Live max is 0.3649, well below 0.4481. The bot would NEVER fire in live under the current
threshold. Confirms calibration drift is real, large, and decisive.

The shift is more extreme than V=0.5 (V=0.5 saw 1% fire rate, V+K7 sees 0%) because the
V+K7 model is sharper at the tail — training median is 0.19 but training p95 is 0.65,
so any population shift collapses the high-score tail more aggressively.

Cause: same as V=0.5. The live bot drops tokens whose first observed `real_sol_reserves`
is >= 3 SOL. Training extractors don't apply that filter. The two populations are
different. Conclusion: option (b) re-extract + retrain is necessary.

## Production deployment (this session, Jun 7)
Two independent systemd services on sol, both `Restart=always` and enabled on boot.

### Architecture
```
gRPC stream  --(insecure :80)-->  pumpfun-grpc-capture  -->  disk (gzipped JSONL, hourly)
                                       (recorder, no model, no wallet)

websocket    --(wss Confirmed)-->  pumpfun-bot --> parser --> V+K7 accumulator -->
                                                              score --> PaperBook
                                       (paper P&L; persistence; restart recovery)
```

The bot does NOT read the capture's disk output. Capture is "free insurance" — keeps
recording regardless of bot health, retrains, or policy changes. The training data the
capture writes can be replayed into any future model, with any future feature set,
without depending on the current bot stack.

### `pumpfun-grpc-capture.service`
- Endpoint: `grpc-fra1-1.erpc.global:80` (PLAINTEXT gRPC, not TLS — `http://` scheme,
  port 80, `grpc.aio.insecure_channel`). Auth via `x-token` metadata header from .env.
  TLS endpoints (port 443) refuse the TCP handshake; only port-80 insecure works.
- Subscribes to pump.fun program TXs, scans `meta.log_messages` for `Program data:` lines,
  decodes via `pumpfun_parse.parse_trade_event`. Writes one JSONL row per parsed
  TradeEvent with all 10 fields + raw base64 (insurance against future parser bugs).
- Hour-rotated; rolled-off file is gzipped in-place. ~274 bytes/event compressed.
  Observed live: 27-40 events/sec. ~1 GB/day. Disk 432 GB free -> 400+ day runway.
- First 7 min of production: 9,270 events, 0 parse failures, 0 gRPC errors.
- File: `bot_shadow/grpc_capture.py`. Unit: `systemd/pumpfun-grpc-capture.service`.

### `pumpfun-bot.service`
- `pumpfun_bot.py` wraps `ShadowHarness` with three additions: PositionStore (atomic
  JSONL log of open/snap/close events), restart recovery (replay store, force-close any
  open positions using snaps collected so far, log `reason:"restart"`, add to
  `closed_mints` so we never re-enter), and status.json (every 30s, atomic rename, for
  systemd / monitoring polling).
- Mode flag is double-gated: live execution requires BOTH `--live` flag AND env
  `PUMPFUN_LIVE_OK=1`. Either alone refuses with `[bot] refusing to start in live mode`.
  Default is paper with `PaperBroker` (no-op).
- `JitoBroker` stub is wired (`jito_exec` import + .env wallet check) but `send_bundle`
  itself is `not_yet_wired` — even in live mode, the broker just logs intent. Real send
  wiring is a separate explicitly-authorized step.
- Recovery TESTED end-to-end on sol: seeded `positions.jsonl` with an unclosed position +
  2 snaps, restarted bot, saw "force-closed 1 position(s)", confirmed `close` event
  appended with `reason:"restart"`, confirmed mint added to no-re-enter set.
- File: `bot_shadow/pumpfun_bot.py`. Unit: `systemd/pumpfun-bot.service`. Persistence
  module: `bot_shadow/position_store.py` (smoke-tested + sol-tested).

### Control script
- `bot_shadow/scripts/pumpfun_ctl.sh` on sol at `scripts/pumpfun_ctl.sh`.
- Commands: `install`, `enable`, `disable`, `start-bot`, `start-capture`,
  `stop-bot`, `stop-capture`, `restart-bot`, `restart-capture`, `status`, `logs-bot`,
  `logs-capture`, `snapshot`, `go-live`.
- `go-live` is print-only — it explains the exact steps required to flip live (edit unit
  to add `--live`, uncomment `Environment=PUMPFUN_LIVE_OK=1`, daemon-reload, restart).
  Refuses to do it automatically.

### Shutdown-bug fix
The first V+K7 deploy inherited a TypeError-on-shutdown bug from the V=0.5 harness
(`log_event("position_close", ..., kind=pos.kind)` collided with the `kind` positional
arg). Fixed in `shadow_harness.py` by renaming the kwarg `kind=pos.kind` -> `exit_kind`
in both the stale-watchdog and shutdown-finalize paths. New units use the fixed code.

### systemd sandboxing gotcha
`ReadWritePaths` requires the named path to EXIST at unit-start time (the namespace is
set up before ExecStart runs). The first capture-start failed with
`status=226/NAMESPACE: /root/.../grpc_capture: No such file or directory`. Fix: the ctl
`install` command now pre-creates `grpc_capture/` and `bot_data/` before daemon-reload.

## On gRPC mode for the bot (planned, FEATURES MAY DIFFER)
Right now the bot is on websocket and the capture is on gRPC. The natural next step is to
let the bot consume the gRPC stream too for the latency uplift (PESSIMISTIC -> PLAUSIBLE,
+0.187/bet per fill). Plumbing-wise it is a small change: add a `--source ws|grpc` flag,
add a `listener_grpc` coroutine that subscribes the same way `grpc_capture` does and
yields parsed `TradeEvent`s to `on_trade`. The bot would have its own independent gRPC
subscription parallel to the capture's (one stream per process, no shared state).

IMPORTANT design note (user-flagged 2026-06-07): when we wire gRPC for the bot, keep in
mind the **feature set may differ** between sources. The websocket `logsSubscribe`
delivers only program log lines; the gRPC `Subscribe` delivers the full TX envelope
(slot, signature, pre/post balances, inner instructions, compute units, fee, etc). That
extra envelope can power features the websocket simply cannot:
  - sub-slot trade ordering (`tx.index`),
  - compute-unit consumption per TX,
  - inner-instruction sequencing (catch creator/dev wallet patterns),
  - exact fee paid (informs the priority-fee distribution we compete in),
  - slot-leader identity at TX time.
So the gRPC-mode entry/recovery heads may eventually be DIFFERENT models with a
SUPERSET feature vector. Don't bake the assumption "same features" into the bot
plumbing — keep the gRPC listener's parsed event type extensible. For v1 we'll start
with the same 22-feature V+K7 model on the gRPC stream (just to capture the latency win)
and reserve the gRPC-native richer feature set for a future model.

## Option (b) — re-extract + retrain on the live population (IN PROGRESS)
The diag finding is unambiguous. The bot's `FRESH_RSOL < 3 SOL` filter excludes tokens
the training extractors include. We must re-extract OLD and OOS for both K7 and V with
the same filter applied, retrain V+K7 entry + recovery on the filtered OLD, validate on
the filtered OOS, and ship new `bot_artifacts_K7V_fresh/`.

Plan:
1. Modify `pumpfun_continuation_value_K7.py` and `pumpfun_continuation_value_V.py` to
   accept `--fresh-rsol-lam <lamports>` (default 0 = no filter). When non-zero, only
   keep tokens whose first observed `real_sol_reserves` is strictly below the threshold;
   tokens that join mid-curve are silently dropped.
2. Re-extract OLD and OOS for both K7 and V at `FRESH_RSOL_LAM=3_000_000_000`, writing
   to `_fresh` sibling dirs.
3. Retrain V+K7 entry on OLD-fresh inner-joined K7+V. New threshold is the top-decile of
   training scores on the FILTERED population. Should be lower than 0.4481 if the
   population shift hypothesis is right.
4. Run the existing `pumpfun_K7_V_book.py` book against `_oos_K7_fresh/` to verify the
   alpha numbers hold on the filtered OOS (the population SHOULD be similar enough that
   PLAUSIBLE mean stays close to +0.43/bet; if it collapses, that's a different finding).
5. Build new `bot_artifacts_K7V_fresh/` pickles + model_spec.json. Swap on sol via
   `systemctl restart pumpfun-bot.service`. Re-run the live diag after another 30 min
   of shadow data; the new threshold should fire ~10% live.

### (b) RESULT — Finding 36: alpha STRENGTHENS on the matched population
All four re-extractions ran in parallel and finished in under 5 minutes
(`data/pumpfun_continuation_{K7,oos_K7,V05,oos_V05}_fresh/`). Token retention 58%
of the unfiltered population (e.g. OLD K7: 116,387 -> 67,421); the dropped ~42%
were mid-curve joins the live bot never sees. Base rate of `peak_ret >= 2x` on the
filtered population is **28.2%** vs ~25% unfiltered — fresh-launch tokens have a
modestly higher 2x continuation rate.

Refit V+K7 entry on OLD-fresh inner-join (60,704 tokens) and ran the OOS book on
OOS-fresh inner-join (86,033 tokens), top-decile = 8,603 bets:

| scenario | mean | median | win% | total | boot_p5 | P(>0) | (vs unfiltered ref) |
|---|---|---|---|---|---|---|---|
| baseline | **+0.705** | +0.089 | 56.1 | 6068 | 5443 | 100% | +0.507 -> +0.198 |
| PLAUSIBLE (el=1, fee=.0015) | **+0.620** | +0.036 | 51.9 | 5337 | 4718 | 100% | +0.430 -> +0.190 |
| PESSIMISTIC (el=2, fee=.003) | **+0.395** | -0.106 | 43.4 | 3394 | 2916 | 100% | +0.243 -> +0.152 |

OOS entry AUC peak 0.7746 (vs 0.7664 unfiltered) — sharper on the cleaner population.
**Bootstrap P(profit>0) = 100% across all three scenarios.** This is not a calibration
patch — it's a strict improvement. The mid-curve-join tokens were adding noise the
model had to fit around; removing them in training gives a cleaner signal AND matches
the live serving population exactly.

Production artifact built: `bot_artifacts_K7V_fresh/{entry_model.pkl,recovery_model.pkl,model_spec.json}`.
- Training: 146,737 tokens (OLD+OOS combined K7+V joined fresh).
- Entry: 22-feat V+K7 stacked HGB, train AUC 0.7743, **new top-decile threshold = 0.5108**
  (vs original 0.4481 on the unfiltered model — the new training distribution has a
  HIGHER tail because the model is sharper on the clean population, so the top-decile
  cutoff sits higher).
- Recovery: 20-feat HGB, train AUC 0.8061, death-cut 0.10 (unchanged policy).
- `fresh_rsol_filtered: true` flag in model_spec.json.

Deployed to sol:
- Old artifacts preserved at `bot_artifacts_K7V_v1_unfiltered/` for audit.
- New artifacts at `bot_artifacts_K7V/` (same path the systemd unit reads, so no unit
  change needed).
- `systemctl restart pumpfun-bot.service`: clean restart, recovery report
  `force-closed 0 position(s)` (no in-flight positions), connected + subscribed.
- 30s after restart: 70 fresh launches / 2 ready events / 0 fires (too early to
  characterize fire rate — need >=30 min of shadow data for that). The live V+K7 diag
  needs to be re-run against the NEW model after another 30+ minutes of accumulation;
  expectation is ~10% of `ready` events should now fire.

Files added this session:
- `pumpfun_K7_V_book_fresh.py` (OOS book on filtered population)
- `build_bot_artifacts_K7V.py` (reusable, parametric via --suffix)
- `pumpfun_continuation_value_K7.py` and `pumpfun_continuation_value_V.py` patched with
  `--fresh-rsol-lam` flag (defaults to 0 = no filter, fully backward compatible).

## Jito execution path — buy + sell + DRY_RUN (this session)
Real Jito sending wired end-to-end. Earlier `JitoBroker` was a stub logging
`status:not_yet_wired`; now it actually assembles signed pump.fun bundles. Default
JITO_DRY_RUN=1 = ASSEMBLE + LOG, never POST. Three new modules.

### `bot_shadow/pump_fun_ix.py` — Anchor instruction builders
- Constants: PUMP_FUN_PROGRAM, SYSTEM, TOKEN, ATA, RENT pubkeys; BUY_DISC = `66063d1201daebea`
  (`global:buy`); SELL_DISC = `33e685a4017f83ad` (`global:sell`).
- PDA derivations: `GLOBAL_PDA` (static), `EVENT_AUTHORITY_PDA` (static),
  `derive_bonding_curve(mint)`, `derive_ata(owner, mint)`.
- AMM math: `tokens_out_for_sol`, `sol_out_for_tokens` (constant-product), plus
  `slippage_max_sol_cost` / `slippage_min_sol_output` (1500 bps = 15% default headroom;
  pump.fun is high-impact, conservative slippage avoids reverts on co-buyers).
- Ix builders: `build_buy_ix(mint, user, fee_recipient, token_amount, max_sol_cost)`,
  `build_sell_ix(...)`, `build_ata_create_idempotent_ix(payer, owner, mint)` for first-time
  ATAs.
- Account layout: 12 accounts (global, fee_recipient, mint, bonding_curve PDA,
  associated_bonding_curve ATA, associated_user ATA, user signer, system_program,
  token_program, rent sysvar, event_authority PDA, pump.fun program).
- Smoke-tested on sol: PDAs derived for a real mint; buy ix data = 24 bytes
  (disc 8 + token_amount 8 + max_sol_cost 8); accounts in correct order with signer/writable
  flags. 1 SOL into a fresh curve buys 34.6T raw tokens.
- **KNOWN LIMITATION**: this layout does NOT pass `creator_vault` (a PDA derived from the
  token's creator) that some newer pump.fun versions require. Tokens with a non-zero
  creator field will need an extra account inserted between user signer and
  event_authority. Before going live on creator-tokens, extend `pumpfun_parse` to extract
  the creator field (at offset >= 177 in the TradeEvent payload) and add the creator_vault
  account to both buy and sell ixs. The DRY_RUN default makes this a non-issue for the
  assembly path; only LIVE submission on a creator-token would revert.

### `bot_shadow/jito_broker.py` — PaperBroker + JitoBroker
- `PaperBroker`: no-op, logs intents to `broker_paper.jsonl`. Used in paper mode.
- `JitoBroker`: async, fire-and-forget. On `buy`/`sell_all`/`sell_slice`, spawns a
  background task that builds the ix via `pump_fun_ix`, reads the cached blockhash
  (background loop started inside `JitoBroker.create`), assembles a versioned tx
  with [ata_create_idempotent, buy_ix, tip_ix] for buys (or [sell_ix, tip_ix] for
  sells), signs with the wallet keypair from `.env` (auto-detects base58 or
  JSON-byte-array format). Records the expected token holdings per-mint so
  subsequent sells know how much to send.
- DRY_RUN path: assembles, signs, logs metadata (bh, bh_age_ms, asm_ms, slot,
  tx_bytes, AMM outputs, slippage caps) to `broker_jito.jsonl`, does NOT POST.
- LIVE path: same assembly, then `jito_client().send_bundle(b58)` to Frankfurt.
- Same-block aim: each call accepts `slot` from the trigger event. WS path passes
  `None` (events have no slot); gRPC path passes the actual `tx.slot`. Logged so
  we can post-hoc measure how often we land in the same slot as detection.
- DRY-RUN smoke confirmed on sol: real wallet loaded, blockhash retrieved (73ms
  fresh), buy bundle assembled (1 SOL → 34.6T tokens, max_sol_cost 1.15 SOL,
  versioned tx 569 bytes, asm_ms=0), then sell_all assembled with correct AMM
  inverse (34.6T → 0.972 SOL out, min_sol 0.826 with 15% slippage). All ops
  sub-millisecond after blockhash is warm.

### `bot_shadow/listener_grpc_bot.py` — switchable listener (default WS)
- Drop-in replacement for `ShadowHarness.listener()` that consumes the same
  `grpc-fra1-1.erpc.global:80` stream as the capture, but as its own independent
  subscription (so the two processes don't share state).
- Attaches `tx.slot` to each parsed `TradeEvent` for same-block aim. WS path
  leaves `slot=None`.
- Selected via `--source grpc` on `pumpfun_bot.py`. Default `--source ws` keeps the
  validated path. Switching is one CLI arg; no model/feature changes required —
  the V+K7 model uses the same 22 features regardless of source.

### Switchable listener + feature-set future (user-flagged)
gRPC mode exposes the FULL tx envelope (tx.index, compute units, pre/post balances,
inner instructions, fee paid). Those are features the WS path cannot supply. When
we eventually train a model that consumes them, that model will be DIFFERENT from
V+K7 — load it under its own artifact dir and switch the bot's `--artifact-dir`.
The current swap is purely a transport change; the model is unchanged.

### Four gates to actually POST a bundle
Live execution is now possible only when ALL of the following are true:
1. `pumpfun_bot.py` started with `--live` flag
2. env `PUMPFUN_LIVE_OK=1` set on the systemd unit (currently commented out)
3. env `JITO_DRY_RUN=0` set (default `1` = assemble + log only)
4. Wallet has enough SOL for `bet + tip + signature fees`
   (test wallet is currently empty — Jito will return InsufficientFundsForRent on
   any LIVE attempt, which still validates the submission path)

The systemd unit file makes this explicit. `pumpfun_ctl.sh go-live` prints the
exact steps and refuses to do them automatically.

### Token-holdings tracking (TODO before real fills)
The broker tracks `self.holdings[mint] += expected_tok` at buy time. This is the
AMM-math estimate — real fills may differ due to slippage from concurrent buyers
inside the same slot. For paper mode the difference is irrelevant. Before flipping
JITO_DRY_RUN=0 with real capital, add a `getTokenAccountBalance` reconcile step
after each buy fill (and treat the chain balance as ground truth for subsequent
sells). Not blocking for DRY_RUN testing.

## ROADMAP — online learning / drift-triggered retrain (user-flagged this session)
The two data assets are already being recorded:
- `grpc_capture/*.jsonl.gz` — every observed TradeEvent (full population, including
  tokens we did not trade)
- `bot_data/shadow_run.jsonl` + `bot_data/positions.jsonl` — every `entry_decision`
  (with score + features-implied outcome) and every paper position (open/snap/close,
  with net_return and kind)

That gives us the four-way matrix needed for online adaptation:
  - TP: fired AND won (positions.jsonl `close` with net_return > 0)
  - FP: fired AND lost (close with net_return < 0)
  - FN: did NOT fire BUT would have been profitable (mints in capture with high
    forward returns that scored below the threshold)
  - TN: did NOT fire AND was bad (the rest of the population)

Roadmap (NOT BUILT YET; this is the design):
1. **Daily drift monitor** — automated `pumpfun_live_score_diag_K7V.py` against the
   prior 24h of bot_data jsonl. If KS p < some threshold AND the mean delta is
   meaningfully negative (live shifted lower), trigger a retrain candidate.
2. **Periodic incremental retrain** — weekly batch: re-extract from the most recent
   trades.csv (or the gRPC capture if we shift to that as the training source) with
   the same fresh_rsol filter, re-train V+K7, validate on a strict-OOS holdout, swap
   artifacts via the symlink pattern if metrics hold.
3. **Hard-negative mining from FN** — periodically replay capture events through the
   accumulator + score with the CURRENT model. Tokens that scored just below
   threshold but had high realized peak_ret are HARD NEGATIVES; bias the next retrain
   to weight them more. Avoid overfitting by keeping the original population balance
   and adding hard negatives as a SECONDARY loss term.
4. **Stability discipline** — never auto-deploy a retrain that doesn't beat the current
   model on the strict-OOS holdout by a min-uplift threshold. Maintain a baseline
   snapshot (current production) and only swap when the new candidate proves out.

Why this matters per user (paraphrased): the live distribution may shift over time
(macro regime, pump.fun program changes, bot competition), and we want to LEARN
from what we got right AND what we missed. Capture being independent of the bot
gives us the FN side cleanly — even the tokens we ignored are recorded with full
forward info.

## NOT YET DONE / explicit follow-ups
1. **Validate JitoBroker in --live + DRY_RUN against real triggers**: start the bot
   with `--live` AND `PUMPFUN_LIVE_OK=1` AND `JITO_DRY_RUN=1`. Wait for a fire
   under the new 0.5108 threshold (expected ~10% of `ready` events). Verify the
   assembled bundle in `broker_jito.jsonl` looks correct.
2. **Verify pump.fun ix layout against a real recent buy TX** before flipping
   `JITO_DRY_RUN=0`. Decode an assembled DRY_RUN tx and compare account ordering +
   discriminator + arg encoding against a known-recent solscan TX. This catches
   any drift in the program's ABI (especially creator_vault on newer tokens).
3. **chain-balance reconcile**: add `getTokenAccountBalance` call after live buys
   to use as ground truth for subsequent sells (replacing the AMM-math estimate).
4. **gRPC listener test**: confirm `--source grpc` actually produces faster
   buys when run against a real trigger. Compare bh_age_ms and asm_ms in
   broker_jito.jsonl between WS and gRPC runs.

## gRPC vs WS latency probe (Jun 7 this session, 30s side-by-side)
`bot_shadow/grpc_latency_probe.py`. Both sources subscribed in parallel, slot poller
running getSlot(Confirmed) every 200ms in background, 411 signatures seen on both.

| measure | result |
|---|---|
| gRPC arrived first vs WS, same TX | **411/411 = 100%** |
| median gRPC advantage | **+306 ms** ahead of WS |
| p10/p90 gRPC advantage | +174ms / +448ms |
| gRPC `receive_t - event_timestamp` | p50 1169ms, p90 1571ms (validator ts is 1s precision, so true lag is sub-second to ~1s) |
| gRPC vs RPC Confirmed slot view | events from slots **1-2 ahead** of `getSlot(Confirmed)` |
| stream throughput | 418 gRPC / 419 WS events in 30s (~14 ev/s) |

**Same-block aim — honest revision.** From these numbers, "submit a buy in the
same slot as the trigger event" is NOT realistic. By the time gRPC delivers the
trigger event (~500-1500ms after on-chain production) and we submit through Jito,
we are already 1-3 slots past the trigger slot. Achievable aim is **N+1 to N+2**
(gRPC) or **N+2 to N+3** (WS). The +306ms gRPC advantage is roughly **one slot's
worth of entry latency**, which matters because:
  - Earlier-slot entry = less AMM impact (fewer co-buyers consumed the curve)
  - Earlier-slot entry = lower price = better realized vs entry assumption
  - Matches the PaperBook entry_lat=1 PLAUSIBLE scenario more often vs entry_lat=2 PESSIMISTIC
Documented previous "same-block" framing as over-optimistic; the real benefit of
gRPC is the +1-slot advantage over WS, not zero-slot.

## Live scale-out / death-cut wiring (Task 26 this session)
**The GAP**: PaperBook does scale-out-into-strength retroactively at close (analytical
schedule: first profit -> linspace 8 slices to end, AMM impact compounded). LIVE
JitoBroker had `sell_slice` and `sell_all` methods that worked in DRY_RUN, but
nothing in `on_trade` called them during a position's lifetime. The bot would buy
and hold until stale-watchdog dumped at 5 min — the ORANGE strategy in the picture,
not GREEN.

**The fix**: wired live scale-out into `ShadowHarness.on_trade` forward-snapshot branch.
On every forward snapshot:
  - If `ret < 0 AND p_rec < death_threshold (0.10)` -> `broker.sell_all()`; mark dead.
  - Else if `ret > 0` -> `broker.sell_slice(frac = 1/(MAX_SLICES - n_sold))`; increment slice counter.
  - After 8 slices, mark dead (no more broker calls).

The fractional schedule sells exactly 1/8 of the ORIGINAL position per slice (verified
DRY_RUN-tested with 8 consecutive sell_slice calls — holdings progress 34.6T -> 30.3
-> 25.96 -> 21.63 -> 17.31 -> 12.98 -> 8.65 -> 4.33 -> 0). PaperBook keeps its
analytical retroactive schedule unchanged (so paper P&L stays the upper-bound reference).

**Race-condition fix**: `JitoBroker.buy/sell_all/sell_slice` now update
`self.holdings[mint]` SYNCHRONOUSLY at call time (before spawning the assembly task)
so subsequent rapid-fire sells see the fresh balance. Pre-fix, all 8 slices saw the
original holdings and the last slice would have sold 100% of position again (massive
over-sell). Post-fix: clean 1/8 increments per slice.

**Differences vs analytical retroactive** (worth knowing):
  - Retroactive picks 8 EVENLY-SPACED snaps from [ip, end] — knows the future
  - Live picks the FIRST 8 profitable snaps in order — front-loaded
  - For pump.fun where peaks are typically <10 trades after entry, front-loaded
    sells while still profitable may actually be closer to optimal than waiting
  - Future enhancement: introduce a pacing parameter so live spreads slices over
    the expected lifetime; would need an estimator of remaining profitable snaps.

`live_slices` and `live_death_cuts` counters added to `stats` for monitoring.

## Calibration deep dive (this session)

### First live fires under threshold override 0.20 (DRY_RUN validation)
Bot in `--live` + `JITO_DRY_RUN=1` + `--entry-threshold 0.20`. First fire was mint
`CYqsQw3iNXSqAT31HWtqK1ZST7T32iHQABfbAEoDpump` at score 0.218 (n_at_ready=7,
cum_buy_sol=4.14 SOL). Full lifecycle in 7.3 seconds:
  - buy: 1 SOL -> 26.8T tokens (max_sol_cost 1.15 SOL, bh 22ms fresh, tx 569 bytes, asm 0ms)
  - 8 scale-out slices at frac 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1.0 -- each removed
    exactly 3.354T tokens (1/8 of original). **Race fix verified end-to-end.** Holdings
    progressed 26.8T -> 23.5 -> 20.1 -> 16.8 -> 13.4 -> 10.1 -> 6.7 -> 3.4 -> 0.
  - Total bundle assembly: 9 signed VersionedTransactions, all DRY_RUN_assembled, none POSTed.
  - blockhash cache stayed within 22-803ms across all 9 (60s validity window so fine).
  - slot=null in all entries (WS source; gRPC needed for slot capture + same-block aim).

### Live winner-rate diagnostic (Task 16-followup)
`bot_shadow/live_winner_rate_diag.py`. For each ready mint in bot's shadow_run.jsonl,
cross-reference forward trade history in grpc_capture archive, compute realized peak_ret.

Result on 88 ready mints with sufficient forward data:

| metric | value |
|---|---|
| LIVE winner rate (peak_ret >= 2x) | **9/88 = 10.2%** |
| LIVE 5x rate | 1/88 = 1.1% |
| LIVE 10x rate | **1/88 = 1.1%** |
| TRAINING (fresh-filtered) base rate | 28% peak >= 2x |
| delta | -17.8 pp (live way below training) |

Score distribution by realized outcome:
  - Winners: n=9, mean 0.1975, median 0.1976, **max 0.3323**
  - Losers:  n=79, mean 0.1374, median 0.1288, **max 0.3215**

The model is sorting in the right direction (+0.06 mean separation winners vs losers),
but the separation is WEAK on live. Highest-scoring loser (0.32) is essentially tied
with highest-scoring winner (0.33). The one 10x token in the sample DID score in the
top (0.33) — model caught it — but the next-highest mints are a coin flip.

### Partial-history theory FALSIFIED (this session, Task 27)
Hypothesized that the calibration drift was caused by the bot connecting mid-stream and
missing pre-observation trades — meaning the bot's K=7 fires after the actual 10th-12th
on-chain trade, with features differing from training. `bot_shadow/partial_history_diag.py`
replays each ready mint's full grpc_capture trade history through TokenState, compares
to the bot's logged trigger state.

Result on 109 mints with sufficient capture coverage:
  - n_pre_obs == 0 (bot saw all trades): **107 / 109 = 98%**
  - n_pre_obs > 0 (bot missed pre-trades): 2 (one outlier with n_pre=12, one with n_pre=1)
  - midK exact match: 103/109 = 94%
  - vsK/vtK exact match: 102-103/109 = 94%
  - 6% mismatches are WS-vs-gRPC trade-ordering jitter (different commitment view), not
    missed history.

**VERDICT: bot's feature pipeline is NOT the problem.** Bot and capture agree on the
trigger features. The calibration gap is downstream.

### Real cause: regime drift between training era and live
Training data was extracted from May trades. Live is June. The drop from 28% -> 10%
winners is consistent with a fresh-launch-quality regime shift on pump.fun: more
competition, more bot-vs-bot, broader memecoin macro, etc. The model is technically
correct (still sorts winners above losers) but its absolute score scale is calibrated
to a population we're not in anymore.

**Options forward (in order of preference):**

1. **Re-extract from grpc_capture archive** once it has 3-7 days of accumulated data.
   This trains on the regime we actually trade in. The capture is already recording
   exactly the firehose we need. Build a "capture-to-extractor" pipe (replays JSONL
   through the same K=7/V=0.5 logic) -> re-run build_bot_artifacts_K7V on the result.
   ETA: 7 days from now (when capture has enough data).

2. **Rolling-window adaptive threshold.** Maintain a rolling-window p90 of recent live
   scores; set threshold = max(0.5108, rolling_p90). Auto-adapts to whatever phase
   we're in. Risk: in a true low-quality phase the bot would fire on weak winners
   that the model already correctly low-scored.

3. **Accept current regime and lower threshold to live-top-decile cutoff.** Currently
   threshold=0.30 -> fires on top ~3-4% of live ready events (~3-5 bets / 100 ready),
   precision likely 30-40% based on score-winner correlation observed. This is the
   PRAGMATIC bet: trade what's in front of us, log everything, retrain when ready.

4. **Online learning loop** (already on roadmap as Task 25). The grpc_capture +
   bot_data jsonl combine into the four-way TP/FP/FN/TN matrix needed for
   drift-triggered retrain. Make this real, not just roadmap.

**Currently active**: option 3 with threshold 0.30, --live + JITO_DRY_RUN=1, scale-out
wired. Bot fires on the top ~3% of live ready events; bundles are assembled+signed but
not POSTed. Will run overnight; tomorrow we look at the broker log and the realized
peak_ret distribution of the fired mints.

### Logging improvements this session
entry_decision rows now log:
  - `threshold`: the threshold the score was compared against (for audit trail when
    overrides are in play)
  - `features`: the full 22-feature dict (K7+V) — so any future drift analysis is a
    single grep
  - `first_seen_rsol`: the rsol at the bot's first observation of this mint
  - `ev_slot`: gRPC envelope slot (None on WS path)

Also added `path_snap` rows logged on every forward snapshot with full path features,
p_rec, and slot. Previously path features were only in positions.jsonl (compact format);
shadow_run.jsonl now has them too for unified grep-debugging.

### Bet size confirmed
`--bet-sol 1.0` (matches PaperBook Q=1.0 analytical reference). The earlier confusion
between `cum_buy_sol` (market-side feature, total SOL bought by everyone on the token
in the V window) and our bet size (1.0 SOL) is just naming overlap. `cum_buy_sol=4.14`
means the market poured 4.14 SOL of buys into this mint by the V trigger moment, NOT
that we bet 4.14 SOL.

### CYqs P&L (first live DRY_RUN fire) — net **-2.1%**
Sum of 8 expected sell_sol_out: 0.97969 SOL. Jito tips 9 * 100K lam = 0.0009 SOL.
Sig fees 9 * 5K = 0.000045 SOL. Spent 1.0 SOL. Net **-0.02125 SOL = -2.1%**. The
position never went meaningfully above midK (8 slices each yielded ~0.122 SOL = near-
break-even per slice). Diagnosis: the bot fired all 8 scale-out slices on micro-positive
wiggles within the first 7 seconds. The token barely climbed. Even with the price not
falling we lost 2% to AMM impact + tips. **This is the front-loaded-exit failure mode**:
on a flat/sideways token we lose ~2% guaranteed because we exit fast on ANY
ret > 0 snap. Fix is the hybrid policy below.

## Hybrid scale-out (Task 29, this session)
Replaces the front-loaded "sell on every profitable snap" with two phases:

**Phase 1 (de-risk, first 4 slices):** sell 1/N of remaining on profitable snaps,
but enforce min `DERISK_MIN_GAP_SEC = 5.0` seconds between slices. So the de-risk
phase always spreads over ≥20s even on fast tokens, instead of the 7s CYqs lifecycle.

**Phase 2 (runner, remaining 4 slices):** hold the position. Trailing stop fires
`sell_all` when retracement-from-running-max-ret hits `RUNNER_RETRACE_FRAC = 0.30`,
but only if running-max-ret >= `RUNNER_MIN_ARM_RET = 0.20` (don't trailing-stop tiny
peaks; let death-cut handle small drawdowns instead). For a 5x peak token, runner
exits when price retraces to 3.5x. For a 10x token, exits at 7x. So **moonshots
get captured at ~70% of peak**, not at the first +0.1% wiggle.

Death-cut still active throughout (ret < 0 AND p_rec < 0.10 -> dump remainder),
regardless of phase. Stale watchdog still dumps anything inactive >5 min.

Logged events expand to include `phase` ("derisk" / "runner"), `n_sold`, `run_max`,
`retrace` (for runner exits). Pure live policy change — PaperBook's analytical
retroactive schedule is unchanged for comparison.

Deployed to sol; bot restarted Jun 7 ~06:51 UTC.

## Capture-to-extractor pipe (Task 30, this session)
`extract_from_capture.py`. Replays `grpc_capture/*.jsonl(.gz)` chronologically through
the SAME K=7 and V=0.5 trigger + feature-extraction logic the offline `trades.csv`
extractors use, emits `token_level.parquet` + `path_snapshots.parquet` to a target
dir. Imports M class + helpers directly from the existing extractors so the feature
math is guaranteed byte-identical — only the input source differs.

Once we have 3-7 days of capture data this lets us re-extract training tables
from the regime we actually trade in (the firehose) and feed
`build_bot_artifacts_K7V.py` to refresh production artifacts. The capture is already
recording continuously (independent systemd unit), so this fix accumulates passively.

### **CRITICAL FINDING from the 50-min smoke test**
Ran on existing ~50 min of capture:
- K=7 stream: 696 tokens, **36.1% peak >= 2x**
- V=0.5 stream: 604 tokens, **35.6% peak >= 2x**

That's HIGHER base rate than the May training data (28%). **The simple "regime drift
= lower winners" hypothesis is FALSE.** Recent capture shows MORE 2x-winners than
training, not fewer.

But the bot's "ready" subset (the 88 mints we fired on via the WS listener over a
similar window) showed only 10% winners. So:
  - capture-ready (gRPC firehose): ~35% winners
  - bot-ready (WS listener): ~10% winners

**Conclusion: the bot is selecting a STRONGLY-BIASED-LOSER subset of the available
ready population.** Likely root cause: WS coverage. The bot is on a single websocket
subscription that drops messages under load (erpc WS is best-effort). Capture sees
the firehose via gRPC. The WS bot probably catches ~18% of the qualifying mints,
and that catch is biased toward quieter moments (loser-typical) because bursts of
buying (winner-typical) overload the WS pipe and get dropped.

### Implication: switch the bot to gRPC
We already wired the switchable listener (`--source grpc` in `pumpfun_bot.py`,
backed by `listener_grpc_bot.py`). Flipping it should:
  - capture the full ready-population stream (vs 18%)
  - eliminate the loser-bias
  - move live winner rate from 10% toward 30-35%
  - make the model's training-derived threshold (0.5108) appropriate again

Before flipping, worth doing one observation pass with the bot on gRPC to confirm
the population shift before recalibrating thresholds. The MOST IMPORTANT NEXT
EXPERIMENT is the WS-vs-gRPC bot AB comparison on the same time window.

### Smoke test caveats
50 min of capture covers only short-lived peaks. Slow-developing tokens (peak after
30 min) are under-represented vs the May training where the trades.csv covered
hours per token. So the 36% peak >= 2x is biased toward QUICK winners. The
proportion of WS bias vs short-window bias in the 10% vs 36% gap is unclear from
this single point. Re-running with multi-day capture will sharpen the answer.

## Hybrid exit policy backtest (Task 31, this session)

User correctly pushed back: I deployed the hybrid scale-out live without backtesting
it. Wrote `pumpfun_K7_V_book_hybrid.py` to run the OOS book three ways: analytical
retroactive (the original +0.62/bet PLAUSIBLE baseline; knows the future), front-
loaded (the ORIGINAL live wiring: sell on every profitable snap, 8 slices total),
and the new hybrid (4 paced de-risk + 4 runner with trailing stop). Same fresh-
filtered OOS top-decile universe (8,603 bets) for all three.

| scenario | analytical | frontload | **hybrid** |
|---|---|---|---|
| baseline mean | +0.705 | +0.363 | **+0.423** |
| baseline median | +0.089 | **+0.304** | +0.140 |
| baseline win% | 56% | **69%** | 58% |
| baseline total SOL | +6068 | +3124 | +3643 |
| PLAUSIBLE mean | +0.620 | +0.325 | **+0.374** |
| PLAUSIBLE median | +0.036 | **+0.259** | +0.075 |
| PLAUSIBLE total SOL | +5337 | +2797 | +3215 |
| PESSIMISTIC mean | +0.395 | +0.176 | **+0.211** |
| PESSIMISTIC median | -0.106 | **+0.111** | -0.087 |
| PESSIMISTIC total SOL | +3394 | +1512 | +1818 |
| all scenarios P(>0) | 100% | 100% | 100% |

### Findings
1. **Hybrid is the best real-time policy by mean P&L**: +17% over frontload baseline
   (+0.423 vs +0.363); +15% over frontload PLAUSIBLE; +20% over frontload PESSIMISTIC.
   The runner phase captures moonshots that frontload exits early on.
2. **Frontload has the best median and win%**: more consistent small wins (+0.304
   median baseline vs +0.140 hybrid), 69% win rate vs 58% hybrid. Frontload exits
   the entire position on the first 8 profitable snaps = very high probability of
   booking a small profit on any token that goes up at all. But misses the tails.
3. **Both real-time policies are ~40% below the analytical upper bound**. The
   analytical retroactive policy linspaces 8 slices across the ENTIRE profitable
   window, capturing peak prices; no real-time can match it because it requires
   knowing the future. The +0.62/bet PLAUSIBLE figure was always the upper bound,
   not a target for any executable policy.
4. **Hybrid is justified live**. Keep it. The +0.374/bet PLAUSIBLE under the
   hybrid on the OOS-fresh population is the realistic alpha number to aim for
   in live trading (once the WS-bias issue is fixed via gRPC).

### Caveat — backtest is on OOS population (28% winners)
Live on WS showed 10% winners (bot's biased subset). Live on gRPC (just switched)
should show ~30-35% winners (matching capture's measured base rate). If gRPC moves
us to OOS-like population, the hybrid PLAUSIBLE +0.374/bet should translate
directly to live. If gRPC still shows substantially lower winners, the population
gap is more than a WS-vs-gRPC artifact and we need a deeper investigation.

### Followup variants worth testing (not done this turn)
- 6 de-risk + 2 runner (more de-risk weight)
- 2 de-risk + 6 runner (less de-risk, more moonshot capture)
- Trailing-stop retrace 40% or 50% (looser stop, holds runner longer)
- Time-based even-spacing (e.g., 8 slices at 30s intervals, the simplest "linspace
  approximation" since we can't know the future end)
- These all benchmark against the +0.620 analytical PLAUSIBLE ceiling.

## Bot now on gRPC (this session)
`--source grpc` flipped on. Bot subscribes via `grpc-fra1-1.erpc.global:80` (the
same endpoint the capture uses, separate subscription). Expected to dramatically
increase ready-event rate and remove the loser-bias from WS dropouts. Will know
by tomorrow whether live winner rate moves from 10% toward 30-35%. If it does, the
hybrid PLAUSIBLE +0.374 OOS becomes the realistic per-bet target.

## Exit-policy sweep (Task this session, 15 policies on OOS-fresh top decile)
After hybrid backtest showed -40% vs analytical, ran a much wider sweep.
`pumpfun_K7_V_book_exit_sweep.py` benchmarks 15 real-time exit policies vs the
analytical upper bound on the same 8,603 OOS-fresh top-decile bets.

PLAUSIBLE (el=1, fee=.0015) ranked by mean:

| rank | policy | mean | median | win% |
|---|---|---|---|---|
| (upper bound) | A_analytical (knows future) | +0.620 | +0.036 | 52% |
| 1 | H_time_spaced 15s | **+0.480** | -0.309 | 38% |
| 2 | J_momentum_aware | +0.431 | -0.108 | 47% |
| 3 | K2_combined 2+6 | +0.422 | -0.193 | 41% |
| 4 | K5_combined 4+4 g20s | +0.412 | -0.072 | 46% |
| 5 | **K_combined 4+4** (now live) | **+0.401** | -0.047 | 48% |
| 6 | K4_combined 4+4 g10s | +0.388 | -0.013 | 50% |
| 7 | K3_combined 6+2 | +0.386 | +0.029 | 51% |
| 8 | F_hybrid 4+4 t50 | +0.385 | +0.009 | 51% |
| 9 | I_pure_runner_t40 | +0.381 | -0.144 | 41% |
| 10 | D_hybrid 6+2 t30 | +0.379 | +0.048 | 52% |
| 11 | C_hybrid 4+4 t30 (was live) | +0.374 | +0.075 | 54% |
| 12 | G_hybrid 4+4 t20 | +0.361 | +0.099 | 55% |
| 13 | E_hybrid 2+6 t30 | +0.344 | +0.030 | 52% |
| 14 | B_frontload | +0.325 | +0.259 | 64% |

### Key findings
1. **Time-based exits beat trailing stops.** H_time_spaced (8 slices every 15s for
   2 minutes total, no ret check, just sell on schedule) is the best real-time
   policy by mean — 77% of the analytical upper bound. Trailing stops fail
   because predicting peak ret correctly is hard; forced time-based selling
   sidesteps prediction.
2. **Hybrid 4+4 trailing-stop variants (C-G) are tightly clustered** at
   +0.36-0.39 PLAUSIBLE. None of them break through into time-based territory.
3. **Combined (K series) sits between hybrid and pure time-spaced**: takes the
   de-risk floor from hybrid + the time-based runner from H. Best of both
   profiles. K_combined 4+4 selected for live.
4. **Trade-off matrix**: more de-risk = positive median + lower mean; more
   runner / time-spaced = higher mean + negative median. No real-time policy
   has both. Total P&L (mean) is what matters for the long run.

### Live policy SWITCH this session: C_hybrid 4+4 t30 -> K_combined 4+4
Implementation in `shadow_harness.py`:
- Phase 1 (4 slices): require ret>0 AND min 5s gap. Same as before.
- Phase 2 (4 slices): forced sell every 15s, NO ret check. Captures moonshots
  via time-based spacing instead of trailing-stop prediction.
- Death-cut active throughout.
- Worst-case lifecycle: 4*5s + 4*15s = 80s before fully exited.

Expected PLAUSIBLE per-bet: **+0.401/bet** (vs previous +0.374, +7%; vs analytical
upper bound +0.620, 65%).

## Jito bundle landing reconciliation — KNOWN GAP, NOT BUILT (Task 32)
Current state: when JitoBroker submits a bundle, holdings are decremented
SYNCHRONOUSLY at call time so subsequent slices see the new balance. But we never
poll back to confirm the bundle actually landed on-chain. If a bundle is dropped
(network, leader skipped, tip too low), holdings are incorrect.

For DRY_RUN this doesn't matter — nothing posts. For LIVE with real capital this
is a HARD PREREQUISITE before flipping `JITO_DRY_RUN=0`. The work:
1. After each `send_bundle` call, poll `getBundleStatuses` (Jito API) or
   `getSignatureStatuses` (Solana RPC) until landed (~10 slot ~4s window)
2. If failed/dropped: roll back the synchronous holdings reservation, optionally
   re-submit with a higher tip
3. Periodic `getTokenAccountBalance` reconcile (every ~5 min): chain balance is
   ground truth; reconcile any drift
4. Add `reconcile_state` event log row per cycle

~200 LOC + careful error handling. Tracked as Task 32, blocking real-capital live.

## Cleanup + git + reconciliation (this session)
sol's `/root/the-distribution-will-manifest` cleaned and version-controlled.

### Layout after two-pass cleanup
- **root** (17 active Python modules + configs): blockhash_cache, config,
  feature_accum, geyser_pb2*, grpc_capture (the recorder), jito_broker, jito_exec,
  listener_grpc_bot, model_serve, paper_book, position_store, pumpfun_bot, pump_fun_ix,
  pumpfun_parse, shadow_harness, solana_storage_pb2*, requirements.txt, .gitignore, .env
- **tools/**: research + diagnostic scripts (extract_from_capture, grpc_latency_probe,
  live_winner_rate_diag, partial_history_diag, pumpfun_continuation_value_K7/_V).
  Run on demand, not part of any running service.
- **logs/**: broker_jito.jsonl, broker_recon.jsonl (broker outputs separated from bot_data/)
- **archive/**: 17 legacy files moved aside (old scaffolds, V=0.5 backups, deal-flow,
  old shadow logs). Not in git.
- **bot_artifacts_K7V/, bot_data/, grpc_capture/, protos/, scripts/, systemd/, venv/**:
  unchanged operational dirs.

### Git history
- `ff3bd0a` Initial commit: V+K7 production bot (paper + DRY_RUN), gRPC capture,
  hybrid scale-out, calibration diagnostics
- `63e2d0c` K_combined 4+4 exit policy live (15s runner gap, no trailing-stop)
- `adc926c` Cleanup: research tools to tools/, broker logs to logs/ + Jito reconciliation

`.gitignore` excludes venv/, *.pkl, *.parquet, grpc_capture/, bot_data/,
broker_*.jsonl, shadow_run*.jsonl, .env, archive/, __pycache__.

## Jito reconciliation (Task 32, done)
The post-submission gap is closed. Three pieces:

### `PendingBundle` dataclass + tracking
Every LIVE submission (buy / sell_slice / sell_all) appends a `PendingBundle` to
`self.pending_bundles[sig]` after `send_bundle` returns. Tracks: sig, mint, op,
tok_delta (signed; + for buy, - for sell), tip_lamports, submitted_t, status,
landed_t, landed_slot, retry_count. DRY_RUN doesn't append (nothing to reconcile).

### `_reconciler_loop` (background)
Polls `getSignatureStatuses` every 2s for all pending sigs (batched up to 200 per
RPC call). 4s grace period before first check (typical landing window).

- **Landed** (status not None, err None): log `landed` event with landed_slot +
  latency_s + tip_lamports. Holdings already match (synchronous reservation was
  correct). Remove from pending.
- **TX err on chain** (status not None, err set): log `failed:tx_err:...`, roll
  back the synchronous holdings reservation (mirror-image of the optimistic
  update), fire `failure_callback(mint, op, reason)`. Remove from pending.
- **Still null at 30s**: log `failed:expired_not_on_chain`, roll back, fire
  callback. Remove from pending. (Note: in rare cases a slow-confirming bundle
  may land after 30s; the chain-balance reconciler catches this.)

### `_holdings_reconcile_loop` (background, 5min cadence)
Skipped in DRY_RUN. For each held mint, calls `getTokenAccountBalance` on the
user's ATA, compares to `self.holdings[mint]`, sets chain balance as ground truth,
logs drift. `CRITICAL_drift` event if >50% mismatch.

### `ShadowHarness._on_broker_failure` callback
Registered as the broker's `failure_callback`. On failed sell:
- decrement `live_slices_sold[mint]` (slice counter rollback)
- reset `last_slice_ts[mint] = 0.0` (next snap can retry immediately, ignoring
  the 5s/15s gap)
- un-mark `live_dead` (so the exit policy will keep working on this mint)
- log `recon_rollback` event

On failed buy: mark mint dead (already in closed_mints). Don't re-enter — buy
timing was the alpha, no point retrying a stale window.

### status.json now exposes recon stats
- `recon`: {n_outcomes, n_landed, n_failed, land_rate, landing_latency_p50_s,
  landing_latency_p90_s, landed_tip_p50, failed_tip_p50}
- `pending_bundles`: count of in-flight reconciliations

### logs/broker_recon.jsonl format
One JSON line per recon event. Useful kinds:
- `landed`: sig, mint, op, landed_slot, latency_s, tip_lam, tok_delta
- `failed`: sig, mint, op, reason, tok_delta, tip_lam, holdings_before, holdings_after, age_s
- `rpc_err`: err, n_sigs
- `holdings_reconcile`: mint, expected, chain, drift, drift_frac
- `CRITICAL_drift`: same fields + critical flag

Once we have ~100+ landed/failed events, can regress `tip_lam` vs `land_rate` for
empirical tip-sizing (Task 33 candidate).

### Hard prereq before flipping JITO_DRY_RUN=0
The reconciler is wired but only exercised in LIVE mode. Smoke test: synthetic
expired-bundle path worked (reconciler task is alive). Real validation requires
either:
  a) submit a tiny-tip real bundle and watch it fail (no SOL on wallet -> Jito
     rejects -> reconciler should roll back holdings), or
  b) fund wallet with a small amount and submit a real buy + sell pair, verifying
     landed events + chain-balance match.
  
This is the actual "ready for real capital" gate.

## Current running state (this turn)
Both systemd services active. Bot: --live + DRY_RUN, gRPC source, K_combined 4+4
exit, threshold 0.30 override, reconciler + holdings-reconcile alive.

Live activity since session start:
- 9 fires under threshold 0.30 (vs ~0 fires on WS without the override)
- 4 fires carry `ev_slot` from gRPC (same-block aim measurable on those)
- K_combined exit pattern: 12 de-risk slices + 39 runner slices + 2 death-cuts on
  9 entries (~5.9 slices avg before exit)
- Paper book: mean **+0.117 SOL/bet**, win% 33%, n=9 (positive but small sample)
- recon: 0 outcomes yet (DRY_RUN; pending_bundles = 0)
- Capture archive: 78MB across 2 files, accumulating ~1 GB/day

## Overnight report tool (Task 33, done)
`tools/overnight_report.py` produces a single comprehensive markdown report
combining everything we log:
- session summary (uptime + event counts)
- score distribution (live ready vs fires vs frozen training reference)
- realized peak_ret per fired mint (via grpc_capture lookup; the ground truth)
- exit-policy mix breakdown
- paper book P&L vs analytical reference
- bundle assembly latency (bh_age_ms, asm_ms, ev_slot coverage)
- reconciliation stats (landed/failed/pending; tip-vs-landing once data available)
Wired into ctl: `./scripts/pumpfun_ctl.sh report`.

## Online learning + A/B tooling (Tasks 25 + 34, done this session)

### `tools/strategy_ab_replay.py` — multi-policy paper replay on actual fires
Reads bot_data/shadow_run.jsonl for actual fires + grpc_capture for forward trades.
Replays each fire through:
  - K_combined 4+4 g15s (current LIVE)
  - H_time_spaced 15s
  - F_hybrid 4+4 t50
  - C_hybrid 4+4 t30
  - B_frontload
Outputs per-policy mean/median/win%/total/best/worst plus the per-fire winning
policy distribution. Empirical evidence beyond OOS backtest projections.
ctl: `./scripts/pumpfun_ctl.sh ab-replay`.

First run on 9 live fires (small sample; revisit after overnight):

| policy | mean | median | win% | total |
|---|---|---|---|---|
| **B_frontload** | **+0.118** | **+0.101** | **78%** | **+1.06** |
| K_combined (LIVE) | +0.081 | +0.057 | 56% | +0.73 |
| H_time_spaced | +0.080 | -0.081 | 44% | +0.72 |
| C_hybrid t30 | +0.075 | +0.042 | 67% | +0.67 |
| F_hybrid t50 | +0.041 | -0.033 | 44% | +0.37 |

B_frontload wins on 5/9 fires (56% winning-policy share). Interesting reversal
from OOS sweep where K_combined > frontload — different population (live fires
under threshold override are lower-scoring than the OOS top-decile cutoff at
~0.51, so different optimal exit). At n=9 sampling noise dominates; needs 50+
fires for statistical power. Built precisely to enable this kind of iteration.

### `tools/drift_monitor.py` — daily KS + winner-rate + location checks
Reads last 24h of entry_decision rows. Flags:
- shape_shift  (KS p-value < 0.01 vs frozen training-score sample)
- location_shift (live_p90 < training_p90 - 0.10)
- low_fire_rate (< 1.0% of ready events)
- low_winner_rate (< 10% on >=30 capture-confirmed outcomes)
Writes `drift_check` events to logs/drift_log.jsonl. Exit 0 = clean, exit 2 =
drift detected (cron-friendly).
ctl: `./scripts/pumpfun_ctl.sh drift`.

First run: location_shift flagged (live_p90=0.22 vs train_p90=0.45). Other gates
healthy: fire_rate 3.6% (vs 1% min), winner_rate 33% (vs 10% min). The location
shift is the same calibration override situation we already know about and will
recur until the model is retrained on capture-derived data.

### `tools/auto_retrain.py` — weekly idempotent retrain candidate
Five-step pipeline, all gated on the previous step succeeding:
1. Gate: capture archive >= 3 days (else skip)
2. Run extract_from_capture.py against grpc_capture
3. Run build_bot_artifacts_K7V.py --suffix _capture
4. Holdout comparison: candidate AUC vs production AUC on candidate's last-20%
   chronological split. Min uplift 0.005, min holdout 200 bets.
5. If uplift criterion met AND --execute: symlink-swap
   bot_artifacts_K7V -> bot_artifacts_K7V_capture_<timestamp>, restart bot
ctl: `./scripts/pumpfun_ctl.sh retrain-check` (dry-run) or `retrain-now`
(--execute). Currently skips (0.05 days of capture; need 3+).

### systemd timer (optional, manual install)
`systemd/pumpfun-drift-monitor.{service,timer}`: daily timer for the drift
monitor. NOT installed by default. To activate:
  `./scripts/pumpfun_ctl.sh install-drift-timer`

The bot + capture services are untouched. All new tooling is offline / read-only
except auto_retrain.py --execute which atomically swaps artifacts and restarts
the bot.

## Git history (current)
```
cbb5f0f tiny-tip test PASS — reconciler validated on real failed bundle
d9f7f98 gitignore: data/ + parquets (522MB May dataset, regenerable)
7182fbe config.yaml + risk limits + tiny-tip test (Tasks 35, 36, 37)
4a534ec Online learning skeleton + A/B replay tools (Tasks 25, 34)
3c02882 fix: pumpfun_ctl.sh report should shift before forwarding args
d9788a6 Overnight report tool (Task 33)
adc926c Cleanup: research tools to tools/, broker logs to logs/ + Jito recon
63e2d0c K_combined 4+4 exit policy live (15s runner gap, no trailing-stop)
ff3bd0a Initial commit: V+K7 production bot
```

## Productionization layer (Tasks 35, 36, 37 — this session)

### config.yaml + bot_config.py (Task 35)
All previously hardcoded knobs now read from `config.yaml` at repo root with
layered overrides: defaults <- config.yaml <- env (`PUMPFUN_<DOTTED_PATH>`).
Modules import `from bot_config import cfg` and read `cfg.broker.tip_lamports`
etc. with dot access. pyyaml soft-dep (defaults if absent). Backward compatible.

config.yaml sections:
- **bot**: artifact_dir, data_dir, bet_sol, entry_threshold_override, status_interval_s
- **harness**: snap_every, stale_sec, fresh_rsol_lam
- **exit**: policy, total_slices, derisk_slices, derisk_min_gap_s, runner_min_gap_s, death_threshold
- **broker**: tip_lamports, slippage_bps, jito_dry_run, jito_endpoint, recon_poll_s, recon_expire_s, holdings_reconcile_s
- **listener**: source (grpc|ws), grpc_endpoint, grpc_insecure
- **paper_book**: cost_bps, fee_per_tx_sol, entry_lat_snaps, max_slices
- **risk**: max_concurrent_positions, max_fires_per_minute, daily_loss_limit_sol, bundle_failure_rate_limit, bundle_failure_window, circuit_breaker_cooldown_s

Refactored to read from `cfg`: shadow_harness.py (SNAP_EVERY, STALE_SEC,
FRESH_RSOL_LAM, MAX_SLICES, DERISK_*, RUNNER_*, RISK_*), jito_broker.py
(tip_lamports, slippage_bps, dry_run defaults), listener_grpc_bot.py
(grpc_endpoint, grpc_insecure).

### Risk limits + circuit breakers (Task 36)
`ShadowHarness._risk_check_before_fire()` called BEFORE `book.open()` on every
entry-decision fire. Refuses on any of:
1. **max_concurrent_positions** (default 10; original design's NUM_SLOTS=16 ish)
2. **max_fires_per_minute** (default 6; rate limit via 64-deep deque)
3. **daily_loss_limit_sol** (default -5.0 SOL cumulative net; arms 10-min
   circuit-breaker cooldown when tripped)
4. **bundle_failure_rate_limit** (default 50% of last 20 outcomes; arms
   circuit-breaker cooldown)

Each refusal increments a stat (`risk_refusal_max_concurrent`,
`risk_refusal_rate_limit`, `risk_refusal_daily_loss`, `risk_refusal_failure_rate`,
`circuit_breaker_active`) and logs a `risk_refusal` event with the reason. Bot
does NOT enter the trade and does NOT add the mint to closed_mints (it can be
retried via a later fire if risk clears).

### Tiny-tip test PASS (Task 37)
`tools/tiny_tip_test.py`. Submitted ONE real bundle with `JITO_DRY_RUN=0` +
`tip_lamports=1` against Jito Frankfurt. Sequence:
1. Bundle submitted; Jito returned HTTP 400 tip-too-low (expected; min tip is
   ~1000 lam, we sent 1)
2. `PendingBundle` added to `broker.pending_bundles` with sig=`61BSm2knSSc4...`
3. Reconciler polled `getSignatureStatuses` every 2s for 30s. Found null (tx
   never made it to chain because Jito rejected at submission).
4. After expire window: `_handle_bundle_failed(reason="expired_not_on_chain")`
5. Optimistic holdings rolled back: 34.6T tokens -> 0
6. `failure_callback(mint, "buy", "expired_not_on_chain")` fired
7. `logs/broker_recon.jsonl` recorded the failure with all fields

Closes the last theoretical-vs-practical gap. The LIVE reconciliation path has
end-to-end empirical evidence. ZERO capital risk (wallet has 0 SOL).

### Disk: May parquets on sol
522MB of fresh-rsol-filtered parquets rsynced to `/root/.../data/`. Available
for any in-place re-extraction, comparison, or auto_retrain holdout step.
Excluded from git (.gitignore: `data/`, `*.parquet`).

## Textual TUI dashboard (Task 38)
`tools/dashboard.py` rendered via `pumpfun_ctl.sh dashboard`. Panels: MODE
(LIVE/PAPER + DRY_RUN gate), activity, paper P&L (mean SOL/bet + win%),
EXEC LATENCY (bh_age p50/p90, asm_ms p50/p90, bundle count by op), RECON
(pending bundles, recent failures), DRIFT (last check result, alert if any),
OPEN POSITIONS (mint, score, age, slice, ret_last, run_max, phase),
RECENT TRADES (time, mint, score, slot, status, exit_kind, exit_ret, net SOL).
Refreshes every 1s. All P&L displayed as **absolute SOL** (fractional ×
cfg.bot.bet_sol) — fixed the "0.144 SOL/bet" confusion. `bet_sol` change
from 1.0 → 0.1 carried through the entire P&L surface.

## Auto-policy selector (Task 39)
`tools/auto_policy.py` (timer: every 4h via pumpfun-auto-policy.timer).
For each fire in the last `dynamic_window_fires` (default 30), replays the
path through all 5 candidate policies (`k_combined`, `h_time_spaced`,
`b_frontload`, `c_hybrid_t30`, `f_hybrid_t50`) using `strategy_ab_replay`
primitives. Picks the highest mean per-bet. Gates: min uplift +0.02 SOL/bet,
6h cooldown, 30-fire min sample. On winner: rewrites `config.yaml`
`exit.policy:` and `systemctl restart pumpfun-bot`. Every decision (winner,
uplift, swap-or-skip) logged to `logs/policy_decisions.jsonl`. `tools/
policy_impact.py` audits past swaps: shows before/after fire distribution +
P&L and grades each as GOOD CALL / MARGINAL / DISAPPOINTING.

## Online learning hardening (Task 40)
`tools/auto_retrain.py` (timer: weekly Sun 02:00 UTC). Combines May parquets
(`_fresh` suffix) + new capture-derived parquets (`_capture` suffix) via
`tools/build_bot_artifacts_K7V.py --inputs _fresh _capture --out _capture`.
Refuses to fire until `grpc_capture/` archive ≥ 3 days. Two gates that must
both pass before swapping the entry head:
1. **OOS holdout AUC uplift ≥ 0.02** — standard cross-validation gate
2. **Live-shadow AUC uplift ≥ 0.02** — `compare_models_on_live_fires()`
   scores the last 30 real live fires under BOTH candidate and production
   models; the candidate must rank them better. This is the "don't trust
   holdout, trust live" guardrail.

On both-gates pass: symlink `bot_artifacts_K7V → bot_artifacts_K7V_capture_<ts>`
and restart bot. Every retrain attempt logged to `logs/auto_retrain.log`.

`build_bot_artifacts_K7V.py` refactored to take `--inputs` (list) and `--out`
(single) so the same script can produce `bot_artifacts_K7V_fresh` (May-only),
`bot_artifacts_K7V_capture` (May + capture), or a custom blend.

## Per-token gate-replay tool (Task 41)
`tools/token_gate_replay.py` (CLI: `pumpfun_ctl.sh gate-replay`). Per-mint
two-block timeline reconstructing both model gates from `shadow_run.jsonl`:

- **ENTRY GATE block**: score, threshold, fire?, margin, K=7 trigger
  reserves, V=0.5 trigger reserves, cum_buy_sol, first_seen_rsol, 14 of 22
  features surfaced.
- **LIFECYCLE GATE block**: per-snap P(recover), each slice fire (phase,
  policy, frac, ret AT FIRE TIME, run_max AT FIRE TIME, slot), death-cuts,
  runner-exits, final close.
- **OUTCOME block**: PaperBook close (fractional + absolute SOL @ bet_sol),
  broker_jito event counts.

Modes: `<mint>`, `--last N`, `--dumped` (only fires that hit death-cut or
closed below −10%).

Revealed two latent PaperBook-close leaks (death_cut and runner_exit paths
never called `book._close_one(pos)`; only slice-exhaustion did) — fixed
together with the slice-exhaustion close fix in commit `2f6cf99`.

## Dashboard exit_ret + asm_ms float-ms (Task 42)
- **exit_ret column added** to RECENT TRADES. Reads `exit_ret` from
  positions.jsonl close records (falls back to last `path_snap.ret` from
  shadow_run.jsonl for older closes that pre-date the patch). Five close
  paths now log `exit_ret`: slice_exhaustion, death_cut, runner_exit,
  stale, shutdown — plus the `position_store.replay()` force-close on
  restart. New helper `_pos_exit_ret(pos)` falls back to deriving from
  `vsC/vtC` vs `vsK/vtK` when the in-memory snap list is empty.
- **asm_ms reported as 0ms — fixed**. Was `int((time.time()-t0)*1000)`
  truncating sub-ms values to 0 (DRY-mode sell is genuinely ~70µs). Now
  `round(ms, 3)`; buy DRY assembly captured AFTER `bytes(tx)` so signing
  + encode time is included (~0.28ms). Dashboard renders with `.3f`.

## Commitment pin + wide capture (Task 43, commit 76b3cae)
Two coupled changes:

**(1) Pinned commitment level explicitly.** `listener_grpc_bot.py` and
`grpc_capture.py` both now set `req.commitment = CommitmentLevel.PROCESSED`
on the gRPC SubscribeRequest. Previously left unset, which means relying
on upstream's default. PROCESSED is intentional — leader-block-included
trades ASAP, accepting <1% reorg risk caught by the holdings reconciler.
Routed via `cfg.listener.commitment` so we can A/B test `confirmed` later
without a code change. Bot startup log line now reads
`[shadow] subscribed (gRPC, commitment=processed)`.

**(2) Widened capture with gRPC-exclusive fields.** `grpc_capture.py` now
persists per-tx fields the historical May snapshot never had:

| Field | Source | New feature surface |
|---|---|---|
| `fee_lam`     | `meta.fee` | priority+base fee per buyer (sniper signature) |
| `cu`          | `meta.compute_units_consumed` | tx complexity per buyer |
| `n_inner_ix`  | `meta.inner_instructions` count | aggregator/bundling proxy |
| `n_keys`      | `tx.message.account_keys` length | wallet complexity |
| `route`       | `account_keys` ∩ KNOWN_ROUTERS | jupiter_v6 / raydium / moonshot |
| `jito_tip_idx`| `account_keys` ∩ JITO_TIP_ACCOUNTS | Jito bundle indicator |
| `jito_tip_lam`| `post_balances[idx] - pre_balances[idx]` | actual tip amount |

Cost: records ~47% bigger uncompressed (956 vs ~650 bytes), ~30% after
gzip rotation. Capture rate ~33 MB/h vs 25 MB/h, ~8 GB/wk vs 5 GB/wk.

Value (first 30s of new capture): **24.8% of all txs are Jito-tipped**,
fee_lam p90=505K lam (20× p50=25K), max fee 20M lam (~$3 priority on
one tx), Jupiter routing 1%. The classic V+K7 7-unique-buyers trigger
now decomposes into a sophistication profile — "7 buyers, 0 tippers"
(manual hype) vs "7 buyers, 7 tippers" (coordinated bot pump) look
identical in May data but very different here.

**Forward path**: `tools/extract_from_capture.py` will need a small
update to surface these new fields as parquet columns when we want a
retrain that actually USES them. For now they are just archived. Older
~100 MB of pre-widening capture remains valid as the parsed TradeEvent
fields are unchanged; new fields are additive.

## Fitted Q-Iteration on May parquets — REAL OOS uplift (commit 8712f92)
`tools/rl_backtest.py`. Frames exit timing as a discrete MDP:
  state s = (ret_bin, runmax_bin, prec_bin, dts_bin, nsold_bin)
  action a ∈ {0, 0.125, 0.25, 0.50, 1.0}  (fraction of remaining to sell)
  reward r = SOL realized this snap

Builds empirical transition+reward tables from 67K K7_fresh mints (1.4M
snaps → 1.34M transitions → 493 visited states), runs tabular value
iteration to convergence (~4s), extracts greedy policy π*. Evaluates on
96K OOS mints via the same ReplayContext used by all other backtests
(same proven AMM math, same cost model).

**Result**: RL π* OOS mean = -0.0344 vs K_combined -0.0616. **Uplift
+0.0272 SOL per bet**. Bootstrap on 2000 resamples gives:
  p05 = +0.0160  (strictly positive — REAL signal)
  p50 = +0.0270
  p95 = +0.0380
  P(uplift > 0) = 100% across all 2000 samples

The learned policy is bang-bang (487/493 states = hold, 6 = sell-all)
and the 6 sell-states cluster into TWO interpretable regimes:

  STOP-LOSS:    ret < -0.5  AND  p_rec ∈ [0, 0.4]  AND various
                (5 sub-states, ~95 transitions total)
  TAKE-PROFIT:  ret > +2.0  AND  runmax > +5.0  AND p_rec ≥ 0.8
                AND dts > 300s  (1 state, 39,445 transitions)

Plain English: hold everything; sell all only if catastrophic break OR
10x+ winner with sustained health. This is a long-volatility /
tail-harvest profile that captures rare 10-50x runners K_combined slice-
exits before they peak.

**Tail-driven**: top 100 of 96K mints = 178% of net uplift; remaining
95K mints lose -78% on average vs K_combined. So RL loses small on most
trades and wins HUGE on rare ones.

**NOT deployable yet — three hardening steps required before any
consideration of swap**:
1. Re-train with the REAL recovery model p_rec instead of the proxy
   `1 - max(0, -ret) / (1 + max(runmax, 0))` we used as a stand-in.
   Effort: ~1h once the recovery model's predictions are appended to
   the path snaps parquet via a small extractor pass.
2. Slot-delay sensitivity: re-eval with entry_lat=2 to simulate
   realistic execution friction. Bang-bang policies eat slot-delay
   slippage every fire. Effort: 5 min code change.
3. Bootstrap on STRATIFIED population (split by peak_ret deciles)
   to check whether the uplift survives in the "no 10x catch" cohorts,
   not just the global. Effort: ~30 min.

Once all three pass, the path forward is:
  4. Implement π* serving in shadow_harness as a 6th exit policy
     ("rl_pi_star") with the policy table loaded from pickle on startup
  5. Add to auto_policy's dynamic_candidates list with the same gating
     (min uplift +0.02, cooldown 6h, sample ≥30 fires)
  6. Auto_policy will swap in/out based on actual live performance,
     same gates as for K_combined vs hybrid variants today

CLI: `pumpfun_ctl.sh rl-backtest [--train-dir D] [--eval-dir D]`

## Pluggable exit-policy framework + rl_layered live (Task 49, commit d821880)
New `bot_shadow/exit_policies/` package abstracts the policy dispatch:

```
exit_policies/
├── base.py          ExitPolicy ABC + ExitDecision dataclass + @register + get_policy
├── k_combined.py    @register("k_combined")
├── h_time_spaced.py @register("h_time_spaced")
├── b_frontload.py   @register("b_frontload")
├── hybrid_trail.py  @register("c_hybrid_t30"), @register("f_hybrid_t50")
└── rl_layered.py    @register("rl_layered")  ← new, deployed
```

**Adding a new policy = create one file, decorate with `@register("name")`,
import in `__init__.py`**. Harness and auto_policy pick it up automatically.

`ExitPolicy.decide()` returns an `ExitDecision(action, frac, phase, ...)`
where action ∈ {hold, slice, sell_all}. The harness owns broker calls,
slice counters, position book, logging; the policy owns the DECISION.

Harness `_dispatch_exit_slice` rewritten to be policy-agnostic — 80 lines
of inline if/elif chain → 50 lines of action dispatch.

**rl_layered** is the first non-heuristic policy: loads pi_disc + pi_hold +
q5_classifier pickles from `cfg.exit.rl_artifact_dir`, calls `on_entry()`
on every fire to cache routing, and `decide()` per snap bins state + does
the table lookup.

`strategy_ab_replay.policy_via_registry()` adapter runs ANY registered
policy against snap arrays — so `auto_policy` automatically A/B-tests
new policies once added to `cfg.exit.dynamic_candidates`. No tool code
change ever needed for new policies.

**First live evaluation (30 fires)**:
  - K_combined  mean = +0.074  ← current
  - rl_layered  mean = -0.429  (worse, EXPECTED)
  - b_frontload mean = +0.117  (best on this slice — but small sample)

The OOS-validated rl_layered uplift (+0.046) doesn't materialize at our
current entry threshold (0.30) because we're firing on Q1-Q4-biased
candidates where RL bleeds. Per-quintile breakdown in Task 47 already
predicted this. **The auto_policy gate is doing its job — refused to
swap (uplift below threshold + in cooldown)**.

`cfg.exit.dynamic_candidates` now includes `rl_layered`. Every 4h
auto_policy.timer re-evaluates; if/when our fire distribution shifts
toward Q5 (e.g., after a tightened threshold or a retrained entry
head), the +0.046 uplift will materialize and rl_layered will be
promoted automatically through the same gate K_combined would.

Pickle-load caching in `get_policy()` (one instance per (name, id(cfg))
to avoid re-loading 3 pickles × 30 fires per auto_policy run.

## Mixture-of-experts RL — REAL +4.6% per-bet uplift (Task 47, commits db6f044 + 20e0324)
**User insight (2026-06-07)**: train RL excluding moonshots so the policy
MUST learn discriminating scale-out (instead of "diamond hands for the
rare 10x"). Then layer with the moonshot-holder via a Q5 classifier at
entry. Two specialists, each trained on its native population.

**Discriminator policy** (trained on Q1-Q4 only, peak < 1.46):
  - 874 visited states, 79 sell-all states (9% — 5× the holder's 1.4%)
  - OOS: win% 44.4 vs K_combined's 25.6, median -0.10 vs -0.21
  - Wins big on Q3 (+0.16) and Q4 (+0.31); bails too early on Q5 (-0.47)

**Holder policy** (current full-data RL): wins on Q5 (+0.51), loses Q1-Q4.

**Per-quintile decomposition (OOS, 96K mints)**:
  | Stratum | n | K_combined | disc | disc-K | hold | hold-K |
  |---|---|---|---|---|---|---|
  | Q1 | 19205 | -0.337 | -0.347 | -0.010 | -0.348 | -0.011 |
  | Q2 | 19204 | -0.342 | -0.380 | -0.039 | -0.393 | -0.051 |
  | **Q3** | 19205 | -0.364 | -0.200 | **+0.164** | -0.485 | -0.121 |
  | **Q4** | 19204 | -0.166 | +0.142 | **+0.308** | -0.361 | -0.195 |
  | **Q5** | 19205 | +0.901 | +0.427 | -0.474 | **+1.415** | **+0.513** |

**Oracle layered (perfect peak knowledge)**: +0.187 per bet uplift — the
ceiling.

**Q5 classifier** trained on 22 V+K7 entry features:
  - HGB, max_depth=3, max_iter=200, lr=0.06, l2=1.0
  - TRAIN AUC 0.7586, OOS AUC 0.7565 (no overfit)
  - Q5 base rate 21.3%

**Realistic layered policy** (Q5-classifier-routed) — threshold sweep:
  | Threshold | % to hold | mean | uplift |
  |---|---|---|---|
  | 0.10 | 65.2% | -0.020 | +0.0411 |
  | **0.20** | 41.9% | -0.015 | **+0.0463** ✓ best |
  | 0.30 | 17.3% | -0.024 | +0.0378 |
  | 0.50 | 4.4% | -0.046 | +0.0153 |

**Best layered: +0.0463 per bet OOS** vs single-holder's +0.0272. That's
**70% MORE uplift** than the single-policy RL we found yesterday, and
**captures ~25% of the oracle ceiling**.

Saved artifacts at `bot_artifacts_K7V_rl_layered/`:
  - `pi_disc.pkl`         — 874-state π* trained on Q1-Q4
  - `pi_hold.pkl`         — 1172-state π* trained on full data
  - `q5_classifier.pkl`   — HGB Q5 predictor + 22-feature spec + threshold 0.20
  - `spec.json`           — metadata + OOS metrics

**Deployment path** (not yet built):
  1. Add `rl_layered` exit policy to `shadow_harness._dispatch_exit_slice`
  2. On startup load all three pickles + the 22-feature names from spec
  3. At entry-decision time: call `clf.predict_proba(features)[:,1]`, store
     the Q5 score on the position (route_score). Then per-snap exit dispatch:
       if route_score >= 0.20: use pi_hold table
       else:                    use pi_disc table
  4. Register `rl_layered` in `auto_policy.py`'s `dynamic_candidates`
  5. Auto_policy A/B replays it against K_combined on actual live fires;
     swaps if live uplift confirms the OOS +0.046 within gate (≥+0.02)

The +0.046 per bet at 0.1 SOL bet = +0.0046 SOL/trade absolute = ~$0.70/trade
at current SOL price. At ~30 fires/day that's ~+0.14 SOL/day = ~$22/day,
~$2000/quarter. Pays the engineering cost easily.

**Status**: research-confirmed, NOT YET DEPLOYED. Deployment is mechanical
once we choose to do it.

## RL hardening pass — TAIL-DRIVEN, conditionally optimal (commit 20e0324)
Three follow-up validations on the FQI policy:

**1. Real recovery model (`--use-recovery-model`)**: identical OOS uplift
+0.0272 with the real `bot_artifacts_K7V/recovery_model.pkl` predictions
instead of the proxy. State count grows from 493 to 1172 — the real model
discriminates "shallow drawdown but doomed" patterns the proxy couldn't.
**Result confirmed not a feature artifact.**

**2. Slot-delay (`--entry-lat 2`)**: uplift only drops from +0.0272 to
+0.0260 (4% of the gain) — bang-bang strategy is execution-robust. Also
fixed a captured-default-arg bug in ReplayContext so the entry_lat
override actually takes effect across all policies.

**3. Stratified bootstrap by peak_ret quintile (OOS, 96K mints)**:

| Stratum | peak_ret range | n | mean K | mean RL | uplift |
|---|---|---|---|---|---|
| Q1 | [-0.57, +0.03]  | 19,205 | -0.337 | -0.348 | **-0.011** |
| Q2 | [+0.03, +0.24]  | 19,204 | -0.342 | -0.393 | **-0.051** |
| Q3 | [+0.24, +0.61]  | 19,205 | -0.364 | -0.485 | **-0.121** |
| Q4 | [+0.61, +1.46]  | 19,204 | -0.166 | -0.361 | **-0.195** |
| Q5 | [+1.46, +279.66] | 19,205 | +0.901 | +1.415 | **+0.513** |

Block bootstrap (1000 reps, blocks of 500): **p05 = +0.0155**, 100%
positive globally.

**Conclusion**: The +2.7% global uplift is **structurally tail-driven**.
On the bottom 80% of tokens (Q1-Q4) RL bleeds 1-19% more than K_combined.
The entire net comes from Q5 (top 20% by peak), where RL captures runners
that K_combined slice-exits early.

**Practical deployment implication**:

  - Naive deployment will bleed money if the entry filter isn't tight
    enough to bias fires toward Q5.
  - Current entry threshold 0.30 (override of trained 0.4481) is firing
    on weaker setups — likely Q1-Q4 heavy. RL would lose on those.
  - To deploy RL exit, would need to ALSO tighten entry threshold back
    to trained value AND validate the joint policy. The two decisions
    are coupled.
  - Alternative: deploy RL only on fires above a higher score cutoff
    (e.g., score > 0.45), keep K_combined for fires in [0.30, 0.45].

**Status**: RL π* exit policy is RESEARCH-CONFIRMED but NOT promoted.
The next step (if we want to pursue this) is to add an "rl_pi_star"
policy to shadow_harness's policy registry, load the pickled π* on
startup, and let auto_policy A/B-replay it against K_combined on actual
live fires. If live fires confirm the +2.7% uplift in production
conditions, auto_policy's existing swap gates (min uplift 0.02 SOL/bet,
6h cooldown) would handle the migration automatically.

## Almgren-Chriss backtest — NEGATIVE result (commit d955ded)
`tools/ac_backtest.py` runs A-C hyperbolic schedule against K_combined on
the full May parquets (67K mints train + 96K mints OOS). Result is a
clean negative: **pure A-C with risk-aversion is uniformly WORSE.**

OOS table (N=96023, mean per-bet):

| Policy | mean | uplift vs K_combined |
|---|---|---|
| K_combined (LIVE)             | -0.0616 | — |
| H_time_spaced 15s             | -0.0570 | **+0.0045** |
| A-C kappa=0.00 T=120s         | -0.0570 | **+0.0045** (≡ H_time_spaced) |
| A-C kappa=0.01 T=120s         | -0.0586 | +0.0030 |
| A-C kappa=0.03 T=120s         | -0.0637 | -0.0021 |
| A-C kappa=0.05 T=120s         | -0.0664 | -0.0048 |
| A-C kappa=0.10 T=120s         | -0.0686 | -0.0070 |
| A-C kappa=0.20 T=120s         | -0.0693 | -0.0077 |
| C_hybrid 4+4 t30              | -0.0630 | -0.0014 |
| B_frontload                   | -0.0705 | -0.0089 |

**Theoretical interpretation**: A-C assumes constant drift + Gaussian
noise; under risk aversion, you liquidate FASTER to reduce variance.
Pump.fun has strong positive drift in the early phase + heavy upside
tails — the very thing A-C's risk-aversion penalizes is exactly what
makes the strategy work. Liquidating faster = missing the tail winners.

**Practical interpretation**: K_combined isn't far from the
schedule-only frontier. The uniform-time 15s schedule (which equals
A-C at kappa=0) beats K_combined by +0.45% per bet OOS — that's real
but below auto_policy's +0.02 swap threshold. The hand-tuned
ret-gating in K_combined is approximately as good as any pure
mathematical schedule.

**Where the real gains live**: state-aware policies (death-cut, RL,
regime detection). Pure scheduling has a small ceiling; getting beyond
~+0.5% per-bet requires the model to learn the regime-switching
boundary that the heuristic only approximates.

**No live action**: auto_policy.timer will pick up H_time_spaced if
live fires confirm the OOS uplift; not worth manually swapping for
+0.45% on the schedule alone.

## Sophistication audit context on entry_decision (Task 44, commit 417f033)
Bot's live listener now extracts the same gRPC-exclusive fields the capture
writes, attaches them to `ev.grpc_extras` (new optional field on `TradeEvent`),
and the harness maintains a per-mint sophistication window. At entry-decision
time the K=7 BUY-subset is aggregated by `_sophistication_summary()` and
stamped onto every `entry_decision` event under the key `"sophistication"`:

```json
"sophistication": {
  "n_buy_in_kwin": 7,
  "fee_p50_lam": 1005000, "fee_p90_lam": 3105000, "fee_max_lam": 3105000,
  "cu_p50": 98079, "cu_mean": 112308,
  "jito_tip_rate": 0.0, "jito_tip_p50_lam": null,
  "routed_rate": 0.0, "route_top": null,
  "n_inner_ix_mean": 11.7, "n_keys_mean": 17.3
}
```

The model itself does NOT consume these — pure annotation so we can later
correlate buyer-sophistication signatures with per-fire P&L BEFORE
committing to a model retrain. First post-deploy sample showed a 7-buyer
window paying fee_p50 = 1M lamports (40× the global median, 25K) with
ZERO Jito tippers — "competitive sniping by individuals who DIDN'T
bundle", a distinct profile from "coordinated bot pump via Jito".

## Sophistication feature integration — INTENDED FUTURE WORK (NOT done)
The audit half (Task 44) just LOGS the sophistication context alongside
fires. To actually have the bot USE these features in entry scoring,
three more steps are needed. **Do not forget:**

### Step 2 — extend `feature_accum.py` for windowed sophistication stats
Currently `TokenState` tracks the K=7 window for the 11 entry features
(win_ret, dir_eff, buy_frac, uniq, etc.). It does NOT track fee/cu/
jito/route stats. Need to either:

- Add a parallel deque inside `TokenState` storing the per-trade extras,
  plus accessor methods like `k_sophistication_features()` returning a
  tuple of 8-12 new stats (jito_tip_rate, fee_p90, cu_mean, routed_rate,
  fee_skew, etc.) computed at the K=7 trigger snapshot
- OR keep the side-car deque in the harness (where Task 44 lives) but
  make it K-anchored: snapshot the stats at the moment of K=7 trigger,
  same way `TokenState._snapshot_entry_feats()` freezes the 11 model
  features. This is cleaner because the model's features are
  POPULATION at K=7, not population RIGHT NOW.

The K-anchored snapshot is the right approach. Effort: ~3-4 hours.

### Step 3 — retrain with the new feature set
Once the K-anchored sophistication stats are in `feature_accum.py`, the
existing `extract_from_capture.py` needs a parallel update to emit these
as parquet columns from the archived capture. Then a new bot artifact
gets built:

```
tools/build_bot_artifacts_K7V_v2.py --inputs _fresh _capture --out _v2
  --extra-features sophistication  # adds 8-12 columns
```

The candidate model then goes through the existing two-gate hardening:
holdout AUC uplift ≥ 0.02 AND live-shadow AUC uplift ≥ 0.02 over the
production V+K7 model. Effort: ~half a day once the extractor + builder
are wired. Prerequisite: ≥3 days of wide capture for the new features
to actually have data, ideally ≥7 days for statistical power.

### Step 4 — `model_serve.py` reads the expanded feature list
The serve loop already drives off `model_spec.json`'s `features` list,
so a new model with 30 features (22 classic + 8 sophistication) gets
picked up automatically when `bot_artifacts_K7V` symlink swaps. The
ONLY change needed is that the harness's `score_entry()` call site
must know how to assemble the new features from `TokenState` accessors
(step 2 prerequisite). If step 2 puts the new features under a method
like `st.k_sophistication_features()`, the harness build of `ef` (the
22-feature dict) gets extended to add those values under the new
column names from `model_spec.json["entry"]["features"]`. Effort:
~1 hour, mostly testing the parity between offline extractor and
live accumulator (same kind of parity test we did for the V+K7 wire-up
in Task 14).

### Gating, the integration triggers
- Step 2 can start any time (independent code work).
- Step 3 needs ≥3-7 days of wide capture archived (currently we have
  hours; gate is `2026-06-10 ~05:00 UTC` for the 3-day minimum).
- Step 4 needs steps 2 + 3 to be done.

The simplest forcing function: **after the first auto_retrain swap
succeeds**, immediately start step 2. The retrain reminder
(`THRESHOLD REVISIT REMINDER`) is a natural prompt to also revisit
"should the next retrain include sophistication features?"

## Threshold-revisit decision (commit abcec8e, 2026-06-07)
User decision: keep `--entry-threshold 0.30` ON PURPOSE for data collection,
accept worse mean per-bet vs the trained 0.4481 operating point. Documented
in two coupled places so it surfaces at natural review points:

- **`systemd/pumpfun-bot.service`**: comment block right next to the flag
  listing the three review triggers (new model swap / drift normalization /
  ≥100 fires accumulated).
- **`tools/auto_retrain.py`**: after a successful retrain swap, prints
  `THRESHOLD REVISIT REMINDER` comparing the new model's native top-decile
  threshold to the live runtime override (scraped from the systemd unit
  via `_extract_runtime_threshold()`). Logged to `logs/auto_retrain.log`
  so the next operator check sees it loudly.

## Where everything stands now (running services)
- **pumpfun-bot.service**: active, K_combined exit, gRPC source, --live + DRY_RUN,
  threshold 0.30 override (documented decision, revisit triggers wired),
  config-driven knobs, risk limits + circuit breakers, reconciler +
  holdings-reconcile alive, restart-recovers via positions.jsonl,
  bet_sol=0.1, max_concurrent_positions=16
- **pumpfun-grpc-capture.service**: active, gRPC firehose recording (~105
  MB / 4h = ~5 GB / week)
- **pumpfun-auto-policy.timer**: every 4h, dry-run unless `policy-now`
- **pumpfun-auto-retrain.timer**: weekly Sun 02:00 UTC (waiting on
  grpc_capture archive ≥ 3 days; will skip until then)
- **pumpfun-drift-monitor.timer**: daily
- all `Restart=always`, enabled on boot
- 44 tasks completed; 0 pending or in-progress
- git: 17+ commits today, latest `417f033` (sophistication audit context).
  Recent: `15f2799` slice-exhaustion close, `d9d04a6` death-cut close +
  gate-replay, `2f6cf99` runner-exit close + exit_ret + asm_ms float,
  `abcec8e` threshold-revisit decision, `76b3cae` commitment pin + wide
  capture, `417f033` sophistication on entry_decision.

## What we have NOT promoted (and why)
- **Live submission to Jito is still gated**: JITO_DRY_RUN=1 in the unit.
  The tiny-tip test (Task 37) proved the reconciler works end-to-end on a
  real failed bundle, but the wallet has 0 SOL and we have not been
  authorized to run with a funded wallet. Going live is a separate
  explicitly-authorized step.
- **Auto-retrain has not yet run successfully**: gates on ≥3 days of
  capture. First eligible Sunday will be once `grpc_capture/` accumulates.
  Until then the timer fires and exits cleanly with "not enough capture".
- **Drift alert still warning**: `location_shift` live p90 0.2365 vs train
  p90 0.4453 — expected because we lowered threshold to 0.30. Auto-retrain
  will re-fit the threshold to the live distribution when it eventually
  swaps.

## Honest P&L state (as of 2026-06-07 ~07:38 UTC, 24 closed paper trades @ bet=0.1)
- mean fractional: +0.047 (≈ +0.0047 SOL / trade absolute)
- wins / losses: 6 / 18 (25%)
- best: +1.864 fractional (+0.186 SOL on a 0.1 bet — a 2.8x winner)
- worst: -0.510 fractional (-0.051 SOL, half the bet wiped)
- mean barely positive, well within sampling noise; one big winner pays
  for ~3.6 worst-losers
- **DRIFT** is the more important signal than the streak: we are firing on
  setups systematically weaker than training because of the lowered
  threshold. Auto-retrain is the path forward.

---

## Catch-up: Tasks 50-54 + adjacent work (2026-06-07 evening → 2026-06-08)

### ERPC HTTP credit drain — fixed (Task 50, commits 314afdc / 55e335b / ea24e91)
8.3M of 10M monthly credits burned in 5 days. Source: blockhash_cache.py
hammering ERPC HTTPS `getLatestBlockhash` at 200ms poll = ~200K calls/day,
projected to burn 550M credits/month at ~43 credits/call. Switched to
**Yellowstone gRPC `GetLatestBlockhash`** on the same endpoint we already pay
for via subscription (`grpc-fra1-1.erpc.global:80`). gRPC traffic is bundled —
zero per-call billing. HTTPS public mainnet kept as fallback only.

Verified ZERO new HTTPS calls post-fix via `ss -tnp` monitoring over 60s
(only CLOSE-WAIT corpse from pre-fix process). ERPC dashboard counter
stopped decreasing within ~30 minutes (pipeline catch-up).

Side benefit: added `bh_slot` + `slot_gap` (trigger_slot − bh_slot) to every
broker_jito event so we can audit "are we keeping up with the chain" in
slot units rather than wall-clock ms.

### Dashboard polish + correctness fixes
Multiple commits across the day:
- **`e116b7b`** Open-positions row shows `silent / await_trades` (dim yellow)
  when a position is older than 10s with no path_snap events — sniper-burst-
  then-vanish pattern, otherwise looked like UI bug.
- **`cca7ebf`** Mint cells in BOTH OPEN POSITIONS and RECENT TRADES rendered
  with OSC 8 hyperlinks to `https://pump.fun/coin/<mint>`. Cmd/Ctrl+click
  opens in browser; non-OSC8 terminals just see styled text.
- **`bae6a22`** Daily P&L panel was UNDER-COUNTING by ~4x. positions.jsonl is
  ~95% snap events; tail_jsonl(5000 lines) returned only ~50 of 167 actual
  closes in last 24h. Switched to full-file scan with cheap `"kind": "close"`
  pre-filter.
- **`7f05b28`** Combined exit_kind + close reason — `hold/stale` vs `cut` is
  now visible, ending the confusion about "did we sell or are we still in?"
- **`7ea570e`** Added `dur` column to RECENT TRADES showing time from entry
  to close (or live age for still-open).
- **`88c7e21`** EXEC LATENCY panel now time-windowed (last 30min) instead of
  last 200 events, so the fix-effect shows immediately after bot restarts.

### live_policy_net wired (Task 51, commit 2eafcef → 2ed533b)
Initial premise was wrong. I claimed the dashboard's `mean +0.0067 SOL/bet`
under-reported by 3x. Reality:

| Source | Today total | Note |
|---|---|---|
| `broker_jito.jsonl` actual (TRUTH) | +1.14 SOL | sum(sell.expected_sol_out) − sum(buy.sol_in) |
| PaperBook GREEN ref (was on dashboard) | +1.37 SOL | overstates ~20% |
| Pure live-policy replay (no death cut) | +0.06 SOL | catastrophic understate |

GREEN ref was close to truth, NOT 3x off. My earlier "3x understate" claim
was based on the broken pure-policy-replay number which misses the death cut.

Switched daily-P&L panel to read `broker_jito.jsonl` directly — ground truth,
no models. Mean panel still shows PaperBook (within 20% of truth — close enough
for now).

Also wired: `live_policy_net` field on every new close event via
`_compute_live_policy_net(pos)` helper (replays the active policy against the
position's snap timeline). Plus `tools/backfill_live_policy_net.py` for any
historical close. Currently the replay is missing the death-cut logic so
its numbers are pessimistic; production-grade fix is to also include death
cut in the replay (todo for later).

### Models still healthy — OOS evaluated on fresh capture (Task 52)
Both production models tested against the 24h gRPC capture data (10,315
ready V+K7 mints):

| Model | Training AUC | OOS AUC (24h fresh) | Decay |
|---|---|---|---|
| Entry (peak >= 2x) | 0.7743 | 0.7639 | -0.010 (1.3%) |
| Recovery (recovers >= 0) | 0.8061 | 0.7956 | -0.011 (1.3%) |

Negligible drift. **Models are still well-calibrated.** The `location_shift`
drift alert is about the score distribution of LIVE FIRES (filtered through
the 0.30 threshold), not the model's predictive AUC. Two different things.

Recovery model's death-cut threshold (0.10) still works:
- 20.6% of drawdown snaps trigger cut
- Of those flagged: only 3.6% would actually recover (precise)
- Of those not flagged: 41.8% recover (correct hold)

### Candidate retrain combining May + 24h capture (Task 53, commit 03f3186)
Built candidate model with `build_bot_artifacts_K7V.py --inputs _fresh
_capture_jun8`. Bug fix: `drop_duplicates(subset=mint, keep=last)` before
`set_index` to handle mints that appear in both suffixes.

Results:
- Training tokens: 146,737 (prod) → 157,063 (cand). +7% data.
- Training AUC: 0.7743 (prod) → 0.7737 (cand). Within noise.
- Live-shadow AUC on 201 fires: 0.4902 (prod) → 0.6097 (cand). +0.12 uplift.
- BUT: cand was trained on June 7 data, fires are from June 7 — contaminated.

Verdict: don't promote this candidate. Real OOS uplift is probably +0.005
to +0.04, not +0.12. Wait for Jun 14 auto-retrain with ~7 days of capture.

Candidate saved at `bot_artifacts_K7V_capture_jun8/`. Available for manual
promotion if desired, not recommended.

### Sophistication features — proper data exploration (Task 54, commit c569af1)
`tools/extract_sophistication.py` computes per-mint K-window aggregates of
the gRPC-exclusive fields (fee_p50/p90, cu_mean, jito_tip_rate, routed_rate,
n_inner_ix_mean, n_keys_mean). 8,868 mints with full coverage from the 16h
of wide capture.

Tested as additions to the existing 22-feature model across multiple
targets (5,453 mints with full data, 70/30 train/test split):

| Target | Base | 22-feat AUC | 31-feat AUC | Uplift |
|---|---|---|---|---|
| peak >= 2x (current entry target) | 40% | 0.7364 | 0.7335 | -0.003 |
| peak >= 5x | 12% | 0.6515 | 0.6515 | 0.000 |
| **peak >= 10x (moonshot)** | 5% | 0.5913 | 0.6416 | **+0.050** ✓ |
| **terminal >= entry (no rug)** | 13% | 0.5953 | 0.6201 | **+0.025** ✓ |
| FAST RUG (peak<30s + dump) | 50% | 0.7370 | 0.7469 | +0.010 |
| hit 2x AND terminal >= +50% | 9% | 0.6049 | 0.6155 | +0.011 |

**Sophistication features add NO value to the standard entry decision.**
The classic features already capture coordinated-burst patterns implicitly
via `single_actor_share`, `trades_per_sec`, etc. Sniper bundles often DO
reach 2x before dumping — Jito tip rate doesn't predict 2x outcomes.

**Sophistication DOES add value for two specific problems:**
1. **Moonshot identification (peak >= 10x)**: +5% AUC. Could sharpen the
   Q5 router in `bot_artifacts_K7V_rl_layered/q5_classifier.pkl`.
2. **Rug detection (terminal < entry)**: +2.5% AUC. Could augment the
   recovery model for death-cut decisions.

Saved auxiliary classifiers at `bot_artifacts_K7V_sophistication_jun8/`:
- `moonshot_classifier.pkl` (target: peak >= 10x)
- `rug_classifier.pkl` (target: terminal < entry)

**Not promoted to production**: only 5,453 mints with full coverage = ~270
moonshot positives. Statistically underpowered. Wait for 7+ days of wide
capture for confident OOS conclusions.

### Live policy: swapped k_combined → c_hybrid_t30 + snap_every=1 (commit 8b251aa)
Throughout the day auto-policy has been cycling:
- Default → k_combined → c_hybrid_t30 (10:49 UTC swap) → h_time_spaced
  (18:54 UTC swap) → c_hybrid_t30 (23:34 UTC swap, --force after cooldown)

Final state: **c_hybrid_t30 — "paced derisk + trail-stop at 30% retrace from peak".**
This is the scale-out-on-the-way-up + lock-in-near-peak hybrid. Best fit
for moonshot capture per the 30-fire-window A/B replay (mean +0.187 fractional).

Concurrent change: **snap_every 3 → 1** so the trail-stop fires on the
actual peak inflection, not 3 events later. Recovery model now scored on
every trade event (3x volume, still ~0.1ms/call — negligible).

Observed moonshots today (peaks we hit):
- C7GCW4q9KU peak +521%, PaperBook net +186%
- EevLvhiyGShh peak +460%, PaperBook net +196%
- 9fZYi3nyccyfH8 peak +446%, PaperBook net +209%
- eJkqSSoBX65rbi peak +445%, PaperBook net +216%
- 2Rurm6GvGaLVoS peak +438%, PaperBook net +237%
- 7 of 202 closed today peaked above 2.5x

The gap between peak (+460%) and realized (+196%) is the under-realization
problem. c_hybrid_t30 + snap_every=1 should narrow that gap on the next
moonshot.

### Honest revisions to earlier claims
Two things I claimed strongly during the session that I later corrected:

1. **"Dashboard understates by 3x"** — wrong. The replay-based "live policy"
   number I compared against was missing death-cut logic. Real ground truth
   is broker_jito.jsonl which is within 20% of GREEN ref, not 3x off.
2. **"+4.85 SOL hypothetical under h_time_spaced"** — replay artifact. Pure
   policy replays without death cut overstate by 5-10x because they hold
   losers indefinitely. True realistic uplift between policies is in the
   ±30% range, not 5x.

Lesson: replay numbers are useful for RANKING policies relative to each
other under the same replay assumptions, but the absolute SOL values are
inflated. Ground-truth is `broker_jito.jsonl`.

## Catch-up: shred-stream + storage-box + parsing audit (2026-06-08 → 2026-06-09)

Big batch of work between the snapshots above and the next operational
state below. Notes here are not exhaustive — see the file-level docstrings
and `shred_bot/intent_capture/SCHEMA.md` for the canonical reference.

### level_tp_100 became the live policy
- New policies in `exit_policies/level_tp.py`: `level_tp_50`, `level_tp_100`,
  `level_tp_200` — sell-all when ret >= threshold.
- Backtest on bot-selected universe (`tools/policy_replay_full.py`) showed
  level_tp_100 dominates the c_hybrid family on c_hybrid_t30's own
  universe (entry_score >= 0.30 override). Promoted to live policy.
- Entry threshold raised from 0.30 → trained 0.5108 (the `--entry-threshold
  override` in the systemd unit removed). Bot now fires at the model's
  actual operating point.
- Auto-policy made **recommend-only**: the `--execute` flag was removed from
  the systemd timer so it never silently flips a policy out from under us
  again. (It once reverted level_tp_100 back to c_hybrid_t30 in the night
  at 09:04 UTC; root cause was the auto-loop still owning the live policy.)
- New rigorous companion: `exit_policies/lsm_continuation.py` — LSM-style
  HGB regressor predicts E[future_ret | s_t] and exits when the expected
  continuation value < 0. Not deployed, kept as a research artifact for
  later promotion.

### Drift root cause — coverage bias, not regime drift
The chronic "live distribution disagrees with training" warning was finally
traced. The training data was generated from gRPC capture; the live bot
sees roughly the **same population** but with a **different filter**:
- Training: every mint that appeared in capture
- Live: only mints whose first event the bot processed before its
  feature-accum window closed (38% of all mints, biased toward slow ones)

`tools/drift_decomposition.py` proves it with a 3-way comparison
(training pool vs. full June capture vs. bot-observed subset). The
coverage gap reproduces the score-distribution gap exactly. Not a feature
bug, not regime drift — coverage. Implication for the drift monitor: use a
**dynamic** train-p90 reference (computed over the bot-observable subset),
not the model-card p90 from training.

### State persistence + capture-replay restart recovery
Bot can now restart without losing in-flight context.

- `feature_accum.py` gains `to_dict()` / `from_dict()` on TokenState.
  `shadow_harness.py` writes a checkpoint every N seconds.
- Recovery is **hybrid**:
  - SHORT downtime (≤5 min): load the checkpoint, then replay the gRPC
    capture from the checkpoint time to "now" to fill the gap.
  - LONG downtime: skip the checkpoint, do a **full capture-replay
    bootstrap** from grpc_capture for the last K minutes (the same source
    we'd use to retrain features). This is the ground-truth restoration
    path; checkpoint is just a speed shortcut.
- Positions stay **OPEN** across restart. The bot re-injects
  `broker.holdings` for any token with an unfilled-exit position recorded
  in `broker_jito.jsonl` and lets the exit policy continue managing them.
  Earlier mistake: a `force_close_on_restart=True` mode that sold-all on
  startup. Correct semantics: restart is not a market event.
- Critical fix in `pumpfun_bot.py` after the first wave: recovered mints
  must also be added to `self.open_paper` because the forward-snap router
  and stale-watchdog check that set, not just `broker.holdings`. Without
  it, a recovered position would be invisible to the routing/watchdog
  paths even though the broker knew it existed. (commit `e25de48`)

### Shred-stream bot — front-run signal feed (NEW infrastructure)
Built a separate process tree that consumes the Jito Shredstream Proxy
**before the block is finalized** (pre-execution). All of it lives under
`shred_bot/`:

- `shred_bot/intent_extractor.py` — decodes the bincode+wire-format hybrid
  payload (Vec<Entry> bincode wrapper + Solana-wire VersionedTransactions
  + ShortVec compact-u16) and emits one record per pump.fun program ix.
- `shred_bot/intent_recorder.py` — long-running daemon, writes JSONL +
  feeds the SHM ring. Hourly rotation, inline gzip on rotation.
- `shred_bot/intent_ring.py` — SPSC shared-memory ring buffer for IPC.
  Lockless via monotonic u64 write_seq (atomic 64-bit on x86_64).
  `RECORD_FMT = "<QQ32s32sQQQQIB3x"` = 120 bytes per record.
  Includes the cpython issue-#82300 `resource_tracker` detach workaround
  on the reader side (otherwise the kernel SHM gets unlinked on reader
  exit which kills the writer).
- `shred_bot/raw_shred_firehose.py` — **separate** archival recorder.
  Filters NOTHING, saves EVERY shred entry message verbatim. Writes
  directly to `/mnt/storagebox/raw_shred_entries/`. See dedicated section
  below.
- Systemd units: `pumpfun-shred-intents.service` (hot path, local disk),
  `pumpfun-shred-firehose.service` (cold path, storage box).

The bot itself does NOT consume the SHM ring yet for trade decisions —
the bot still uses the executed-tx gRPC capture (Yellowstone) as its
primary signal. The shred stream is being recorded so a **future** front-
run capability can be added on top of established signal flow.

### Hetzner storage box (BX21) — mounted as sshfs
- 5 TB (not 1 TB — earlier estimate was wrong; runway revised below).
- Mounted via sshfs (SFTP-backed) at `/mnt/storagebox` using deployment-local
  connection settings and an untracked SSH identity.
- Mount line in `/etc/fstab` (key bits): `:./backup` (relative path — SFTP
  defaults to the Storage Box user's home, NOT `/`), `reconnect`,
  `ServerAliveInterval=30`.
- The firehose writes directly to the mount. If the mount drops, the
  recorder logs an error and falls back to local `shred_bot/raw_shred_entries/`
  (small disk, low runway — alert path only).

### Raw firehose — what we save + compression cadence
- One file per hour, named `raw-shreds-YYYYMMDDTHHMMSSZ.bin`.
- On rotation, the previous `.bin` is **gzipped inline (compresslevel=3)**.
  This briefly blocks the recorder (~30-60s for a typical 5 GB file via
  sshfs).
- Observed compression: ~5.3 GB raw → ~2.6 GB gzipped (~49% size).
- Throughput: ~60-100 frames/s, ~5-6 GB/h uncompressed, ~1.5 GB/h gzipped.
- 5 TB storage / 1.5 GB/h gzipped = ~**90-100 days of archival runway**.
- Schemas on disk: `shred_bot/intent_capture/SCHEMA.md` (filtered JSONL)
  and `shred_bot/raw_shred_entries/SCHEMA.md` (raw binary frames). The
  raw-firehose schema is also mirrored to `/mnt/storagebox/raw_shred_entries/SCHEMA.md`
  so the data and its format are colocated.

### Spoof-resistance signals (peer review)
Added two structural filters that detect "free spoof" txs (sandwich-bait
that costs the sender ~0 because it's guaranteed to revert):
- `bonding_curve_writable` — accs[3] (v1) / accs[10] (v2) must have its
  write-lock set in the message header. If it doesn't, the tx structurally
  cannot mutate AMM state → guaranteed revert. The on-chain priority fee
  is debited UPFRONT regardless of revert, but the spoofer pays close to
  zero because they set `cu_limit` tiny.
- `cu_limit_too_low` — real pump.fun buys need 30-50k CU. <25k CU
  guarantees `ComputationalBudgetExceeded` revert (free spoof).
- `probable_spoof` = (NOT bonding_curve_writable) OR (cu_limit < 25k).

These are JSONL fields only — the bot does not gate on them yet.

### Clock sync — chrony swap-in
Replaced systemd-timesyncd with **chrony** (multi-pool stratum-1 anycast:
time.cloudflare.com, time.google.com, ntp.hetzner.com — iburst, minpoll 6,
maxpoll 8, makestep 1.0 3, rtcsync). Sub-ms accuracy. Matters less for
intra-server timing than cross-server correlation work (which we don't
yet need), but useful hygiene.

### Pump.fun discriminator audit — 100% coverage (commits `f024d58`, `ad1ac1c`)
Brute-forced + IDL-verified every observed discriminator. The complete
mapping lives in `shred_bot/intent_extractor.py:PUMPFUN_IX_NAMES` and in
`shred_bot/intent_capture/SCHEMA.md`.

After the IDL audit (against `pump-fun/pump-public-docs` official IDLs):
- All 23 distinct discriminators in the captured stream are now labeled.
- `pumpfun_other` went from ~10.3% to **0.0%** in live data.
- Trade-shaped: `buy`, `sell` (v1 + v2 disambiguated by `version` field),
  `buy_quote` (USDC-quote router), `buy_sol_in` (legacy SOL-in router).
- Non-trade plumbing: `create`, `migrate`, `claim_cashback` (v1+v2),
  `init_volume_accum`, `sync_volume_accum`, `close_volume_accum`,
  `collect_creator_fee` (v1+v2), `distribute_creator` (v1+v2),
  `extend_account`, `set_params`, `initialize`, `withdraw`.

### CRITICAL parsing bug — v2 account layout shift (fixed 2026-06-09)

During a parsing audit, discovered that the v2 buy/sell aliasing was
using the v1 account positions. Per the official IDL:

| field | v1 buy/sell | v2 buy/sell |
|-------|-------------|-------------|
| target token mint | accs[2] | **accs[1]** (`base_mint`) |
| (quote mint) | implicit SOL | **accs[2]** (`quote_mint`: SOL or USDC) |
| user (signer) | accs[6] | **accs[13]** |
| bonding_curve | accs[3] | accs[10] |

Empirical impact on data captured before the fix: of 341 buy+sell records
in the sample window, **26.4% had the wrong `mint` value** — specifically,
17% had SOL and 9.4% had USDC written where the base mint should have been.
These are the v2 records where we extracted accs[2]=quote_mint and called
it the mint.

**Fix:**
- Dispatch by discriminator: `is_v2 = (disc == BUY_V2_DISC or disc == SELL_V2_DISC)`.
- Use IDL-correct positions per version.
- Added `version` (1 | 2), `quote_mint` (SOL/USDC for v2, null for v1),
  `ix_disc_hex`, `ix_accounts` to the JSONL record schema. The last two
  let us repair any future ABI shift retroactively from saved data
  without needing fresh capture.

Post-fix verification on 1888 records: mint is 96.2% pump-suffix, 0% SOL/USDC.

**Historical data is corrupted for v2 records.** Repair path: cross-
reference `first_sig` against grpc_capture (which has executed txs with
meta = correct mint) and overwrite the mint field. Not done yet.

### Auto-policy + drift-monitor systemd hardening
- `pumpfun-auto-policy.timer`: 4h cadence, **recommend-only** (no
  `--execute`). Writes `logs/policy_decisions.jsonl` so we can see what it
  *would* have done.
- `pumpfun-drift-monitor.timer`: 00:07 UTC daily, dynamic train_p90
  reference (computed over bot-observable population, not training pool).

## Current operational state (2026-06-09 ~05:40 UTC)

**Running services on sol:**
- `pumpfun-bot.service`: `level_tp_100` exit policy, entry threshold 0.5108
  (trained), gRPC source on commitment=PROCESSED, blockhash via gRPC.
- `pumpfun-grpc-capture.service`: wide capture (fee_lam/cu/jito_tip/route),
  ~33 MB/h.
- `pumpfun-shred-intents.service`: shred-stream → JSONL + SHM ring,
  hourly rotation, ~5-20 MB/h. **v2 fix deployed 2026-06-09 ~05:32 UTC.**
- `pumpfun-shred-firehose.service`: raw-shreds → `/mnt/storagebox`,
  ~5-6 GB/h raw, ~1.5 GB/h gzipped. Last rotation at 03:34:23 Z gzipped
  to 2.6 GB.
- `pumpfun-auto-policy.timer`: 4h, recommend-only.
- `pumpfun-drift-monitor.timer`: daily 00:07 UTC, dynamic reference.

**Disk runway (storage box):**
- ~1.5 GB/h gzipped, 5 TB total = ~140 days at current rate.

**Open known issues:**
- v2 historical data corruption (26% of buy/sell records pre-2026-06-09 05:32).
  Repair via grpc_capture cross-ref pending.
- Program-id ambiguity for `buy`/`sell` disc collision between bonding curve
  and PumpSwap AMM. `ix_program_id` is currently None in JSONL; if we want
  to split pre-grad from post-grad activity we need to capture it.
- The intent-extractor's pre-2026-06-09 records lack `ix_disc_hex` and
  `ix_accounts` on buy/sell rows so we can't always disambiguate after
  the fact without grpc_capture.

---

## Original 2026-06-08 ~01:33 UTC snapshot (kept for history)

**Running:**
- `pumpfun-bot.service`: c_hybrid_t30 exit policy, snap_every=1, threshold 0.30
  override (data collection mode), gRPC source on commitment=PROCESSED,
  blockhash via gRPC GetLatestBlockhash (zero HTTP credit usage)
- `pumpfun-grpc-capture.service`: wide capture (fee_lam/cu/jito_tip/route)
  recording at ~33 MB/h, archive currently ~539 MB (24h+ of wide data + earlier
  narrow data)
- `pumpfun-auto-policy.timer`: every 4h, dynamic mode (will swap if gates pass)
- `pumpfun-auto-retrain.timer`: weekly Sun 02:00 UTC — first eligible attempt
  Jun 14 ~02:00 UTC with ~7 days of capture

**Stats today (Jun 7-8, partial 2 days):**
- 208 closed positions, mean +0.0067 SOL/bet (PaperBook GREEN ref)
- Actual realized P&L from broker_jito.jsonl: ~+1.14 SOL absolute
- Win rate ~40% (PaperBook GREEN scheme)
- 5 tokens reached +5x peak, 0 reached +10x

**Tasks completed:** 54.
**Total git commits this session: 30+** (see `git log` on sol).

---

## exit_ret=0 root cause + snap_every=1 silent bug (2026-06-09 ~06:30 UTC, commit `f49dfc9`)

User noticed in the dashboard that every closed position under
`level_tp_100` was showing `exit_ret: +0.000` exactly, with `net SOL`
showing only the entry fee. Investigated.

### What was happening
In `shadow_harness.py` the hot-path snap update is gated on
`fwd_counter[mint] % SNAP_EVERY == 1`. With `snap_every: 1` in
config.yaml (the "react to every tick" change we made earlier),
`n % 1 == 1` is mathematically impossible (anything mod 1 = 0), so the
condition silently evaluated False on every event. The block under it
contains:
- the `book.add_snapshot()` call that appends to `snaps_ret_vs_midV`
- the `pos.vsC = ev.virtual_sol_reserves` / `pos.vtC = ev.virtual_token_reserves` updates
- the live exit policy dispatch (`_dispatch_exit_slice`)
- the death-cut check (`p_rec < death_threshold`)

So for every entry fired in the last ~24h: nothing post-entry updated.
The stale watchdog eventually closed the position, calling
`_pos_exit_ret(pos)`. With `snaps_ret_vs_midV` empty AND vsC/vtC frozen
at entry-time vsK/vtK, the fallback returns `(vsK/vtK)/(vsK/vtK) - 1.0
= 0.0 EXACTLY`. That's the `+0.000` in the dashboard.

Empirical confirmation in the 24h before the fix:
```
7,027 fires  ->  0 snaps  ->  0 exit_slices  ->  8 stale_closes
exit_ret distribution: {exact_0.0: 8, NONE: 2}
```

### Fix
Same guard `tools/extract_from_capture.py` already used:
```python
if SNAP_EVERY == 1 or self.fwd_counter[mint] % SNAP_EVERY == 1:
```
Special-cases the snap_every=1 mode. Net result: snap path fires on
every forward event when snap_every=1; original snap_every=3 cadence
of `fwd=1,4,7,...` preserved otherwise.

### Practical impact while broken
- Bot kept buying (entry path works fine — entry features are at
  trigger time, not at snap time)
- Live policy `level_tp_100` never got consulted post-entry, so the
  +100% take-profit rule physically couldn't fire
- Stale watchdog still closed positions and sold via broker, but using
  the stale vsC/vtC (entry-time price). In paper mode this means net
  return tracking shows -0.006 SOL per trade (just the entry fee).
  In live mode, the sell-side slippage params would be wrong but the
  sell would still go through.

## Sophistication features: methodology re-test (2026-06-09 ~06:55 UTC, draft artifact)

Re-tested whether the 9 sophistication features (fee percentiles, CU,
jito_tip_rate, jito_tip_p50, routed_rate, n_inner_ix_mean, n_keys_mean)
add OOS uplift to the entry head, after the user pointed out that
`bot_artifacts_K7V_wide_jun8` had reported `sophistication_uplift_oos = 0.0`
and asked whether the original test was set up correctly.

### What was wrong with wide_jun8
- Trained on `n_train = 86,033` rows where ~90% had NaN sophistication
  features (the join was LEFT, and `sophistication_capture_jun8.parquet`
  only had 8,868 mints with real soph data — all from a short window of
  June 8 wide capture).
- HGB tolerates NaN at prediction time but can't learn signal from a
  feature that is NaN in 90% of training rows.
- Resulted in the model not bothering to use the soph features → 0.0
  uplift was a methodology artifact, not "soph features don't help".

### Wide_v2 fix + result (tools/train_wide_v2.py)
- Re-ran `extract_sophistication.py` on current grpc_capture (2.8x more
  mints than jun8 had: 25,175 vs 8,868). More wide data accumulated.
- INNER JOIN K7+V05 with the soph parquet so every training row has
  real soph values. Drops 93% of K7+V rows (those without wide capture
  coverage); the surviving 21,772 rows form the cleaner train+test set.
- 80/20 stratified split, same indices applied to baseline (22 features)
  and wide (31 features). Apples-to-apples.

Result:

|                          | n_test | baseline AUC | wide AUC | OOS uplift |
|--------------------------|--------|--------------|----------|------------|
| wide_jun8 (NaN-diluted)  | 7,667  | 0.7626       | 0.7626   | **0.0**    |
| wide_v2 (inner-join)     | 4,355  | 0.7868       | 0.7992   | **+0.0124**|

The +0.0124 OOS uplift is roughly 1.5-2σ above zero given n=4,355 — a
real positive signal, not as tight as we'd like but not noise either.
Baseline AUC is itself higher in wide_v2 (0.787 vs 0.763) because the
inner-join population is the "future-like" subset (post-wide-capture
mints, June onwards) without May-vs-June regime mixing.

Output: `bot_artifacts_K7V_wide_v2/` (entry_model.pkl, entry_model_baseline.pkl,
model_spec.json). **NOT symlinked live** — kept as a research artifact.

### Plan for the NEXT rebuild round (~few days from now)

Today the gRPC capture is also producing **gap A-H features** (turned
on 2026-06-09 ~06:08 UTC, commit `047ab0b`). After a few days of
accumulation the augmented capture will give us a feature menu we
haven't tried yet:

| Gap | Field | Hypothesis for use in entry/recovery |
|-----|-------|---------------------------------------|
| A | All 23 bonding curve events | `CreateEvent` timing → age-of-token feature; `CompleteEvent` / `CompletePumpAmmMigrationEvent` → graduation flag (binary feature + time-to-graduation). `SetCreatorEvent` / `MigrateBondingCurveCreatorEvent` → creator-change-before-rug signal. |
| B | PumpSwap (pAMMBay) events | Post-grad market activity per mint, can label tokens that graduated even if we don't trade them. Useful for survivorship-controlled training labels. |
| C | Failed txs with `failed` flag | Per-mint failed-tx rate at entry-time = spoof prevalence in the K-window. Could be a sophistication feature. |
| D | `pre_tb` / `post_tb` (pre/post token balances) | Exact who-got-how-many ledger. Can compute precise SPL transfer graphs for sandwich detection at entry. |
| E | `inner_ix` summary (CPI tree) | Aggregator-routing reconstruction more precise than current `route` header match. Detect Jupiter/Photon/etc. by their CPI signature. |
| F | `loaded_w` / `loaded_r` (ALT-loaded addresses) | Complete account view for V0 txs that use lookup tables. Fixes the `n_keys` undercount we currently have on sophisticated routes. |
| G | `cu_limit` (requested) | Already captured in shred-intent path; now also in gRPC. Lets us compute headroom = cu_limit - cu (CU slack), a known sophistication signal. |
| H | `block_time` | Validator-witnessed timestamp. Currently `t` is our wall-clock receive time. Replace `t` with `block_time` for cross-replay consistency. |

Re-running `train_wide_v2.py` in 3-7 days with these added features
(after we extract per-mint aggregates from the new wide capture)
should give a much richer feature menu. Expected uplift candidates,
ordered by my prior:

1. **graduation flag** (gap B) — direct evidence that a token reached
   the post-grad regime, which strongly correlates with surviving the
   bonding-curve phase. Highest-prior win.
2. **failed-tx rate at entry** (gap C) — sophisticated buyers' fail
   rate is a strong actor-quality signal in adjacent research.
3. **cu_headroom** (gap G) — direct sophistication proxy, already
   shown useful in shred-path analysis.
4. **CPI-tree-based router signature** (gap E) — finer-grained than
   the current `route` match.

We should re-run `extract_sophistication.py` then (it currently looks at
`fee_lam`, `cu`, `n_inner_ix`, `n_keys`, `jito_tip_idx/lam`, `route`)
and EXTEND it to also aggregate per-mint failed rate, cu_headroom mean,
graduation_seen flag, and the new `pre_tb`/`post_tb` summaries. Then
train_wide_v2 just consumes the wider parquet — no other code change.

Decision rule for shipping the next round:
- OOS uplift over the 22-feature baseline >= +0.02 → ship
- Uplift +0.01 to +0.02 → ship with `--entry-threshold` recalibration
  to match the new score distribution (avoid the same drift problem
  we have today)
- Uplift < +0.01 → keep researching, don't ship.

The current `wide_v2` artifact (+0.0124 OOS uplift) sits at the
borderline of that rule. We're deliberately NOT shipping it yet
because the wider-feature rebuild in a few days should beat it
clearly, and one swap is better than two.

---

## SHIPPED: wide_v2 model + calibrated threshold (2026-06-09 ~07:17 UTC, commit `fccd9fe`)

Plan changed after a closer backtest. The wide_v2 candidate beats
production by +0.0180 OOS AUC (not just the +0.0124 on its own holdout)
and the threshold was solvable. Shipped.

### What's live now
- `bot_artifacts_K7V/` symlink → `bot_artifacts_K7V_wide_v2/`
- `bot_artifacts_K7V_pre_wide_v2_swap_1780981049/` kept as rollback
- Bot restarted on the new artifact, no errors in startup, scoring
  with the 31-feature head verified on the first entry_decisions

### Final feature set in production

**Entry head (31 features, HGB):**
| # | Feature | Origin |
|---|---------|--------|
| 1-11 | win_ret, dir_eff, buy_frac, uniq, net_sol, tot_sol, single_actor_share, trades_per_sec, entry_sol, win_drawup, win_drawdown | K-window aggregates at K=7 trigger |
| 12-22 | (same names, _v suffix) | V-window aggregates at V=0.5 trigger |
| 23-24 | soph_fee_p50_lam, soph_fee_p90_lam | gRPC fee distribution in K-window |
| 25-26 | soph_cu_p50, soph_cu_mean | Compute units used |
| 27-28 | soph_jito_tip_rate, soph_jito_tip_p50_lam | Fraction of buys with Jito tip, median tip lamports |
| 29 | soph_routed_rate | Fraction of buys via known aggregator |
| 30-31 | soph_n_inner_ix_mean, soph_n_keys_mean | Inner-instruction count + total key count means |

**Recovery head (20 features, HGB):**
- 9 path features: ret, run_max_ret, dd, fill_k, buy_frac_w, nsell_w, solo_sell_w, vel_w, dts
- 11 K-entry features (same as 1-11 above, frozen at entry)

### Thresholds in production
- Entry: 0.4134 (calibrated to ~2% live fire rate, NOT the training top-decile
  of 0.6357 — the live distribution is shifted lower so the training top-decile
  would fire too rarely)
- Death cut: 0.10 (unchanged)

### How the integrated rebuild worked
- `extract_sophistication.py` re-run on current grpc_capture → 25,175 mints
  with soph features (2.8x more than the original jun8 8,868 mints — that's
  why this round showed +0.0046 OOS uplift from soph vs jun8's 0.0)
- `tools/train_integrated_v2.py` inner-joins K7+V05+soph (21,691 mints clean
  training set), trains both heads in one run
- Entry: 80/20 stratified split, baseline (22) vs wide (31) on same indices
- Recovery: trained on _fresh + _snap1 path-snapshots combined (2.9M
  drawdown-snap rows, 5.4x the production model's 537k)
- `shadow_harness.py` updated to map `_sophistication_summary()` output to
  the model's `soph_*` feature names with NaN fallback (HGB-friendly)

### Future feature set (gap A-H — augmented gRPC capture)

The augmented gRPC capture (commit `047ab0b`) is producing these as we
speak. Need 3-7 days of accumulation before a meaningful retrain.

| Gap | New field(s) | Hypothesis for use |
|-----|--------------|---------------------|
| A | All 23 bonding-curve events | `CreateEvent` → age-of-token; `CompleteEvent` / `CompletePumpAmmMigration` → graduation flag; `SetCreatorEvent` → rug-prep signal |
| B | PumpSwap (pAMMBay) events | Post-grad activity → survivorship-controlled training labels |
| C | Failed txs (`failed` flag) | Per-mint failed-tx rate at entry → spoof prevalence |
| D | `pre_tb` / `post_tb` | Exact SPL transfer graph → sandwich detection |
| E | `inner_ix` summary | Aggregator-routing reconstruction more precise than current `route` |
| F | `loaded_w` / `loaded_r` | Complete V0-tx account view |
| G | `cu_limit` (requested) | cu_headroom = cu_limit − cu, direct sophistication proxy |
| H | `block_time` | Validator-witnessed timestamp |

### Ship-decision rule (locked)
- OOS uplift over baseline ≥ +0.02 → ship
- +0.01 to +0.02 → ship with `--entry-threshold` recalibrated
- < +0.01 → keep researching

---

## Shred-stream ring buffer — open question, not a plan

`shred_bot/intent_recorder.py` writes every pump.fun buy/sell ix to a
SPSC SHM ring (`pumpfun_intents`, 120B records) the moment it arrives
in a shred, which is ~100-400ms BEFORE the executed-tx Yellowstone
gRPC stream sees the same tx land in a block. Records carry
`is_buy`, `slot`, `mint`, `user`, `signer`, `first_sig`, `recv_ns`,
`token_amount`, `sol_limit_lam`, `priority_fee_micro`, `cu_limit`,
`jito_tip_lam`, plus the spoof flags (`bonding_curve_writable`,
`cu_limit_too_low`, `probable_spoof`) and v2 disambiguation fields.

The bot doesn't read this ring yet — it's archival-only today.

The lead time is potentially valuable as raw material for the entry
model, the recovery model, exit policy overlays, slice timing,
risk-circuit-breaker logic — or some combination we haven't thought
of yet. The shape of how we'd use it is still open; no roadmap or
specific feature design here yet.

The intent JSONLs are on disk (intent_capture/*.jsonl.gz) and the
raw shred firehose (raw_shred_entries/*.bin.gz) is being archived to
the Hetzner storage box, so whatever we eventually decide to do, the
historical data is available to backtest against.

---

## TODO: revisit hardcoded exit-policy values (NOT empirically validated)

The following parameters in `config.yaml` and the policy classes were set as
round-number defaults and have NOT been properly empirically swept:

- `exit.derisk_min_gap_s` (currently 5.0)
- `exit.runner_min_gap_s` (currently 15.0)
- `exit.runner_retrace_frac` (currently 0.30)
- `exit.runner_min_arm_ret` (currently 0.20)
- `exit.death_threshold` (currently 0.10 — this one came from the trained
  recovery model so it's at least somewhat principled)
- `exit.total_slices` (currently 8)
- `exit.derisk_slices` (currently 4)
- entry threshold override (currently `--entry-threshold 0.30` in systemd
  unit, vs trained 0.5108)

These need a proper rigorous sweep with the right framing:
- Single-sweep on aggregate population is misleading (mean is dominated by
  loser tokens; optimal value differs by token outcome)
- The right frame is conditional: optimal exit timing depends on what kind
  of token it is (moonshot vs slow grind vs early rug)
- A single global gap is a bad compromise; routing to different gap values
  by an entry-time classifier is the cleaner architecture

Until a proper sweep is done with the right methodology (stratified by
outcome, or routed via classifier), treat these values as placeholders.

## 2026-06-11 — cost tuning + rent recovery + shred reality check

- **cu_limit 200k -> 160k.** Measured real buy CU via simulateTransaction across recent mints: worst case WITH Token-2022 ATA create ~99k (range 88-99k), without ~81k. 160k clears the worst case by ~60k (room for a fresh-launch's first-buy PDA inits). Cuts the priority fee ~20% (0.0004 -> 0.00032/tx). Priority RANKING is unchanged: it's set by the per-CU price (2M micro), not the limit. No revert risk. cfg.broker.cu_limit=160000.
- **ATA rent recovery.** Each new-token buy locks ~0.00207 SOL rent in the Token-2022 ATA; reclaimable by closeAccount (ix 9) once empty. Did NOT couple the close into the sell tx — a close failure on an exotic Token-2022 mint would revert the *critical* sell, re-introducing the sell-failure mode we just hardened. Instead: standalone tools/ata_sweep.py (enumerate balance==0 token accts, batch-close via sendTransaction), run MANUALLY. Validated: closed 2 empty ATAs, reclaimed +0.004108 SOL. A systemd timer (pumpfun-ata-sweep.timer) is installed but DISABLED — per MB, an autonomous process submitting live txs from the bot's wallet *during* trading is a moving part not worth the risk in real territory, and rent is recoverable anytime. The sweep is *probably* race-safe (the buy tx creates+fills the ATA atomically, so it's empty only post-sell; closed_mints prevents re-entry) but "probably" isn't worth it when running it by hand is free. Run on demand / when the bot is idle.
- **Total live-test cost now ~0.0032 SOL** (was 0.0073; reclaimed 0.0041 of locked rent). All of it is execution overhead (base fee + tip + priority fee + pump 1% + tiny slippage) plus the deliberate forced-sell-failure test. Zero strategy losses.
- **Shred reality check** (spoofing concern). 14,199 recent buy intents: probable_spoof 0.1%, cu_limit_too_low 0%. Sample of 60 shred buy-intent sigs checked on-chain: 85% landed-ok, 15% landed-but-reverted, 0% never-on-chain. Shreds are trustworthy — no phantom/bait txs.
- **Shred-trigger assessment: DEFERRED (not a config flip).** Bot --source is ws/grpc only; shreds are consumed only at fire-time for tip sizing (shred_window.signal + intent_features). Our edge = the K+V pattern on EXECUTED reserves (gRPC TradeEvents give vsol/vtok). Shreds = pre-execution intents with no reserves, so the existing model literally cannot run on them. Options: (A) shred-native front-run model = a NEW strategy needing its own OOS validation; (B) shred head-start on the gRPC trigger = keeps our edge but only ~12ms median (p90 107ms). Dominant latency is submit->land, not detect. Recommendation: if chasing same-block, the lever is the submission/landing path, not shred-detect. Shred-native is a future research direction (intent_features infra exists as a starting point).

## 2026-06-11 (cont.) — ATA close-after-sell + git hygiene/first GitHub backup

**Supersedes the "manual sweep only" ATA note in the entry above** — per MB we then DID add an automatic close.

- **bot-closes-after-its-own-sell (off the hot path).** `JitoBroker._close_ata` fires from the reconciler's `_log_actual_fill`, i.e. AFTER a sell signature is confirmed landed AND the post-sell token balance is 0 (full exit). Triple-guarded: (1) only reached on a landed sell, (2) only when post-bal == 0, (3) re-verifies balance == 0 on-chain before sending `closeAccount`. Runs as a fire-and-forget `create_task` so it never touches the fire/submit path; any failure is swallowed (rent stays reclaimable by `tools/ata_sweep.py`). Added `pump_fun_ix.build_close_account_ix` (CloseAccount = ix 9). Regression test `tests/test_actual_fill.py::test_close_fires_only_on_full_exit` asserts the close fires on a full exit but NOT on a buy or a partial sell. The manual `tools/ata_sweep.py` + the DISABLED `pumpfun-ata-sweep.timer` remain as a backstop. Validated: 46 tests pass; bot restarts clean (dry-run, hook dormant until live).

- **git hygiene + FIRST off-host backup.** Discovered `git add -A` had been sweeping live collector buffers into git — a **10GB** grpc-firehose `.bin` was committed into HEAD and `.git` had bloated to **15GB**. Fixes, none of which touched a collector file: gitignored + `git rm --cached` the buffers (`buffer/`, `shred_bot/intent_capture/`, `*.bin`) and all binary/model formats — repo is now **code-only (0 binary files tracked)**; stripped all >50MB blobs from history with `git-filter-repo` (**15G → 981M**, code byte-identical by md5, bot+collectors undisturbed, run at low IO/CPU prio so it couldn't starve the collectors; filter-repo also did reflog-expire + gc-prune). Force-pushed the blob-free history to `origin`; re-synced the local file mirror via `git archive`.

- **Shred-alpha question, ANSWERED (the harness existed and had already been run).** `tools/train_june_causal_sweep.py` IS exactly this test: a leakage-safe, chronologically-split (train/val/test) (K,V) entry-model sweep with an include-intent toggle (`--skip-intent`), trained WITH vs WITHOUT shred intent-features. It was run Jun 9 (`data/june_causal_sweep_20260609T161436Z/`: results_base.csv vs results_intent.csv, REPORT.md). **Verdict: shred intent-features do NOT add robust alpha over K+V.** Matched-setting test AUCs are comparable and mixed (~0.83 to 0.90 both ways; e.g. k3/v0.3/peak_ge_200: base AUC 0.895 on n=2940 vs intent 0.858 on n=1945), deltas are within noise, and the intent-overlap requirement only shrinks the usable sample. The sweep auto-selected an intent model as "best" only because its selection score leans on noisy ~20-sample validation net-returns (a "validation selection bug" is flagged in the artifact filenames), not on robust AUC. Read: K+V (executed reserves) already captures what the pre-execution shred intents would signal, so shreds add no incremental discrimination. Caveat: single day (Jun 9, ~11h, 35k candidate rows); a re-run on more/current capture would confirm. (This corrects the earlier "was NOT built" note in this entry, which was wrong.)

## 2026-06-11 (cont. 2) — pre-arm audit: fresh-mint meta-fetch regression, slippage split, no unattended swaps

- **Audit context.** MB ordered a full pre-arm audit, then fixes. Mirror==host verified (176 files md5-identical), 46 tests green, collectors healthy, journal clean. Wallet staging and the deliberate 0-SOL gate were confirmed using deployment-local, untracked configuration. NOT armed; JITO_DRY_RUN stays 1.

- **REGRESSION (GOTCHA-class): every organic buy failed assembly since the t22 rework went live (~18:49).** `_get_mint_meta` queried the mint + bonding-curve accounts at the solana-py client DEFAULT commitment = FINALIZED. A fresh mint's accounts are invisible there for ~13s, and the bot fires 1-3s after creation, so all 8 fires of the 18:49 era errored "'NoneType' object has no attribute 'owner'" (buy and the same-second sell together). Proof pair: GprwzJHB's buy failed at +0s while its sell assembled at +13s, exactly the finalization boundary. The Jun-11 live canary (35FwyPFD) could not catch this: it used an old, long-finalized mint. RULE: on the hot path, pass commitment EXPLICITLY; lookups that must see what the processed-level feed sees must themselves be processed. FIX: `_get_mint_meta` now Processed + one 250ms retry + 2s-per-attempt timeout; `_get_curve_reserves` and `_token_balance` also Processed. Probe on a 1.5s-old mint: the real code path returned Token-2022 meta in 233ms.

- **Slippage split shipped (the arming-prep item).** New cfg keys `slippage_bps_buy=20000` / `slippage_bps_sell=1500`; legacy shared `slippage_bps` stays as a fallback. Basis, exec_sim cap sweep re-run on the 157 usable fires: winner and loser fires have the SAME entry-slip distribution (p50 +0.85, p95 +1.11, max +1.34), so a binding entry cap rejects winners pro-rata, it cannot select. At the old 1500bps only 6/157 fires survive and total net is NEGATIVE (-0.84); >=13400bps keeps all 157 (+6.24). 20000bps keeps every observed fire with ~50% headroom and bounds worst-case spend at 3x bet (the cap is a SAFETY BOUND, not an EV optimizer; the +7.52 at 10000bps is noise given identical distributions). Note the split is STRUCTURALLY required: with the shared knob, anything >=10000bps drives `slippage_min_sol_output()` to <=0 and every sell refuses as bad_amm_state.

- **Assembly-failure rollback (audit find).** The synchronous holdings reservation leaked on buy assembly failures: bad_amm_state rolled back, but no_blockhash and the generic except did NOT, so every meta-fetch failure left phantom expected_tok that the later sell sold as air. Buys now roll back on all three no-submit paths. Sells: assembly failures (incl. bad_amm_state / no_blockhash) now route a synthetic PendingBundle into `_fail_or_retry`, so a failed sell_all ASSEMBLY gets the same chain-balance retry ladder as a failed LANDING instead of silently stranding the decremented holdings.

- **No unattended swaps (MB: "don't allow to change model").** `pumpfun-auto-retrain.timer` (ExecStart with --execute, would have swapped the model Sun 04:17) and `pumpfun-auto-policy.timer` disabled --now. auto-policy was a no-op under exit.mode=static but burned ~1h CPU per run; the standing rule is OFF.

- **"dur 0s" on the dashboard: real, not a bug.** positions.jsonl open->close deltas: farm fires close in 6-83 MILLISECONDS (the farm's own buy burst blows through TP50 within one bundle; the dashboard truncates to whole seconds). GprwzJHB 12.5s, the one stop fire 0.96s, stales 8m+ prove the column computes correctly. Live implication: on this cohort our buy lands AFTER the burst (the +84% slip mechanism) and the TP sell is dispatched before our buy even lands; the retry ladder covers the ordering race but exits will land seconds late vs paper. That ordering gap is a first-class paper-vs-real calibration metric once armed.

- Deployed: backups `*_pre_slipsplit_20260611T191535Z`; 46 tests green post-change; bot restarted 21:16 CEST on the new code (startup print: slippage_bps buy=20000 sell=1500). Verdict era cut unchanged (--since 1781047096).

## 2026-06-11 (cont. 3) — latency changeset: warm pools, fire-time meta prefetch + dedup, async submit

MB asked why the three measured decision-to-submit leaks were not fixed immediately; answer: reviewed-window discipline, now approved and landed. Measured before: cold per-mint meta fetch 500-600ms at fire time (fresh AsyncClient per call, two sequential RPCs), sync urllib POST blocking the whole event loop (10s worst case), per-call TLS handshakes everywhere. Against a 400ms slot, with the replay saying slots are where the edge lives.

- **Shared persistent RPC client** (`JitoBroker._rpc_client`): one AsyncClient reused by mint-meta, curve-reserves, and token-balance lookups. Cold fetch 285ms -> warm 90ms.
- **Parallel meta pair**: the mint-account and curve-account lookups are independent; `asyncio.gather` cuts warm fetch 90ms -> ~54ms.
- **Fire-time prefetch + in-flight dedup** (`prefetch_mint_meta` / `_spawn_meta_fetch`): the harness kicks the fetch off at FIRE time so it overlaps entry bookkeeping, and the same-second TP sell SHARES the buy's in-flight task (the EMqjH6Hd fire showed buy 606ms + sell 505ms racing duplicate fetches). DELIBERATELY NOT at K/V partial-trigger time: that would be ~34k RPC calls/day against the ~10M/month eRPC credit pool (same math that killed HTTP blockhash polling) for mints that mostly never fire; fire-time is ~100 calls/day.
- **Async pooled submit** (`jito_exec.send_transaction_b64_async` via persistent httpx): no event-loop stall during live POSTs, no per-call handshake (pooled request measured 6.7ms; warmup 114ms paid once at startup). Sync `send_transaction_b64` kept for tools. 5s total / 2s connect timeout instead of 10s.
- **Startup warmers in `create()`**: Jito TLS connection, shared RPC pool (`get_version`), tip-account cache (sync SDK call moved to a thread). First fire pays no one-time costs.

Net effect: decision-to-submit ~600ms+stall -> ~60ms (54ms meta overlapped with bookkeeping + ~1ms sign + ~7ms pooled POST), with the event loop free the whole time. Deferred deliberately: dual-submit (Jito + direct RPC) until armed calibration data (land-in-decision-slot rate) shows it is needed; CreateEvent-parsed creator cache (zero-RPC meta) as a future zero-credit option. Backups `*_pre_latency_20260611T193909Z`; 46 tests green; bot restarted 21:44 dry-run; verdict era cut unchanged.

## 2026-06-11 (cont. 4) — CreateEvent-seeded mint meta: fire-time RPC eliminated

The cont.-3 changeset still showed asm_ms ~470ms on the first post-deploy fire: httpx keep-alive is ~5s while fires are 20-30 min apart, so the warmed pool is COLD again by the next fire, and keeping it warm by pinging would burn the same eRPC credit pool that rules out trigger-time prefetch. Root fix: the feed itself announces every token birth.

- **pumpfun_create_parse.py** (new module, deliberately NOT in frozen pumpfun_parse.py): parses pump.fun CreateEvent (disc 1b72a94ddeeb6376; borsh name/symbol/uri then mint/bonding_curve/user/creator). Layout verified LIVE on samples: bonding_curve field == PDA derivation AND creator field == curve account offset 49 via RPC (3/3), token program identifiable in static+ALUT-loaded keys 12/12 (all Token-2022; creates routed via lookup tables carry it ONLY in loaded_readonly/writable_addresses, which is why static-only scanning saw 1/3).
- **listener_grpc_bot** seeds `JitoBroker.seed_mint_meta(mint, token_program, creator)` on every CreateEvent (~20-25/min; stats counter `creates_seeded`, 12 in the first 30s live). FIFO prune at 150k entries.
- Fire-time `_get_mint_meta` is now a dict hit for any mint born while the bot runs: ZERO RPC, ~0ms. The processed-commitment RPC fetch (with warm-pool ~54ms parallel pair) remains as fallback for mints created before process start.
- Decision-to-submit estimate now ~10ms (sign ~1ms + pooled POST ~7ms); the next organic fire's asm_ms is the in-situ confirmation.

## 2026-06-12 — blockhash goes push (blocks_meta stream); Jito keepalive

MB asked why we poll blockhash at 200ms when gRPC streams blocks, and whether HTTP to Jito is the fastest channel. He was right on the first: the 200ms unary poll was a leftover shape from the credit-pool migration (ERPC HTTP poll -> free gRPC unary poll). The latest blockhash only CHANGES once per block (~400ms), so polling at 200ms re-reads the same hash; the stream delivers each hash at birth.

- **blockhash_cache.py STREAM-FIRST**: subscribes blocks_meta (processed); each block's hash cached at production time with its real slot and last_valid_block_height (+150). Maximum remaining validity window, slot_gap logging now meaningful (bh_slot ~= event slot). Fallback chain: stream error -> rebuild channel -> 10s of legacy 200ms unary polling -> re-stream; HTTPS only if the unary also fails. NOTE: bh_age_ms in broker logs now measures true hash age (0-400ms typical), not poll recency; the 500ms freshness gate keeps its meaning (>500ms = skipped slots or stalled stream -> on-demand refresh).
- **jito_exec.jito_keepalive_loop** (60s GET, pool keepalive_expiry 90s): fires are 20-30min apart vs seconds-scale idle timeouts, so every fire's submit was paying a fresh TLS handshake despite the pool; now the connection stays warm permanently (~1.4k cheap requests/day to the public endpoint, no credit pool involved). First iteration doubles as the startup warm.
- **On "is HTTP to Jito the fastest":** the public block engine API is HTTP JSON-RPC; Jito's gRPC searcher API is authenticated and bundle-oriented (we deliberately moved OFF bundles: public 1-tx sendBundle goes Invalid and never lands, verified 2026-06-11), and shredstream is their inbound-data interface (already consumed by our shred services). At 0.2ms from the Frankfurt engine a pooled HTTP request measures ~2-7ms total; transport is not the bottleneck, engine->leader forwarding and scheduling dominate and are identical regardless of how the tx reaches the engine. The real next rung, if armed land-rate disappoints, is dual-submit via a staked-connection sender alongside Jito (deferred to calibration data).

## 2026-06-12 (cont.) — regional sendTransaction fan-out (docs re-read per MB)

MB corrected the searcher-API claim and pointed at docs.jito.wtf/lowlatencytxnsend. Verified from the page: auth keys are NO LONGER required for default sends; the documented send rails are the regional HTTPS JSON-RPC endpoints only (no gRPC endpoints documented for submission); sendTransaction is a direct proxy to the validator with skip_preflight=true and MEV protection; bundles batch through tip auctions (and our live finding stands: public 1-tx sendBundle goes Invalid, sendTransaction lands). KEY NEW FACT: default rate limit = 1 request/second/IP/REGION; our instant-TP fires submit buy+sell in the same second = guaranteed 429 on one region (observed in the Jun-11 canary: 429 at 16:27).

- jito_exec now keeps a per-region 1.05s budget across [frankfurt, amsterdam, london, dublin] (cfg.broker.jito_endpoints overrides); send_transaction_b64_async picks the nearest region with budget, pick-and-reserve atomic under asyncio, short-slice waits only when all 4 are exhausted in one second. Chosen region recorded as _region in jito_resp (per-region land-rate data for calibration falls out for free).
- keepalive loop warms ALL regions (4 GETs/min).
- Sanity: 4 same-second picks -> 4 distinct regions; 5th waits 1.05s. 46 tests green; restart clean.
- Tip note from docs: recommended 70/30 priority-fee/tip split for sendTransaction; ours is 0.00032/0.0001 SOL = 76/24, already in line.
- Latency budget after tonight (decision -> submitted): meta 0ms (CreateEvent-seeded), assemble+sign ~0.4ms, POST ~2-7ms on permanently-warm pooled TLS, blockhash slot_gap=1 (push). Remaining unowned terms: detect floor (gRPC node replay of the processed block) and engine->leader forwarding/scheduling = measured only when armed (landed_slot - ev_slot per region is THE calibration deliverable).

## 2026-06-12 (cont. 2) — entry broadcast to all regions; gRPC researched and ruled out for sends

MB pushed two things: (1) why does picking a region matter if Jito relays internally over a better backbone, and (2) is 1-tx-bundle really disallowed or did we mess up. Both researched properly.

**gRPC for sends — ruled out, from the proto source.** Pulled jito-labs mev-protos searcher.proto: SearcherService exposes SendBundle, SubscribeBundleResults, GetNextScheduledLeader, GetConnectedLeaders(Regioned), GetTipAccounts, GetRegions — there is NO SendTransaction over gRPC. Auth/whitelist IS removed (SDKs ship NewNoAuth). So gRPC = bundles only, and bundles run 50ms-tick auctions (+~25ms) which our 1-tx bundle loses (Invalid, verified live). gRPC transport would save ~1-3ms vs our warm pooled HTTPS (~7ms) but cost the ~25ms auction => WORSE for single-tx. The HTTPS sendTransaction direct-proxy (skip_preflight, no auction) is the fastest rail Jito sells for what we send. gRPC's real value is GetNextScheduledLeader / GetConnectedLeadersRegioned (no-auth) for leader-aware routing — a future lever gated on calibration.

**1-tx bundle: docs ambiguous, our live test authoritative, and it does not matter.** QuickNode says no stated minimum (max 5); another source says min 2; the canonical behavior we OBSERVED is a 1-tx sendBundle goes Invalid and never lands on the public endpoint. Mechanism: bundles must WIN the 50ms tip auction; a lone 1-tx bundle with a tiny tip just loses/expires. We are NOT messing up — sendTransaction is the correct + documented single-tx rail and it lands. (If we ever want revert-protection, sendTransaction?bundleOnly=true wraps a 1-tx bundle, trading the auction delay for it; not needed at our generous buy slippage.)

**Routing answer + the real fix — BROADCAST entries.** MB's "Jito forwards internally so region shouldn't matter" is the wrong half: GetConnectedLeadersRegioned exists and the documented HFT practice is "shotgun-blast the same signed tx to ALL regions in parallel" — proof that ONE regional engine does NOT reach all leaders; each forwards only to the leaders it peers with. We can't beat Jito's engine->leader hop (his right half) but we DO pick which engines ingest, and broadcasting hits whichever is best-peered to the current leader. Duplicates are idempotent (Solana dedupes by signature, lands once). Measured warm RTT from sol: frankfurt 6.9 / amsterdam 12.4 / london 18.1 / dublin 29.5 / ny 82.4 / slc 124 / singapore 156 / tokyo 226 ms; beyond ~85ms the ingest hop eats the N+1 window, so broadcast set = EU4 + NY.

- jito_exec.send_transaction_b64_broadcast: fires the same signed tx to all budgeted regions in parallel (atomic per-region reserve), returns first ACK (fastest engine ~7ms) annotated _region + _broadcast_n; raises only if all fail; falls back to single-region when all are rate-reserved in-second. _do_buy (ENTRY, latency-critical) now broadcasts; _do_sell stays single-region rotation (less critical, preserves budget). Per-region 1-rps respected (broadcast = 1 req each). keepalive warms all 5. Validated live with an invalid throwaway payload: all 5 regions reserved+fired in parallel, clean raise, same-second 2nd call fell back to single-region with the rate wait. 46 tests green; restart clean; still dry-run (broadcast only runs in the LIVE submit branch, so not exercised by dry-run fires — unit-verified instead).

## 2026-06-12 (cont. 3) — sells broadcast too; same-second buy+sell contention handled (MB caught it)

MB caught two flaws in the cont.-2 broadcast: (1) the buy broadcast reserves ALL region budgets for ~1s, starving a same-second sell; (2) the SELL has the SAME leader-reachability need as the buy (a regional engine forwards only to leaders it peers with; the sell targets a LATER slot = a different leader), so making the sell single-region was wrong — it can miss the leader's fast path exactly like a single-region buy. The latency-vs-reachability distinction I used to justify single-region sells was the error.

Fix:
- BOTH _do_buy and _do_sell now use send_transaction_b64_broadcast.
- broadcast rewritten: send to all regions FREE NOW; if none free, wait for the soonest to free (capped 2s, sleep-slices <=1.2s) then broadcast the freed regions — never silently drops to one region. Last-resort forces the soonest region only if the 2s cap is hit.
- Validated live (invalid throwaway payload): buy reserved all 5 regions; same-second sell waited ~0.87s for budget then broadcast the freed region(s), NOT a single-region drop.

Why the ~1s sell wait is acceptable without more work: under the documented 1-rps-PER-REGION limit, 5 regions = 5 sends/sec total, so a full buy-broadcast + full sell-broadcast cannot both fit in one second. But in LIVE the sell cannot fill until the buy CONFIRMS on-chain (we hold no tokens until then, ~slot N+1), which naturally spaces the legs; and the sell's target leader is a later slot anyway. The unconstrained fix (both legs always fully broadcast simultaneously) needs a Jito UUID/API key lifting the per-region limit, OR leader-aware routing (GetConnectedLeadersRegioned, no-auth gRPC) so buy and sell target DIFFERENT region-subsets and never contend — both are arming-prep items, not blockers. 46 tests green; restart clean; dry-run (broadcast only runs in the LIVE submit branch).

## 2026-06-12 (cont. 4) — RETRACTION: broadcast was built on a wrong premise; reverted to nearest-region + failover

MB pushed back with an empirical fact I had not reconciled: our 2026-06-11 live canary sent ONLY to Frankfurt and every buy+sell landed. If a regional engine only forwarded to leaders it peers with (the cont.-2/cont.-3 reachability claim), Frankfurt-only would have missed leaders and sometimes failed. It never did. VERIFIED (docs.jito.wtf + Anza/HFT sources): the sendTransaction proxy "forwards your transaction directly to the validator" using the NETWORK-WIDE leader schedule + current slot, so ANY region reaches the CURRENT leader. Region choice is a LATENCY tweak, NOT a reachability gate. My "single-region misses leaders" justification was WRONG.

Consequence: broadcast's reachability rationale is void, and on pure latency it barely helps when we are geographically fixed (the distance to a far leader is paid on either the us->engine leg or the engine->leader leg regardless of which engine we hit; the long hop happens somewhere). Meanwhile broadcast CREATED the buy/sell 1-rps contention MB flagged. Net negative.

REVERTED to nearest-region-with-failover (send_transaction_b64_async rewritten; send_transaction_b64_broadcast + _pick_region + _asyncio_sleep_min removed): try the nearest region with 1-rps budget (Frankfurt, 6.9ms), fail over to the next region ONLY on send error or rate-exhaustion. This (a) reaches every leader (proxy forwards globally), (b) keeps the lowest local hop, (c) gives redundancy vs a transient single-path drop, (d) dissolves the contention — a same-second buy+sell use frankfurt then amsterdam with no artificial wait. Order = measured warm RTT from sol: frankfurt 6.9 / amsterdam 12.4 / london 18.1 / dublin 29.5 / ny 82.4ms. Validated: invalid-payload cascades through all 5 then raises (failover proven); valid tx returns at frankfurt; same-second 2nd send skips rate-limited frankfurt -> amsterdam. keepalive still warms all 5 (failover targets stay warm). 46 tests green; restart clean; dry-run.

Arc for the record: nearest-region (cont.1, ~right) -> broadcast (cont.2/3, wrong premise) -> nearest+failover (here, correct). The genuine far-leader latency lever, IF calibration shows far-leader fills systematically late, is LEADER-AWARE routing (GetConnectedLeadersRegioned, no-auth gRPC): pre-send the tx to the region nearest the UPCOMING leader. Deferred to calibration data. [[feedback-modeling-rigor]]: an empirical observation (Frankfurt always landed) overturned an inferred mechanism; the deployed path was corrected, not rationalized.

## 2026-06-12 (cont. 5) — pre-arm final audit + slot_gap logging (before a 0.05 overnight calibration run)

MB proposing: arm 0.05 overnight to gather the missing EXECUTION data. Final audit before flipping:
- Mirror == sol for all 8 changed files (md5). 47 tests green.
- Effective runtime: bet 0.1 (unit; -> 0.05 to arm), dry_run TRUE, slippage buy 20000 / sell 1500, tip 100k + adaptive, prio 2M micro, cu 160k, exit level_tp_50_stop30_cap120 static, risk max_concurrent 16 / 6 per min / daily_loss -1.0 SOL / fail_rate 0.5.
- Wallets: the funded wallet and empty gate wallet are deployment-local and untracked (arming = change the local wallet configuration + JITO_DRY_RUN=0 + bet 0.05).
- Timers: auto-retrain + auto-policy DISABLED (confirmed); only archive-sync (copy-only) + drift-monitor (read-only) remain = safe unattended.
- Disk 335G free; shadow_run.jsonl 90M lifetime, ~50-100MB/night growth = negligible. RSS ~550MB/26min (overnight ~1-1.5GB, box 62GB, fine).
- Realized-P&L circuit wired: realized_pnl_lam accrues on sell fills (paired buy cost), realized_net_sol() gates the live daily-loss circuit (shadow_harness:530).

LOGGING COVERAGE (the "are we logging everything" check) — COMPLETE for the execution gap:
- shadow_run.jsonl: entry_decision, position_close (+policy_nets +live_policy_net), k/v_trigger, path_snap, front_run_tip_bump, live_runner_exit, shadow_death_cut, model_loaded, recovery, shutdown.
- broker_jito.jsonl: buy/sell LIVE_submitted (sig, slot, bh_age, asm_ms, tip, jito_resp._region) / error / no_holdings.
- broker_recon.jsonl: landed, fill (actual vs expected tok+sol, fee, tip), failed, sell_will_retry/sell_retry/skip, holdings_reconcile, ata_closed.
- GAP FOUND + FIXED: PendingBundle carried landed_slot but not the DECISION slot, so land-in-decision-slot (THE calibration metric) needed a fragile cross-file join. Added target_slot + bh_slot to PendingBundle (set at submit for buy+sell) and logged target_slot/bh_slot/slot_gap in BOTH the `landed` and `fill` recon records. slot_gap = landed_slot - target_slot (0=same slot, 1=next slot = the lat1 our economics assume). Used getattr (fill logging must never drop a record on a missing attr). test_actual_fill extended to assert slot_gap (buy same-slot=0, sell next-slot=1). 47 tests green.

FIRST-RUN RISK (flagged for the go/no-go): the new live submit path (async failover sender send_transaction_b64_async, create-seeded meta, push blockhash) has only been DRY-tested + unit-tested; the Jun-11 canary that actually landed used the OLD sync sender. So the recommendation is: arm, WATCH the first organic live fire's recon end-to-end (landed + fill + slot_gap) to validate the new submit path for real, THEN let it run unattended overnight. Do NOT walk away before the first fill confirms.

## 2026-06-12 (cont. 6) — gRPC-native wallet reconcile (eRPC getAccountInfo/getTransaction outage) + live canary PASS

eRPC re-checked thoroughly (MB gave the endpoint): SELECTIVE outage — getHealth/getSlot/getSignatureStatuses(96ms)/getTokenAccountBalance(95ms) UP; getAccountInfo + getTransaction HANG (8-15s, http 000). Public mainnet-beta serves the same getAccountInfo in 23ms. So the two account/transaction-data methods (backing fill logging + retry-ladder curve fetch) are down on eRPC; the rest work.

MB directive: "run exclusively grpc wallet info with fallback erpc; we have max stream connection on the grpc endpoint." BUILT (commit pending):
- listener_grpc_bot: fold OUR WALLET into the EXISTING listener subscription as a 2nd transaction filter (failed=True to catch our reverts) — NO new gRPC stream (respects max-stream limit). When "wallet" in resp.filters, b58 the sig + call broker.reconcile_grpc_tx. Only our own txs pay the cost (dict lookup otherwise). Stat grpc_wallet_recon.
- jito_broker.reconcile_grpc_tx(sig,slot,meta): PRIMARY confirmation+fill from the feed's tx meta (pre/post SOL+token balances, err, fee, slot) = landed/reverted + actual fill + slot_gap, ZERO HTTP. Mirrors _log_actual_fill parsing. Updates realized_pnl. Fires ata-close on full exit (_close_ata_shared via the working getTokenAccountBalance+getLatestBlockhash). Single-threaded asyncio so the pending_bundles.pop is atomic vs the poll reconciler (added a `sig in pending_bundles` race-guard there).
- Poll reconciler (getSignatureStatuses, works on eRPC) DEMOTED to expiry/timeout backstop + eRPC fallback; its getTransaction fill is now secondary (gRPC primary).
- Retry ladder resilient to getAccountInfo-down: _get_curve_reserves None -> force a MARKET sell (min_sol=1) using PendingBundle.vsol_submit/vtok_submit (stored at submit) instead of terminal abort; _do_sell guard relaxed so market sells proceed with stale/zero reserves. Bag-hold-on-RPC-outage risk closed.
- 4 new unit tests (test_grpc_reconcile.py: buy fill, sell fill+realized-pnl, reverted->fail+rollback, foreign-sig noop); 50 tests total green. Subscription accepted by endpoint (wallet-recon ON ... subscribed, 0 errors).

LIVE CANARY through the NEW code (`tools/tiny_live_roundtrip`, with wallet and endpoint supplied only through local environment): BUY and SELL both landed, actual buy output matched the quote, the full position exited, and the observed sell slip was ~6.6%. New submit + reconcile + fill-parse validated on real chain. (Canary fills came via the poll path on public RPC; the gRPC reconcile path is unit-tested + will first live-fire when the BOT is armed.)

Bot resilient to the eRPC outage now: assemble (create-seeded meta, no getAccountInfo) -> submit (Jito) -> confirm+fill (gRPC wallet feed, no getTransaction) -> retry (market fallback) -> daily-loss circuit (realized_net_sol from fills). Money-safety (stop/retry/position-cap) does NOT depend on the degraded eRPC methods. Still dry-run; NOT armed. The gRPC reconcile has never live-fired on a real BOT tx (only canary-via-poll + unit) -> watch the first live fire when armed.
