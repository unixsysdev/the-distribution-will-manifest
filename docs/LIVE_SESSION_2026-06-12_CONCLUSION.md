# Live Session Conclusion — 2026-06-12 (k3v03_final, 0.05 SOL bet)

## TL;DR
We armed the bot live for the first time, traded ~9 hours, and lost **−0.4003 SOL
(−3.4% of the session's initial bankroll)**. We then **paused live (flipped to dry-run)**.
The session was net-negative, but it delivered the thing offline sim could not:
**the truth that the strategy, as built, has no tradeable live edge after real
fills** — and *why*. The losses were the tuition for that finding.

Three fixes shipped during the session turned catastrophic losses into bounded
ones and built a genuine win side, but they **cannot make the strategy +EV**,
because the remaining loss is structural and lives in the **entry**, not the exit.

## Timeline
- **~01:54 UTC** first live buy (H7M1L), armed at 0.05 SOL/bet, 5M tip era.
- Through the day: root-caused losses layer by layer, shipped fixes live.
- **20:57 UTC** paused live → dry-run (`JITO_DRY_RUN=0→1`). Final **−0.4003 SOL**.

## What we fixed (and what each proved)
1. **4× swarm skip** (`skip_fire_shred_4x`) — don't fire when ≥4 buys/500ms or
   ≥8 buys/2s. The 4× tip-tier was **0-for-6 live, −0.258 SOL** (the bulk of the
   early loss): we were filling late at high slip into competing-bot swarms and
   eating the dump. *Result: removed the worst cohort.*
2. **Exit slippage 1500 → 9000 bps** — the −30% stop was firing with only a 15%
   sell-slippage floor, so the first sell **reverted into the dump** and we chased
   it down a 3000/5000/8000 ladder (14 reverts logged; GX3QyyPL realized −84%).
   *Result: first stop-sell lands instead of chasing.*
3. **Exit tip 100k → 1.5M (sells only)** — a slow sell (slot_gap 4) during a
   congested dump (BEjwYub4) let the price crater 0.065→0.0168, doubling the loss.
   Buy speed only wins slot *inclusion* (H7M1L landed slot_gap 0 with a 5M tip and
   still filled +116% behind in-block whales), but **sell speed saves money on a
   falling price**. *Result: sells now land slot_gap 0–1.*

## Root causes found (in order of discovery)
- **The mirage / late fill.** The offline sim anchors at the first post-decision
  slot (~+36% slip). Our **real fills land 1–3 slots later at +90–130%**. The
  entire offline edge (median +70% run) happens *between* those two prices —
  before we fill. We arrive near the top.
- **Swarm buying.** High-shred (4×) launches are competing-bot farms; we fill last
  and eat the reversal. (Fixed by the skip.)
- **Slow / reverting exits.** (Fixed by slippage + tip.)
- **One-block dumps.** ~22% of quiet fills collapse −75/−85% in a single block.
  **No exit can save these** — confirmed: C9p9Xz5 sold slot_gap 0 with the 1.5M
  tip and still realized −0.085 (quoted 0.046, filled 0.024).

## The decisive finding (entry feature analysis, tools/entry_split.py)
**Decision-time features do not separate dumpers from runners.** Feature-identical
entries produced both big dumps and wins:

| mint | net | score | vsK | win_ret | shred 500/2k | outcome |
|------|-----|-------|-----|---------|--------------|---------|
| BEjwYub4 | −0.096 | 0.522 | 51.6 | 1.95 | 1/1 | DUMP |
| C9p9Xz5  | −0.085 | 0.522 | 51.6 | 1.95 | 1/1 | DUMP |
| CQMwk88  | +0.034 | 0.522 | 51.6 | 1.95 | 1/1 | win  |
| 7ACAG    | +0.034 | 0.522 | 52.2 | 2.02 | 1/1 | win  |

Across all live fills, **nothing** — score, vsK, win_ret, cum_buy, shred counts,
reserves — predicts which way a fill resolves. The same fingerprint dumps, goes
flat, and wins.

## Conclusion
- The **offline edge was the mirage anchor**; the live realizable outcomes are
  coin-flips on identical features, and the ~22% one-block-dump tail (≈ −0.09)
  makes that flip **−EV** (wins are time-cap harvests of only ~+0.02–0.03).
- **Exit tuning is exhausted** — bounded the losses, can't remove the dump tail.
- **No entry filter on current K+V features can avoid the dumps** — they're
  invisible to the model at decision time.
- **The wall is the entry.** Only two real paths forward, both research-grade:
  1. **New separating signals** — first-second post-launch order-flow (bundled
     snipers vs organic, holder concentration, dev-wallet history, net flow /
     velocity in the opening second). Requires proper OOS validation.
  2. **Fundamentally earlier/cheaper fills** — hard infra fight; H7M1L shows even
     slot-0 fills sit behind in-block whales.

## Final state
- Live **paused** (dry-run); bot still evaluates + logs for data. **−0.4003 SOL**.
- Model + exit policy **unchanged** throughout (per directive).
- Pre-existing wallet bags (J5ij, Bktj) are **not** bot positions.
- Re-arm: set `JITO_DRY_RUN=1→0` in the systemd unit, `daemon-reload`, restart.

## Next step
Offline feature research on **opening-second order-flow** as a dumper/runner
separator, validated OOS on the 157-fire forward dataset before any live re-arm.

---

## Entry research — Study 1: opening-window order-flow (tools/orderflow_study.py)
Tested whether richer microstructure in the first 1.5s (buy/sell mix, sell
pressure, event velocity, price move, net SOL flow, competing-tip fraction)
separates the eventual fill-anchored outcome. 149 catchable fires, 48 dumps / 100 runs.

| feature | AUC (run vs dump) | verdict |
|---|---|---|
| n_events / price_move / net_sol / evt_rate | 0.43–0.48 | no signal |
| sell_frac (early sell pressure) | 0.35 | weak: dumpers sell more early |
| tip_frac (competing tips) | 0.28 | weak: dumpers more contested |

**Result: largely null.** The only off-0.5 features are weak (AUC ~0.28–0.35) and
point the intuitive way (early selling + contention → dump). `tip_frac` is the same
axis as the existing 4× skip. The features that should carry signal (velocity,
price move, net flow) are flat. The opening order-flow does NOT cleanly separate
dumpers from runners — reinforcing the live finding that the rug decision is made
later and isn't telegraphed in the opening second.

**Next escalation (bigger, different data):** holder concentration at launch,
dev-wallet reputation/history, and bundle/sniper detection. These require
assembling data beyond the reserves forward-path and proper OOS validation before
any live re-arm.

---

## Entry research — Study 2: do ANY existing pre-decision features separate? (tools/feature_auc.py)
Tested all 26 matched-dataset features (incl. the concentration/holder ones we
hoped would help) against the realizable fill-anchored outcome. 149 fires, 48 dumps / 100 runs.

**Two findings:**
1. **~10 of the 22 model features are CONSTANT in the fired regime** — `uniq`=3,
   `buy_frac`=1.0, `dir_eff`=1.0, `trades_per_sec`=saturated, `win_drawdown`=0
   (these ARE the trigger conditions). They carry zero discriminative info among
   fires. The model nominally has 22 features but effectively ranks on ~6.
2. **`single_actor_share` (concentration) is degenerate at the trigger** — range
   only 0.496–0.594 (≈always 0.5), AUC 0.57 → no separation. The holder-concentration
   signal we hoped to mine is flat where it matters.
   - The only features that vary AND weakly separate: `win_ret` / `net_sol` /
     `entry_sol` (AUC ≈0.34, |sep|≈0.16) — "more extended / higher volume → more
     dump," weak, and they did NOT separate the live dumps from runs (identical win_ret).
   - (`peak_ret`/`n_fwd` AUC ≈0.95 are outcome leakage, not usable.)

**Verdict: the existing feature set — including concentration/holder/flow — does not
contain a strong dumper/runner separator.** Combined with Study 1, order-flow and
concentration are both ruled out.

## Entry research — the one untested signal: DEV-WALLET REPUTATION
Creator's prior-launch / rug history is the only candidate not in any dataset.
**The creator pubkey was parsed by the bot but never logged**, and backfilling it
for 149 historical mints (paging each to its oldest tx through a rate-limited RPC)
is prohibitive. So: **added creator logging** (`listener_grpc_bot.py` →
`bot_data/creators.jsonl`: mint, creator, token_program, ts) — now accumulating in
dry-run. Prospective plan: as creators repeat (serial deployers become visible),
join fires→creator→prior-outcomes and test whether dev reputation separates the
dump tail. This needs accumulation time + a (cached, batched) creator-history fetch.

---

## Entry research — Studies 3–5: raw-capture mining (MB's suggestion)
We DO have the raw per-tx data: a 2.1 GB gRPC capture + a long history of intent
captures (per-wallet buy intents: user, mint, size, tip, slot). Mined it for the
signals the aggregated dataset discarded. Tested on the live fills with known real
outcomes (6 dumps / 5 wins / flats).

**Study 3 — buyer breadth (total intents):** STRONG separation but **hindsight-only.**
Winners mean 2467 total intents vs dumpers 127 (~19×; 3Svcun 6626, 7ACAG 2882 vs
dumpers 26–323). BUT we fire at the leading edge (`intent_smin == decision_slot`,
zero pre-decision intents) — the breadth accumulates over minutes *after* we commit.

**Study 4 — early-window demand (first 3 / 10 slots after decision):** NULL.
Dump mean n@+3=18, n@+10=27 vs win n@+3=15, n@+10=21 — dumpers are if anything
slightly busier early. The breadth signal is NOT present in the actionable window;
the dump happens (tens of slots) faster than the demand signal forms (thousands of slots).

**Study 5 — wallet / bot-cluster reputation of early buyers:** NULL.
Built a wallet→#tokens ubiquity map (47,705 wallets). Dumpers' early buyers
mean-ubiquity 115.7, high-bot-frac 0.44 vs winners' 105.6 / 0.48. **The same farm
bots buy the dumps and the runs** (one super-bot, ubiquity 1456, appears in nearly
all). Early-buyer identity carries no dump signal.

## Decisive meta-conclusion
**Five distinct signal families — order-flow microstructure, concentration/holders,
buyer breadth, early-demand rate, and buyer/bot-cluster identity — are all NULL in
the actionable (pre-decision / early-post-fill) window.** Feature-identical,
flow-identical, buyer-identical launches produce both −0.09 dumps and +0.03 wins.
The rug is the operator's later choice and is **not telegraphed in any market /
flow / buyer data we can observe early.** This is a strong, multiply-confirmed
negative result: the strategy is not made viable by any entry/early signal in the
data we have.

## The sole remaining hypothesis: CREATOR (dev) reputation
The one untested signal is operator-side, not flow-side: the *creator's own*
prior-launch rug history (intent-to-rug lives in the dev's history, not the token's
flow). It can't be cleanly backfilled (creator absent from intent captures; firehose
covers only the last hour, no creator key, needs raw-ix parsing; n=few). **Wired up
prospectively instead:** `creators.jsonl` now logs mint→creator for every launch
(dry-run, zero risk). Plan: accumulate dry-run fires + creators + path-outcomes over
hours, then test whether serial-rugger creators separate the dump tail. If creator
reputation is ALSO null, the strategy is not viable with any available signal and
should stay shelved.

---

## Entry research — Study 6: dev (creator) reputation — POWERED (tools/dev_rep_offline.py)
Built a creator registry from the 105 GB gRPC capture archive (CreateEvents, Jun 9–12,
120,276 launches / 24,412 creators; biggest farm = 10,017 launches). Joined to the
149 offline fires (50 matched; the rest launched before CreateEvent logging began Jun 9).

**Result: NULL.** Our fires' creators almost all launched exactly 1 token (run mean
1.2 vs dump mean 1.0, both ≈1; AUC 0.135 is degenerate from near-constant values).
The mega-farms exist but we never fire on them — we fire on one-off-creator launches,
and among those dump-vs-run is unpredictable. Same as the within-day live result.

## FINAL VERDICT
**Seven signal families tested, all null in the actionable window:** order-flow
microstructure, concentration/holders, buyer breadth (hindsight-only), early-demand
rate, buyer/bot-cluster identity, within-day dev reputation, and powered (n=149)
creator launch-frequency. Flow-identical, buyer-identical, creator-identical launches
produce both −0.09 dumps and +0.03 wins. **The pump.fun rug is the operator's later
choice and is not predictable from any signal in the data we have.**

The live experiment did its job: it proved the offline edge was a fill mirage and
that the realizable outcome is an unforecastable coin-flip with a −EV dump tail.
The three exit fixes (4× skip, exit-slip 9000, exit-tip 1.5M) bounded the losses but
cannot create an edge. **Recommendation: keep the strategy shelved.** Re-arming would
require genuinely new data we don't have (archival holder-graph at launch, long-horizon
creator histories via RPC, or fundamentally faster/cheaper fills) — all high-effort,
uncertain, and now against a strongly negative prior (7/7 null).

Final realized: **−0.4003 SOL** over the live session. Bot remains in dry-run.

---

## THE PIVOT (2026-06-13, MB): continuation, not launch — VALIDATED
Reframe: don't predict the launch rug (7/7 null, unforecastable). Instead ENTER once a
token clears a milestone (+M% from launch, survived) and predict whether it reaches the
NEXT milestone (+K%) before failing (-30%). This conditions on a survivor population and
makes the demand/momentum history (hindsight at launch) an OBSERVABLE input.

**Validation (tools/milestone_study.py, exec_sim_fwd n=24835):** unlike launch-entry
(every feature AUC ~0.50), milestone-entry features separate continuers from failures:

| A→B | reach A | P(B) | top features (AUC) |
|---|---|---|---|
| +50→+100 | 16% | 17% | pump_vel 0.60, buy_frac 0.58 |
| +100→+100 | 9% | 14% | drawdown 0.65, pump_vel 0.62, buy_frac 0.61, n_evt 0.34 |
| +100→+200 | 9% | 3% | buy_frac 0.65, drawdown 0.64, n_evt 0.29 |
| +200→+100 | 3% | 10% | buy_frac 0.70, drawdown 0.63, n_evt 0.28 |

Coherent signature: a FAST, EFFICIENT (few trades), CLEAN (low drawdown), BUY-DOMINATED
ascent continues; a churny/dippy/sell-heavy one fails. Real momentum-quality alpha.

**Build path (rigorous, OOS-first — this is the lesson from the launch failure):**
1. Pick milestone + target (start: enter +100%, target +100% more, stop -30%).
2. Build features-at-A on the full continuation dataset (path_snapshots, tens of thousands):
   velocity, drawdown, buy_frac, trade-efficiency, + intent-breadth-to-A (the 19× signal,
   now observable), time-to-A, reserves/vSOL at A.
3. Train P(continue | features-at-A); validate **OOS by time** (no leakage).
4. **EXECUTION REALITY CHECK (what killed the launch strategy):** measure the real fill
   slip when entering AT a milestone (not racing the launch swarm — may be far cleaner),
   and the continuation exit. Re-run the fill-anchored P&L on REAL fills, not the mirage.
5. Backtest model+execution OOS; only then consider dry-run → live.

---

## Continuation pivot — POWERED on Alchemy data (369k launches / 28M trades, Apr29-May4)
tools_local/milestone_alchemy.py. Milestone +100% -> target +100% (stop -30%).
68,905 mints reached +100%; P(continue to +100% more) = 23%.

| feature | AUC | note |
|---|---|---|
| ddown (clean ascent) | **0.630** | the one moderate separator |
| buy_frac | 0.564 | weak |
| pump_vel | 0.561 | weak |
| slots_to_A / ntr | 0.443 / 0.452 | weak (faster/fewer-trades) |
| uniq (breadth) / top1 (conc) / vol / cre_n (dev-rep) | ~0.49 / .49 / .51 / .44 | **NULL** |

**Honest correction to the path-only floor:** the edge is REAL but MODEST and narrow —
drawdown-led (clean/fast/efficient/buy-heavy ascent continues), the rich per-wallet
features (breadth, concentration, dev-rep) are NULL for predicting continuation.
A combined model ~0.65-0.68 (features correlated). NOT the broad 0.6-0.7 first implied.

**Economics:** base 23% continuation at +100%->+100% is ~BREAKEVEN
(0.23*1.0 - 0.77*0.30 ~= 0). Viability requires BOTH (unproven): (1) a model lifting
hit-rate to ~30%+ (at 33% -> +0.13/trade gross), validated OOS on June data; (2) clean
milestone-entry execution (re-run fill-anchored P&L on real entries -- the slip that
killed the launch edge applies to buying an actively-doubling token). First genuinely
real lead in the effort, but thin; not a sure thing, not dead.

## Concrete next step
1. Train a continuation classifier (drawdown + buy_frac + pump_vel + ntr + slots) on
   Apr-May Alchemy; validate OOS on June capture data -> does it lift hit-rate >30%?
2. Execution reality: simulate milestone entries on real fills (the slip at +100%).
3. Only if both clear -> dry-run -> live.

---

## Continuation — TRAJECTORY-WIDE sweep (MB: don't hardcode +100%->+100%)
tools_local/panel_continuation.py: entry at first crossing of each stage (1.3-8x),
target +50% before -30%, across the Alchemy 28M trades.

| stage | n | P(+50%) | ddown AUC | buy_frac | recent | vsol |
|---|---|---|---|---|---|---|
| 1.3x | 148k | 34.6% | 0.447 | 0.566 | 0.531 | 0.503 |
| 1.5x | 90k | 34.0% | 0.492 | 0.569 | 0.514 | 0.492 |
| 2.0x | 66k | 36.9% | 0.551 | 0.547 | 0.501 | 0.457 |
| 3.0x | 37k | 37.4% | 0.571 | 0.543 | 0.505 | 0.453 |
| 5.0x | 19k | 37.0% | 0.578 | 0.538 | 0.486 | 0.458 |
| 8.0x | 10k | 29.9% | 0.541 | 0.514 | 0.554 | 0.520 |

**Finding: the edge is thin EVERYWHERE, no hidden strong regime.** Drawdown
(clean-ascent) is the best feature and strengthens through the 2-5x consolidation
zone (0.55->0.58); buy_frac weak (~0.55) and fades; age/vsol/recent ~null. Nothing
breaks 0.6 trajectory-wide.

**Economics at breakeven:** +50%/-30% needs 37.5% hit; base is 34-37% (just under).
+100%/-30% was 23% vs 23.1% breakeven. So base rates are ~breakeven-to-slightly-neg;
viability requires a model to push hit-rate a few pts higher OOS. MARGINAL, not fat.

**Refined levers:** (a) payoff geometry (target/stop ratio) matters as much as features;
(b) do stops even HOLD on these tokens? (launch dumps gapped -30%->-75%). Next: build
the classifier (drawdown+buy_frac, best in 2-5x zone), sweep target/stop, OOS-validate
the hit-rate lift on June data, and execution/stop-integrity check before any dry-run.

---

## Continuation — CLASSIFIER OOS + ROBUSTNESS (the first robust edge)
Entry at 2x/3x crossing, target +50% before -30%, Alchemy 28M trades, time-split OOS
(train earlier half, test later half). tools_local/classifier_continuation.py + continuation_robustness.py.

OOS AUC 0.534 (weak overall; edge in the top tail). base hit 37.3%, breakeven 37.5%.
Model top-tier OOS hit-rates: top25% 42.7%, top10% 51.8%, top5% 61.4%.

ROBUSTNESS (both passed):
- STOP-GAP REALITY: 60,638 stop-outs, -30% stop actually fills mean -37% / median -34%
  (p10 -48%, 8% worse than -50%). MODEST gap -- nothing like launch's -75%; 2-3x
  survivors are liquid/less-explosive so the stop largely holds.
- DROP-AGE: removing the dominant age feature changes nothing (AUC 0.534, top tiers
  identical) -> NOT an age artifact; momentum-quality features carry it redundantly.
- EV at the REAL -37% gap stop: top10% +0.080, top5% +0.163 (top25% ~breakeven).

VERDICT: first OOS-validated, age-artifact-free, stop-gap-surviving +EV signal in the
whole project. Real but thin (lives in the top 5-10% of 2-3x movers).

REMAINING GATES before live: (1) JUNE cross-regime OOS (all above is within Apr-May;
apply Apr-May model to June host captures); (2) ENTRY-SLIP reality (slip from entering
AT the 2-3x milestone -- the launch-killer analog; the -37% stop-gap vs -75% at launch
suggests these are far more tractable, but must measure).

---

## Continuation — ENTRY-SLIP GATE (the launch-killer) → MIRAGE, edge dies here
tools_local/entry_slip.py. Fill ~2 slots after the 2-3x cross (live latency), outcome
measured FROM THE REAL FILL, EV net of entry slip.

Entry slip: median 0.4%, but mean 4.2%, p75 13%, p90 29.5%. FILL-ANCHORED OOS AUC
collapses 0.534 -> 0.509 (~random). Top tiers net of slip: top25% EV -0.073, top10%
-0.121, top5% -0.195. The MODEL'S TOP PICKS carry the HIGHEST slip (top5% slip 23.5%)
because momentum (what predicts continuation) is exactly what prevents a clean fill.

**The signal and the execution cost are the SAME coin.** Tokens that continue move
fast -> we fill late at high slip -> the +EV becomes -EV. Median token is cheap to
enter (0.4%) but those are the ones that DON'T continue. Anchored to the real fill,
predictability vanishes.

## FINAL VERDICT (whole project)
- LAUNCH: not predictable (7/7 signal families null, AUC ~0.50).
- CONTINUATION: weakly predictable (momentum-quality, OOS top-tier +EV optimistically,
  survives stop-gap & age-drop) BUT NOT REALIZABLE -- entry slip on the momentum picks
  (the launch-killer) collapses fill-anchored AUC to 0.509 and turns top-tier EV negative.
Both die on the SAME structural fact: on pump.fun, the predictable signal is momentum,
and momentum is exactly what you cannot fill into cleanly with our execution. Not a
tuning problem -- a structural one. No robustly +EV strategy exists here for us.
RECOMMENDATION: shelved, definitively. Bot stays dry-run. Final realized -0.4003 SOL.
