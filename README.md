# The Distribution Will Manifest

Research code and model-building workflows for an empirical study of Solana launch microstructure: what can be observed before execution, what can be decoded from the network, what remains predictable out of sample, and what survives actual landing, fees, slippage, and discontinuous price paths.

The accompanying paper is:

> Marcel Butucea, **“From Shreds to Fills: An Empirical Study of Solana Launch Microstructure, Bonding-Curve Graduation, and Execution Realizability.”** July 2026. DOI: [10.13140/RG.2.2.18552.20481](https://doi.org/10.13140/RG.2.2.18552.20481). [Read the paper on ResearchGate](https://www.researchgate.net/publication/409285433_COMPUTATIONAL_FINANCE_AND_NETWORK_MICROSTRUCTURE_From_Shreds_to_Fills_An_Empirical_Study_of_Solana_Launch_Microstructure_Bonding-Curve_Graduation_and_Execution_Realizability_A_month_of_full-firehose_o).

The paper itself is intentionally **not** stored in this repository. This repository contains code, schemas, tests, model specifications, and selected research notes; it excludes private credentials, wallet material, raw firehose archives, runtime state, and trained binary models.

## Research question and main result

The project began as a launch-trading system and became a measurement instrument. It followed the entire causal chain:

```text
shreds / gRPC events
        ↓
causal decoding and feature construction
        ↓
chronological model evaluation
        ↓
transaction assembly and submission
        ↓
landed transaction and fill reconciliation
        ↓
fees, slippage, exits, and realized outcome
```

The central result is methodological: statistical discrimination is not the same thing as executable economic value. Several model families produced useful chronological ranking, but that signal weakened—sometimes reversed—when decisions were anchored to attainable fills and charged for actual execution. Fast access alone was not enough to rescue a slot-level race. Failure cases such as leakage, cohort mismatch, stale schema assumptions, zero-filled serving features, and same-slot price discontinuities are therefore part of the result, not details to hide.

This is research software, not a profitability claim, trading recommendation, or turnkey production bot.

## What we did

1. **Captured the market at multiple layers.** The collectors preserved decoded Pump program events, transaction metadata, direct shred-stream observations, and pre-execution intent records.
2. **Reconstructed causal state.** Parsers and trackers built reserve paths, trade flow, participant concentration, acceleration, drawdown, and timing features using only information available at each decision time.
3. **Built live-matched datasets.** Extraction was repeatedly corrected so the training population, trigger, schema era, and live serving path matched the population the bot could actually trade.
4. **Validated chronologically.** Model comparisons used time-ordered or cross-day splits, explicit population checks, feature-parity tests, shuffled-identity controls, and failure-aware reporting rather than random-row validation alone.
5. **Separated prediction from execution.** Paper outcomes were re-anchored to confirmed fills and charged for priority fees, tips, Pump fees, slippage, failed submissions, and exit constraints.
6. **Iterated through three bot families.** Each generation moved the decision point later in the token lifecycle and exposed a different observability/execution boundary.

## The three bot generations

### 1. Launch bot

The first family observes new Pump launches on the bonding curve. Early versions used milestone triggers and compact reserve/path features (the `K` and `V` families); later variants added richer flow, signer-concentration, intent, and exit-policy features. The repository preserves:

- ingestion and parsing in `grpc_capture.py`, `listener_grpc_bot.py`, `pumpfun_parse.py`, and `shred_bot/`;
- live-matched feature construction in `feature_accum.py`, `rich_entry_features.py`, and `shadow_harness.py`;
- entry/model experiments under `tools/train_*`, `tools/extract_*`, and the archived specifications in `artifacts/model-specs/launch/`;
- execution assembly and reconciliation in `pump_fun_ix.py`, `jito_exec.py`, `jito_broker.py`, and `position_store.py`;
- exit-policy comparisons in `exit_policies/`, `paper_book.py`, and the replay tools.

The final launch work emphasized population parity, chronological ranking, Token-2022 compatibility, actual-fill anchoring, and the difference between a paper decision slot and a landed fill.

### 2. Bonding-curve continuation bot

The second family asks a later question: after a token has already crossed a launch milestone, can it continue far enough to reach a defined take-profit before a stop? It adds continuation-specific trackers, capacity-aware sizing, reputation experiments, raw-firehose and shred-intent augmentation, replay evaluators, deployment trainers, a dry-run/live harness, dashboard, and an independent loss watchdog.

The full generation now lives in `research/continuation/`; its final live research harness is `continuation_live_rep_bot.py`. Supporting execution corrections and fill-reconciliation tests are in `jito_broker.py`, `tests/test_actual_fill.py`, and `tests/test_grpc_reconcile.py`.

The reproducible build path is preserved alongside the harness: `entry_aug_extract.py` and `cont_2x_aug_extract.py` construct causal panels from decoded and raw firehose archives; `cont_2x_shred_rep.py` and `cont_2x_build_deploy.py` add shred-flow and as-of reputation features; and `cont_2x_train.py`, `cont_2x_deploy_train.py`, and `continuation_rep_train.py` fit and serialize the reported continuation model families.

### 3. Post-graduation bot

The third family moves to the PumpSwap phase after bonding-curve graduation. It listens for migration/graduation, tracks AMM reserve state, builds a 22-feature rich representation at the post-graduation trigger, and evaluates whether the move is both predictable and sellable across slot boundaries. A second “slow climber” gate explicitly targets paths whose apparent win does not collapse inside the same block.

The core files are:

- `research/graduation/graduation_listener.py` — graduation/migration observation;
- `research/graduation/grad_cont_extract.py` — post-graduation panel construction;
- `research/graduation/grad_deploy_train.py` — rich gradient-boosted deployment model;
- `research/graduation/grad_climber_train.py` and `grad_climber_filter.py` — catchability/slow-climber gate;
- `research/graduation/graduation_live_bot.py` — dry-run-by-default research harness with fill anchoring and recovery controls;
- `research/graduation/graduation_dashboard.py` and `grad_cont_*` — monitoring and failure-aware analyses.

The paper reports useful discrimination for post-graduation models, while showing that many attractive same-block outcomes were not realizable at the first attainable fill. That distinction is encoded directly in the later scripts.

## Repository map

| Area | Purpose |
|---|---|
| `pumpbot/` | Small package facade for ingestion, features, models, exits, execution, and harness commands |
| `shred_bot/` | Shred-stream ingestion, intent decoding, ring buffer, and schemas |
| `protos/`, `grpc_stubs/` | Geyser protocol definitions and generated Python stubs |
| `tools/` | Dataset extraction, training, replay, parity checks, drift diagnostics, and execution studies |
| `exit_policies/` | Counterfactual and active exit-policy implementations |
| `artifacts/model-specs/` | Archived text model specifications and training notes; binary estimators are deliberately excluded |
| `research/continuation/` | Second-generation continuation research, evaluators, dashboard, and bot harness |
| `research/graduation/` | Third-generation post-graduation research, trainers, analyses, and bot harness |
| `tests/` | Collector freeze, feature, execution, reconciliation, and package-surface checks |
| `docs/notes/` | Architecture notes, state snapshots, roadmap, and chronological laboratory notebook |
| `docs/schemas/` | Standalone capture/schema documentation |
| `docs/LIVE_SESSION_2026-06-12_CONCLUSION.md` | Fill-anchored live-session conclusion used in the evidence trail |

## Reproducing the code environment

Python 3.12 is the reference runtime.

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,dash]'
pytest
```

The checked-in tests exercise code and invariants that do not require the private archive. Full empirical reproduction additionally requires the original versioned capture shards and derived panels. Those datasets are large and are not distributed here.

Typical offline stages are:

```bash
# Build or inspect live-matched launch datasets (see each command's --help/source).
python tools/extract_live_matched.py
python tools/verify_entry_claims.py

# Continuation analyses.
python -m research.continuation.continuation_eval
python -m research.continuation.continuation_rich_eval
python -m research.continuation.cont_2x_aug_extract --help
python -m research.continuation.cont_2x_train --help

# Build a post-graduation panel and train the final rich model.
python -m research.graduation.grad_cont_extract --help
python -m research.graduation.grad_deploy_train /path/to/grad_cont_panel.jsonl
python -m research.graduation.grad_climber_train /path/to/grad_climber_panel.jsonl
```

Many research scripts retain the original machine layout (`/root/the-distribution-will-manifest`) to preserve provenance. Use the documented path, pass an input path where supported, or adapt the `ROOT` constant for a separate reproduction environment. Generated `.pkl`, Parquet, JSONL, capture, and log files are ignored by Git.

## Configuration and safety

Copy `.env.example` to `.env` only on a private machine. Keep all values local. In particular, never commit a wallet private key, RPC token, endpoint containing a query-string credential, or a provider allowlist.

Live-capable code is dry-run by default and requires multiple explicit gates. Those gates reduce accidents; they do not make the system safe or profitable. Prefer offline replay and paper execution. If you study the live path, use a disposable low-value wallet and independently verify every program ID, account layout, fee assumption, and transaction effect.

The public repository intentionally excludes:

- `.env` files, wallet/keypair material, credentials, and authenticated endpoints;
- raw gRPC and shred firehose captures;
- intent archives, participant-level runtime state, and broker/reconciliation logs;
- trained pickle/joblib/ONNX artifacts and local position journals;
- the paper PDF (use the ResearchGate/DOI links above).

## References

The full bibliography and the internal evidence/checksum register are in the paper. The implementation also points directly to the primary technical references it relied on:

- [Solana whitepaper](https://solana.com/solana-whitepaper.pdf)
- [Solana fee structure](https://solana.com/docs/core/fees/fee-structure)
- [Solana confirmation and expiration guidance](https://solana.com/uk/developers/guides/advanced/confirmation)
- [Solana slot lifecycle / skipped-slot guidance](https://solana.com/docs/rpc/websocket/slotsupdatessubscribe)
- [Jito low-latency transaction send](https://docs.jito.wtf/lowlatencytxnsend/)
- [Pump public documentation](https://github.com/pump-fun/pump-public-docs)
- Bailey, Borwein, López de Prado, and Zhu, [“The Probability of Backtest Overfitting”](https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting)
- White, [“A Reality Check for Data Snooping”](https://doi.org/10.1111/1468-0262.00152)
- Hansen, [“A Test for Superior Predictive Ability”](https://doi.org/10.1198/073500105000000063)
- Politis and Romano, [“The Stationary Bootstrap”](https://doi.org/10.1080/01621459.1994.10476870)

## Citation

If this repository informs academic work, cite the paper using the metadata on the [ResearchGate publication page](https://www.researchgate.net/publication/409285433_COMPUTATIONAL_FINANCE_AND_NETWORK_MICROSTRUCTURE_From_Shreds_to_Fills_An_Empirical_Study_of_Solana_Launch_Microstructure_Bonding-Curve_Graduation_and_Execution_Realizability_A_month_of_full-firehose_o) or DOI `10.13140/RG.2.2.18552.20481`.

## License

The code is released under the [MIT License](LICENSE). The linked paper and third-party datasets, protocols, services, and trademarks remain subject to their respective terms.
