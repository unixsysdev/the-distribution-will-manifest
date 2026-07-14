# pumpbot architecture and migration plan

Status: stages 1 + 2a applied 2026-06-10 (tooling + tests + additive package
surface). No file moves yet; the collector frozen set is guarded by tests.
The tree still runs as flat modules; this document is the contract for
turning it into a proper package without breaking a live trading system.

## What runs in production (today)

```
systemd units                      entry point                 reads/writes
  pumpfun-bot.service              pumpfun_bot.py              bot_data/, logs/broker_jito.jsonl
  pumpfun-grpc-capture.service     grpc_capture.py             grpc_capture/
  pumpfun-grpc-firehose.service    grpc_firehose.py            buffer/
  pumpfun-shred-firehose.service   shred_bot/raw_shred_firehose.py
  pumpfun-shred-intents.service    shred_bot/intent_recorder.py  shred_bot/intent_capture/, SHM ring
  pumpfun-storagebox-shipper.service shred_bot/storagebox_shipper.py
  (dormant: auto-retrain, auto-policy, drift-monitor — keep OFF; the first two swap models)
```

Core dataflow:

```
gRPC processed feed ──> pumpfun_bot/shadow_harness ──> feature_accum.TokenState (K=3 AND V=0.3 joint trigger)
shredstream ──> intent_recorder ──> SHM ring (intent_ring) ──> shred_window (drain_now at decision)
                                                 │
   ModelServer(bot_artifacts_K7V symlink) <── 22-feat entry score, thr from model_spec.json
                                                 │ fire
   exit_policies registry (config.yaml exit.policy) + stale watchdog (300s) + PaperBook
                                                 │
   JitoBroker (DRY_RUN): blockhash cache, bundle assemble+sign, adaptive tip, recon
```

## Target package layout (stage 2+)

```
pumpbot/
  ingest/        grpc_capture, grpc_firehose, pumpfun_parse, listener_grpc_bot
  features/      feature_accum, rich_entry_features
  models/        model_serve, training (extract_live_matched, train_crossday/era, sweeps)
  execution/     jito_broker, jito_exec, blockhash_cache, pump_fun_ix
  shred/         intent_ring, intent_recorder, intent_extractor, shred_window
  harness/       shadow_harness, paper_book, position_store, exit_policies/
  ops/           dashboard, live_bucket_diag, shred_coverage_probe, pumpfun_ctl
  cli.py         single `pumpbot` entry: bot | capture | recorder | dashboard | diag | extract | train
tests/
artifacts/       bot_artifacts_* (unchanged; symlink contract stays)
```

## Migration stages (each independently shippable, bot keeps running)

0. COLLECTOR FROZEN SET (standing rule). The five collector services have
   Restart=always; a refactor that breaks their import closure turns the next
   incidental crash into a capture-losing crashloop. Their closure (15 files,
   tests/collector_frozen_manifest.txt) is FROZEN: tests/test_collector_freeze.py
   fails any commit that changes it. Changes to the manifest happen only in a
   planned migration window with a restart drill per collector.
   tools/collector_health.py (`python -m pumpbot health`) is the read-only
   health check: service state, output freshness + growth, ring write_seq
   advance, disk, storagebox mount.

1. DONE — tooling + tests. pyproject.toml (deps pinned where load-bearing:
   sklearn==1.8.0 matches the pickles), pytest suite locking in the bug
   classes we actually hit (rich-path misdetection, ring roundtrip, exit
   registry, presence flags, era accounting), ruff with correctness-only
   rules. No imports change.
2a. DONE — additive package surface. pumpbot/ with PEP 562 LAZY re-export
   submodules (features/models/execution/shred/harness/exits/ingest) over the
   flat tree: zero import-time side effects, zero file moves, zero service
   impact. `python -m pumpbot <bot|dashboard|diag|health|probe-shreds|extract>`
   dispatches to the existing scripts via runpy (byte-identical behavior).
   Collector entry points are deliberately NOT exposed in the CLI (a second
   hand-started instance would fight the daemon: duplicate ring writer,
   double capture files).

2b. Package move with import shims. `git mv` modules into pumpbot/, leave
   one-line legacy shims at the old paths (`from pumpbot.features.feature_accum
   import *`), update systemd ExecStart to `python -m pumpbot.cli bot ...`,
   restart services one at a time, delete shims after one clean week.
   Risk note: sklearn pickles store only sklearn class paths (safe to move
   our modules), but any pickle that references project classes must be
   re-checked before its loader moves.
3. CLI consolidation. argparse subcommands in pumpbot/cli.py replacing the
   per-script __main__ blocks; pumpfun_ctl.sh shrinks to systemctl wrappers.
4. Typed config. pydantic Settings replacing ad-hoc config.yaml access
   (bot_config._C), env-var overrides documented in one place, the
   K_TRIGGER/V_TRIGGER env contract made explicit and validated against
   the loaded model_spec (refuse to start on mismatch — this exact
   mismatch class caused the TP50/K=5 catastrophe).
5. CI. pytest + ruff on every push (repo already has git); a deploy
   checklist script (alignment check, fire-rate band, parity test) as a
   gate before any symlink flip.

## Invariants the structure must never break

- The bot reads the model through the `bot_artifacts_K7V` symlink; deploys
  are symlink flips with timestamped backup links. Never edit in place.
- Trigger semantics live in env (K_TRIGGER/V_TRIGGER) + feature_accum;
  training extraction MUST run the same TokenState code (train==live).
- Append-only logs span eras; anything reporting "current" stats must cut
  by era (status.json era block / model_loaded events).
- Double-gated live: --live AND PUMPFUN_LIVE_OK=1; JITO_DRY_RUN=1 separately
  gates real submission. Tests must never touch these.
- SHADOW_HARNESS_LOG.md GOTCHAS is the canonical failure record; new
  failure classes get a numbered rule there, and a regression test here.
