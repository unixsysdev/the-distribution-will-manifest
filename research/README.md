# Research generations

The executable research is grouped by the lifecycle stage at which it makes a decision:

| Package | Decision point | Contents |
|---|---|---|
| `continuation/` | A launch has crossed a bonding-curve milestone | causal trackers, sizing, reputation experiments, replay evaluators, dashboard, watchdog, and dry-run/live harness |
| `graduation/` | A token has migrated to PumpSwap | graduation listener, AMM panel extraction, rich and slow-climber trainers, realizability analyses, dashboard, and dry-run/live harness |

Run modules from the repository root so the shared launch/runtime modules remain importable:

```bash
python -m research.continuation.continuation_eval
python -m research.continuation.continuation_dashboard
python -m research.graduation.grad_cont_extract --help
python -m research.graduation.graduation_dashboard
```

Both generations reuse shared execution, parsing, and reconciliation code from the repository root. Generated panels, logs, and model binaries belong under ignored local data directories and must not be committed.

## Continuation build chain

The published continuation sources cover the full path from archived observations to deployable-model output:

| Stage | Modules |
|---|---|
| Early-entry augmentation | `continuation/entry_aug_extract.py`, `continuation/entry_aug_train.py` |
| Raw binary gRPC-firehose decoding | `continuation/cont_2x_aug_extract.py` |
| Shred flow, signer reputation, and recency | `continuation/cont_2x_shred_rep.py`, `continuation/cont_2x_recency_extract.py`, `continuation/cont_aug_features.py` |
| Train/compare augmented models | `continuation/cont_2x_train.py` |
| Rebuild the shared serving panel and seed | `continuation/cont_2x_build_deploy.py` |
| Fit final deployment artifacts | `continuation/cont_2x_deploy_train.py`, `continuation/continuation_rep_train.py` |

The extractors accept explicit input/output paths. Their defaults use `PUMPFUN_ROOT` for the repository-local data tree and retain the original Storage Box archive location as provenance. Trained pickle files and source data remain excluded from Git.
