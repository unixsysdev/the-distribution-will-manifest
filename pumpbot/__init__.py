"""pumpbot — the package surface over the (still flat) production tree.

Migration stage 2a (additive, 2026-06-10): every submodule here is a LAZY
re-export of the canonical flat module, so importing pumpbot never executes
side-effectful module code until an attribute is actually used, and nothing
on disk moved. The five data collectors' import closure is FROZEN (see
docs/notes/ARCHITECTURE.md and tests/test_collector_freeze.py); file moves happen in
stage 2b inside a planned migration window only.

Use:
    from pumpbot.features import TokenState, ENTRY_FEATURE_NAMES
    from pumpbot.exits import get_policy
    python -m pumpbot bot --help
"""
__version__ = "0.2.0"
