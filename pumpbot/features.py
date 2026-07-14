"""Entry feature computation (canonical: feature_accum.py, rich_entry_features.py).

Trigger semantics (K_TRIGGER/V_TRIGGER env) are resolved at feature_accum
import time; set the env before first attribute access.
"""
from ._lazy import make_lazy

make_lazy(__name__, {
    "TokenState": ("feature_accum", "TokenState"),
    "ENTRY_FEATURE_NAMES": ("feature_accum", "ENTRY_FEATURE_NAMES"),
    "K_TRIGGER": ("feature_accum", "K_TRIGGER"),
    "V_TRIGGER": ("feature_accum", "V_TRIGGER"),
    "build_entry_features": ("rich_entry_features", "build_entry_features"),
    "decision_path_features": ("rich_entry_features", "decision_path_features"),
})
