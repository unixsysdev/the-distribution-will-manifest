"""Pluggable exit-policy registry.

Import side-effects: importing this package registers all built-in policies
under their NAME via the @register decorator. To add a new policy:

    1. Create bot_shadow/exit_policies/<name>.py
    2. Subclass ExitPolicy, decorate with @register("name")
    3. Import the module here so the @register fires
    4. Set cfg.exit.policy = "name" — the harness picks it up automatically

No harness code change needed for any of those steps.
"""
from .base import ExitPolicy, ExitDecision, HarnessConsts, get_policy, list_policies

# Side-effect imports — these register their classes
from . import k_combined        # noqa: F401
from . import h_time_spaced     # noqa: F401
from . import b_frontload       # noqa: F401
from . import hybrid_trail      # noqa: F401
from . import rl_layered        # noqa: F401
from . import level_tp          # noqa: F401
from . import lsm_continuation  # noqa: F401

__all__ = ["ExitPolicy", "ExitDecision", "HarnessConsts",
           "get_policy", "list_policies"]
