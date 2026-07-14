"""ExitPolicy — pluggable per-position exit dispatcher.

Each policy implements:
    on_entry(mint, ev, entry_features, score)     — fires once when position opens
    decide(...)  -> ExitDecision                    — fires on every path snap
    on_close(mint)                                  — cleanup hook

The harness owns the broker, paper book, slice counters, logging. The policy
only decides WHAT to do; the harness DOES it. Adding a new policy = subclass
ExitPolicy, decorate with @register("name"), no harness changes needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# Action that decide() returns. Three legal shapes:
#   "hold"     — do nothing this snap
#   "slice"    — fire a partial sell at frac of REMAINING (n_sold tracked by harness)
#   "sell_all" — liquidate the remaining position immediately (e.g. trailing stop)
@dataclass
class ExitDecision:
    action: str                   # "hold" | "slice" | "sell_all"
    frac: float = 0.0             # fraction of remaining (0,1]. Ignored for "hold"/"sell_all"
    phase: str = "?"              # "derisk" | "runner" | "rl_disc" | "rl_hold" | etc — for logs
    reason: str = ""              # human-readable why (logged on sell_all)
    extra: dict = field(default_factory=dict)  # policy-specific telemetry (logged verbatim)


# Constants the harness owns and passes into decide() so policies can stay
# pure-functional w.r.t. the harness internals. Kept tiny on purpose — if a
# policy needs more, add a field here and pass it through.
@dataclass
class HarnessConsts:
    max_slices:        int
    derisk_slices:     int
    derisk_min_gap_s:  float
    runner_min_gap_s:  float
    runner_retrace_frac: float
    runner_min_arm_ret: float
    death_threshold:   float


class ExitPolicy:
    """Base class. Subclasses override the three hooks.

    Per-position state lives on self.per_mint[mint] — a dict the harness will
    NOT touch. Cleanup via on_close().
    """
    NAME: str = "base"           # match key for cfg.exit.policy

    def __init__(self, cfg, **kw):
        self.cfg = cfg
        self.per_mint: dict[str, dict[str, Any]] = {}

    # ---------- lifecycle ----------

    def on_entry(self, mint: str, ev, entry_features: dict, score: float) -> None:
        """Called by harness once when book.open() fires. Use to compute and
        cache anything that depends on entry-time features (e.g. classifier
        scores) so per-snap decide() stays fast."""
        pass

    def decide(self, mint: str, n_sold: int, last_slice_t: float,
               now: float, pf: dict, run_max: float, p_rec: float, fwd_n: int,
               consts: HarnessConsts) -> ExitDecision:
        """Per-snap dispatch. Returns ExitDecision; harness executes."""
        return ExitDecision("hold")

    def on_close(self, mint: str) -> None:
        """Called by harness after the position is formally closed
        (slice_exhaustion, death_cut, runner_exit, stale, shutdown).
        Default: clear per-mint state."""
        self.per_mint.pop(mint, None)


# ----------------------- Registry -----------------------------------
# Subclasses decorate with @register("name") to make themselves discoverable
# by cfg.exit.policy = "name". The harness calls get_policy(name, cfg) once
# at startup; the same instance is reused for all positions.

_REGISTRY: dict[str, type[ExitPolicy]] = {}
# Per-(name, cfg-id) instance cache. The harness instantiates once at startup,
# but offline replay (auto_policy + ac_backtest etc.) may call get_policy()
# many times within a single tool run. Caching avoids re-loading expensive
# artifacts (pickled RL tables, classifiers) per call.
_INSTANCE_CACHE: dict[tuple[str, int], ExitPolicy] = {}


def register(name: str):
    """Class decorator. Registers the policy class under `name`."""
    def wrap(cls):
        cls.NAME = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def get_policy(name: str, cfg, *, fresh: bool = False, **kw) -> ExitPolicy:
    """Return the registered policy. By default cache one instance per
    (name, cfg-id) so repeated calls with the same cfg reuse loaded artifacts.
    Pass fresh=True to bypass the cache (useful in tests)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown exit policy '{name}' — registered: "
                         f"{sorted(_REGISTRY.keys())}")
    if fresh:
        return _REGISTRY[name](cfg, **kw)
    key = (name, id(cfg))
    inst = _INSTANCE_CACHE.get(key)
    if inst is None:
        inst = _REGISTRY[name](cfg, **kw)
        _INSTANCE_CACHE[key] = inst
    return inst


def list_policies() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_instance_cache() -> None:
    _INSTANCE_CACHE.clear()
