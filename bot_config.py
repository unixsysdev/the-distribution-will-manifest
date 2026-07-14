"""Single source of truth for bot knobs.

Layered overrides (later wins):
  1. defaults baked into this module
  2. config.yaml at repo root (if present)
  3. env vars: PUMPFUN_<UPPERCASE_DOTTED_PATH>=value
     (e.g. PUMPFUN_BROKER_TIP_LAMPORTS=200000)

Modules import what they need:
  from bot_config import cfg
  tip_lam = cfg.broker.tip_lamports

The Config object is a frozen dataclass-like with dot access. Backward compatible:
config.yaml absent = use defaults. Existing CLI flags (--bet-sol, --entry-threshold,
--source, --artifact-dir, --data-dir) still win over the config values when set.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH_ENV = "PUMPFUN_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("config.yaml")


# ---------- Defaults (also written to config.yaml on first run) ----------

DEFAULTS = {
    "bot": {
        "artifact_dir": "bot_artifacts_K7V",
        "data_dir": "bot_data",
        "bet_sol": 1.0,
        "entry_threshold_override": None,   # null = use model spec
        "status_interval_s": 30,
    },
    "harness": {
        "snap_every": 3,
        "stale_sec": 300,
        "fresh_rsol_lam": 3_000_000_000,    # 3 SOL
    },
    "exit": {
        "policy": "k_combined",             # k_combined | h_time_spaced | b_frontload | c_hybrid_t30 | f_hybrid_t50
        "mode": "static",                   # static | dynamic (dynamic = auto-policy selector picks)
        "total_slices": 8,
        "derisk_slices": 4,
        "derisk_min_gap_s": 5.0,
        "runner_min_gap_s": 15.0,
        "runner_retrace_frac": 0.30,        # used by c_hybrid_t30 / f_hybrid_t50
        "runner_min_arm_ret": 0.20,
        "death_threshold": 0.10,
        # Auto-policy selector knobs (only active when mode=dynamic)
        "dynamic_window_fires": 30,         # last N fires to score
        "dynamic_min_uplift_sol": 0.02,     # min uplift/bet to actually swap
        "dynamic_cooldown_h": 6,            # min hours between swaps
        "dynamic_min_sample": 30,           # min fires before considering a swap
        "dynamic_candidates": ["k_combined","h_time_spaced","b_frontload",
                                "c_hybrid_t30","f_hybrid_t50"],
    },
    "broker": {
        "tip_lamports": 100_000,            # 0.0001 SOL per bundle
        "slippage_bps": 1500,               # LEGACY shared fallback (used only if the split keys below are absent)
        # Split slippage (2026-06-11). ENTRY cap = SAFETY BOUND on worst-case
        # spend (max_sol_cost = bet*(1+bps/1e4)), NOT an EV optimizer: exec_sim
        # measured identical slip distributions for winners and losers (p50
        # +0.85, max +1.34), so a binding cap rejects winners pro-rata; at
        # 1500bps only 6/157 fires survive and total net is NEGATIVE. 20000bps
        # keeps every observed fire with ~50% headroom, bounds spend at 3x bet.
        "slippage_bps_buy": 20000,
        # SELL protection unchanged at 15%; the validated retry ladder
        # (3000/5000/8000 -> market) handles reverts on dumps.
        "slippage_bps_sell": 1500,
        "jito_dry_run": True,               # mirror of JITO_DRY_RUN env
        "jito_endpoint": "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1",
        "recon_poll_s": 2.0,
        "recon_expire_s": 30.0,
        "holdings_reconcile_s": 300.0,
        "blockhash_poll_s": 0.2,            # aggressive poll -> p50 bh_age ~100ms
        # Freshness cap for blockhash: if cached entry older than this when a
        # bundle is being assembled, broker blocks on an on-demand RPC refresh
        # before using it. Bounds worst-case bh_age regardless of asyncio
        # scheduling stalls in the background poll loop.
        "bh_max_age_ms": 500.0,
    },
    "listener": {
        "source": "grpc",                   # grpc | ws
        "grpc_endpoint": "grpc-fra1-1.erpc.global:80",
        "grpc_insecure": True,
        # commitment level for gRPC subscription. PROCESSED = leader-block-included,
        # fastest, ~0.5-1% reorg risk caught by holdings reconciler. Pin explicitly
        # so we don't silently slow down if upstream's default ever changes.
        "commitment": "processed",          # processed | confirmed | finalized
    },
    "paper_book": {
        "cost_bps": 250.0,
        "fee_per_tx_sol": 0.0015,
        "entry_lat_snaps": 1,
        "max_slices": 8,
    },
    "risk": {
        "max_concurrent_positions": 10,     # was NUM_SLOTS=16 in original design
        "max_fires_per_minute": 6,          # rate limit
        "daily_loss_limit_sol": -5.0,       # auto-pause when cumulative net <= this
        "bundle_failure_rate_limit": 0.50,  # auto-pause if >50% of recent bundles fail
        "bundle_failure_window": 20,        # over the last N bundles
        "circuit_breaker_cooldown_s": 600,  # wait 10min after a breaker fires before re-arming
        "skip_fire_vsol_lam": 55_000_000_000,  # don't fire if decision vsol>=55 SOL (late entry = validated loser; catchable_edge.py)
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(d: dict, prefix: str = "PUMPFUN") -> dict:
    """Override any leaf value via env: PUMPFUN_BROKER_TIP_LAMPORTS=200000.
    Type-coerces string env values to int/float/bool when the default is one."""
    def walk(node: dict, path: list[str]):
        for k, v in list(node.items()):
            new_path = path + [k]
            if isinstance(v, dict):
                walk(v, new_path)
            else:
                env_key = "_".join([prefix] + [p.upper() for p in new_path])
                if env_key in os.environ:
                    raw = os.environ[env_key]
                    if isinstance(v, bool):
                        node[k] = raw.strip().lower() in ("1","true","yes","on")
                    elif isinstance(v, int) and not isinstance(v, bool):
                        try: node[k] = int(raw)
                        except ValueError: pass
                    elif isinstance(v, float):
                        try: node[k] = float(raw)
                        except ValueError: pass
                    else:
                        node[k] = raw
    walk(d, [])
    return d


class _Ns:
    """Lightweight dot-access namespace over a dict."""
    def __init__(self, d: dict):
        self._d = d
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _Ns(v))
            else:
                setattr(self, k, v)
    def as_dict(self) -> dict: return self._d
    def get(self, key: str, default=None): return self._d.get(key, default)


def load_config(path: str | Path | None = None) -> _Ns:
    """Load and return the active config. Defaults if no file present."""
    p = Path(path) if path else Path(os.environ.get(CONFIG_PATH_ENV, str(DEFAULT_CONFIG_PATH)))
    merged = dict(DEFAULTS)
    # Re-deepcopy
    import copy
    merged = copy.deepcopy(DEFAULTS)
    if p.exists():
        try:
            import yaml
            with open(p) as f:
                file_cfg = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, file_cfg)
        except ImportError:
            # pyyaml not installed; fall back to defaults silently
            pass
        except Exception as e:
            print(f"[bot_config] error loading {p}: {e}; using defaults")
    merged = _apply_env_overrides(merged)
    return _Ns(merged)


# Module-level singleton; modules `from bot_config import cfg`
cfg = load_config()


def write_default_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Write the DEFAULTS dict to disk as YAML. Called on demand."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml not installed (`pip install pyyaml`)")
    p = Path(path)
    with open(p, "w") as f:
        yaml.safe_dump(DEFAULTS, f, default_flow_style=False, sort_keys=False)
    return p


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1 and sys.argv[1] == "write-default":
        out = write_default_config(sys.argv[2] if len(sys.argv) > 2 else "config.yaml")
        print(f"wrote defaults to {out}")
    else:
        print(json.dumps(cfg.as_dict(), indent=2, default=str))
