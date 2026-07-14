"""Live shadow harness — V+K7 stacked entry head (Finding 35).

Pipeline on each pump.fun log notification:
  parse Program data -> TradeEvent
    -> gate: classic curve (vsol-rsol=30) + fresh launch (first-seen at rsol<3 SOL)
    -> per-mint dual-trigger accumulator (TokenState tracks K=7 AND V=0.5 in parallel)
    -> on 'k_only' / 'v_only': log partial trigger fire
    -> on 'ready' (both fired): ModelServer.score_entry; if fire, open PaperPosition
       at K=7-anchored reserves (vsK, vtK)
    -> on 'fwd' (every SNAP_EVERY trades): K-anchored path features,
       ModelServer.score_recovery (uses 11 K-features + 9 path features), route to
       PaperBook (scale-out / death-cut)
  Stale watchdog closes positions inactive > STALE_SEC.
  All events logged to JSONL.

NO EXECUTION. No Jito calls. No wallet usage. wallet_configured() result ignored.
"""
from __future__ import annotations
import asyncio, json, signal, sys, time
from collections import defaultdict, deque
from pathlib import Path

# allow running from /root/the-distribution-will-manifest with bot_shadow alongside
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))

from solders.pubkey import Pubkey
from solana.rpc.commitment import Confirmed
from solana.rpc.websocket_api import RpcTransactionLogsFilterMentions, connect

import config
from pumpfun_parse import parse_program_data_line
from feature_accum import (TokenState, ENTRY_FEATURE_NAMES,
                           ENTRY_FEATURE_NAMES_K, ENTRY_FEATURE_NAMES_V)
from model_serve import ModelServer
from paper_book import PaperBook
from rich_entry_features import (build_entry_features, decision_path_features,
                                 trade_row_from_event)

# Config knobs (loaded from config.yaml or env overrides; see bot_config.py)
try:
    from bot_config import cfg as _C
    SNAP_EVERY         = _C.harness.snap_every
    STALE_SEC          = _C.harness.stale_sec
    FRESH_RSOL_LAM     = _C.harness.fresh_rsol_lam
    MAX_SLICES         = _C.exit.total_slices
    DERISK_SLICES      = _C.exit.derisk_slices
    DERISK_MIN_GAP_SEC = _C.exit.derisk_min_gap_s
    RUNNER_MIN_GAP_SEC = _C.exit.runner_min_gap_s
    EXIT_POLICY        = getattr(_C.exit, "policy", "k_combined")
    DEATH_THRESHOLD    = float(_C.exit.death_threshold)
    # Trailing stop params (used by hybrid_trail family)
    RUNNER_RETRACE_FRAC = float(getattr(_C.exit, "runner_retrace_frac", 0.30))
    RUNNER_MIN_ARM_RET  = float(getattr(_C.exit, "runner_min_arm_ret", 0.20))
    # Risk limits (circuit breakers)
    RISK_MAX_CONCURRENT          = _C.risk.max_concurrent_positions
    RISK_MAX_FIRES_PER_MIN       = _C.risk.max_fires_per_minute
    RISK_DAILY_LOSS_LIMIT        = _C.risk.daily_loss_limit_sol
    RISK_BUNDLE_FAILURE_RATE     = _C.risk.bundle_failure_rate_limit
    RISK_BUNDLE_FAILURE_WINDOW   = _C.risk.bundle_failure_window
    RISK_BREAKER_COOLDOWN_S      = _C.risk.circuit_breaker_cooldown_s
    # Late-entry skip: don't fire if decision virtual-SOL reserves >= this
    # (token already too far up the curve = late entry = validated loser +
    # graduates before we land). 0 disables. See catchable_edge.py.
    SKIP_FIRE_VSOL_LAM           = int(getattr(_C.risk, "skip_fire_vsol_lam", 55_000_000_000))
except Exception:
    # fallback defaults if bot_config not available (e.g. local dev)
    SNAP_EVERY = 3; STALE_SEC = 300; FRESH_RSOL_LAM = 3_000_000_000
    MAX_SLICES = 8; DERISK_SLICES = 4
    DERISK_MIN_GAP_SEC = 5.0; RUNNER_MIN_GAP_SEC = 15.0
    EXIT_POLICY = "k_combined"; DEATH_THRESHOLD = 0.10
    RUNNER_RETRACE_FRAC = 0.30; RUNNER_MIN_ARM_RET = 0.20
    RISK_MAX_CONCURRENT = 10; RISK_MAX_FIRES_PER_MIN = 6
    RISK_DAILY_LOSS_LIMIT = -5.0
    RISK_BUNDLE_FAILURE_RATE = 0.50; RISK_BUNDLE_FAILURE_WINDOW = 20
    RISK_BREAKER_COOLDOWN_S = 600
    SKIP_FIRE_VSOL_LAM = 55_000_000_000


class ShadowHarness:
    def __init__(self, artifact_dir="bot_artifacts_K7V", log_path="shadow_run.jsonl",
                 position_store=None, broker=None, closed_mints: set | None = None):
        """
        position_store: optional PositionStore (bot_shadow.position_store.PositionStore).
                        When provided, every open/snap/close also writes to the store
                        for restart recovery.
        broker:         optional broker with .buy(mint, sol)/.sell(mint, slice_tok) methods.
                        Defaults to None (pure paper, no execution). Pass a JitoBroker
                        for live mode.
        closed_mints:   optional set of mints to skip (don't re-enter). Used by the bot
                        to avoid re-entering positions that were force-closed on restart.
        """
        self.srv = ModelServer(artifact_dir)
        # Shadow recovery head (2026-06-10): candidate death-cut scored on
        # drawdown snaps of OPEN positions and LOGGED ONLY — it never acts.
        # Accumulates the live would-cut record needed to judge the cut
        # policy on forward data before arming.
        self.shadow_recovery = None
        self.shadow_cut_flagged = set()
        try:
            import pickle as _pk
            from pathlib import Path as _P
            _rc = _P(artifact_dir) / "recovery_candidate.pkl"
            if _rc.exists():
                with open(_rc, "rb") as _fh:
                    self.shadow_recovery = _pk.load(_fh)
                print("[shadow] recovery candidate loaded (LOG-ONLY death-cut shadow, thr 0.20)",
                      flush=True)
        except Exception as _e:
            print(f"[shadow] recovery candidate load failed: {_e}", flush=True)
        self.rich_entry_enabled = bool(getattr(self.srv, "rich_entry", False))
        # Rich shadow scorer (2026-06-10): scores each decision with the rich
        # 192-feat model in parallel, logs shadow_rich_score, never acts. Gives
        # live rich score-alignment + the intent-TIMING gap (structural intent
        # parity already verified clean) at zero deploy risk.
        self.shadow_rich = None
        self.shadow_rich_feats = None
        try:
            import pickle as _pk2, json as _json2
            from pathlib import Path as _P2
            _rd = _P2("bot_artifacts_rich_shadow")
            if (_rd / "entry_model.pkl").exists():
                with open(_rd / "entry_model.pkl", "rb") as _fh2:
                    self.shadow_rich = _pk2.load(_fh2)
                self.shadow_rich_feats = _json2.loads(
                    (_rd / "model_spec.json").read_text())["entry"]["features"]
                print(f"[shadow] rich shadow scorer loaded "
                      f"({len(self.shadow_rich_feats)} feats, LOG-ONLY)", flush=True)
        except Exception as _e:
            print(f"[shadow] rich shadow load failed: {_e}", flush=True)
        self.rich_hist_needed = bool(self.rich_entry_enabled) or (self.shadow_rich is not None)
        # PaperBook q_sol matches the broker bet_sol so paper P&L numbers are in
        # the same units as real per-bet outcomes. Read from config.
        try:
            from bot_config import cfg as _C
            q_sol_for_book = float(_C.bot.bet_sol)
            book_cost_bps = float(_C.paper_book.cost_bps)
            book_fee = float(_C.paper_book.fee_per_tx_sol)
            book_max_sl = int(_C.paper_book.max_slices)
            book_entry_lat = int(_C.paper_book.entry_lat_snaps)
            book_c_death = float(_C.exit.death_threshold)
        except Exception:
            q_sol_for_book = 1.0; book_cost_bps = 250.0; book_fee = 0.0015
            book_max_sl = 8; book_entry_lat = 1; book_c_death = 0.10
        self.book = PaperBook(q_sol=q_sol_for_book, cost_bps=book_cost_bps,
                              fee_per_tx_sol=book_fee, max_slices=book_max_sl,
                              entry_lat_snaps=book_entry_lat, c_death=book_c_death)
        self.position_store = position_store
        self.broker = broker
        self.closed_mints: set = closed_mints if closed_mints is not None else set()
        self.states: dict = {}
        self.first_seen_rsol: dict = {}
        self.open_paper: set = set()
        self.fwd_counter = defaultdict(int)
        self.snap_count = defaultdict(int)
        self.last_trade_ts: dict = {}
        # Side-car sophistication context (Task 44 audit half).
        # The gRPC stream exposes per-tx fields (fee_lam, cu, jito_tip_lam, route,
        # n_inner_ix, n_keys) that capture buyer sophistication. The current V+K7
        # model does NOT consume them, but we track a rolling per-mint window
        # so each entry_decision event can be annotated with the K-window
        # sophistication signature. That gives us a paired (model-input, hidden-
        # context) dataset for offline correlation analysis without changing the
        # model's behavior.
        self.sophistication_win: dict = defaultdict(lambda: deque(maxlen=15))
        self.rich_trade_hist: dict = defaultdict(list)
        self.rich_entry_mid: dict = {}
        self.rich_run_max_ret = defaultdict(float)
        # Live scale-out scheduling (drives broker.sell_slice calls in real time;
        # PaperBook still does the retroactive analytical schedule for paper P&L)
        self.live_slices_sold = defaultdict(int)   # mint -> n slices sold this run
        self.live_dead = set()                     # mints fully exited (no more selling)
        self.last_slice_ts = defaultdict(float)    # mint -> wall ts of last slice (de-risk pacing)
        # Pluggable exit policy. Instantiated once per harness; same instance
        # services all positions (per-position state lives inside the policy's
        # `per_mint` dict). Adding a new policy = add an exit_policies/<name>.py
        # subclass with @register("name"); no harness changes needed.
        from exit_policies import get_policy, HarnessConsts, list_policies
        try:
            self.exit_policy = get_policy(EXIT_POLICY, _C)
            print(f"[shadow] exit policy = {EXIT_POLICY!r}  "
                  f"(registered: {list_policies()})", flush=True)
        except Exception as e:
            # Don't crash startup; fall back to k_combined.
            print(f"[shadow] WARN failed to load policy {EXIT_POLICY!r}: {e}; "
                  f"falling back to k_combined", flush=True)
            self.exit_policy = get_policy("k_combined", _C)
        self._exit_consts = HarnessConsts(
            max_slices=MAX_SLICES, derisk_slices=DERISK_SLICES,
            derisk_min_gap_s=DERISK_MIN_GAP_SEC, runner_min_gap_s=RUNNER_MIN_GAP_SEC,
            runner_retrace_frac=RUNNER_RETRACE_FRAC,
            runner_min_arm_ret=RUNNER_MIN_ARM_RET,
            death_threshold=DEATH_THRESHOLD,
        )
        # If the broker has a reconciliation callback hook, register it so the harness
        # can roll back optimistic slice state on bundle-landing failure.
        if self.broker is not None and hasattr(self.broker, "set_failure_callback"):
            self.broker.set_failure_callback(self._on_broker_failure)
        # Fill-anchored exits (2026-06-12). LIVE ONLY. The exit policy's +50%/-30%
        # is measured from the DECISION mid (midK) -- but on slipped fills we buy
        # well above that, so the TP fires below our actual cost (the first live
        # fire sold at +98% over decision = a loss over our +106% fill). exec_sim
        # validated taking profit from the FILL price (Model B, positive); the
        # live bot must do the same. On a live buy we hold exits until the fill
        # reconciles, then anchor ret to the real fill mid. Decision anchor is
        # kept for the PaperBook's analytical accounting (paper/dry-run unchanged).
        self.live_exit_anchor_mid: dict = {}   # mint -> realized fill mid (lamports/token-unit)
        self.awaiting_fill: set = set()        # live mints whose buy fill isn't reconciled yet
        if self.broker is not None and hasattr(self.broker, "set_fill_callback"):
            self.broker.set_fill_callback(self._on_buy_fill)
        self.log = open(log_path, "a", buffering=1)
        # Feature-accumulator checkpoint path (sibling of log_path). Loaded at
        # init, saved every CKPT_INTERVAL_S seconds by _checkpoint_loop. Restores
        # in-flight mid-window mints across restarts to prevent the fresh-rsol
        # bias that systematically dropped fast-moving high-scoring mints
        # before this landed (coverage_diag.py 2026-06-08).
        from pathlib import Path as _Path
        self._ckpt_path = _Path(log_path).parent / "feature_accum_checkpoint.json"
        self._ckpt_interval_s = 30.0
        self._ckpt_loaded_count = 0
        # Hybrid restoration strategy:
        # - SHORT downtime (<60s, i.e. quick restart): load checkpoint to get all
        #   in-flight TokenStates that existed pre-restart (their state may have
        #   been mid-window with NO trades in the small gap, so they wouldn't
        #   re-appear via capture-replay alone). Then replay capture from
        #   saved_at onward to apply the trades that arrived during downtime.
        # - LONG downtime (>=60s): checkpoint state is stale enough that
        #   completed/dead mints would pollute fresh state — do a full
        #   capture-replay (default 300s lookback) and ignore checkpoint.
        # - NO checkpoint: cold start, full capture-replay.
        self._capture_dir = _Path("/root/the-distribution-will-manifest/grpc_capture")
        self._bootstrap_lookback_s = 300  # 5 min cap for capture-replay window
        self._short_downtime_s = 60       # threshold separating short/long restart
        self._restore_hybrid()
        self.stats = dict(events=0, trade_events=0, classic=0, fresh=0,
                          k_fires=0, v_fires=0, both_ready=0,
                          entry_fire=0, snaps_routed=0, stale_close=0,
                          skipped_already_closed=0, broker_calls=0,
                          live_slices=0, live_death_cuts=0,
                          # Risk-limit refusals
                          risk_refusal_max_concurrent=0,
                          risk_refusal_rate_limit=0,
                          risk_refusal_daily_loss=0,
                          risk_refusal_failure_rate=0,
                          circuit_breaker_active=0)
        # Risk-limit tracking
        from collections import deque as _dq
        self.recent_fire_ts = _dq(maxlen=64)  # for per-minute rate limit
        self.circuit_breaker_until = 0.0       # epoch ts; bot refuses to fire while now < this
        self.start_ts = time.time()
        # Era P&L tracking: closes since THIS process start, in both
        # accountings. status.json publishes these so the dashboard can show
        # this-run stats instead of mixing restored cross-era book history.
        self.era_book_nets: list[float] = []
        self.era_policy_nets: list[float] = []

    def log_event(self, kind, **kw):
        if kind == "position_close":
            ln = getattr(self, "_last_policy_nets", None)
            if ln is not None and ln[0] == kw.get("mint") and "policy_nets" not in kw:
                kw["policy_nets"] = ln[1]
            try:
                if kw.get("net") is not None:
                    self.era_book_nets.append(float(kw["net"]))
                lp = kw.get("live_policy_net")
                # policy series stays pure-policy: closes without a policy net
                # (e.g. shutdown force-closes) are counted in the book series
                # only, never silently mixed into the policy mean.
                if lp is not None:
                    self.era_policy_nets.append(float(lp))
            except (TypeError, ValueError):
                pass
        rec = {"t": time.time(), "kind": kind, **kw}
        try: self.log.write(json.dumps(rec, default=str) + "\n")
        except Exception: pass

    # ---------- Hybrid restoration: checkpoint + capture-replay ----------
    def _restore_hybrid(self) -> None:
        """Combine checkpoint (fast, full prior state) with capture-replay
        (ground-truth, fills the downtime gap). The logic picks one of three
        modes based on how long the bot was down per the checkpoint's saved_at.
        Prints exactly one diagnostic line summarizing what was loaded."""
        # Read checkpoint metadata (downtime) without committing to using it
        downtime_s = None
        saved_at = 0.0
        if self._ckpt_path.exists():
            try:
                payload = json.loads(self._ckpt_path.read_text())
                saved_at = float(payload.get("saved_at", 0))
                if saved_at > 0:
                    downtime_s = max(0.0, time.time() - saved_at)
            except Exception:
                pass

        # Mode select
        if downtime_s is not None and downtime_s < self._short_downtime_s:
            # SHORT restart: checkpoint + capture-replay gap
            n_ckpt = self._load_checkpoint()
            # Replay from saved_at onward (plus 10s safety buffer)
            gap_lookback_s = downtime_s + 10.0
            n_capture, n_trades = self._bootstrap_from_capture(lookback_s=gap_lookback_s)
            print(f"[shadow] SHORT-restart restore: checkpoint={n_ckpt} mints + "
                  f"capture-gap-replay {gap_lookback_s:.1f}s ({n_trades:,} trades, "
                  f"{n_capture} mints touched) -> total {len(self.states)} mints "
                  f"({downtime_s:.1f}s downtime)", flush=True)
        else:
            # LONG restart or cold start: capture-replay only, full lookback
            n_capture, n_trades = self._bootstrap_from_capture(
                lookback_s=self._bootstrap_lookback_s)
            why = (f"LONG-restart ({downtime_s:.0f}s downtime)" if downtime_s
                   else "cold start (no checkpoint)")
            print(f"[shadow] {why} restore: full capture-replay "
                  f"({n_trades:,} trades from {self._bootstrap_lookback_s}s) -> "
                  f"{n_capture} mints", flush=True)

    # ---------- FeatureAccum bootstrap from grpc_capture (ground-truth) ----------
    def _bootstrap_from_capture(self, lookback_s: float | None = None) -> tuple[int, int]:
        """Replay last `lookback_s` seconds of grpc_capture/*.jsonl(.gz) through
        existing TokenStates (creating new ones if not present). Applies the
        same fresh-rsol filter as live on_trade().

        Returns (mints_touched, trades_replayed). Counted as "touched" if the
        mint already existed in self.states OR was newly created during replay.
        The caller (`_restore_hybrid`) prints the consolidated summary."""
        import glob, gzip, os
        if not self._capture_dir.exists():
            return 0, 0
        if lookback_s is None:
            lookback_s = self._bootstrap_lookback_s
        cutoff = time.time() - lookback_s
        # Files mtime within (lookback + slop) are candidates. Capture rotates
        # every ~hour. Active jsonl files can be multiple GB, so read only the
        # tail on restart; this preserves recent in-flight trigger state without
        # blocking startup on a full-file scan.
        files = sorted(glob.glob(str(self._capture_dir / "*.jsonl*")))
        active = [p for p in files if not p.endswith(".gz")]
        if active and os.stat(active[-1]).st_mtime > cutoff - 60:
            recent = [active[-1]]
        else:
            recent = [p for p in files
                      if os.stat(p).st_mtime > cutoff - 3600][-2:]
        if not recent:
            return 0, 0
        n_trades = n_skipped_fresh = n_skipped_closed = 0
        for path in recent:
            opener = gzip.open if path.endswith(".gz") else open
            try:
                with opener(path, "rt") as f:
                    if not path.endswith(".gz"):
                        tail_bytes = int(os.getenv("BOOTSTRAP_CAPTURE_TAIL_BYTES",
                                                    str(1536 * 1024 * 1024)))
                        size = os.stat(path).st_size
                        if size > tail_bytes:
                            f.seek(size - tail_bytes)
                            f.readline()
                    for ln in f:
                        try: r = json.loads(ln)
                        except Exception: continue
                        ts = float(r.get("ev_ts") or 0)
                        if ts < cutoff: continue
                        mint = r.get("mint")
                        if not mint: continue
                        if mint in self.closed_mints:
                            n_skipped_closed += 1; continue
                        vsol = float(r.get("vsol") or 0)
                        vtok = float(r.get("vtok") or 0)
                        if vsol <= 0 or vtok <= 0: continue
                        sol = float(r.get("sol") or 0)
                        is_buy = bool(r.get("is_buy"))
                        user = r.get("user") or ""
                        rsol = float(r.get("rsol") or 0)
                        # Apply fresh-rsol filter at first observation —
                        # matches the live on_trade() logic exactly.
                        if mint not in self.first_seen_rsol:
                            self.first_seen_rsol[mint] = rsol
                            if rsol >= FRESH_RSOL_LAM:
                                n_skipped_fresh += 1
                                continue
                        if self.first_seen_rsol[mint] >= FRESH_RSOL_LAM:
                            continue
                        # Update or create TokenState
                        st = self.states.get(mint)
                        if st is None:
                            self.states[mint] = TokenState(vsol, vtok, sol, is_buy, user, ts)
                        else:
                            st.update(vsol, vtok, sol, is_buy, user, ts)
                        n_trades += 1
            except Exception as e:
                print(f"[shadow] bootstrap read err on {path}: {e}", flush=True)
                continue
        # Drop mints that already reached forward terminal beyond a small window
        # (these are already past the entry-decision moment and don't help us
        # observe new entries on restart).
        to_drop = [m for m, st in self.states.items() if st.fwd > 200]
        for m in to_drop: del self.states[m]
        return len(self.states), n_trades

    # ---------- FeatureAccum checkpoint (used by SHORT-restart path) ----------
    def _load_checkpoint(self) -> int:
        """Load TokenState dict + first_seen_rsol from disk if checkpoint exists.
        Silently skips on any error (we'd rather start fresh than crash).
        Returns number of mints loaded."""
        try:
            if not self._ckpt_path.exists(): return 0
            payload = json.loads(self._ckpt_path.read_text())
            mints = payload.get("states", {})
            for mint, sd in mints.items():
                if mint in self.closed_mints: continue
                try: self.states[mint] = TokenState.from_dict(sd)
                except Exception: continue
            for mint, rsol in payload.get("first_seen_rsol", {}).items():
                if mint in self.closed_mints: continue
                self.first_seen_rsol[mint] = float(rsol)
            self._ckpt_loaded_count = len(self.states)
            return len(self.states)
        except Exception as e:
            print(f"[shadow] checkpoint load failed: {e}", flush=True)
            return 0

    def _save_checkpoint(self) -> None:
        """Atomic dump of current TokenState dict + first_seen_rsol.
        Excludes mints already in closed_mints (won't be re-entered anyway).
        Excludes mints with k_fired AND v_fired AND fwd > 200 (forward-phase
        positions where the model decision has been made; carrying their full
        history forward is unnecessary and would bloat the checkpoint)."""
        ts = time.time()
        keep_states = {}
        for mint, st in self.states.items():
            if mint in self.closed_mints: continue
            # Skip very-old forward-phase mints (already fired or already missed)
            if st.k_fired and st.v_fired and st.fwd > 200: continue
            try: keep_states[mint] = st.to_dict()
            except Exception: continue
        payload = {
            "saved_at": ts,
            "states": keep_states,
            "first_seen_rsol": {m: r for m, r in self.first_seen_rsol.items()
                                if m not in self.closed_mints},
        }
        tmp = self._ckpt_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._ckpt_path)
        except Exception as e:
            print(f"[shadow] checkpoint save failed: {e}", flush=True)

    async def _checkpoint_loop(self) -> None:
        """Periodic atomic checkpoint of in-flight TokenStates."""
        while True:
            await asyncio.sleep(self._ckpt_interval_s)
            try: self._save_checkpoint()
            except Exception as e:
                print(f"[shadow] _checkpoint_loop tick failed: {e}", flush=True)

    def _compute_live_policy_net(self, pos) -> float | None:
        """Replay the CURRENTLY ACTIVE exit policy against this position's snap
        timeline to get the realized fractional P&L under what we ACTUALLY did
        (vs PaperBook's GREEN reference scheme in pos.net_return).

        Returns the fractional net (multiply by bet_sol for absolute SOL), or
        None if replay fails / not enough data. Best-effort: never raises into
        the close path, so logging never breaks even if the replay errors."""
        try:
            if not pos.snaps_vs or len(pos.snaps_vs) < 1:
                return None
            snaps = list(zip(pos.snaps_vs, pos.snaps_vt))
            # dts = seconds since first snap (matches what the policies expect)
            if pos.snaps_t and len(pos.snaps_t) == len(pos.snaps_vs):
                t0 = pos.snaps_t[0]
                dts = [t - t0 for t in pos.snaps_t]
            else:
                # No timestamp data — degrade to fwd-index-based dts (approximate).
                # Each fwd advance is roughly 1 trade event ≈ 0.5-2s on active tokens.
                dts = [float(i) for i in range(len(pos.snaps_vs))]
            # Lazy import the registry adapter (avoids circular imports)
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
            from strategy_ab_replay import policy_via_registry
            active = policy_via_registry(
                pos.vsK, pos.vtK, pos.vsC, pos.vtC, snaps, dts,
                policy_name=self.exit_policy.NAME, cfg=_C, mint=pos.mint)
            # Shadow exit accounting (2026-06-11): counterfactual nets for the
            # level-TP family on the same snap timeline. exit_lab's tournament
            # put tp_100 ahead of the incumbent tp_50 on the offline test fold;
            # the live deduped comparison of these decides the swap.
            nets = {}
            for pname in ("level_tp_50", "level_tp_100", "level_tp_200", "level_tp_100_t120", "level_tp_50_stop30_cap120", "level_tp_100_stop30_cap120"):
                if pname == self.exit_policy.NAME:
                    nets[pname] = active
                    continue
                try:
                    nets[pname] = policy_via_registry(
                        pos.vsK, pos.vtK, pos.vsC, pos.vtC, snaps, dts,
                        policy_name=pname, cfg=_C, mint=pos.mint)
                except Exception:
                    nets[pname] = None
            self._last_policy_nets = (pos.mint, nets)
            return active
        except Exception:
            return None

    def _prefetch_broker_meta(self, mint: str) -> None:
        """Start the broker's (token_program, creator) fetch at FIRE time so it
        overlaps the entry bookkeeping and the same-second TP sell shares the
        buy's in-flight fetch (broker dedups per mint). Deliberately NOT called
        at K/V partial-trigger time: that would be ~34k RPC calls/day against
        the eRPC credit pool (~10M/month) for mints that mostly never fire,
        while the warm pooled connection already makes the fire-time fetch
        ~10-30ms. No-op for PaperBroker; failures are swallowed (the buy's own
        fetch retries with full error handling)."""
        b = self.broker
        if b is not None and hasattr(b, "prefetch_mint_meta"):
            try:
                b.prefetch_mint_meta(mint)
            except Exception:
                pass

    def _risk_check_before_fire(self) -> tuple[bool, str]:
        """Returns (allow_fire, refusal_reason). Refuses on any tripped limit.
        Called once per entry-decision fire BEFORE book.open + broker.buy.
        Each refusal increments a stat counter so the report can show why we held back."""
        now = time.time()
        # Circuit breaker timeout in effect?
        if now < self.circuit_breaker_until:
            self.stats["circuit_breaker_active"] += 1
            return False, f"circuit_breaker_active (until {self.circuit_breaker_until:.0f})"
        # Concurrent positions cap
        if len(self.open_paper) >= RISK_MAX_CONCURRENT:
            self.stats["risk_refusal_max_concurrent"] += 1
            return False, f"max_concurrent_positions ({len(self.open_paper)} >= {RISK_MAX_CONCURRENT})"
        # Rate limit (per minute)
        cutoff = now - 60.0
        while self.recent_fire_ts and self.recent_fire_ts[0] < cutoff:
            self.recent_fire_ts.popleft()
        if len(self.recent_fire_ts) >= RISK_MAX_FIRES_PER_MIN:
            self.stats["risk_refusal_rate_limit"] += 1
            return False, f"rate_limit ({len(self.recent_fire_ts)} fires in last 60s >= {RISK_MAX_FIRES_PER_MIN})"
        # Daily loss limit — ERA-scoped and in SOL. Two bugs fixed 2026-06-10:
        # (1) book.returns() includes RESTORED cross-era positions, so a prior
        # model's record could gate, or loosen the gate for, the current one;
        # (2) the config value is SOL but the sum was FRACTIONAL returns,
        # a 10x mismatch at bet=0.1. Era closes live in era_book_nets.
        # LIVE: gate on REAL realized P&L (paired buy+sell wallet deltas), active from
        # the FIRST fire (no cold start). DRY_RUN: paper-book era accounting (shadow only).
        if self.broker is not None and not getattr(self.broker, "dry_run", True):
            cum_sol = self.broker.realized_net_sol()
            if cum_sol <= RISK_DAILY_LOSS_LIMIT:
                self.stats["risk_refusal_daily_loss"] += 1
                self.circuit_breaker_until = now + RISK_BREAKER_COOLDOWN_S
                return False, (f"daily_loss_limit LIVE (realized {cum_sol:+.3f} SOL "
                               f"<= {RISK_DAILY_LOSS_LIMIT})")
        else:
            era_nets = getattr(self, "era_book_nets", [])
            if len(era_nets) >= 5:
                bet = float(getattr(self.broker, "bet_sol", 1.0) or 1.0)
                cum_sol = float(sum(era_nets)) * bet
                if cum_sol <= RISK_DAILY_LOSS_LIMIT:
                    self.stats["risk_refusal_daily_loss"] += 1
                    self.circuit_breaker_until = now + RISK_BREAKER_COOLDOWN_S
                    return False, (f"daily_loss_limit (era cum={cum_sol:+.3f} SOL "
                                   f"<= {RISK_DAILY_LOSS_LIMIT})")
        # Bundle failure rate (only meaningful in LIVE; DRY_RUN has no recon outcomes)
        if self.broker is not None and hasattr(self.broker, "recon_summary"):
            summ = self.broker.recon_summary()
            n = summ.get("n_outcomes", 0)
            if n >= RISK_BUNDLE_FAILURE_WINDOW:
                fail_rate = summ.get("n_failed", 0) / n
                if fail_rate > RISK_BUNDLE_FAILURE_RATE:
                    self.stats["risk_refusal_failure_rate"] += 1
                    self.circuit_breaker_until = now + RISK_BREAKER_COOLDOWN_S
                    return False, f"bundle_failure_rate ({fail_rate:.2f} > {RISK_BUNDLE_FAILURE_RATE} on n={n})"
        return True, ""

    async def _dispatch_exit_slice(self, mint, n_sold, now, pf, run_max, p_rec,
                                    vsol_i, vtok_i, slot, fwd_n):
        """Pluggable dispatch via self.exit_policy.decide(). The policy decides
        WHAT to do (hold/slice/sell_all + fraction + phase); the harness DOES
        it here (broker call + state tracking + logging + book close on exit).
        Adding a new policy = add a subclass in exit_policies/, no harness
        change. p_rec is the recovery-model output for this snap; the harness
        passes it through so RL-style policies can use it as a state feature."""
        last_t = self.last_slice_ts.get(mint, 0.0)
        try:
            dec = self.exit_policy.decide(mint, n_sold, last_t, now, pf,
                                           run_max, p_rec, fwd_n, self._exit_consts)
        except Exception as e:
            self.log_event("exit_policy_error", mint=mint, policy=self.exit_policy.NAME,
                           err=str(e))
            return

        if dec.action == "hold":
            return

        if dec.action == "sell_all":
            # Liquidate the remainder. Used by trailing-stop / RL bang-bang.
            self.live_dead.add(mint)
            self.stats["live_slices"] += 1
            try:
                await self.broker.sell_all(mint, vsol_lam=vsol_i, vtok=vtok_i, slot=slot)
                self.stats["broker_calls"] += 1
                self.log_event("live_runner_exit", mint=mint, fwd=fwd_n,
                                policy=self.exit_policy.NAME, n_sold=n_sold,
                                ret=pf["ret"], run_max=run_max,
                                phase=dec.phase, reason=dec.reason, slot=slot,
                                extra=dec.extra)
            except Exception as e:
                self.log_event("broker_error", mint=mint, op="sell_all_runner", err=str(e))
            # Formally close PaperBook (same leak we fixed earlier; sell_all
            # without close left positions "open" in the dashboard)
            pos = self.book.positions.get(mint)
            if pos is not None and not pos.closed:
                self.book._close_one(pos)
                exit_ret_val = pf["ret"]
                live_net = self._compute_live_policy_net(pos)
                if self.position_store is not None:
                    self.position_store.record_close(
                        mint, pos.net_return, pos.kind, reason="runner_exit",
                        exit_ret=exit_ret_val,
                        live_policy_net=live_net,
                        live_policy_name=self.exit_policy.NAME)
                self.log_event("position_close", mint=mint, exit_kind=pos.kind,
                               net=pos.net_return, reason="runner_exit",
                               exit_ret=exit_ret_val,
                               live_policy_net=live_net,
                               live_policy_name=self.exit_policy.NAME)
                self.open_paper.discard(mint)
                self.closed_mints.add(mint)
            try: self.exit_policy.on_close(mint)
            except Exception: pass
            return

        # ---- dec.action == "slice" ----
        if dec.action != "slice":
            self.log_event("unknown_exit_action", mint=mint,
                           action=dec.action, policy=self.exit_policy.NAME)
            return
        fire, frac, phase = True, float(dec.frac), dec.phase
        policy = self.exit_policy.NAME  # used in later logs

        # ---- Fire the slice ----
        self.live_slices_sold[mint] = n_sold + 1
        self.last_slice_ts[mint] = now
        self.stats["live_slices"] += 1
        try:
            await self.broker.sell_slice(mint, frac, vsol_lam=vsol_i, vtok=vtok_i, slot=slot)
            self.stats["broker_calls"] += 1
            self.log_event("live_scale_slice", mint=mint, fwd=fwd_n,
                           policy=policy, phase=phase, slice_n=n_sold + 1, frac=frac,
                           ret=pf["ret"], run_max=run_max, slot=slot)
            if self.live_slices_sold[mint] >= MAX_SLICES:
                # All slices fired; formally close the PaperBook position so the
                # dashboard stops showing it as open with stale current-market ret.
                # PaperBook._close_one re-runs the analytical retroactive scale-out
                # on the routed snaps — its net_return is the analytical estimate
                # (not the exact live realized; the snap-and-broker sequences differ).
                # Live realized aggregate is in broker_jito.jsonl per-slice.
                self.live_dead.add(mint)
                pos = self.book.positions.get(mint)
                if pos is not None and not pos.closed:
                    self.book._close_one(pos)
                    exit_ret_val = pf["ret"]
                    live_net = self._compute_live_policy_net(pos)
                    if self.position_store is not None:
                        self.position_store.record_close(
                            mint, pos.net_return, pos.kind, reason="slices_exhausted",
                            exit_ret=exit_ret_val,
                            live_policy_net=live_net,
                            live_policy_name=self.exit_policy.NAME)
                    self.log_event("position_close", mint=mint, exit_kind=pos.kind,
                                   net=pos.net_return, reason="slices_exhausted",
                                   exit_ret=exit_ret_val,
                                   live_policy_net=live_net,
                                   live_policy_name=self.exit_policy.NAME)
                    self.open_paper.discard(mint)
                    self.closed_mints.add(mint)
                    try: self.exit_policy.on_close(mint)
                    except Exception: pass
        except Exception as e:
            self.log_event("broker_error", mint=mint, op="sell_slice", err=str(e))

    def _on_broker_failure(self, mint: str, op: str, reason: str) -> None:
        """Called by JitoBroker's reconciler when a submitted bundle does NOT land
        on chain. Rolls back optimistic slice state so the exit policy can retry
        on the next forward snap. Runs from the reconciler task's context (still
        single-threaded asyncio so safe to mutate state).

        For sells: decrement n_sold and reset the slice timer (so the exit policy
        will retry the same slice immediately on the next snap, ignoring the gap).
        For buys: a failed buy is timing-sensitive and we don't try to re-enter;
        we just log it and leave the mint in closed_mints so no second attempt.
        """
        if op in ("sell_slice", "sell_all"):
            prev_sold = self.live_slices_sold.get(mint, 0)
            if prev_sold > 0:
                self.live_slices_sold[mint] = prev_sold - 1
            self.last_slice_ts[mint] = 0.0   # reset; next snap can fire immediately
            was_dead = mint in self.live_dead
            self.live_dead.discard(mint)
            self.log_event("recon_rollback", mint=mint, op=op, reason=reason,
                           prev_n_sold=prev_sold,
                           new_n_sold=self.live_slices_sold.get(mint, 0),
                           was_dead=was_dead)
            self.stats["recon_rollbacks"] = self.stats.get("recon_rollbacks", 0) + 1
        elif op == "buy":
            # We optimistically opened the position in the paper book + position store.
            # Mark it as failed-on-chain; don't try to re-enter the same mint.
            self.live_dead.add(mint)
            self.awaiting_fill.discard(mint)        # buy never landed -> no fill anchor coming
            self.live_exit_anchor_mid.pop(mint, None)
            self.log_event("recon_buy_failed", mint=mint, reason=reason)
            self.stats["recon_buy_failed"] = self.stats.get("recon_buy_failed", 0) + 1
            # Close the optimistic book entry SYNCHRONOUSLY (we hold no tokens).
            # Otherwise the phantom lingers in open_paper and silently eats a
            # concurrency slot (n_open lies to the rate gate). net=0: we never
            # held the position; the failed-tx fee lives in the broker's realized.
            pos = self.book.positions.get(mint)
            if pos is not None and not pos.closed:
                pos.net_return = 0.0; pos.kind = "buy_failed"; pos.closed = True
                if self.position_store is not None:
                    self.position_store.record_close(mint, 0.0, "buy_failed", reason="buy_failed")
                self.log_event("position_close", mint=mint, exit_kind="buy_failed",
                               net=None, reason="buy_failed")  # net=None -> not counted in era P&L
            self.open_paper.discard(mint)
            self.closed_mints.add(mint)
            try: self.exit_policy.on_close(mint)
            except Exception: pass

    def _on_buy_fill(self, mint: str, fill_mid: float) -> None:
        """Called by the broker when a BUY reconciles (gRPC feed or poll), with the
        realized fill mid (lamports per token-unit = curve SOL paid / tokens got).
        Anchors the live exit ret to it and releases the exit hold. Single-threaded
        asyncio, so safe to mutate. Best-effort: if it never fires (lost bundle),
        the stale watchdog still force-closes the position after STALE_SEC."""
        try:
            if fill_mid and fill_mid > 0:
                self.live_exit_anchor_mid[mint] = float(fill_mid)
            self.awaiting_fill.discard(mint)
            self.log_event("live_exit_anchor_set", mint=mint, fill_mid=fill_mid)
        except Exception:
            self.awaiting_fill.discard(mint)

    async def on_trade(self, ev) -> None:
        self.stats["trade_events"] += 1
        mint = ev.mint
        self.last_trade_ts[mint] = time.time()
        if not ev.is_classic_curve:
            return
        self.stats["classic"] += 1
        if mint not in self.first_seen_rsol:
            self.first_seen_rsol[mint] = ev.real_sol_reserves
            if ev.real_sol_reserves >= FRESH_RSOL_LAM:
                return  # joined mid-curve, not a fresh launch
        if self.first_seen_rsol[mint] >= FRESH_RSOL_LAM:
            return
        self.stats["fresh"] += 1
        # Side-car: stash per-trade sophistication context (set by the gRPC
        # listener on the event as ev.grpc_extras). Cheap append; bounded
        # deque so memory stays flat.
        extras = getattr(ev, "grpc_extras", None)
        if extras is not None:
            self.sophistication_win[mint].append({
                "is_buy": ev.is_buy, "user": ev.user, **extras
            })
        if self.rich_hist_needed:
            self.rich_trade_hist[mint].append(trade_row_from_event(ev))
        st = self.states.get(mint)
        if st is None:
            self.states[mint] = TokenState(ev.virtual_sol_reserves, ev.virtual_token_reserves,
                                            ev.sol, ev.is_buy, ev.user, ev.timestamp)
            return
        result = st.update(ev.virtual_sol_reserves, ev.virtual_token_reserves,
                           ev.sol, ev.is_buy, ev.user, ev.timestamp)
        if result == "k_only":
            self.stats["k_fires"] += 1
            self.log_event("k_trigger", mint=mint, midK=st.midK, vsK=st.vsK, vtK=st.vtK,
                           n_at_trigger=st.n, cum_buy_sol=st.cum_buy_sol)
        elif result == "v_only":
            self.stats["v_fires"] += 1
            self.log_event("v_trigger", mint=mint, midV=st.midV, vsV=st.vsV, vtV=st.vtV,
                           n_at_trigger=st.n, cum_buy_sol=st.cum_buy_sol)
        elif result == "ready":
            self.stats["both_ready"] += 1
            # If both fired on this same trade we log the partial events too
            if st.k_window_last_ts == ev.timestamp:
                self.stats["k_fires"] += 1
            if st.v_window_last_ts == ev.timestamp:
                self.stats["v_fires"] += 1
            # Shred-stream front-run signal. Reads the SHM ring (filled by
            # intent_recorder ~1 ms after the shred entry hits us) and
            # counts pending intents for THIS mint in two windows (500ms
            # and 2000ms). When the signal is strong we'll bump the Jito
            # tip on the broker.buy below so our bundle lands in the same
            # slot as the snipers. In DRY_RUN the bump just gets logged.
            shred_sig = {}
            intent_feats = {}
            if self.shred_window is not None:
                try:
                    self.shred_window.drain_now()
                    shred_sig = self.shred_window.signal(mint)
                    if self.rich_hist_needed:
                        intent_feats = self.shred_window.intent_features(mint)
                except Exception as e:
                    shred_sig = {"err": str(e)[:80]}
            if self.rich_entry_enabled:
                try:
                    ef, rich_debug = build_entry_features(
                        self.rich_trade_hist.get(mint, []),
                        k=self.srv.entry_k,
                        v_sol=self.srv.entry_v_sol,
                        expected_features=self.srv.entry_features,
                        intent_features=intent_feats,
                    )
                except Exception as e:
                    self.log_event("entry_feature_error", mint=mint, err=str(e),
                                   model="rich_entry", n_hist=len(self.rich_trade_hist.get(mint, [])),
                                   shred_signal=shred_sig, intent_features=intent_feats)
                    self.rich_trade_hist.pop(mint, None)
                    return
                entry_vs = float(rich_debug["entry_vsol"])
                entry_vt = float(rich_debug["entry_vtok"])
                entry_mid = float(rich_debug["entry_mid"])
                soph = _sophistication_summary(list(self.sophistication_win.get(mint, [])))
            else:
                # Build the entry input dict. Always includes the 22 K+V features.
                # If the live model is the wide (31-feat) variant, ALSO include the
                # 9 sophistication features pulled from the per-mint sophistication
                # window the bot already accumulates. HGB tolerates NaN for missing
                # soph fields (e.g. mints with no Jito tip in the K-window).
                k_vals = st.k_entry_features()
                v_vals = st.v_feats  # already a tuple of 11
                ef = {}
                for name, val in zip(ENTRY_FEATURE_NAMES_K, k_vals): ef[name] = val
                for name, val in zip(ENTRY_FEATURE_NAMES_V, v_vals): ef[name] = val
                # Soph features. We always compute the summary (used for logging
                # below); if the model expects soph_* features, populate them with
                # the soph_ prefix mapping. Compute once, used twice.
                _soph_summary = _sophistication_summary(list(self.sophistication_win.get(mint, [])))
                _SOPH_KEYS = ("fee_p50_lam","fee_p90_lam","cu_p50","cu_mean",
                              "jito_tip_rate","jito_tip_p50_lam","routed_rate",
                              "n_inner_ix_mean","n_keys_mean")
                for k in _SOPH_KEYS:
                    v = _soph_summary.get(k)
                    ef[f"soph_{k}"] = float(v) if v is not None else float("nan")
                entry_vs = st.vsK
                entry_vt = st.vtK
                entry_mid = st.midK
                soph = _soph_summary
            score, fire = self.srv.score_entry(ef)
            if self.shadow_rich is not None and not self.rich_entry_enabled:
                try:
                    _, _snon = build_entry_features(
                        self.rich_trade_hist.get(mint, []),
                        k=self.srv.entry_k, v_sol=self.srv.entry_v_sol,
                        expected_features=[])
                    _merged = dict(_snon)
                    _merged.update(intent_feats or {})
                    import numpy as _np
                    _vec = _np.array([[float(_merged.get(f, _np.nan))
                                       for f in self.shadow_rich_feats]], dtype=float)
                    _rs = float(self.shadow_rich.predict_proba(_vec)[0, 1])
                    _nmiss = sum(1 for f in self.shadow_rich_feats if f not in _merged)
                    self.log_event("shadow_rich_score", mint=mint, rich_score=_rs,
                                   entry_score=score, n_missing_feats=_nmiss,
                                   intent_present=bool(intent_feats),
                                   intent_2s_n=(intent_feats or {}).get("intent_2p0s_n"))
                except Exception as _e:
                    self.log_event("shadow_rich_error", mint=mint, err=str(_e)[:120])
                finally:
                    self.rich_trade_hist.pop(mint, None)
            if self.rich_entry_enabled:
                self.rich_trade_hist.pop(mint, None)
            # log features + threshold + active exit policy + first-seen state
            # so future debugging, drift diagnostics, and policy-impact
            # correlation can be done from JSONL without offline reconstruction.
            self.log_event("entry_decision", mint=mint, score=score, fire=fire,
                           threshold=self.srv.entry_threshold,
                           exit_policy=EXIT_POLICY,
                           bet_sol=getattr(self.broker, "bet_sol", None),
                           midK=st.midK, vsK=st.vsK, vtK=st.vtK,
                           midV=st.midV, vsV=st.vsV, vtV=st.vtV,
                           entry_vs=entry_vs, entry_vt=entry_vt,
                           entry_anchor=("decision" if self.rich_entry_enabled else "k"),
                           n_at_ready=st.n, cum_buy_sol=st.cum_buy_sol,
                           first_seen_rsol=self.first_seen_rsol.get(mint),
                           k_window_last_ts=st.k_window_last_ts,
                           v_window_last_ts=st.v_window_last_ts,
                           ev_slot=getattr(ev, "slot", None),
                           features=ef,
                           sophistication=soph,
                           shred_signal=shred_sig)
            if fire:
                if mint in self.closed_mints:
                    self.stats["skipped_already_closed"] += 1
                elif SKIP_FIRE_VSOL_LAM and entry_vs >= SKIP_FIRE_VSOL_LAM:
                    # LATE-ENTRY SKIP (2026-06-12, catchable_edge.py on 157 fires):
                    # the edge is in EARLY entries -- decision vsK<55 SOL netted
                    # +0.108/fire 68% win, while vsK 55-80 netted -0.322 12% win
                    # (you're buying near the top of the curve, no room for +50%
                    # before it graduates/dumps). Live, these high-vsK fires also
                    # graduate-revert before we land. So skip them: removes a
                    # validated-negative subset, saves the revert fee, keeps the
                    # failure-rate breaker clean. Do NOT add to closed_mints
                    # (a later, earlier-vsK trigger on the same mint could be fine).
                    self.stats["skipped_late_entry"] = self.stats.get("skipped_late_entry", 0) + 1
                    self.log_event("late_entry_skip", mint=mint, score=score,
                                   entry_vs=entry_vs, vsol_sol=entry_vs / 1e9,
                                   threshold_lam=SKIP_FIRE_VSOL_LAM)
                    return
                else:
                    # Risk-limit / circuit-breaker gate BEFORE we open + buy
                    allow, refuse_reason = self._risk_check_before_fire()
                    if not allow:
                        self.log_event("risk_refusal", mint=mint, score=score,
                                       reason=refuse_reason)
                        # do NOT enter, do NOT add to closed_mints either (we may retry
                        # this same mint via a different fire later if risk clears)
                        return
                    self.stats["entry_fire"] += 1
                    self.recent_fire_ts.append(time.time())
                    # Kick off the broker meta fetch immediately: it overlaps
                    # the book/store/tip bookkeeping below and the same-second
                    # TP sell will share it via the broker's in-flight dedup.
                    self._prefetch_broker_meta(mint)
                    # Old artifacts enter at the K anchor. Rich June artifacts
                    # enter at the joint decision reserves, matching the offline
                    # tp*_net labels.
                    self.book.open(mint, vsK=entry_vs, vtK=entry_vt, vsC=entry_vs, vtC=entry_vt)
                    if self.rich_entry_enabled:
                        self.rich_entry_mid[mint] = entry_mid
                        self.rich_run_max_ret[mint] = 0.0
                    self.open_paper.add(mint)
                    # LIVE only: hold exits until the buy fill reconciles, then the
                    # exit ret is anchored to our real fill (see _on_buy_fill). In
                    # paper/dry-run there is no fill, so we leave the decision anchor.
                    if self.broker is not None and not getattr(self.broker, "dry_run", True):
                        self.awaiting_fill.add(mint)
                    # Notify the pluggable exit policy so it can cache anything
                    # that depends on entry-time features (e.g. RL routing
                    # classifier scores). Best-effort; never block on errors.
                    try:
                        self.exit_policy.on_entry(mint, ev, ef, score)
                    except Exception as e:
                        self.log_event("exit_policy_entry_error", mint=mint,
                                       err=str(e), policy=self.exit_policy.NAME)
                    if self.position_store is not None:
                        self.position_store.record_open(mint, vsK=entry_vs, vtK=entry_vt,
                                                        vsC=entry_vs, vtC=entry_vt,
                                                        score=score)
                    # Decide on a Jito-tip override based on the shred signal.
                    # If snipers are visibly forming on this mint in the
                    # pre-execution window, we bump the tip so our bundle
                    # competes for the same slot. Conservative thresholds
                    # for now; tunable as we collect data.
                    # Adaptive tip: outbid the VISIBLE competition, capped by
                    # marginal value. The replay latency gradient prices one
                    # trade of latency at ~24M lam on 0.1 SOL bets; the p99
                    # cluster tip observed in the shred stream is ~4M. Paying
                    # up to 5M lam when a real cluster forms costs <=10% of the
                    # expected edge per bet. (2026-06-10; dry-run logged only
                    # until armed.)
                    tip_override = None
                    tip_tier = 0
                    base_tip = getattr(self.broker, "tip_lamports", 100_000)
                    sn_buy_500  = shred_sig.get("shred_buy_500ms", 0)
                    sn_buy_2k   = shred_sig.get("shred_buy_2000ms", 0)
                    sn_jito_pct = shred_sig.get("shred_jito_tip_rate_2000ms", 0.0)
                    p90_tip     = shred_sig.get("shred_jito_tip_p90_2000ms", 0) or 0
                    # Data-driven tip (tools/tip_model.py 2026-06-11): replaces base*2/base*4 with
                    # contention-mapped floors from the competing jito-tip distribution (p75~200k,
                    # p90~1.5M; only ~10% of the field even uses a jito tip). Bid the data floor
                    # for the observed contention AND above the visible p90.
                    # 2026-06-12 RECALIBRATED at 0.05 bet from the first live fire: a tier-2
                    # 5M tip was 65% of a -0.0077 SOL loss on a hot farm we entered post-pump.
                    # The 21k-intent study says tip ~= tax (speed wins the slot; we landed
                    # slot_gap=0), and 5M is 10% of a 0.05 bet. Cap cut 5M->800k, floors
                    # 1.5M->300k / 400k->150k, and drop the outbid-visible-p90 term (it chased
                    # a 12M competitor into the toxic high-contention farms = the net-negative
                    # fires). Scale the cap with bet size so it stays ~<=2% of the bet.
                    _bet = float(getattr(self.broker, "bet_sol", 0.05) or 0.05)
                    _tip_cap = max(200_000, min(800_000, int(_bet * 1e9 * 0.02)))
                    if sn_buy_500 >= 4 or sn_buy_2k >= 8:
                        tip_tier = 2; tip_floor = 300_000        # hot cluster
                    elif (sn_buy_500 >= 2 and sn_jito_pct > 0) or sn_buy_2k >= 3:
                        tip_tier = 1; tip_floor = 150_000        # mild
                    if tip_tier > 0:
                        tip_override = min(max(base_tip, tip_floor), _tip_cap)
                    if tip_override is not None:
                        self.log_event("front_run_tip_bump", mint=mint,
                                       base_tip_lam=base_tip,
                                       override_tip_lam=tip_override,
                                       tier=tip_tier,
                                       p90_visible_tip_lam=p90_tip,
                                       shred_sig=shred_sig)
                    if self.broker is not None:
                        try:
                            # broker.buy needs entry AMM reserves to compute token
                            # amount + slippage. Use the same reserves as the
                            # offline model entry anchor.
                            slot = getattr(ev, "slot", None)
                            buy_sol = float(getattr(self.broker, "bet_sol", 1.0))
                            await self.broker.buy(mint, sol=buy_sol,
                                                  vsol_lam=int(entry_vs), vtok=int(entry_vt),
                                                  slot=slot,
                                                  tip_lamports_override=tip_override)
                            self.stats["broker_calls"] += 1
                        except Exception as e:
                            self.log_event("broker_error", mint=mint, op="buy", err=str(e))
        elif result == "fwd" and mint in self.open_paper:
            self.fwd_counter[mint] += 1
            # Snap cadence guard. With SNAP_EVERY=3, intended sequence is
            # fwd=1,4,7,... (`% 3 == 1`). With SNAP_EVERY=1 the intent is
            # "snap every event", but `n % 1 == 1` is impossible (anything
            # mod 1 = 0), so the original condition silently disabled the
            # entire snap path when snap_every=1 was set in config.yaml.
            # Result: pos.vsC/vtC stuck at entry values, snaps_ret_vs_midV
            # stayed empty, the exit policy never got consulted, and every
            # position closed via stale watchdog with exit_ret = 0 exactly.
            # Same guard is used in tools/extract_from_capture.py for the
            # offline path.
            if SNAP_EVERY == 1 or self.fwd_counter[mint] % SNAP_EVERY == 1:
                pf = st.path_features(ev.virtual_sol_reserves, ev.virtual_token_reserves)
                if self.rich_entry_enabled and mint in self.rich_entry_mid:
                    pf, self.rich_run_max_ret[mint] = decision_path_features(
                        self.rich_entry_mid[mint],
                        self.rich_run_max_ret[mint],
                        ev.virtual_sol_reserves,
                        ev.virtual_token_reserves,
                        pf,
                    )
                kef = dict(zip(ENTRY_FEATURE_NAMES_K, st.k_entry_features()))
                p_rec = 1.0
                if pf["ret"] < 0:
                    p_rec, _ = self.srv.score_recovery(kef, pf)
                self.snap_count[mint] += 1
                fwd_n = self.snap_count[mint]
                self.book.add_snapshot(mint, fwd_n,
                                       ev.virtual_sol_reserves, ev.virtual_token_reserves,
                                       pf["ret"], p_rec, t=time.time())
                self.stats["snaps_routed"] += 1
                if self.position_store is not None:
                    self.position_store.record_snap(mint, fwd_n,
                                                     ev.virtual_sol_reserves,
                                                     ev.virtual_token_reserves,
                                                     pf["ret"], p_rec)
                self.log_event("path_snap", mint=mint, fwd=fwd_n, p_rec=p_rec,
                               vs=ev.virtual_sol_reserves, vt=ev.virtual_token_reserves,
                               path_feats=pf, ev_slot=getattr(ev, "slot", None))
                if (self.shadow_recovery is not None and pf["ret"] < 0
                        and mint not in self.shadow_cut_flagged):
                    try:
                        _vec = [[pf[k] for k in ("ret", "run_max_ret", "dd", "fill_k",
                                                 "buy_frac_w", "nsell_w", "solo_sell_w",
                                                 "vel_w", "dts")]
                                + list(st.k_entry_features())]
                        _psh = float(self.shadow_recovery.predict_proba(_vec)[0, 1])
                        if _psh < 0.20:
                            if len(self.shadow_cut_flagged) > 5000:
                                self.shadow_cut_flagged.clear()
                            self.shadow_cut_flagged.add(mint)
                            self.log_event("shadow_death_cut", mint=mint, p_rec=_psh,
                                           ret=pf["ret"], dd=pf["dd"], fwd=fwd_n, thr=0.20)
                    except Exception:
                        pass
                pos = self.book.positions.get(mint)
                if pos:
                    pos.vsC = ev.virtual_sol_reserves
                    pos.vtC = ev.virtual_token_reserves
                # ---- LIVE exit dispatch by cfg.exit.policy ----
                # Supports: k_combined (4 paced + 4 forced-time), h_time_spaced
                # (8 slices at fixed gaps), b_frontload (sell on every profitable
                # snap, 8 slices), c_hybrid_t30 / f_hybrid_t50 (paced de-risk +
                # trailing stop). Each policy uses the SAME risk machinery, death
                # cut, broker, holdings tracking — only the trigger pattern differs.
                # Fill-anchored exit (LIVE): hold while the buy fill isn't known,
                # then judge the exit on ret measured from our REAL fill mid, not
                # the decision mid. exit_pf is a copy so the PaperBook/logging keep
                # the decision-anchored pf untouched.
                exit_pf = pf
                _anchor = self.live_exit_anchor_mid.get(mint)
                if _anchor and _anchor > 0 and ev.virtual_token_reserves:
                    _cur_mid = ev.virtual_sol_reserves / ev.virtual_token_reserves
                    exit_pf = dict(pf)
                    exit_pf["ret"] = _cur_mid / _anchor - 1.0
                if (self.broker is not None and mint not in self.live_dead
                        and mint not in self.awaiting_fill):
                    slot = getattr(ev, "slot", None)
                    vsol_i = int(ev.virtual_sol_reserves)
                    vtok_i = int(ev.virtual_token_reserves)
                    now = time.time()
                    n_sold = self.live_slices_sold[mint]
                    run_max = exit_pf["run_max_ret"]
                    if exit_pf["ret"] < 0 and p_rec < self.srv.death_threshold:
                        # DEATH CUT — anywhere in lifecycle
                        self.live_dead.add(mint)
                        self.stats["live_death_cuts"] += 1
                        try:
                            await self.broker.sell_all(mint, vsol_lam=vsol_i, vtok=vtok_i, slot=slot)
                            self.stats["broker_calls"] += 1
                            self.log_event("live_death_cut", mint=mint, fwd=fwd_n,
                                           phase=("derisk" if n_sold < DERISK_SLICES else "runner"),
                                           n_sold=n_sold, ret=exit_pf["ret"], p_rec=p_rec,
                                           run_max=run_max, slot=slot)
                        except Exception as e:
                            self.log_event("broker_error", mint=mint, op="sell_all_death", err=str(e))
                        # Formally close the PaperBook position so the dashboard
                        # stops showing post-crash stale ret. We've exited live;
                        # let the analytical book record realize the position.
                        pos = self.book.positions.get(mint)
                        if pos is not None and not pos.closed:
                            self.book._close_one(pos)
                            exit_ret_val = pf["ret"]
                            live_net = self._compute_live_policy_net(pos)
                            if self.position_store is not None:
                                self.position_store.record_close(
                                    mint, pos.net_return, pos.kind, reason="death_cut",
                                    exit_ret=exit_ret_val,
                                    live_policy_net=live_net,
                                    live_policy_name=self.exit_policy.NAME)
                            self.log_event("position_close", mint=mint, exit_kind=pos.kind,
                                           net=pos.net_return, reason="death_cut",
                                           exit_ret=exit_ret_val,
                                           live_policy_net=live_net,
                                           live_policy_name=self.exit_policy.NAME)
                            self.open_paper.discard(mint)
                            self.closed_mints.add(mint)
                            try: self.exit_policy.on_close(mint)
                            except Exception: pass
                    else:
                        await self._dispatch_exit_slice(mint, n_sold, now, exit_pf, run_max,
                                                        p_rec, vsol_i, vtok_i, slot, fwd_n)

    async def stale_watchdog(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = time.time()
            to_close = [m for m in list(self.open_paper)
                        if now - self.last_trade_ts.get(m, now) > STALE_SEC]
            for mint in to_close:
                pos = self.book.positions.get(mint)
                if pos is None: continue
                self.book._close_one(pos)
                exit_ret_val = _pos_exit_ret(pos)
                live_net = self._compute_live_policy_net(pos)
                self.log_event("position_close", mint=mint, exit_kind=pos.kind,
                               net=pos.net_return, reason="stale", exit_ret=exit_ret_val,
                               live_policy_net=live_net,
                               live_policy_name=self.exit_policy.NAME)
                if self.position_store is not None:
                    self.position_store.record_close(mint, pos.net_return, pos.kind, "stale",
                                                     exit_ret=exit_ret_val,
                                                     live_policy_net=live_net,
                                                     live_policy_name=self.exit_policy.NAME)
                if self.broker is not None and pos.kind in ("rider", "cut", "hold"):
                    try:
                        # use the current AMM state (vsC/vtC = last seen reserves)
                        await self.broker.sell_all(mint,
                                                    vsol_lam=int(pos.vsC), vtok=int(pos.vtC),
                                                    slot=None)
                        self.stats["broker_calls"] += 1
                    except Exception as e:
                        self.log_event("broker_error", mint=mint, op="sell_all", err=str(e))
                self.closed_mints.add(mint)
                self.open_paper.discard(mint)
                self.stats["stale_close"] += 1
                try: self.exit_policy.on_close(mint)
                except Exception: pass

    async def listener(self) -> None:
        program = Pubkey.from_string(config.PUMP_FUN_PROGRAM)
        ws_url = config.rpc_ws_url()
        masked = ws_url.split("api-key=")[0] + "api-key=***" if "api-key=" in ws_url else ws_url
        print(f"[shadow] connecting {masked}", flush=True)
        while True:
            try:
                async with connect(ws_url) as ws:
                    await ws.logs_subscribe(RpcTransactionLogsFilterMentions(program),
                                            commitment=Confirmed)
                    await ws.recv()
                    print(f"[shadow] subscribed", flush=True)
                    while True:
                        msg = await ws.recv()
                        if not isinstance(msg, list) or not msg: continue
                        item = msg[0]
                        val = getattr(item.result, "value", None)
                        if val is None or val.err is not None: continue
                        self.stats["events"] += 1
                        for ln in val.logs:
                            ev = parse_program_data_line(ln)
                            if ev: await self.on_trade(ev)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[shadow] stream error: {exc}; reconnect in 3s", flush=True)
                await asyncio.sleep(3)

    async def stats_printer(self) -> None:
        while True:
            await asyncio.sleep(15)
            rets = self.book.returns()
            n = len(rets)
            elapsed = time.time() - self.start_ts
            print(f"[shadow] {elapsed:.0f}s  evt={self.stats['events']}  trades={self.stats['trade_events']}  "
                  f"fresh={self.stats['fresh']}  k={self.stats['k_fires']}  v={self.stats['v_fires']}  "
                  f"ready={self.stats['both_ready']}  fired={self.stats['entry_fire']}  "
                  f"open={len(self.open_paper)}  closed={n}  "
                  f"mean={rets.mean() if n else 0:+.3f}  win={100*(rets>0).mean() if n else 0:.0f}%",
                  flush=True)

    async def run(self) -> None:
        # Attach to the shred intent SHM ring (best-effort; ring may not
        # exist if intent_recorder isn't running). On success, the
        # ShredWindow background task drains records into a per-mint deque
        # that we query at entry-trigger time for front-run signal.
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "shred_bot"))
            from shred_window import ShredWindow
            self.shred_window = ShredWindow()
            self.shred_window.start()
        except Exception as e:
            print(f"[shadow] shred_window unavailable ({e}); "
                  f"continuing without front-run signal", flush=True)
            self.shred_window = None
        await asyncio.gather(self.listener(), self.stale_watchdog(),
                             self.stats_printer(), self._checkpoint_loop())


def _sophistication_summary(rows: list, k_window: int = 7) -> dict:
    """Summarize buyer sophistication over the last `k_window` BUY trades.
    Each row is a dict like {"is_buy": bool, "user": str, "fee_lam": int,
    "cu": int, "n_inner_ix": int, "n_keys": int, "route": str|None,
    "jito_tip_idx": int|None, "jito_tip_lam": int|None}.

    Returned dict is JSON-safe and bounded in size. Keys:
      n_buy_in_kwin        how many buys were captured in the window
      fee_p50, fee_p90     priority+base fee distribution
      cu_p50, cu_mean      compute units used
      jito_tip_rate        fraction of buys with a Jito tip account in tx
      jito_tip_p50_lam     median tip lamports among Jito buys
      routed_rate          fraction of buys routed via aggregator (jupiter/etc)
      route_top            most common route name (or null)

    None / missing fields are skipped (older capture without extras). Returns
    an empty dict when the window has no buys with extras — caller can decide
    what to log."""
    buys = [r for r in rows if r.get("is_buy")]
    if not buys: return {}
    buys = buys[-k_window:]
    fees = [r["fee_lam"] for r in buys if r.get("fee_lam") is not None]
    cus  = [r["cu"]      for r in buys if r.get("cu")      is not None]
    tips = [r["jito_tip_lam"] for r in buys if r.get("jito_tip_lam") is not None]
    n_tipped = sum(1 for r in buys if r.get("jito_tip_idx") is not None)
    routes = [r["route"] for r in buys if r.get("route")]
    summary = {"n_buy_in_kwin": len(buys)}
    if fees:
        q = sorted(fees)
        summary["fee_p50_lam"] = q[len(q)//2]
        summary["fee_p90_lam"] = q[int(len(q)*0.9)] if len(q) > 1 else q[-1]
        summary["fee_max_lam"] = q[-1]
    if cus:
        summary["cu_p50"] = sorted(cus)[len(cus)//2]
        summary["cu_mean"] = sum(cus) // len(cus)
    summary["jito_tip_rate"] = round(n_tipped / len(buys), 3)
    if tips:
        summary["jito_tip_p50_lam"] = sorted(tips)[len(tips)//2]
    summary["routed_rate"] = round(len(routes) / len(buys), 3)
    if routes:
        from collections import Counter
        summary["route_top"] = Counter(routes).most_common(1)[0][0]
    # inner ix / account_keys averages — proxies for tx complexity
    n_inner = [r["n_inner_ix"] for r in buys if r.get("n_inner_ix") is not None]
    if n_inner:
        summary["n_inner_ix_mean"] = round(sum(n_inner) / len(n_inner), 1)
    n_keys = [r["n_keys"] for r in buys if r.get("n_keys") is not None]
    if n_keys:
        summary["n_keys_mean"] = round(sum(n_keys) / len(n_keys), 1)
    return summary


def _pos_exit_ret(pos) -> float | None:
    """Last observed AMM ret for `pos` — preferring the recorded snap stream,
    falling back to deriving it from vsC/vtC (current AMM reserves) vs vsK/vtK
    (entry K=7 reserves). The fallback covers edge cases where the in-memory
    snap list is empty even though we know the live current AMM state.
    Returns None only if BOTH sources are missing.
    """
    if getattr(pos, "snaps_ret_vs_midV", None):
        return pos.snaps_ret_vs_midV[-1]
    try:
        # ret = (vsC/vtC) / (vsK/vtK) - 1
        if pos.vsC and pos.vtC and pos.vsK and pos.vtK:
            return (pos.vsC / pos.vtC) / (pos.vsK / pos.vtK) - 1.0
    except Exception:
        pass
    return None


async def amain():
    h = ShadowHarness()
    stop = asyncio.Event()
    def shutdown(*_): stop.set()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, shutdown)
        except NotImplementedError: pass
    runner = asyncio.create_task(h.run())
    closer = asyncio.create_task(stop.wait())
    done, pending = await asyncio.wait([runner, closer], return_when=asyncio.FIRST_COMPLETED)
    for t in pending: t.cancel()
    # final close: close any still-open positions and dump stats
    for m in list(h.open_paper):
        pos = h.book.positions.get(m)
        if pos is not None and not pos.closed:
            h.book._close_one(pos)
            exit_ret_val = _pos_exit_ret(pos)
            live_net = h._compute_live_policy_net(pos)
            h.log_event("position_close", mint=m, exit_kind=pos.kind,
                        net=pos.net_return, reason="shutdown", exit_ret=exit_ret_val,
                        live_policy_net=live_net,
                        live_policy_name=h.exit_policy.NAME)
            if h.position_store is not None:
                h.position_store.record_close(m, pos.net_return, pos.kind, "shutdown",
                                              exit_ret=exit_ret_val,
                                              live_policy_net=live_net,
                                              live_policy_name=h.exit_policy.NAME)
            try: h.exit_policy.on_close(m)
            except Exception: pass
    rets = h.book.returns()
    h.log_event("shutdown", stats=h.stats, n_closed=len(rets),
                mean_net=float(rets.mean()) if len(rets) else None,
                win_pct=float(100*(rets>0).mean()) if len(rets) else None)
    print(f"[shadow] shutdown — closed={len(rets)} mean={rets.mean() if len(rets) else 0:+.3f}",
          flush=True)
    h.log.close()


if __name__ == "__main__":
    asyncio.run(amain())
