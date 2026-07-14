"""Broker layer for the bot.

PaperBroker:  no-op, logs intent only. Default. Used in paper-mode.
JitoBroker:   assembles pump.fun buy/sell bundles, signs with the wallet, and (when
              JITO_DRY_RUN=0) POSTs to Jito Frankfurt block engine.

Hot path: on_trade calls broker.buy(...) which spawns an async task and returns
immediately, so the listener loop never blocks on bundle submission. The task
itself reads the cached blockhash, builds the ix via pump_fun_ix, assembles a
versioned tx, signs, and either logs (DRY_RUN) or POSTs.

Token-holdings tracking: when a buy task completes we record the expected token
amount (from AMM math at submission time) in `self.holdings[mint]`. Sells use
this as the amount to sell. This is an APPROXIMATION — real fill may differ due
to slippage from other concurrent buyers in the same slot. For paper mode we
ignore the difference; for live we'd need to reconcile against chain balance via
getTokenAccountBalance after each fill (TODO before live).

Same-block aim: when the trigger event carries a slot (gRPC mode), the broker
records target slot + submission slot estimate in the log so we can measure how
often we land in the same slot as detection. In WS mode slot is None and we
implicitly aim for "next-slot or sooner".

FOUR GATES TO ACTUALLY FIRE A REAL BUNDLE:
  1) ShadowHarness was constructed with broker=JitoBroker(...) (not PaperBroker)
  2) pumpfun_bot.py was started with --live  (refuses without)
  3) env PUMPFUN_LIVE_OK=1                   (refuses without)
  4) env JITO_DRY_RUN=0                      (DRY_RUN=1 = log, don't POST)
The wallet also has to have enough SOL for the bet + tip + signature fees. With
the test wallet at zero capital Jito will return InsufficientFundsForRent or
similar; that still validates the signing+submission path without risk.
"""
from __future__ import annotations
import asyncio, json, os, time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, Awaitable


# Sell-failure recovery: a failed sell_all is re-submitted with fresh reserves and
# escalating slippage, ending in a market sell, so we never hold a dumping bag.
MAX_SELL_RETRIES = 4
SELL_RETRY_SLIPPAGE_BPS = [3000, 5000, 8000]  # attempts 1,2,3; attempt 4 = market


@dataclass
class PendingBundle:
    """In-flight bundle awaiting on-chain confirmation. Lives in
    JitoBroker.pending_bundles until reconciled (landed / failed / expired)."""
    sig: str                       # base58 tx signature (computed locally on sign)
    mint: str
    op: str                        # 'buy' / 'sell_slice' / 'sell_all'
    tok_delta: int                 # signed: +tokens for buy, -tokens for sell
    tip_lamports: int
    submitted_t: float
    bet_sol: float = 0.0
    expected_sol_out: int = 0      # for sells
    expected_sol_cost: int = 0     # for buys
    status: str = "pending"        # pending / landed / failed / expired
    landed_t: Optional[float] = None
    landed_slot: Optional[int] = None
    retry_count: int = 0
    # Detection-time slots, carried so the recon record alone tells the latency
    # story: target_slot = the slot we DETECTED at (gRPC event slot), bh_slot =
    # the slot of the blockhash we signed with. landed_slot - target_slot is
    # THE land-in-decision-slot metric (0 = same slot, 1 = next slot); the
    # armed-phase deliverable. Without these, that metric needs a fragile
    # cross-file join on signature.
    target_slot: Optional[int] = None
    bh_slot: Optional[int] = None
    # Reserves at submit, kept so the sell-retry ladder can still price (or
    # market-sell) when the eRPC getAccountInfo curve refetch is down — the
    # outage that motivated the gRPC-native reconcile (2026-06-12).
    vsol_submit: int = 0
    vtok_submit: int = 0


class PaperBroker:
    """No-op broker for paper mode. Logs intents to a JSONL for diagnostic completeness."""
    def __init__(self, log_path: str = "broker_paper.jsonl"):
        self.fh = open(log_path, "a", buffering=1)
        self.holdings: dict[str, int] = {}

    async def buy(self, mint: str, sol: float, vsol_lam: int, vtok: int,
                  slot: int | None = None) -> None:
        self.fh.write(json.dumps({"t": time.time(), "op": "buy", "mint": mint,
                                  "sol": sol, "slot": slot}) + "\n")

    async def sell_all(self, mint: str, vsol_lam: int, vtok: int,
                       slot: int | None = None) -> None:
        self.fh.write(json.dumps({"t": time.time(), "op": "sell_all", "mint": mint,
                                  "slot": slot}) + "\n")

    async def sell_slice(self, mint: str, frac: float, vsol_lam: int, vtok: int,
                         slot: int | None = None) -> None:
        self.fh.write(json.dumps({"t": time.time(), "op": "sell_slice", "mint": mint,
                                  "frac": frac, "slot": slot}) + "\n")

    def close(self) -> None:
        try: self.fh.close()
        except Exception: pass


class JitoBroker:
    """Live broker. Assembles + (optionally) submits pump.fun bundles via Jito Frankfurt.

    Construct via JitoBroker.create() — needs to await the wallet load and
    blockhash-cache start. Constructor is sync, so use the create() classmethod.
    """
    def __init__(self, wallet, fee_recipient_str: str,
                 get_blockhash: Callable[[], Awaitable],
                 bet_sol: float | None = None,
                 dry_run: bool | None = None,
                 tip_lamports: int | None = None,
                 slippage_bps: int | None = None,
                 log_path: str = "logs/broker_jito.jsonl"):
        from pathlib import Path as _P
        _P(log_path).parent.mkdir(parents=True, exist_ok=True)
        # Load config-driven defaults if not explicitly passed
        slippage_bps_buy = slippage_bps_sell = None
        try:
            from bot_config import cfg as _C
            if bet_sol is None:       bet_sol       = _C.bot.bet_sol
            if dry_run is None:       dry_run       = _C.broker.jito_dry_run
            if tip_lamports is None:  tip_lamports  = _C.broker.tip_lamports
            if slippage_bps is None:  slippage_bps  = _C.broker.slippage_bps
            slippage_bps_buy  = getattr(_C.broker, "slippage_bps_buy",  None)
            slippage_bps_sell = getattr(_C.broker, "slippage_bps_sell", None)
        except Exception:
            if bet_sol is None:       bet_sol       = 1.0
            if dry_run is None:       dry_run       = True
            if tip_lamports is None:  tip_lamports  = 100_000
            if slippage_bps is None:  slippage_bps  = 1500
        from solders.pubkey import Pubkey
        self.wallet = wallet
        self.user_pk = wallet.pubkey()
        self.fee_recipient = Pubkey.from_string(fee_recipient_str)
        self.get_blockhash = get_blockhash
        self.bet_sol = bet_sol
        self.dry_run = dry_run
        self.tip_lamports = tip_lamports
        self.slippage_bps = slippage_bps
        # Split slippage (2026-06-11). The ENTRY cap is a SAFETY BOUND on
        # worst-case spend (max_sol_cost = bet * (1 + bps/1e4)), NOT an EV
        # optimizer: exec_sim showed winner and loser fires have the same
        # entry-slip distribution (p50 +0.85, max +1.34), so any cap tight
        # enough to bind rejects winners pro-rata (1500bps kept 6/157 fires,
        # total net NEGATIVE). 20000bps keeps every observed fire with ~50%
        # headroom and bounds spend at 3x bet. Sells keep the tighter
        # protection; the validated retry ladder (3000/5000/8000 -> market)
        # handles reverts. Falls back to the legacy shared slippage_bps if
        # the split keys are absent.
        # Per-bot ENV override (wins over config). The continuation bot sets
        # BROKER_SLIPPAGE_BPS_BUY in its own systemd drop-in so its tight entry
        # cap (15% = bound a 0.1 bet to <=0.115 spend) does NOT alter the shared
        # launch-bot config (which deliberately runs 20000bps per exec_sim).
        _eb = os.getenv("BROKER_SLIPPAGE_BPS_BUY")
        if _eb is not None and _eb.strip() != "":
            slippage_bps_buy = int(_eb)
        _es = os.getenv("BROKER_SLIPPAGE_BPS_SELL")
        if _es is not None and _es.strip() != "":
            slippage_bps_sell = int(_es)
        self.slippage_bps_buy  = int(slippage_bps_buy)  if slippage_bps_buy  is not None else int(slippage_bps)
        self.slippage_bps_sell = int(slippage_bps_sell) if slippage_bps_sell is not None else int(slippage_bps)
        # EXIT tip (2026-06-12, BEjwYub4 + MB directive): sells pay a HIGHER tip
        # than buys. Buy speed only wins slot INCLUSION, not a better price (H7M1L
        # landed slot_gap=0 with a 5M tip and still filled +116% behind in-slot
        # whales). But SELL speed directly saves money on a dump: BEjwYub4's
        # stop-sell landed slot_gap=4 (100k tip, congested dump slot) and the
        # price fell 0.065->0.0168 in those 4 slots -- the slow exit doubled the
        # loss to -0.096. A fat exit tip lands the panic-sell next block. ~0.0015
        # SOL/sell (3% of a 0.05 bet); cheap insurance against the dump tail.
        try:
            from bot_config import cfg as _Cet
            self.exit_tip_lamports = int(getattr(_Cet.broker, "exit_tip_lamports", 1_500_000))
        except Exception:
            self.exit_tip_lamports = 1_500_000
        # LOW base tip for the INITIAL (calm) sell; the panic-retry ladder escalates to exit_tip_lamports.
        # The priority fee (the real race lever) lands a calm exit -> reserve the fat tip for dumps only.
        self.exit_tip_base = int(os.getenv("CONT_EXIT_TIP_BASE", "150000"))
        try:
            from bot_config import cfg as _Cpf
            self.priority_fee_micro = int(getattr(_Cpf.broker, "priority_fee_micro", 2_000_000))
            self.cu_limit = int(getattr(_Cpf.broker, "cu_limit", 200_000))
        except Exception:
            self.priority_fee_micro, self.cu_limit = 2_000_000, 200_000
        # Per-bot ENV override (continuation drop-in) so the operator's intended priority
        # fee / CU actually drive on-chain spend, instead of the shared launch-bot config.
        _pf = os.getenv("BROKER_PRIORITY_FEE_MICRO")
        if _pf and _pf.strip(): self.priority_fee_micro = int(_pf)
        _cu = os.getenv("BROKER_CU_LIMIT")
        if _cu and _cu.strip(): self.cu_limit = int(_cu)
        self.holdings: dict[str, int] = {}                  # mint -> expected raw token balance
        self.bg_tasks: set[asyncio.Task] = set()
        self.fh = open(log_path, "a", buffering=1)
        # Reconciliation state
        self.pending_bundles: dict[str, PendingBundle] = {}  # sig -> bundle
        self._mint_meta: dict[str, tuple] = {}   # mint -> (token_program, creator)
        self._meta_tasks: dict[str, asyncio.Task] = {}   # mint -> in-flight meta fetch
        self._rpc = None                          # shared AsyncClient (lazy, persistent)
        self.failure_callback: Optional[Callable[[str, str, str], None]] = None
        self.fill_callback: Optional[Callable[[str, float], None]] = None
        recon_path = str(Path(log_path).parent / "broker_recon.jsonl")
        self.recon_fh = open(recon_path, "a", buffering=1)
        self.reconcile_task: Optional[asyncio.Task] = None
        # Track tip-vs-landing stats for later analysis (capped circular buffer)
        self.recent_outcomes: deque = deque(maxlen=500)
        self.realized_pnl_lam = 0      # real realized P&L (paired buy+sell), LIVE
        self._buy_cost_lam: dict[str, int] = {}   # mint -> buy lamport delta (open)
        print(f"[jito] wallet={self.user_pk}  bet={bet_sol}  tip_lam={tip_lamports}  "
              f"dry_run={dry_run}  slippage_bps buy={self.slippage_bps_buy} "
              f"sell={self.slippage_bps_sell}", flush=True)

    @classmethod
    async def create(cls, *, bet_sol: float = 1.0,
                     dry_run: bool | None = None) -> "JitoBroker":
        """Async factory. Loads wallet, starts blockhash cache, returns broker."""
        import config
        from solders.keypair import Keypair
        if not config.wallet_configured():
            raise RuntimeError("wallet not configured (.env WALLET_PRIVATE_KEY missing)")
        # Try base58 first (most common), then JSON byte-array
        wpk = config.WALLET_PRIVATE_KEY.strip()
        wallet = None
        try:
            wallet = Keypair.from_base58_string(wpk)
        except Exception:
            try:
                arr = json.loads(wpk)
                wallet = Keypair.from_bytes(bytes(arr))
            except Exception as e:
                raise RuntimeError(f"could not parse WALLET_PRIVATE_KEY: {e}")
        # Fee recipient — pump.fun current fee_recipient address. Override via env if needed.
        fee_str = os.getenv("PUMPFUN_FEE_RECIPIENT",
                            "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
        # Blockhash cache: start the background loop and use the FRESH-ON-DEMAND
        # getter (caps worst-case bh_age regardless of asyncio scheduling stalls).
        # Background loop still runs at poll_s cadence (cheap RPC, keeps cache
        # warm); the getter only blocks when cache is stale > max_age_ms.
        from blockhash_cache import blockhash_shadow_loop, get_cached_blockhash, get_fresh_blockhash
        bh_task = asyncio.create_task(blockhash_shadow_loop(), name="blockhash-cache")
        # Give the cache one cycle to warm up (use the plain cached getter here —
        # the broker hasn't started yet so no scheduler pressure)
        for _ in range(40):
            entry = await get_cached_blockhash()
            if entry is not None: break
            await asyncio.sleep(0.1)
        if entry is None:
            print("[jito] WARNING — blockhash cache failed to warm in 4s; "
                  "bundles will queue until it does", flush=True)
        # Configurable freshness cap. Default 500ms — if cached bh older than
        # that, refresh synchronously before assembling the bundle. Lower =
        # bh_age tighter but more on-demand refreshes; higher = more reliance
        # on the background loop staying scheduled. Tune via cfg.broker.bh_max_age_ms.
        try:
            from bot_config import cfg as _C
            bh_max_age_ms = float(getattr(_C.broker, "bh_max_age_ms", 500.0))
        except Exception:
            bh_max_age_ms = 500.0
        print(f"[jito] blockhash freshness cap: {bh_max_age_ms:.0f}ms "
              f"(stale -> on-demand RPC refresh before bundle assembly)",
              flush=True)
        async def _bh_fresh():
            return await get_fresh_blockhash(max_age_ms=bh_max_age_ms)
        dry = dry_run if dry_run is not None else os.getenv("JITO_DRY_RUN", "1") == "1"
        broker = cls(wallet=wallet, fee_recipient_str=fee_str,
                     get_blockhash=_bh_fresh,
                     bet_sol=bet_sol, dry_run=dry)
        broker._bh_task = bh_task   # keep reference so it isn't GC'd
        # Start the reconciler loop (only meaningful in LIVE mode, but harmless in DRY_RUN
        # because no bundles get added to pending_bundles in DRY_RUN).
        broker.reconcile_task = asyncio.create_task(broker._reconciler_loop(),
                                                     name="jito-reconciler")
        broker.holdings_reconcile_task = asyncio.create_task(
            broker._holdings_reconcile_loop(), name="jito-holdings-reconcile")
        # Warm the execution path off the hot path so fires never pay one-time
        # costs: a KEEPALIVE loop holds the pooled TLS to the block engine warm
        # between fires (20-30min gaps vs seconds-scale idle timeouts; first
        # iteration doubles as the startup warm), plus the shared RPC pool
        # (cold fetch measured 285ms vs 90ms warm) and the tip-account cache
        # (a sync SDK call on first use; do it in a thread).
        from jito_exec import jito_keepalive_loop, get_cached_tip_account
        async def _warm_rpc():
            try:
                cli = await broker._rpc_client()
                await cli.get_version()
            except Exception:
                pass
        for warm in (jito_keepalive_loop(),
                     _warm_rpc(),
                     asyncio.to_thread(get_cached_tip_account)):
            wt = asyncio.create_task(warm)
            broker.bg_tasks.add(wt); wt.add_done_callback(broker.bg_tasks.discard)
        return broker

    # ---------- Reconciliation: bundle landing + chain holdings ground truth ----------

    def set_failure_callback(self, fn: Callable[[str, str, str], None]) -> None:
        """Register a callback fn(mint, op_label, reason) called when a bundle FAILS to
        land. Used by ShadowHarness to roll back optimistic slice state so the exit
        policy can retry on the next forward snap."""
        self.failure_callback = fn

    def set_fill_callback(self, fn: Callable[[str, float], None]) -> None:
        """Register fn(mint, fill_mid) called when a BUY reconciles, with our realized
        fill mid (lamports per token-unit = curve SOL paid / tokens received). The
        harness anchors the live take-profit/stop to it (fill-anchored exits)."""
        self.fill_callback = fn

    def _emit_buy_fill(self, mint, sol_delta_lam, tip_lam, fee_lam, tok_delta) -> None:
        """Compute the realized fill mid from a landed BUY and notify the harness.
        curve_in = SOL that bought tokens = |wallet delta| - tip - base/priority fee
        (still includes the pump 1% fee = part of the cost basis). fill_mid =
        curve_in / tokens, same units as the AMM mid (vsol/vtok)."""
        try:
            cb = getattr(self, "fill_callback", None)
            if cb is None or sol_delta_lam is None or not tok_delta or tok_delta <= 0:
                return
            curve_in = (-int(sol_delta_lam)) - int(tip_lam or 0) - int(fee_lam or 0)
            if curve_in <= 0:
                return
            cb(mint, curve_in / float(tok_delta))
        except Exception as e:
            self._recon_log("fill_cb_err", mint=mint, err=str(e)[:100])

    def _recon_log(self, kind: str, **kw) -> None:
        try:
            self.recon_fh.write(json.dumps({"t": time.time(), "kind": kind, **kw},
                                            default=str) + "\n")
        except Exception:
            pass

    def _handle_bundle_failed(self, b: PendingBundle, reason: str) -> None:
        """Roll back the optimistic holdings reservation, fire the harness callback,
        log."""
        rolled = max(0, self.holdings.get(b.mint, 0) - b.tok_delta) if b.tok_delta > 0 else \
                  self.holdings.get(b.mint, 0) + abs(b.tok_delta)
        # tok_delta > 0 means BUY (we'd added expected tokens; subtract them back)
        # tok_delta < 0 means SELL (we'd subtracted tokens; add them back)
        prev = self.holdings.get(b.mint, 0)
        self.holdings[b.mint] = rolled
        b.status = "failed"
        self.recent_outcomes.append({"op": b.op, "tip": b.tip_lamports,
                                      "landed": False, "reason": reason,
                                      "age_s": time.time() - b.submitted_t})
        self._recon_log("failed", sig=b.sig, mint=b.mint, op=b.op, reason=reason,
                        tok_delta=b.tok_delta, tip_lam=b.tip_lamports,
                        holdings_before=prev, holdings_after=rolled,
                        age_s=time.time() - b.submitted_t)
        if self.failure_callback is not None:
            try:
                self.failure_callback(b.mint, b.op, reason)
            except Exception as e:
                self._recon_log("callback_err", err=str(e))

    def _handle_bundle_landed(self, b: PendingBundle, slot: int) -> None:
        b.status = "landed"
        b.landed_t = time.time()
        b.landed_slot = slot
        latency_s = b.landed_t - b.submitted_t
        self.recent_outcomes.append({"op": b.op, "tip": b.tip_lamports,
                                      "landed": True, "latency_s": latency_s,
                                      "slot": slot})
        _tgt = getattr(b, "target_slot", None)
        self._recon_log("landed", sig=b.sig, mint=b.mint, op=b.op,
                        landed_slot=slot, target_slot=_tgt,
                        bh_slot=getattr(b, "bh_slot", None),
                        slot_gap=(slot - _tgt) if _tgt else None,
                        latency_s=latency_s,
                        tip_lam=b.tip_lamports, tok_delta=b.tok_delta)

    async def _log_actual_fill(self, cli, sig_str, b):
        """Best-effort: pull the landed tx meta and log the ACTUAL on-chain fill
        for our wallet (token delta, lamport delta, fee) vs the expected quote,
        so we can measure REAL slippage. Never raises into the reconciler; if it
        fails we still have the signature for post-hoc reconciliation. Closes the
        'reconcile against chain after each fill' TODO. LIVE-only in practice
        (DRY_RUN never submits, so pending_bundles is empty)."""
        try:
            from solders.signature import Signature
            resp = await asyncio.wait_for(
                cli.get_transaction(Signature.from_string(sig_str),
                                    encoding="jsonParsed",
                                    commitment="confirmed",
                                    max_supported_transaction_version=0),
                timeout=5.0)
            tx = getattr(resp, "value", None)
            meta = getattr(getattr(tx, "transaction", None), "meta", None) if tx is not None else None
            if meta is None:
                self._recon_log("fill_no_meta", sig=sig_str, mint=b.mint, op=b.op,
                                landed_slot=getattr(b, "landed_slot", None))
                return
            owner = str(self.user_pk)
            def _tok_sum(balances):
                tot = 0
                for tb in (balances or []):
                    try:
                        if str(getattr(tb, "owner", "")) == owner and str(getattr(tb, "mint", "")) == b.mint:
                            amt = getattr(tb, "ui_token_amount", None)
                            tot += int(getattr(amt, "amount", 0) or 0)
                    except Exception:
                        pass
                return tot
            pre_tok = _tok_sum(getattr(meta, "pre_token_balances", None))
            post_tok = _tok_sum(getattr(meta, "post_token_balances", None))
            pre_bal = list(getattr(meta, "pre_balances", []) or [])
            post_bal = list(getattr(meta, "post_balances", []) or [])
            sol_delta_lam = (int(post_bal[0]) - int(pre_bal[0])) if (pre_bal and post_bal) else None
            fee_lam = int(getattr(meta, "fee", 0) or 0)
            if sol_delta_lam is not None:
                if b.op == "buy":
                    self._buy_cost_lam[b.mint] = int(sol_delta_lam)
                    self._emit_buy_fill(b.mint, sol_delta_lam, b.tip_lamports, fee_lam,
                                        post_tok - pre_tok)
                elif b.op in ("sell_all", "sell_slice"):
                    self.realized_pnl_lam += int(sol_delta_lam) + self._buy_cost_lam.pop(b.mint, 0)
            _land = getattr(b, "landed_slot", None)
            _tgt = getattr(b, "target_slot", None)
            self._recon_log("fill", sig=sig_str, mint=b.mint, op=b.op,
                            landed_slot=_land,
                            target_slot=_tgt,
                            bh_slot=getattr(b, "bh_slot", None),
                            slot_gap=((_land - _tgt) if (_land and _tgt) else None),
                            actual_tok_delta=(post_tok - pre_tok),
                            expected_tok_delta=getattr(b, "tok_delta", None),
                            actual_sol_delta_lam=sol_delta_lam,
                            expected_sol_out=getattr(b, "expected_sol_out", 0),
                            expected_sol_cost=getattr(b, "expected_sol_cost", 0),
                            fee_lam=fee_lam, tip_lam=b.tip_lamports)
            # Off-hot-path rent recovery: fire ONLY after a sell that LANDED (we are
            # inside the landed-fill path) AND fully exited the position (held>0 ->
            # now 0). Best-effort; a failure never touches trading and the rent stays
            # reclaimable by tools/ata_sweep.py.
            if (not self.dry_run) and b.op in ("sell_all", "sell_slice") and pre_tok > 0 and post_tok == 0:
                _ct = asyncio.create_task(self._close_ata(cli, b.mint))
                self.bg_tasks.add(_ct); _ct.add_done_callback(self.bg_tasks.discard)
        except Exception as e:
            self._recon_log("fill_err", sig=sig_str, mint=b.mint, op=b.op, err=str(e))

    def reconcile_grpc_tx(self, sig_b58: str, slot: int, meta) -> bool:
        """PRIMARY confirmation + fill path (2026-06-12), fed by the gRPC wallet
        subscription on the EXISTING listener stream (no new connection, no
        HTTP). eRPC getTransaction/getAccountInfo are unreliable (selective
        outage 2026-06-12), so the feed's own tx meta is the source of truth for
        landed/reverted + the actual fill (token+SOL deltas, fee). The polling
        reconciler stays only as the expiry/timeout backstop (a tx that NEVER
        lands produces no feed event) and an eRPC fallback.

        meta is the geyser TransactionStatusMeta (same shape grpc_capture reads).
        Returns True if sig was one of ours (and was reconciled), else False.
        Runs in the listener's asyncio context — single-threaded, so the
        pending_bundles.pop is atomic w.r.t. the poll reconciler."""
        b = self.pending_bundles.get(sig_b58)
        if b is None:
            return False
        self.pending_bundles.pop(sig_b58, None)
        try:
            failed = bool(meta.err.err)
        except Exception:
            failed = False
        if failed:
            self._fail_or_retry(b, "grpc_tx_reverted")
            return True
        self._handle_bundle_landed(b, int(slot))
        try:
            owner = str(self.user_pk)
            def _tok_sum(balances):
                tot = 0
                for tb in (balances or []):
                    try:
                        if str(tb.owner) == owner and str(tb.mint) == b.mint:
                            tot += int(tb.ui_token_amount.amount or 0)
                    except Exception:
                        pass
                return tot
            pre_tok = _tok_sum(meta.pre_token_balances)
            post_tok = _tok_sum(meta.post_token_balances)
            pre_bal = list(meta.pre_balances or [])
            post_bal = list(meta.post_balances or [])
            sol_delta_lam = (int(post_bal[0]) - int(pre_bal[0])) if (pre_bal and post_bal) else None
            fee_lam = int(meta.fee or 0)
            if sol_delta_lam is not None:
                if b.op == "buy":
                    self._buy_cost_lam[b.mint] = int(sol_delta_lam)
                    self._emit_buy_fill(b.mint, sol_delta_lam, b.tip_lamports, fee_lam,
                                        post_tok - pre_tok)
                elif b.op in ("sell_all", "sell_slice"):
                    self.realized_pnl_lam += int(sol_delta_lam) + self._buy_cost_lam.pop(b.mint, 0)
            _tgt = getattr(b, "target_slot", None)
            self._recon_log("fill", sig=sig_b58, mint=b.mint, op=b.op, source="grpc",
                            landed_slot=int(slot), target_slot=_tgt,
                            bh_slot=getattr(b, "bh_slot", None),
                            slot_gap=(int(slot) - _tgt) if _tgt else None,
                            actual_tok_delta=(post_tok - pre_tok),
                            expected_tok_delta=getattr(b, "tok_delta", None),
                            actual_sol_delta_lam=sol_delta_lam,
                            expected_sol_out=getattr(b, "expected_sol_out", 0),
                            expected_sol_cost=getattr(b, "expected_sol_cost", 0),
                            fee_lam=fee_lam, tip_lam=b.tip_lamports)
            if (not self.dry_run) and b.op in ("sell_all", "sell_slice") and pre_tok > 0 and post_tok == 0:
                _ct = asyncio.create_task(self._close_ata_shared(b.mint))
                self.bg_tasks.add(_ct); _ct.add_done_callback(self.bg_tasks.discard)
        except Exception as e:
            self._recon_log("grpc_fill_err", sig=sig_b58, mint=b.mint, op=b.op, err=str(e)[:120])
        return True

    async def _close_ata_shared(self, mint: str) -> None:
        """ATA-close from the gRPC reconcile path (no cli arg). Uses the shared
        RPC pool; getTokenAccountBalance + getLatestBlockhash both work on eRPC
        (only getAccountInfo/getTransaction are degraded). Best-effort."""
        try:
            cli = await self._rpc_client()
            await self._close_ata(cli, mint)
        except Exception:
            pass

    async def _reconciler_loop(self, poll_s: float = 2.0, expire_s: float = 30.0) -> None:
        """Background coroutine. Every poll_s, checks every pending bundle's signature
        via getSignatureStatuses. If landed: success. If not on-chain after expire_s:
        treated as dropped, holdings rolled back, harness notified."""
        # Defer imports until run-time so importing jito_broker doesn't pull in solana
        # heavyweights at module load.
        try:
            from solana.rpc.async_api import AsyncClient
            from solders.signature import Signature
            import config
        except ImportError as e:
            self._recon_log("startup_err", err=f"missing deps for reconciler: {e}")
            return
        async with AsyncClient(config.rpc_http_url()) as cli:
            while True:
                await asyncio.sleep(poll_s)
                if not self.pending_bundles:
                    continue
                now = time.time()
                # Take a snapshot of sigs to check (deque keys may mutate during await)
                to_check = list(self.pending_bundles.items())
                # Batch up to 256 sigs per call (RPC limit is typically 256)
                BATCH = 200
                for i in range(0, len(to_check), BATCH):
                    chunk = to_check[i:i + BATCH]
                    sigs = []
                    bundles = []
                    for sig_str, b in chunk:
                        if now - b.submitted_t < 4.0:
                            continue   # too early to even check
                        try:
                            sigs.append(Signature.from_string(sig_str))
                            bundles.append((sig_str, b))
                        except Exception:
                            pass
                    if not sigs:
                        continue
                    try:
                        resp = await cli.get_signature_statuses(sigs, search_transaction_history=False)
                    except Exception as e:
                        self._recon_log("rpc_err", err=str(e), n_sigs=len(sigs))
                        await asyncio.sleep(2.0)
                        continue
                    statuses = resp.value
                    for (sig_str, b), st in zip(bundles, statuses):
                        # The gRPC wallet feed is PRIMARY and may have reconciled
                        # (popped) this sig during the await above; skip if so.
                        if sig_str not in self.pending_bundles:
                            continue
                        age = now - b.submitted_t
                        if st is None:
                            if age >= expire_s:
                                self._fail_or_retry(b, "expired_not_on_chain")
                                self.pending_bundles.pop(sig_str, None)
                            # else still pending, leave it
                        else:
                            if st.err is not None:
                                self._fail_or_retry(b, f"tx_err:{st.err}")
                                self.pending_bundles.pop(sig_str, None)
                            else:
                                self._handle_bundle_landed(b, int(st.slot))
                                await self._log_actual_fill(cli, sig_str, b)
                                self.pending_bundles.pop(sig_str, None)

    async def _holdings_reconcile_loop(self, poll_s: float = 300.0) -> None:
        """Every poll_s (5 min default), query the chain for our wallet's token balance
        on each held mint and reconcile against self.holdings. Chain is ground truth.
        Logs any drift; in extreme cases (drift > 50%) emit a critical warning.
        Skipped in DRY_RUN mode (nothing actually executed -> nothing to reconcile)."""
        if self.dry_run:
            return
        try:
            from solana.rpc.async_api import AsyncClient
            from solders.pubkey import Pubkey
            import config
            from pump_fun_ix import derive_ata
        except ImportError:
            return
        async with AsyncClient(config.rpc_http_url()) as cli:
            while True:
                await asyncio.sleep(poll_s)
                held = {m: v for m, v in self.holdings.items() if v > 0}
                if not held:
                    continue
                for mint_str, expected in held.items():
                    # Do NOT resurrect a position zeroed since the snapshot: a sell in flight
                    # zeroes holdings synchronously while the chain still shows the old balance
                    # for a few seconds, so writing chain_bal back would re-add a bag we are
                    # actively selling -> a spurious second sell. The sell's own gRPC reconcile
                    # is the truth. Skip zeroed-or-in-flight mints.
                    if self.holdings.get(mint_str, 0) <= 0:
                        continue
                    if any(b.mint == mint_str for b in self.pending_bundles.values()):
                        continue
                    try:
                        mint_pk = Pubkey.from_string(mint_str)
                        tp, _ = await self._get_mint_meta(mint_str)   # correct token program (Token-2022 for current mints)
                        ata = derive_ata(self.user_pk, mint_pk, tp)
                        resp = await cli.get_token_account_balance(ata)
                        chain_bal = int(resp.value.amount) if resp.value else 0
                    except Exception as e:
                        self._recon_log("chain_query_err", mint=mint_str, err=str(e))
                        continue
                    drift = chain_bal - expected
                    drift_frac = abs(drift) / max(expected, 1)
                    self._recon_log("holdings_reconcile", mint=mint_str,
                                    expected=expected, chain=chain_bal,
                                    drift=drift, drift_frac=drift_frac)
                    if drift_frac > 0.5:
                        self._recon_log("CRITICAL_drift", mint=mint_str,
                                        expected=expected, chain=chain_bal)
                    # truth is chain; reconcile
                    self.holdings[mint_str] = chain_bal

    def recon_summary(self) -> dict:
        """Snapshot of recent landing stats for status.json."""
        recent = list(self.recent_outcomes)
        if not recent:
            return {"n_outcomes": 0}
        landed = [r for r in recent if r["landed"]]
        failed = [r for r in recent if not r["landed"]]
        out = {"n_outcomes": len(recent),
               "n_landed": len(landed), "n_failed": len(failed),
               "land_rate": len(landed) / len(recent)}
        if landed:
            lats = [r["latency_s"] for r in landed]
            out["landing_latency_p50_s"] = sorted(lats)[len(lats)//2]
            out["landing_latency_p90_s"] = sorted(lats)[int(len(lats)*0.9)]
            tips = [r["tip"] for r in landed]
            out["landed_tip_p50"] = sorted(tips)[len(tips)//2]
        if failed:
            tips = [r["tip"] for r in failed]
            out["failed_tip_p50"] = sorted(tips)[len(tips)//2]
        return out

    # ---------- Hot-path methods: fire-and-forget tasks ----------

    async def buy(self, mint: str, sol: float, vsol_lam: int, vtok: int,
                  slot: int | None = None,
                  tip_lamports_override: int | None = None) -> None:
        # tip_lamports_override: when set, this single buy uses the supplied
        # tip lamports instead of self.tip_lamports. Caller (the harness)
        # passes this when the shred-window signal says snipers are forming
        # on the same mint and we want our bundle to land in the same slot.
        # In DRY_RUN mode it just changes what gets logged; in LIVE mode it
        # changes the actual Jito bundle submission.
        from pump_fun_ix import tokens_out_for_sol
        expected_tok = tokens_out_for_sol(vsol_lam, vtok, int(sol * 1e9))
        if expected_tok > 0:
            self.holdings[mint] = self.holdings.get(mint, 0) + expected_tok
        t = asyncio.create_task(self._do_buy(mint, sol, vsol_lam, vtok, slot,
                                             reserved_tok=expected_tok,
                                             tip_lamports_override=tip_lamports_override))
        self.bg_tasks.add(t); t.add_done_callback(self.bg_tasks.discard)

    async def sell_all(self, mint: str, vsol_lam: int, vtok: int,
                       slot: int | None = None) -> None:
        amt = self.holdings.get(mint, 0)
        if amt <= 0:
            self._log(op="sell_all", mint=mint, slot=slot, status="no_holdings")
            return
        # decrement synchronously
        self.holdings[mint] = 0
        t = asyncio.create_task(self._do_sell(mint, amt, vsol_lam, vtok, slot,
                                              op_label="sell_all", tip_override=self.exit_tip_base))
        self.bg_tasks.add(t); t.add_done_callback(self.bg_tasks.discard)

    async def sell_slice(self, mint: str, frac: float, vsol_lam: int, vtok: int,
                         slot: int | None = None) -> None:
        held = self.holdings.get(mint, 0)
        amt = int(held * frac)
        if amt <= 0:
            self._log(op="sell_slice", mint=mint, slot=slot, frac=frac, status="no_holdings")
            return
        # decrement synchronously so rapid-fire calls see fresh balance
        self.holdings[mint] = max(0, held - amt)
        t = asyncio.create_task(self._do_sell(mint, amt, vsol_lam, vtok, slot,
                                              op_label="sell_slice", frac=frac))
        self.bg_tasks.add(t); t.add_done_callback(self.bg_tasks.discard)

    def _log(self, **kw):
        self.fh.write(json.dumps({"t": time.time(), **kw}, default=str) + "\n")

    # ---------- Actual assembly + submission ----------


    async def _bh_or_none(self, op: str, mint: str, slot=None):
        """get_blockhash with a hard timeout. A stalled on-demand refresh was
        the asm_ms tail (2.15s max observed 2026-06-10); fail the bundle fast
        and log it instead of stalling the fire path."""
        try:
            return await asyncio.wait_for(self.get_blockhash(), timeout=1.5)
        except asyncio.TimeoutError:
            self._log(op=op, mint=mint, status="bh_stall_timeout", slot=slot)
            return None

    async def _rpc_client(self):
        """Shared long-lived AsyncClient for hot-path lookups (mint meta, curve
        reserves, token balance). A persistent connection pool removes the
        per-call TCP+TLS handshake that dominated cold fetches (~200-500ms).
        Left open for the process lifetime; the OS reaps it at exit."""
        if self._rpc is None:
            from solana.rpc.async_api import AsyncClient
            import config as _cfg
            self._rpc = AsyncClient(_cfg.rpc_http_url())
        return self._rpc

    def _spawn_meta_fetch(self, mint: str) -> asyncio.Task:
        """Register a single in-flight fetch per mint so concurrent callers
        (prefetch, the buy, the same-second TP sell) share ONE fetch."""
        t = asyncio.create_task(self._fetch_mint_meta(mint))
        self._meta_tasks[mint] = t
        self.bg_tasks.add(t)
        def _done(task: asyncio.Task) -> None:
            self.bg_tasks.discard(task)
            if self._meta_tasks.get(mint) is task:
                self._meta_tasks.pop(mint, None)
            if not task.cancelled():
                task.exception()   # consume: a failed PREFETCH must never warn
        t.add_done_callback(_done)
        return t

    def seed_mint_meta(self, mint: str, token_program: str, creator: str) -> None:
        """Seed the (token_program, creator) cache from the feed's CreateEvent
        at token birth (zero RPC, zero fire-time latency). Called by the gRPC
        listener; layout live-verified against the curve account's offset-49
        creator (2026-06-11). FIFO-prunes so a day of creates (~30-60k mints)
        cannot grow the cache unbounded."""
        if mint in self._mint_meta:
            return
        try:
            from solders.pubkey import Pubkey
            self._mint_meta[mint] = (Pubkey.from_string(token_program),
                                     Pubkey.from_string(creator))
        except Exception:
            return
        if len(self._mint_meta) > 150_000:
            for k in list(self._mint_meta)[:50_000]:
                self._mint_meta.pop(k, None)

    def prefetch_mint_meta(self, mint: str) -> None:
        """Fire-and-forget cache warm. The harness calls this at K/V
        partial-trigger time, 1-2s before a possible fire, so fire-time
        assembly skips the 2-RPC meta fetch (measured 500-600ms cold = more
        than a slot). Best-effort: on failure the fire-time fetch retries."""
        if mint in self._mint_meta:
            return
        t = self._meta_tasks.get(mint)
        if t is not None and not t.done():
            return
        self._spawn_meta_fetch(mint)

    async def _get_mint_meta(self, mint: str):
        """(token_program, creator) for a mint: cache, then any in-flight
        prefetch, then a fresh fetch. Raises if the mint stays unresolvable."""
        m = self._mint_meta.get(mint)
        if m is not None:
            return m
        t = self._meta_tasks.get(mint)
        if t is None or (t.done() and mint not in self._mint_meta):
            t = self._spawn_meta_fetch(mint)
        try:
            await asyncio.shield(t)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"mint_meta_unavailable: {e}") from e
        m = self._mint_meta.get(mint)
        if m is None:
            raise RuntimeError("mint_meta_unavailable (fetch completed without caching)")
        return m

    async def _fetch_mint_meta(self, mint: str):
        """The actual 2-RPC fetch. The current pump.fun ABI needs the mint's
        token program (Token-2022 for current mints) and the bonding-curve
        creator (account data offset 49). Fetched once per mint, cached.

        COMMITMENT MATTERS (regression found 2026-06-11): the bot buys mints
        that are 1-3s old. At the client default (finalized) a fresh mint's
        accounts are invisible for ~13s, so every organic buy failed assembly
        with \"'NoneType' object has no attribute 'owner'\" (proof: GprwzJHB buy
        failed at +0s, its sell assembled at +13s). Query at PROCESSED, the
        same commitment as the trade feed that triggered us, with one short
        retry for RPC-node lag and a hard timeout per attempt so a hung RPC
        can't stall the fire path indefinitely."""
        from solders.pubkey import Pubkey
        from solana.rpc.commitment import Processed
        from pump_fun_ix import derive_bonding_curve
        mint_pk = Pubkey.from_string(mint)
        curve_pk = derive_bonding_curve(mint_pk)
        cli = await self._rpc_client()
        last_err = None
        for attempt in (1, 2):
            try:
                # the two lookups are independent: run them concurrently
                # (measured: 2 x ~45ms sequential -> ~45ms total on warm pool)
                acc_resp, curve_resp = await asyncio.gather(
                    asyncio.wait_for(
                        cli.get_account_info(mint_pk, commitment=Processed),
                        timeout=2.0),
                    asyncio.wait_for(
                        cli.get_account_info(curve_pk, commitment=Processed,
                                             encoding="base64"),
                        timeout=2.0))
                acc, curve = acc_resp.value, curve_resp.value
                if acc is None or curve is None:
                    raise RuntimeError(
                        f"account not visible at processed yet "
                        f"(mint_ok={acc is not None}, curve_ok={curve is not None})")
                tp = acc.owner
                creator = Pubkey.from_bytes(bytes(curve.data)[49:81])
                self._mint_meta[mint] = (tp, creator)
                return tp, creator
            except Exception as e:
                last_err = e
                if attempt == 1:
                    await asyncio.sleep(0.25)
        raise RuntimeError(f"mint_meta_unavailable after 2 processed attempts: {last_err}")

    def realized_net_sol(self) -> float:
        """Real realized net P&L in SOL this process: sum over CLOSED round-trips of
        (sell wallet lamport delta + that mint's buy lamport delta). Open positions do
        NOT drag it (only realized losses gate). LIVE only (DRY_RUN never fills)."""
        return self.realized_pnl_lam / 1e9

    async def _get_curve_reserves(self, mint: str):
        """(vsol_lam, vtok, complete) from the bonding-curve account; None on error."""
        try:
            import struct
            from solders.pubkey import Pubkey
            from solana.rpc.commitment import Processed
            from pump_fun_ix import derive_bonding_curve
            cli = await self._rpc_client()
            # PROCESSED: freshest reserves for retry pricing, and consistent
            # with the feed (finalized lags ~13s = stale prices on a mover).
            v = (await cli.get_account_info(derive_bonding_curve(Pubkey.from_string(mint)),
                                            encoding="base64", commitment=Processed)).value
            if v is None:
                return None
            d = bytes(v.data)
            vtok, vsol = struct.unpack_from("<QQ", d, 8)
            return vsol, vtok, (bool(d[48]) if len(d) > 48 else False)
        except Exception as e:
            self._recon_log("curve_fetch_err", mint=mint, err=str(e))
            return None

    async def _token_balance(self, mint: str) -> int:
        """Our current on-chain token balance for `mint` (0 on error/none)."""
        try:
            from solders.pubkey import Pubkey
            from solana.rpc.commitment import Processed
            from pump_fun_ix import derive_ata
            tp, _ = await self._get_mint_meta(mint)
            cli = await self._rpc_client()
            r = await cli.get_token_account_balance(
                derive_ata(self.user_pk, Pubkey.from_string(mint), tp),
                commitment=Processed)
            return int(r.value.amount) if r.value else 0
        except Exception:
            return 0

    async def list_token_holdings(self) -> list:
        """One-shot enumeration of the wallet's token holdings across BOTH token programs
        (legacy SPL + Token-2022 — current pump.fun mints are Token-2022, so a legacy-only
        scan misses them). Startup-only restart-recovery, NOT a hot path. Excludes WSOL and
        zero balances. Best-effort: returns [] on failure so the caller falls back to the
        on-disk position journal. Returns [{mint, raw, decimals, token_program}]."""
        out = []
        try:
            from solana.rpc.types import TokenAccountOpts
            from solana.rpc.commitment import Processed
            from pump_fun_ix import TOKEN_PROGRAM, TOKEN_2022_PROGRAM
            WSOL = "So11111111111111111111111111111111111111112"
            cli = await self._rpc_client()
            for prog in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
                try:
                    # PROCESSED so a buy that landed seconds before a restart is visible
                    # (finalized lags ~13s -> would miss a fresh position).
                    try:
                        resp = await cli.get_token_accounts_by_owner_json_parsed(
                            self.user_pk, TokenAccountOpts(program_id=prog), commitment=Processed)
                    except TypeError:
                        resp = await cli.get_token_accounts_by_owner_json_parsed(
                            self.user_pk, TokenAccountOpts(program_id=prog))
                except Exception as e:
                    self._recon_log("holdings_enum_err", program=str(prog), err=str(e)[:120])
                    continue
                for a in (resp.value or []):
                    try:
                        info = a.account.data.parsed["info"]
                        raw = int(info["tokenAmount"]["amount"])
                        mint = info["mint"]
                        if raw <= 0 or mint == WSOL:
                            continue
                        out.append({"mint": mint, "raw": raw,
                                    "decimals": int(info["tokenAmount"]["decimals"]),
                                    "token_program": str(prog)})
                    except Exception:
                        continue
        except Exception as e:
            self._recon_log("holdings_enum_fatal", err=str(e)[:120])
        return out

    async def _retry_sell_all(self, b) -> None:
        """A sell_all failed -> re-submit with the ACTUAL chain balance, FRESH reserves,
        and escalating slippage (final attempt = market sell). Bounded by MAX_SELL_RETRIES."""
        mint = b.mint
        attempt = b.retry_count + 1
        bal = await self._token_balance(mint)
        if bal <= 0:
            self._recon_log("sell_retry_skip", mint=mint, attempt=attempt, reason="zero_balance")
            return
        res = await self._get_curve_reserves(mint)
        if res is None:
            # eRPC getAccountInfo down (the 2026-06-12 outage). Do NOT abort and
            # strand a dumping bag: force a MARKET sell (min_sol=1 ignores
            # reserves) using the reserves we stored at submit, so the panic
            # exit still fires. We lose the migrated-curve check, but a migrated
            # curve's sell just fails on-chain -> retry exhausts -> terminal.
            vsol = int(getattr(b, "vsol_submit", 0)) or 30_000_000_000
            vtok = int(getattr(b, "vtok_submit", 0)) or 1
            self._recon_log("sell_retry_no_curve_market", mint=mint, attempt=attempt,
                            reason="curve_fetch_down_force_market")
            self.holdings[mint] = 0
            await self._do_sell(mint, bal, vsol, vtok, None, op_label="sell_all",
                                retry_count=attempt, slippage_bps_override=None, market=True)
            return
        vsol, vtok, complete = res
        if complete:
            self._recon_log("sell_retry_abort", mint=mint, attempt=attempt, reason="curve_migrated")
            self._handle_bundle_failed(b, reason="migrated_cannot_curve_sell")
            return
        market = attempt >= MAX_SELL_RETRIES
        slip = None if market else SELL_RETRY_SLIPPAGE_BPS[min(attempt - 1, len(SELL_RETRY_SLIPPAGE_BPS) - 1)]
        self._recon_log("sell_retry", mint=mint, attempt=attempt, market=market,
                        slippage_bps=slip, balance=bal)
        self.holdings[mint] = 0
        await self._do_sell(mint, bal, vsol, vtok, None, op_label="sell_all",
                            retry_count=attempt, slippage_bps_override=slip, market=market)

    def _fail_or_retry(self, b, reason: str) -> None:
        """Retry a failed sell_all (panic exit); else terminal failure."""
        if b.op == "sell_all" and b.retry_count < MAX_SELL_RETRIES:
            self._recon_log("sell_will_retry", mint=b.mint, reason=reason, next_attempt=b.retry_count + 1)
            t = asyncio.create_task(self._retry_sell_all(b))
            self.bg_tasks.add(t); t.add_done_callback(self.bg_tasks.discard)
        else:
            self._handle_bundle_failed(b, reason=reason)

    def _compute_budget_ixs(self):
        """ComputeBudget priority-fee ixs (CU limit + price). 90% of the competing
        pump.fun field uses a priority fee; our bundles previously set none.
        price = broker.priority_fee_micro (data p75 ~2M micro-lamports)."""
        from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
        return [set_compute_unit_limit(self.cu_limit),
                set_compute_unit_price(self.priority_fee_micro)]

    async def _do_buy(self, mint: str, sol: float, vsol_lam: int, vtok: int,
                      slot: int | None, reserved_tok: int = 0,
                      tip_lamports_override: int | None = None) -> None:
        """reserved_tok was already added to self.holdings synchronously in buy()
        for fast subsequent-sell visibility. If the assembly fails below, we roll
        the reservation back."""
        from solders.pubkey import Pubkey
        from pump_fun_ix import (build_buy_ix, build_ata_create_idempotent_ix,
                                  tokens_out_for_sol, slippage_max_sol_cost)
        t0 = time.time()
        mint_pk = Pubkey.from_string(mint)
        sol_in_lam = int(sol * 1e9)
        expected_tok = tokens_out_for_sol(vsol_lam, vtok, sol_in_lam)
        max_cost = slippage_max_sol_cost(sol_in_lam, self.slippage_bps_buy)
        if expected_tok <= 0 or max_cost <= 0:
            # roll back the tentative reservation
            if reserved_tok > 0:
                self.holdings[mint] = max(0, self.holdings.get(mint, 0) - reserved_tok)
            self._log(op="buy", mint=mint, status="bad_amm_state",
                       vsol=vsol_lam, vtok=vtok, sol_in=sol_in_lam)
            return
        bh = await self._bh_or_none("buy", mint, slot)
        if bh is None:
            # nothing submitted -> roll back the reservation (was leaking phantom
            # holdings that a later sell would have tried to sell)
            if reserved_tok > 0:
                self.holdings[mint] = max(0, self.holdings.get(mint, 0) - reserved_tok)
            self._log(op="buy", mint=mint, status="no_blockhash", slot=slot)
            return
        bh_age_ms = int((time.time() - bh.fetched_at) * 1000)
        try:
            _tp, _creator = await self._get_mint_meta(mint)
            ata_ix = build_ata_create_idempotent_ix(self.user_pk, self.user_pk, mint_pk, _tp)
            buy_ix = build_buy_ix(mint_pk, self.user_pk, self.fee_recipient,
                                   expected_tok, max_cost,
                                   token_program=_tp, creator=_creator)
            ixs = self._compute_budget_ixs() + [ata_ix, buy_ix]
            if self.dry_run:
                # DRY: log assembled-bundle metadata, do NOT POST
                from solders.hash import Hash
                from solders.message import MessageV0
                from solders.transaction import VersionedTransaction
                msg = MessageV0.try_compile(payer=self.user_pk, instructions=ixs,
                                             address_lookup_table_accounts=[],
                                             recent_blockhash=Hash.from_string(bh.blockhash))
                tx = VersionedTransaction(msg, [self.wallet])
                tx_bytes = len(bytes(tx))
                # measure AFTER tx serialization so signing+encode time is included;
                # use float-ms (sub-ms is real, int() was truncating to 0).
                asm_ms = round((time.time() - t0) * 1000, 3)
                # holdings already reserved synchronously in buy()
                # Log the tip we'd HAVE used (incl. any front-run override)
                # so DRY_RUN diagnostics show the actual bundle-cost we'd
                # incur in LIVE mode.
                tip_lam_for_this_buy = (tip_lamports_override
                                         if tip_lamports_override is not None
                                         else self.tip_lamports)
                self._log(op="buy", mint=mint, status="DRY_RUN_assembled",
                           sol=sol, expected_tok=expected_tok, max_sol_cost_lam=max_cost,
                           tip_lam=tip_lam_for_this_buy,
                           bh=bh.blockhash, bh_age_ms=bh_age_ms,
                           bh_slot=getattr(bh, "slot", None),
                           slot_gap=(slot - bh.slot) if (slot is not None
                                                          and getattr(bh, "slot", 0)) else None,
                           asm_ms=asm_ms, slot=slot,
                           tx_bytes=tx_bytes)
                return
            # LIVE submit
            from jito_exec import execute_jito_bundle
            # execute_jito_bundle wants a SINGLE swap_ix + builds tip+versioned; we have
            # two ixs (ata create + buy). Inline-assemble like execute_jito_bundle but
            # with our 2-ix list:
            from solders.system_program import TransferParams, transfer
            from solders.hash import Hash
            from solders.message import MessageV0
            from solders.transaction import VersionedTransaction
            from jito_exec import get_cached_tip_account, send_transaction_b64_async
            tip_account = Pubkey.from_string(get_cached_tip_account())
            tip_lam_for_this_buy = (tip_lamports_override
                                     if tip_lamports_override is not None
                                     else self.tip_lamports)
            tip_ix = transfer(TransferParams(from_pubkey=self.user_pk,
                                              to_pubkey=tip_account,
                                              lamports=tip_lam_for_this_buy))
            msg = MessageV0.try_compile(payer=self.user_pk,
                                         instructions=ixs + [tip_ix],
                                         address_lookup_table_accounts=[],
                                         recent_blockhash=Hash.from_string(bh.blockhash))
            tx = VersionedTransaction(msg, [self.wallet])
            import base64
            b64 = base64.b64encode(bytes(tx)).decode()
            sig_str = str(tx.signatures[0])
            # Submit via the nearest region with budget, failing over to the
            # next on error/rate-exhaustion. The proxy forwards to the current
            # leader regardless of region, so this reaches every leader at the
            # lowest local hop; async + pooled keeps the event loop running.
            resp = await send_transaction_b64_async(b64)
            # Record pending bundle for reconciliation
            pb = PendingBundle(sig=sig_str, mint=mint, op="buy",
                               tok_delta=expected_tok,
                               tip_lamports=tip_lam_for_this_buy,
                               submitted_t=time.time(),
                               bet_sol=sol, expected_sol_cost=sol_in_lam,
                               target_slot=slot, bh_slot=getattr(bh, "slot", None),
                               vsol_submit=int(vsol_lam), vtok_submit=int(vtok))
            self.pending_bundles[sig_str] = pb
            # holdings already reserved synchronously in buy()
            self._log(op="buy", mint=mint, status="LIVE_submitted",
                       sol=sol, expected_tok=expected_tok, max_sol_cost_lam=max_cost,
                       bh=bh.blockhash, bh_age_ms=bh_age_ms,
                           bh_slot=getattr(bh, "slot", None),
                           slot_gap=(slot - bh.slot) if (slot is not None
                                                          and getattr(bh, "slot", 0)) else None,
                       asm_ms=round((time.time()-t0)*1000, 3), slot=slot,
                       sig=sig_str, jito_resp=resp)
        except Exception as e:
            # Assembly/submit raised -> nothing on chain. Roll back the
            # synchronous reservation; the reconciler only rolls back bundles
            # that were SUBMITTED, so without this an assembly failure left
            # phantom holdings (found in the 2026-06-11 audit: every meta-fetch
            # failure left expected_tok reserved and the later sell sold air).
            if reserved_tok > 0:
                self.holdings[mint] = max(0, self.holdings.get(mint, 0) - reserved_tok)
            self._log(op="buy", mint=mint, status="error", err=str(e), slot=slot)

    async def _do_sell(self, mint: str, token_amount: int, vsol_lam: int, vtok: int,
                       slot: int | None, op_label: str, frac: float | None = None,
                       retry_count: int = 0, slippage_bps_override=None,
                       market: bool = False, tip_override: int | None = None) -> None:
        from solders.pubkey import Pubkey
        from pump_fun_ix import (build_sell_ix, sol_out_for_tokens, slippage_min_sol_output)
        t0 = time.time()
        mint_pk = Pubkey.from_string(mint)

        def _abort_rollback(reason: str) -> None:
            # Nothing was submitted. Route through the SAME machinery as a
            # bundle that failed to land (synthetic PendingBundle into
            # _fail_or_retry): sell_all gets the validated retry ladder (chain
            # balance + fresh reserves + escalating slippage -> market);
            # sell_slice / exhausted retries get holdings restored + the
            # harness callback. Before the 2026-06-11 audit, assembly failures
            # silently stranded the synchronously decremented holdings.
            pb = PendingBundle(sig="", mint=mint, op=op_label,
                               tok_delta=-int(token_amount),
                               tip_lamports=self.tip_lamports,
                               submitted_t=time.time(),
                               expected_sol_out=int(expected_sol_out) if expected_sol_out and expected_sol_out > 0 else 0,
                               retry_count=retry_count)
            self._fail_or_retry(pb, reason)

        expected_sol_out = sol_out_for_tokens(vsol_lam, vtok, token_amount)
        _slip = slippage_bps_override if slippage_bps_override is not None else self.slippage_bps_sell
        min_sol = 1 if market else slippage_min_sol_output(expected_sol_out, _slip)
        # A MARKET sell (min_sol=1) must proceed even with stale/zero reserves
        # (expected_sol_out is then just a logging estimate) — that is the
        # panic-exit when the curve refetch is down. Only block non-market sells
        # on a bad AMM state.
        if min_sol <= 0 or (not market and expected_sol_out <= 0):
            self._log(op=op_label, mint=mint, status="bad_amm_state",
                       vsol=vsol_lam, vtok=vtok, token_amount=token_amount)
            _abort_rollback("assembly_bad_amm_state")
            return
        bh = await self._bh_or_none("sell", mint, slot)
        if bh is None:
            self._log(op=op_label, mint=mint, status="no_blockhash", slot=slot)
            _abort_rollback("assembly_no_blockhash")
            return
        bh_age_ms = int((time.time() - bh.fetched_at) * 1000)
        try:
            _tp, _creator = await self._get_mint_meta(mint)
            sell_ix = build_sell_ix(mint_pk, self.user_pk, self.fee_recipient,
                                     token_amount, min_sol,
                                     token_program=_tp, creator=_creator)
            if self.dry_run:
                # holdings already decremented synchronously by sell_all / sell_slice
                # DRY path skips MessageV0 compile + tx encode, so timing here is
                # AMM math + 1 ix build only — still measure as float-ms for parity.
                self._log(op=op_label, mint=mint, status="DRY_RUN_assembled",
                           frac=frac, token_amount=token_amount,
                           expected_sol_out=expected_sol_out, min_sol=min_sol,
                           bh=bh.blockhash, bh_age_ms=bh_age_ms,
                           bh_slot=getattr(bh, "slot", None),
                           slot_gap=(slot - bh.slot) if (slot is not None
                                                          and getattr(bh, "slot", 0)) else None,
                           asm_ms=round((time.time()-t0)*1000, 3), slot=slot)
                return
            from solders.system_program import TransferParams, transfer
            from solders.hash import Hash
            from solders.message import MessageV0
            from solders.transaction import VersionedTransaction
            from jito_exec import get_cached_tip_account, send_transaction_b64_async
            tip_account = Pubkey.from_string(get_cached_tip_account())
            _tip = int(tip_override) if tip_override is not None else self.exit_tip_lamports
            tip_ix = transfer(TransferParams(from_pubkey=self.user_pk,
                                              to_pubkey=tip_account,
                                              lamports=_tip))
            msg = MessageV0.try_compile(payer=self.user_pk,
                                         instructions=self._compute_budget_ixs() + [sell_ix, tip_ix],
                                         address_lookup_table_accounts=[],
                                         recent_blockhash=Hash.from_string(bh.blockhash))
            tx = VersionedTransaction(msg, [self.wallet])
            import base64
            b64 = base64.b64encode(bytes(tx)).decode()
            sig_str = str(tx.signatures[0])
            # Same nearest-region-with-failover sender as the buy. A
            # same-second buy+sell naturally use different regions (buy takes
            # Frankfurt, sell finds it rate-limited and fails over to
            # Amsterdam), so neither waits and both reach the current leader.
            resp = await send_transaction_b64_async(b64)
            # Record pending bundle for reconciliation (negative tok_delta = sell)
            pb = PendingBundle(sig=sig_str, mint=mint, op=op_label,
                               tok_delta=-int(token_amount),
                               tip_lamports=_tip,
                               submitted_t=time.time(),
                               expected_sol_out=int(expected_sol_out), retry_count=retry_count,
                               target_slot=slot, bh_slot=getattr(bh, "slot", None),
                               vsol_submit=int(vsol_lam), vtok_submit=int(vtok))
            self.pending_bundles[sig_str] = pb
            # holdings already decremented synchronously by sell_all / sell_slice
            self._log(op=op_label, mint=mint, status="LIVE_submitted",
                       frac=frac, token_amount=token_amount,
                       expected_sol_out=expected_sol_out, min_sol=min_sol,
                       bh=bh.blockhash, bh_age_ms=bh_age_ms,
                           bh_slot=getattr(bh, "slot", None),
                           slot_gap=(slot - bh.slot) if (slot is not None
                                                          and getattr(bh, "slot", 0)) else None,
                       asm_ms=round((time.time()-t0)*1000, 3), slot=slot,
                       sig=sig_str, jito_resp=resp)
        except Exception as e:
            self._log(op=op_label, mint=mint, status="error", err=str(e), slot=slot)
            _abort_rollback(f"assembly_error:{str(e)[:80]}")

    async def _close_ata(self, cli, mint: str) -> None:
        """Reclaim ~0.002 SOL ATA rent after a position is FULLY exited. Off the hot
        path (fired from the reconciler's fill logging, never the submit path).
        Triple-guarded: reached only after the sell LANDED, only when post-sell
        balance is 0, and re-verified == 0 on-chain here before closing. A failure
        NEVER affects trading -- rent stays reclaimable by tools/ata_sweep.py."""
        try:
            from solders.pubkey import Pubkey
            from solders.message import MessageV0
            from solders.transaction import VersionedTransaction
            from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
            from jito_exec import send_transaction_b64
            from pump_fun_ix import derive_ata, build_close_account_ix
            import base64
            tp, _creator = await self._get_mint_meta(mint)
            ata = derive_ata(self.user_pk, Pubkey.from_string(mint), tp)
            try:
                bal = await asyncio.wait_for(
                    cli.get_token_account_balance(ata, commitment="confirmed"), timeout=5.0)
                if int(bal.value.amount) != 0:
                    self._recon_log("ata_close_skip", mint=mint, reason="nonzero",
                                    amount=bal.value.amount)
                    return
            except Exception:
                return  # account already gone / unreadable -> nothing to reclaim
            bh = (await cli.get_latest_blockhash()).value.blockhash
            ixs = [set_compute_unit_limit(12_000), set_compute_unit_price(500_000),
                   build_close_account_ix(ata, self.user_pk, tp)]
            msg = MessageV0.try_compile(payer=self.user_pk, instructions=ixs,
                                        address_lookup_table_accounts=[], recent_blockhash=bh)
            tx = VersionedTransaction(msg, [self.wallet])
            resp = send_transaction_b64(base64.b64encode(bytes(tx)).decode())
            self._recon_log("ata_closed", mint=mint, ata=str(ata),
                            sig=str(tx.signatures[0]), jito_resp=resp)
        except Exception as e:
            self._recon_log("ata_close_err", mint=mint, err=str(e)[:140])

    def close(self) -> None:
        try: self.fh.close()
        except Exception: pass
        for t in list(self.bg_tasks):
            t.cancel()
