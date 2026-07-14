"""graduation_live_bot.py — post-graduation PumpSwap research bot (2026-06-15).

The research question: after a token graduates from its bonding curve, can a rich 22-feature
tracker identify a first 2x continuation that remains attainable after slot and fill constraints?
Driven by the direct gRPC listener and intent SHM ring; scores with the trained graduation model.

SAFETY: DRY-RUN by default (pure sim via continuation_sizing, NO broker/wallet/submit). LIVE is
triple-gated (--live + PUMPFUN_LIVE_OK=1 + JITO_DRY_RUN=0) and uses jito_broker, fill-anchored.
Writes bot_data/continuation_status.json (the existing dashboard reads it) + continuation_rep.jsonl.

Reuses: continuation_tracker_rich.RichTracker, continuation_reputation.Reputation,
the trained cont_leanrep_model.pkl, continuation_sizing, listener_grpc_bot, shred_bot/ShredWindow,
and ContinuationBot's _bump/_block/_cut/write_status/setup (same status schema, same dashboard).
"""
import asyncio, collections, json, math, os, sys, time, pickle, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/the-distribution-will-manifest/shred_bot")
from research.continuation.continuation_bot import (
    ContinuationBot,
    parse_args,
    _empty,
    PUMP_RT,
    NARROW,
    LAMPORTS_PER_SOL,
    STATUS,
)
from research.continuation.continuation_tracker_rich import RichTracker
from research.continuation.continuation_reputation import Reputation
from research.continuation.continuation_sizing import plan_buy, simulate_fill, realized_return

ROOT = "/root/the-distribution-will-manifest"
MODEL = f"{ROOT}/bot_data/grad_deploy_model.pkl"    # GRADUATION RICH model (PumpSwap AMM, OOS AUC 0.814)
SPEC = f"{ROOT}/bot_data/grad_model_spec.json"      # features=RICH(22), lean=RICH, rep=[] (rep unused on graduation)
SEED = f"{ROOT}/bot_data/cont_reputation_seed.json" # rep map; UNUSED for scoring (FE=RICH) but Reputation.load needs a path
SAME_BLOCK_S = float(os.getenv("CONT_SAME_BLOCK_S", "0.4"))  # closes resolving faster than ~1 slot are same-block gap-0 illusions -> tracked as UNREALIZABLE
PANEL = f"{ROOT}/bot_data/grad_cont_panel_4d.jsonl"          # runtime-unused
REP_LOG = f"{ROOT}/bot_data/graduation_events.jsonl"        # extensive per-cross/decision/fill/outcome log
TIMING_LOG = f"{ROOT}/bot_data/graduation_timing.jsonl"     # per-cross hot-path stage latencies
POS_JOURNAL = f"{ROOT}/bot_data/graduation_open_positions.json"
IGNORE_MINTS_FILE = f"{ROOT}/graduation_ignore_mints.txt"
# This bot defines its OWN write_status() (below) that uses this module-level STATUS name (imported
# from continuation_bot at the top). Shadow it with the graduation path so the continuation bot's
# status file + its dashboard stay completely untouched and graduation writes its own.
STATUS = f"{ROOT}/bot_data/graduation_status.json"


class RepLive(ContinuationBot):
    def __init__(self, args):
        # fresh init (we swap model + tracker + backfill vs the base bot; do NOT call super)
        self.args = args
        self.WIDE = args.tier
        self.bet = args.bet_sol; self.tip = args.tip_sol
        self.bet_lam = int(args.bet_sol * LAMPORTS_PER_SOL)
        self.tip_lam = int(args.tip_sol * LAMPORTS_PER_SOL)
        self.buy_tip_base = int(os.getenv("CONT_BUY_TIP_BASE", "250000"))   # base buy tip (priority fee is the real lever; tip is secondary)
        self.buy_tip_cap = int(os.getenv("CONT_BUY_TIP_CAP", "1500000"))    # cap the contention-driven up-bump
        self.prio_lam = int(args.prio_fee_micro * args.cu_limit / 1e6)
        # Honest fixed round-trip cost: the ACTUAL tips paid are buy=buy_tip_base and
        # sell=exit_tip_base (CONT_EXIT_TIP_BASE) on a calm exit, NOT self.tip_lam (which is
        # unused in live submission). Panic exits escalate the sell tip to 1.5M (rare); the
        # -1 SOL watchdog and realized_net_sol use REAL chain deltas, so this is display-only.
        _exit_tip_base = int(os.getenv("CONT_EXIT_TIP_BASE", "150000"))
        self.fixed_rt = (self.buy_tip_base + _exit_tip_base + 2 * self.prio_lam + 2 * 5000) / LAMPORTS_PER_SOL
        spec = json.load(open(SPEC))
        self.FE = spec["features"]; self.LEAN = spec["lean"]; self.REP = spec["rep"]
        self.clf = pickle.load(open(MODEL, "rb"))
        # FAST EXACT predictor: sklearn predict_proba is 5-250ms/call single-sample (per-call overhead +
        # x2's 2-core contention). Extract the float64 trees and traverse directly: ~0.17ms, bit-identical
        # to predict_proba (verified 0 diff / 0 decision-flips on 3000 real crosses). No onnx/runtime dep.
        self._base = float(np.ravel(self.clf._baseline_prediction)[0])
        self._trees = []
        for _it in self.clf._predictors:
            _nd = _it[0].nodes
            _tf = "num_threshold" if "num_threshold" in _nd.dtype.names else "threshold"
            self._trees.append((_nd["feature_idx"].tolist(), _nd[_tf].astype(np.float64).tolist(),
                                _nd["left"].tolist(), _nd["right"].tolist(), _nd["is_leaf"].astype(bool).tolist(),
                                _nd["value"].astype(np.float64).tolist(), _nd["missing_go_to_left"].astype(bool).tolist()))
        self.rep = Reputation.load(SEED)
        self.trk = RichTracker(k=2.0)
        self.mint_creator = {}
        self.live = bool(args.live) and os.getenv("PUMPFUN_LIVE_OK") == "1" and os.getenv("JITO_DRY_RUN", "1") == "0"
        self.broker = None
        self.crosses = 0
        self._prev_qres = {}     # mint -> last pool quote reserve; AMM trade-size = |delta quote_res|
        self.ignore_mints = self._load_ignore_mints()    # old stranded bags to never recover/adopt/manage
        self.recovered = {"recovered": 0, "adopted": 0, "dropped": 0, "unmanageable": 0, "ignored": 0}  # startup restart-recovery tally
        self.max_loss_sol = float(os.getenv("CONT_MAX_LOSS_SOL", "1.0"))   # halt NEW buys when realized P&L <= -this (defense-in-depth; the external watchdog stops the service)
        self.halted = False
        self.max_concurrent = int(os.getenv("CONT_MAX_CONCURRENT", "5"))    # cap concurrent open+awaiting positions -> bounds total live exposure ~ max_concurrent * bet * (1+buy_slip)
        self.band_only = os.getenv("CONT_BAND_ONLY", "0") == "1"            # fire ONLY the 5-10% band (skip top-5%): the top-5% are the explosive runners that mostly revert/mirage at gap-1
        self.t = _empty(); self.bf = _empty(); self.bf_n = 0; self.bf_test_frac = 1.0
        self.t_rz = _empty()   # REALIZABLE forward stats: closes EXCLUDING same-block (<1 slot) gap-0 illusions
        self.sel = {}; self.pos = {}
        self.recent_p = collections.deque(maxlen=500)
        self.recent_closes = collections.deque(maxlen=40)
        self.warm = {"5": 0.7, "10": 0.5}
        self.t0 = time.time()
        self.out = open(REP_LOG, "a")
        self._timing_out = open(TIMING_LOG, "a")           # instrumentation: per-cross stage latencies
        self._score_us = (0.0, 0.0)                        # (feature_build_us, predict_us) from last score()
        self._ev_us_sum = 0.0; self._ev_n = 0              # per-event tracker-update cost aggregate
        self.stats = collections.Counter()
        self.shred_window = None
        self.awaiting_fill = set()
        self.live_exit_anchor_mid = {}
        self.awaiting_ts = {}            # mint -> buy submit ts (awaiting-fill timeout; anti-zombie)
        self._exiting = set()            # mints with a sell in flight (double-exit guard)
        self.TP = self.trk.tp; self.STOP = self.trk.stop      # +0.5x / -0.3x, same thresholds as the tracker
        self.max_hold_s = float(os.getenv("CONT_MAX_HOLD_S", "300"))         # force-exit a position held this long (zombie guard)
        self.fill_timeout_s = float(os.getenv("CONT_FILL_TIMEOUT_S", "45"))  # drop an unfilled buy after this (broker expires ~30s)
        self.rep_backfill()

    # --- scoring: assemble LEAN (from the rich cross) + REP (from the reputation map) ---
    def _proba_fast(self, x):
        """Exact float64 traversal of the HGB trees -> p(class 1). x = list indexed by feature_idx.
        ~0.17ms vs predict_proba's 5-250ms; bit-identical to sklearn (verified 0-diff/0-flips)."""
        raw = self._base
        for feat, thr, left, right, leaf, val, mgl in self._trees:
            n = 0
            while not leaf[n]:
                xv = x[feat[n]]
                if xv != xv:                       # NaN -> follow the trained missing direction
                    n = left[n] if mgl[n] else right[n]
                elif xv <= thr[n]:
                    n = left[n]
                else:
                    n = right[n]
            raw += val[n]
        return 1.0 / (1.0 + math.exp(-raw))

    def score(self, rich_feats, buyers, creator):
        _t0 = time.perf_counter()
        d = dict(rich_feats); d.update(self.rep.features(buyers, creator))
        x = [float(d.get(f, 0.0)) for f in self.FE]    # plain list -> fast scalar indexing in the traversal
        _t1 = time.perf_counter()
        pr = self._proba_fast(x)
        _t2 = time.perf_counter()
        self._score_us = ((_t1 - _t0) * 1e6, (_t2 - _t1) * 1e6)
        return pr, d

    def _log_timing(self, slot, recv_ts, mint, ring_us, score_us, dq):
        feat_us, pred_us = self._score_us
        try:
            self._timing_out.write(json.dumps({
                "slot": slot, "recv_ts": recv_ts, "mint": mint,
                "ring_us": round(ring_us, 1), "feat_us": round(feat_us, 1),
                "predict_us": round(pred_us, 1), "score_us": round(score_us, 1),
                "decide_us": round(ring_us + score_us, 1), "dq": dq,
                "ev_avg_us": round(self._ev_us_sum / self._ev_n, 2) if self._ev_n else 0.0,
                "ev_n": self._ev_n}) + "\n")
            self._timing_out.flush()
        except Exception:
            pass

    def buyers_of(self, mint, ts):
        """Distinct buy-intent signers for this mint, causal (recv <= cross). FORCE a drain first
        (like the launch bot's shadow_harness) so the ring isn't stale for fast/burst coins.
        Returns (buyers, dq_size, buy_raw) for diagnosis."""
        if not self.shred_window:
            return [], 0, 0
        try: self.shred_window.drain_now()
        except Exception: pass
        dq = self.shred_window._by_mint.get(mint)
        if not dq:
            return [], 0, 0
        buy_raw = sum(1 for it in dq if it.get("is_buy"))
        buyers = sorted({it.get("user") for it in dq
                         if it.get("is_buy") and it.get("user") and (it.get("recv_ns", 0) / 1e9 <= ts + 0.5)})
        return buyers, len(dq), buy_raw

    def _bid_tip(self, mint):
        """Dynamic buy tip: base, bumped to stay competitive when the intent ring shows a swarm of
        TIPPING rivals on this mint (most of the field competes via priority fee, so tips are usually
        ~0 -> base). Drains the ring fresh first (the performant non-blocking drain). Bounded [base, cap]."""
        base = self.buy_tip_base
        if not self.shred_window:
            return base
        try:
            self.shred_window.drain_now()
            dq = self.shred_window._by_mint.get(mint)
            if not dq:
                return base
            tips = sorted(int(it.get("jito_tip_lam", 0) or 0) for it in dq if it.get("is_buy"))
            if not tips or tips[-1] <= 0:
                return base
            p75 = tips[min(len(tips) - 1, int(0.75 * len(tips)))]
            return max(base, min(self.buy_tip_cap, int(p75 * 1.2)))
        except Exception:
            return base

    def seed_mint_meta(self, mint, token_program=None, creator=None):
        if creator:
            self.mint_creator[mint] = creator
        if self.broker is not None and hasattr(self.broker, "seed_mint_meta"):
            try: self.broker.seed_mint_meta(mint, token_program, creator)
            except Exception: pass

    def _on_buy_fill(self, mint, fill_mid):
        """Broker callback: a BUY landed. RE-ANCHOR to the ACTUAL fill mid (replacing any speculative
        slipped-entry anchor set at submit), with fill-anchored TP/STOP. If a speculative exit already
        cleared the position before the fill confirmed (holdings drained), there is nothing left to
        manage -> skip (no ghost position). LIVE only (dormant in dry-run)."""
        try:
            now = time.time()
            self.awaiting_fill.discard(mint); self.awaiting_ts.pop(mint, None)
            if not fill_mid or fill_mid <= 0:
                self.emit({"kind": "buy_fill_bad", "mint": mint, "t": now, "fill_mid": fill_mid}); return
            fm = float(fill_mid); self.live_exit_anchor_mid[mint] = fm
            prev = self.pos.get(mint)
            if prev is None and self.broker.holdings.get(mint, 0) <= 0:
                self.emit({"kind": "fill_post_exit", "mint": mint, "t": now, "fill_mid": fm}); return   # squeezed out same-block
            s = self.sel.get(mint, {}); base = prev or {}
            self.pos[mint] = {"p": base.get("p", s.get("p")), "is5": bool(base.get("is5", s.get("is5"))),
                              "fill_t": now, "exec_slip": 0.0,
                              "fill_mid": fm, "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP),
                              "buyers": base.get("buyers", s.get("buyers", [])), "creator": base.get("creator", s.get("creator")),
                              "buy_rep": base.get("buy_rep", s.get("buy_rep", 0.5)), "last_vsol": 0, "last_vtok": 0, "spec": False}
            self._bump(self.t, self.pos[mint]["is5"], "filled")
            self.emit({"kind": "live_fill", "mint": mint, "t": now, "fill_mid": fm, "reanchored": prev is not None,
                       "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP), "p": self.pos[mint]["p"]})
            self._persist_pos()
        except Exception as ex:
            self.awaiting_fill.discard(mint)
            self.emit({"kind": "buy_fill_err", "mint": mint, "err": str(ex)[:160]})

    def _on_buy_fail(self, mint, op, reason):
        """Broker callback: a bundle FAILED (reverted / expired / dropped). BUY -> drop the awaiting-fill so
        we never wait on a dead order (anti-zombie); SELL -> log (the broker's tip ladder retries the exit)."""
        try:
            now = time.time()
            if "sell" not in str(op).lower():
                self.awaiting_fill.discard(mint); self.awaiting_ts.pop(mint, None); self.sel.pop(mint, None)
                # buy reverted/expired -> drop the unconfirmed spec position (broker already rolled holdings
                # back). Since we now only sell after confirmation, a dead buy must never leave a managed pos.
                self.pos.pop(mint, None); self.live_exit_anchor_mid.pop(mint, None); self._persist_pos()
                self.emit({"kind": "buy_fail", "mint": mint, "t": now, "op": op, "reason": reason})
            else:
                self.emit({"kind": "sell_fail", "mint": mint, "t": now, "op": op, "reason": reason})
        except Exception as ex:
            self.emit({"kind": "fail_cb_err", "mint": mint, "err": str(ex)[:160]})

    async def _check_live_exit(self, mint, vsol, vtok, ts, slot):
        """LIVE fill-anchored exit. Runs on every trade for an open-position mint, INDEPENDENT of the
        tracker (which prunes a mint after ITS own outcome and would otherwise strand the position)."""
        pos = self.pos.get(mint)
        if not pos or vtok <= 0 or mint in self._exiting or pos.get("spec"):   # NEVER sell an unconfirmed (spec) position; wait for _on_buy_fill
            return
        pos["last_vsol"] = vsol; pos["last_vtok"] = vtok          # keep fresh reserves for the time-stop sweep
        mid = vsol / vtok
        reason = ("tp" if mid >= pos["tp"] else
                  ("stop" if mid <= pos["stop"] else
                   ("timestop" if ts - pos["fill_t"] >= self.max_hold_s else None)))
        if reason:
            await self._live_exit(mint, pos, vsol, vtok, ts, slot, reason, mid)

    async def _live_exit(self, mint, pos, vsol, vtok, ts, slot, reason, mid):
        """Fire the live sell, close the position, log fully. Exact proceeds reconcile in the broker recon
        log (broker_recon.jsonl); here we book the TRIGGER (approx net from the trigger mid)."""
        if mint in self._exiting:
            return
        self._exiting.add(mint)
        fm = pos.get("fill_mid") or mid
        ret = (mid / fm - 1.0) if fm else 0.0
        dur = ts - pos["fill_t"]; is5 = bool(pos.get("is5")); y = 1 if reason == "tp" else 0
        same_block = dur < SAME_BLOCK_S
        self.emit({"kind": "exit_trigger", "mint": mint, "t": ts, "reason": reason, "fill_mid": fm, "cur_mid": mid,
                   "ret": round(ret, 4), "dur_s": round(dur, 2), "tp": pos.get("tp"), "stop": pos.get("stop"), "slot": slot})
        try:
            await self.broker.sell_all(mint, vsol, vtok, slot=slot)
            self.emit({"kind": "sell_submit", "mint": mint, "t": ts, "reason": reason, "slot": slot})
        except Exception as ex:
            self.emit({"kind": "sell_err", "mint": mint, "err": str(ex)[:160]})
        net = self.bet * ret - PUMP_RT * self.bet - self.fixed_rt     # APPROX (trigger mid); broker recon = truth
        self._bump(self.t, is5, "closed"); self._bump(self.t, is5, "win" if y == 1 else "loss")
        self._bump(self.t, is5, "net_lam", int(net * LAMPORTS_PER_SOL))
        if not same_block:
            self._bump(self.t_rz, is5, "closed"); self._bump(self.t_rz, is5, "win" if y == 1 else "loss")
            self._bump(self.t_rz, is5, "net_lam", int(net * LAMPORTS_PER_SOL))
        try: self.rep.update(pos.get("buyers", []), pos.get("creator"), y)
        except Exception: pass
        self.recent_closes.appendleft({"mint": mint, "p": round(pos.get("p") or 0, 4), "is5": is5, "y": y, "bf": False,
                                       "ret_curve": round(ret, 4), "ret_outlay": round(ret, 4), "net_sol": round(net, 5),
                                       "dur_s": round(dur, 1), "same_block": same_block, "reason": reason,
                                       "buy_rep": round(pos.get("buy_rep", 0.5), 3)})
        self.emit({"kind": "live_close", "mint": mint, "t": ts, "reason": reason, "y": y, "ret": round(ret, 4),
                   "net_sol_approx": round(net, 5), "dur_s": round(dur, 1), "same_block": same_block})
        self.pos.pop(mint, None); self.sel.pop(mint, None); self.live_exit_anchor_mid.pop(mint, None)
        self._exiting.discard(mint)
        self._persist_pos()    # position closed -> drop it from the journal so a restart doesn't re-adopt a sold bag

    async def _sweep_live(self):
        """Zombie guard (every status tick): force-exit positions held past max_hold_s even if the coin
        STOPPED trading (no on_trade to fire the per-trade check), and drop buys that never filled."""
        if not self.live:
            return
        now = time.time()
        for m in list(self.awaiting_ts):
            t0 = self.awaiting_ts.get(m)
            if t0 and now - t0 >= self.fill_timeout_s:
                self.awaiting_fill.discard(m); self.awaiting_ts.pop(m, None); self.sel.pop(m, None)
                if self.pos.get(m, {}).get("spec"):     # unconfirmed buy timed out -> drop its spec position (never sell unconfirmed)
                    self.pos.pop(m, None); self.live_exit_anchor_mid.pop(m, None); self._persist_pos()
                self.emit({"kind": "buy_timeout", "mint": m, "t": now, "waited_s": round(now - t0, 1)})
        for m in list(self.pos):
            pos = self.pos.get(m)
            if not pos or m in self._exiting or pos.get("spec"):   # never timestop-SELL an unconfirmed position (dropped via buy_fail/timeout instead)
                continue
            if now - pos["fill_t"] >= self.max_hold_s:
                vs = pos.get("last_vsol") or 0; vt = pos.get("last_vtok") or 0
                mid = (vs / vt) if vt else (pos.get("fill_mid") or 0.0)
                await self._live_exit(m, pos, vs, vt, now, None, "timestop", mid)
        # orphan-holdings guard: a buy that confirmed AFTER its speculative sell reverted (the narrow
        # holdings-rollback race) leaves tokens with no managing position. Re-anchor so the normal
        # exit/timestop manages it -> we can never silently hold a zombie bag.
        for m, h in list(self.broker.holdings.items()):
            if h <= 0 or m in self.pos or m in self.awaiting_fill or m in self._exiting or m in self.ignore_mints:
                continue
            fm = self.live_exit_anchor_mid.get(m)
            if not fm:
                continue
            self.pos[m] = {"p": None, "is5": False, "fill_t": now, "exec_slip": 0.0,
                           "fill_mid": fm, "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP),
                           "buyers": [], "creator": None, "buy_rep": 0.5,
                           "last_vsol": 0, "last_vtok": 0, "spec": False}
            self.emit({"kind": "orphan_reanchor", "mint": m, "t": now, "fill_mid": fm, "holdings": h})
        self._persist_pos()    # also the 5s periodic safety write (a missed mutation-site call still lands here)

    def _load_ignore_mints(self):
        """Mints the bot must NEVER touch in recovery (old stranded bags the operator has written
        off). Read once at startup from IGNORE_MINTS_FILE (one mint/line, '#' comments). A restart
        re-reads it. Edit the file + restart to change the set."""
        s = set()
        try:
            if os.path.exists(IGNORE_MINTS_FILE):
                for line in open(IGNORE_MINTS_FILE):
                    tok = line.split("#", 1)[0].strip()
                    if tok:
                        s.add(tok)
        except Exception as ex:
            print(f"[rep-bot] ignore-list load error: {ex}", flush=True)
        if s:
            print(f"[rep-bot] ignore-list: {len(s)} mint(s) will be skipped in recovery", flush=True)
        return s

    def _persist_pos(self):
        """Atomically journal the open LIVE positions so a restart can resume managing them
        (the in-memory self.pos / broker.holdings both reset on restart -> without this every
        restart orphans open bags; that is exactly the failure we hit 2026-06-16). gRPC/eRPC
        independent: pure local disk. Carries the token amount (broker.holdings = chain-true
        after the gRPC fill reconcile) so a chainless restart can still seed the exit path."""
        if not self.live:
            return
        try:
            snap = {}
            for m, v in self.pos.items():
                snap[m] = {"p": v.get("p"), "is5": bool(v.get("is5")),
                           "fill_t": v.get("fill_t"), "orig_fill_t": v.get("orig_fill_t") or v.get("fill_t"),
                           "fill_mid": v.get("fill_mid"), "tp": v.get("tp"), "stop": v.get("stop"),
                           "buyers": list(v.get("buyers") or []), "creator": v.get("creator"),
                           "buy_rep": v.get("buy_rep", 0.5), "spec": bool(v.get("spec")),
                           "exec_slip": v.get("exec_slip", 0.0),
                           "tokens": int(self.broker.holdings.get(m, 0)) if self.broker else 0}
            tmp = POS_JOURNAL + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(snap, fh)
            os.replace(tmp, POS_JOURNAL)
        except Exception as ex:
            self.emit({"kind": "persist_pos_err", "err": str(ex)[:160]})

    async def _recover_positions(self):
        """STARTUP restart-recovery (LIVE only). Rebuild open positions so a restart never
        abandons a live bag. Primary truth = on-chain holdings (BOTH token programs); the
        journal supplies the original fill-anchored TP/STOP. Reconcile:
          - chain-held & journaled   -> restore with the ORIGINAL fill_mid/tp/stop (resume managing)
          - chain-held, not journaled-> adopt at the CURRENT curve mid if the curve is live;
                                         flag (no auto-exit) if the curve has migrated/closed
          - journaled, not on chain  -> sold while we were down; drop
        If the chain enumeration is unavailable (eRPC degraded), fall back to journal-only so we
        still resume managing what we believed we held (amounts from the journal)."""
        if not self.live or self.broker is None:
            return
        now = time.time()
        journ = {}
        try:
            if os.path.exists(POS_JOURNAL):
                journ = json.load(open(POS_JOURNAL)) or {}
        except Exception as ex:
            self.emit({"kind": "recover_journal_err", "err": str(ex)[:160]})
        try:
            chain = await self.broker.list_token_holdings()
        except Exception as ex:
            chain = []
            self.emit({"kind": "recover_chain_err", "err": str(ex)[:160]})
        chain_mints = {h["mint"] for h in chain}

        def _restore(m, j, raw, src):
            fm = float(j["fill_mid"])
            self.broker.holdings[m] = int(raw)
            self.pos[m] = {"p": j.get("p"), "is5": bool(j.get("is5")), "fill_t": now,
                           "orig_fill_t": j.get("orig_fill_t") or j.get("fill_t") or now, "exec_slip": j.get("exec_slip", 0.0),
                           "fill_mid": fm, "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP),
                           "buyers": list(j.get("buyers") or []), "creator": j.get("creator"),
                           "buy_rep": j.get("buy_rep", 0.5), "last_vsol": 0, "last_vtok": 0, "spec": False}
            self.live_exit_anchor_mid[m] = fm
            self.recovered["recovered"] += 1
            self.emit({"kind": "pos_recovered", "mint": m, "src": src, "fill_mid": fm, "tokens": int(raw),
                       "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP), "orig_fill_t": self.pos[m]["orig_fill_t"]})

        if chain:
            for h in chain:
                m = h["mint"]; raw = int(h["raw"])
                if raw <= 0:
                    continue
                if m in self.ignore_mints:        # old stranded bag: never seed holdings / adopt / flag
                    self.recovered["ignored"] += 1
                    self.emit({"kind": "recover_ignored", "mint": m, "tokens": raw})
                    continue
                j = journ.get(m)
                if j and j.get("fill_mid"):
                    _restore(m, j, raw, "journal+chain")
                    continue
                # un-journaled holding: adopt at the current mid if the curve is still live
                res = await self.broker._get_curve_reserves(m)        # (vsol, vtok, complete) | None
                if res and res[1] > 0 and not res[2]:
                    fm = res[0] / res[1]
                    self.broker.holdings[m] = raw
                    self.pos[m] = {"p": None, "is5": False, "fill_t": now, "orig_fill_t": now, "exec_slip": 0.0,
                                   "fill_mid": fm, "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP),
                                   "buyers": [], "creator": None, "buy_rep": 0.5,
                                   "last_vsol": res[0], "last_vtok": res[1], "spec": False}
                    self.live_exit_anchor_mid[m] = fm
                    self.recovered["adopted"] += 1
                    self.emit({"kind": "orphan_adopt", "mint": m, "fill_mid": fm, "tokens": raw,
                               "tp": fm * (1 + self.TP), "stop": fm * (1 - self.STOP)})
                else:
                    # migrated / closed curve -> cannot exit via the pump.fun path. Seed holdings
                    # so the reconcile loop tracks it, but do NOT fabricate a managing position
                    # (its TP/STOP would never fire and a sell would just abort). Surface loudly.
                    self.broker.holdings[m] = raw
                    self.recovered["unmanageable"] += 1
                    self.emit({"kind": "orphan_unmanageable", "mint": m, "tokens": raw, "reason": "no_live_curve", "curve": res})
            for m in journ:
                if m not in chain_mints:
                    self.recovered["dropped"] += 1
                    self.emit({"kind": "pos_dropped_not_on_chain", "mint": m})
        elif journ:
            # chain enumeration unavailable -> journal-only: resume managing what we believed we held
            for m, j in journ.items():
                if m in self.ignore_mints:
                    self.recovered["ignored"] += 1
                    continue
                if j.get("fill_mid") and int(j.get("tokens", 0)) > 0:
                    _restore(m, j, int(j["tokens"]), "journal_only")
            self.emit({"kind": "recover_journal_only", "n": self.recovered["recovered"]})

        self._persist_pos()
        print(f"[rep-bot] position recovery: {self.recovered} (journal {len(journ)}, chain {len(chain)})", flush=True)

    def rep_backfill(self):
        """Seed the dashboard baseline from the rich+rep panel scored by the LEAN+REP model
        (by-coin held-out for an honest OOS baseline), + recent_p/warm calibration."""
        if not os.path.exists(PANEL):
            print("[rep-bot] no panel for backfill", flush=True); return
        cr = {}; fl = {}; oc = {}
        for l in open(PANEL):
            try: e = json.loads(l)
            except Exception: continue
            k = e.get("kind"); m = e.get("mint")
            if k == "cross": cr[m] = e
            elif k == "fill": fl[m] = e
            elif k == "outcome": oc[m] = e
        keys = [m for m in oc if m in cr]
        if not keys:
            return
        self.bf_n = len(keys)
        X = np.array([[cr[m].get(f, 0.0) for f in self.FE] for m in keys], float)
        y = np.array([1 if oc[m]["y"] == 1 else 0 for m in keys], int)
        P = self.clf.predict_proba(X)[:, 1]                      # deployed model -> LIVE-cutoff calibration
        for p in P[-500:].tolist():
            self.recent_p.append(float(p))
        self.warm = {"5": float(np.quantile(P, 1 - NARROW)), "10": float(np.quantile(P, 1 - self.WIDE))}
        bk = np.array([int(hashlib.md5(m.encode()).hexdigest(), 16) % 100 for m in keys])
        tr = bk < 70; idx = np.where(bk >= 70)[0]; self.bf_test_frac = float(len(idx) / len(keys))
        if int(tr.sum()) < 50 or len(idx) < 20:
            return
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf_bf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                                l2_regularization=1.0).fit(X[tr], y[tr])
        pte = clf_bf.predict_proba(X[idx])[:, 1]                 # by-coin OOS -> HONEST baseline (not in-sample)
        cut5 = float(np.quantile(pte, 1 - NARROW)); cut10 = float(np.quantile(pte, 1 - self.WIDE))
        cap_frac = self.args.cap_bps / 10000.0; revert_lam = int(0.0006 * LAMPORTS_PER_SOL)
        closes = []
        for jj, gi in enumerate(idx):
            p = float(pte[jj])
            if p < cut10:
                continue
            m = keys[gi]; is5 = bool(p >= cut5)
            self._bump(self.bf, is5, "selected")
            c = cr[m]; f = fl.get(m); o = oc[m]
            slip = (f["fill_mid"] / c["cross_mid"] - 1.0) if (f and c.get("cross_mid")) else 0.0
            if slip > cap_frac:
                self._bump(self.bf, is5, "reverted"); self._bump(self.bf, is5, "net_lam", -revert_lam); continue
            self._bump(self.bf, is5, "filled")
            ret = o["ret"]; ro = ret - PUMP_RT - self.fixed_rt / self.bet
            net = self.bet * ret - PUMP_RT * self.bet - self.fixed_rt
            self._bump(self.bf, is5, "closed"); self._bump(self.bf, is5, "win" if o["y"] == 1 else "loss")
            self._bump(self.bf, is5, "net_lam", int(net * LAMPORTS_PER_SOL))
            self._bump(self.bf, is5, "sum_curve", ret); self._bump(self.bf, is5, "sum_outlay", ro)
            closes.append((p, is5, o["y"], ret, ro, net))
        for (p, is5, y0, ret, ro, net) in closes[-12:]:
            self.recent_closes.append({"mint": "(backfill)", "p": round(p, 3), "is5": is5, "y": y0,
                                       "ret_curve": round(ret, 3), "ret_outlay": round(ro, 3),
                                       "net_sol": round(net, 5), "dur_s": 0, "bf": True})
        print(f"[rep-bot] backfilled OOS {len(idx)}/{len(keys)} (frac {self.bf_test_frac:.2f}): "
              f"top5 {self.bf['5']['closed']} / top10 {self.bf['10']['closed']} closes", flush=True)

    async def on_trade(self, ev):
        self.stats["events"] += 1
        if ev is None or not getattr(ev, "is_classic_curve", False) or ev.virtual_token_reserves <= 0:
            return
        ts = time.time()
        if self.live and ev.mint in self.pos:        # LIVE exit check, independent of the tracker's prune lifecycle
            await self._check_live_exit(ev.mint, ev.virtual_sol_reserves, ev.virtual_token_reserves, ts, getattr(ev, "slot", None))
        _eu0 = time.perf_counter()
        evs = self.trk.update(ev.mint, ev.virtual_sol_reserves, ev.virtual_token_reserves,
                              ev.real_sol_reserves, ev.sol_amount, ev.is_buy, ts, getattr(ev, "user", ""))
        self._ev_us_sum += (time.perf_counter() - _eu0) * 1e6; self._ev_n += 1
        if evs:
            await self.on_events(evs, ev.virtual_sol_reserves, ev.virtual_token_reserves, ts, getattr(ev, "slot", None))

    async def on_trade_amm(self, ev):
        """GRADUATION feed adapter: a PumpSwap AMM trade (graduation_listener.AmmEvent). Maps the pool
        reserves into the venue-agnostic tracker (vsol<-quote_res, vtok<-base_res, real_sol<-quote_res)
        and routes crosses through the SAME on_events path (constant-product, so plan_buy/simulate_fill
        are identical). Trade-size = |delta quote reserve| (exact; no trusted sol_amount field on AMM)."""
        self.stats["events"] += 1
        if ev is None or ev.base_res <= 0 or ev.quote_res <= 0:
            return
        ts = ev.t
        prev = self._prev_qres.get(ev.mint)
        sol_amount = abs(ev.quote_res - prev) if prev is not None else 0
        self._prev_qres[ev.mint] = ev.quote_res
        if self.live and ev.mint in self.pos:        # LIVE exit check (dormant in dry-run; broker=None)
            await self._check_live_exit(ev.mint, ev.quote_res, ev.base_res, ts, ev.slot)
        _eu0 = time.perf_counter()
        evs = self.trk.update(ev.mint, ev.quote_res, ev.base_res, ev.quote_res, sol_amount, ev.is_buy, ts, ev.user)
        self._ev_us_sum += (time.perf_counter() - _eu0) * 1e6; self._ev_n += 1
        if evs:
            await self.on_events(evs, ev.quote_res, ev.base_res, ts, ev.slot)

    async def on_events(self, events, vsol, vtok, ts, slot):
        for e in events:
            k = e["kind"]; m = e["mint"]
            if k == "cross":
                self.crosses += 1
                _tc0 = time.perf_counter()
                buyers, dq_size, buy_raw = self.buyers_of(m, ts); creator = self.mint_creator.get(m)
                _tc1 = time.perf_counter()
                rich = {f: e.get(f, 0.0) for f in self.LEAN}
                p, full = self.score(rich, buyers, creator)
                _tc2 = time.perf_counter()
                self._log_timing(slot, ts, m, (_tc1 - _tc0) * 1e6, (_tc2 - _tc1) * 1e6, dq_size)
                self.recent_p.append(p)
                cut10 = self._cut("10", self.WIDE)
                if p < cut10:
                    self.emit({"kind": "cross_skip", "mint": m, "t": ts, "p": round(p, 4), "cut10": round(cut10, 4),
                               "buy_rep": round(full.get("buy_rep_mean", 0.5), 3), "buy_n": full.get("buy_n", 0),
                               "dq": dq_size, "buy_raw": buy_raw,
                               "vol_sol": round(rich.get("vol_sol", 0.0), 2), "mcap_sol": round(rich.get("mcap_sol", 0.0), 2)})
                    continue
                is5 = bool(p >= self._cut("5", NARROW))
                if self.band_only and is5:                         # band-only: skip the top-5%, fire only cut10<=p<cut5
                    self.emit({"kind": "skip_top5", "mint": m, "t": ts, "p": round(p, 4)})
                    continue
                self._bump(self.t, is5, "selected")
                plan = plan_buy(vsol, vtok, self.bet_lam, self.args.cap_bps)
                self.sel[m] = {"plan": plan, "p": p, "is5": is5, "t": ts, "buyers": buyers,
                               "creator": creator, "buy_rep": full.get("buy_rep_mean", 0.5)}
                self.emit({"kind": "decision", "mint": m, "t": ts, "p": round(p, 4), "is5": is5,
                           "buy_rep_mean": round(full.get("buy_rep_mean", 0.5), 3),
                           "buy_known_frac": round(full.get("buy_known_frac", 0.0), 3),
                           "buy_n": full.get("buy_n", 0), "dq": dq_size, "buy_raw": buy_raw,
                           "cre_winrate": round(full.get("cre_winrate", 0.5), 3)})
                if self.live:
                    if self.broker and self.broker.realized_net_sol() <= -self.max_loss_sol:
                        if not self.halted:
                            self.halted = True
                            print(f"[rep-bot] HALT: realized {self.broker.realized_net_sol():.4f} SOL <= -{self.max_loss_sol} -> no new buys", flush=True)
                        self.emit({"kind": "halt_new_buys", "mint": m, "t": ts, "realized_net_sol": round(self.broker.realized_net_sol(), 6)})
                        continue
                    if len(set(self.pos) | self.awaiting_fill) >= self.max_concurrent:   # bound total live exposure
                        self.emit({"kind": "skip_max_concurrent", "mint": m, "t": ts,
                                   "open": len(self.pos), "awaiting": len(self.awaiting_fill), "cap": self.max_concurrent})
                        continue
                    self.awaiting_fill.add(m); self.awaiting_ts[m] = ts
                    tip = self._bid_tip(m)
                    self.emit({"kind": "submit", "mint": m, "t": ts, "slot": slot, "p": round(p, 4), "is5": is5,
                               "bet_sol": self.bet, "vsol": vsol, "vtok": vtok, "tip_lam": tip, "tip_base": self.buy_tip_base})
                    try:
                        await self.broker.buy(m, self.bet, vsol, vtok, slot=slot, tip_lamports_override=tip)
                        # PENDING anchor (2026-06-17: speculative same-block SELL removed). Anchor the position at
                        # the worst-case slipped entry mid so _on_buy_fill has a base, but DO NOT sell it until the
                        # buy is CONFIRMED filled — spec=True is skipped by _check_live_exit and _sweep_live, and a
                        # reverted/expired buy is dropped by _on_buy_fail. This kills the fee-bleed of selling
                        # unconfirmed buys that revert (~96% at gap-1). _on_buy_fill flips spec->False + re-anchors.
                        bslip = getattr(self.broker, "slippage_bps_buy", 1500) / 1e4
                        amid = float(e.get("cross_mid", 0.0)) * (1.0 + bslip)
                        if amid > 0:
                            self.pos[m] = {"p": p, "is5": is5, "fill_t": ts, "exec_slip": 0.0,
                                           "fill_mid": amid, "tp": amid * (1 + self.TP), "stop": amid * (1 - self.STOP),
                                           "buyers": buyers, "creator": creator, "buy_rep": full.get("buy_rep_mean", 0.5),
                                           "last_vsol": vsol, "last_vtok": vtok, "spec": True}
                            self.live_exit_anchor_mid[m] = amid
                            self.emit({"kind": "spec_anchor", "mint": m, "t": ts, "assumed_mid": amid, "buy_slip_bps": int(bslip * 1e4),
                                       "tp": amid * (1 + self.TP), "stop": amid * (1 - self.STOP)})
                            self._persist_pos()    # journal the commitment immediately (a crash now must not orphan it)
                    except Exception as ex:
                        self.awaiting_fill.discard(m); self.awaiting_ts.pop(m, None); self.pos.pop(m, None)   # don't strand an awaiting/spec on submit error
                        self.emit({"kind": "submit_err", "mint": m, "err": str(ex)[:160]})
            elif k == "fill" and m in self.sel:
                if self.live:
                    continue                          # LIVE: the real fill arrives via the broker callback (_on_buy_fill)
                s = self.sel[m]; is5 = s["is5"]
                fr = simulate_fill(vsol, vtok, s["plan"], tip_lam=self.tip_lam, priority_fee_lam=self.prio_lam)
                if fr.filled:
                    self._bump(self.t, is5, "filled")
                    self.pos[m] = {"p": s["p"], "is5": is5, "fill": fr, "plan": s["plan"], "fill_t": ts,
                                   "exec_slip": fr.exec_slip, "buyers": s["buyers"], "creator": s["creator"],
                                   "buy_rep": s["buy_rep"]}
                else:
                    self._bump(self.t, is5, "reverted")
                    self._bump(self.t, is5, "net_lam", -int(0.0006 * LAMPORTS_PER_SOL))
                    self.sel.pop(m, None)
                self.emit({"kind": "fill", "mint": m, "t": ts, "is5": is5, "filled": fr.filled,
                           "exec_slip": round(fr.exec_slip, 4) if fr.filled else None})
            elif k == "outcome" and m in self.pos:
                if self.live:
                    continue                          # LIVE: exit is fill-anchored + bot-managed (_check_live_exit / _sweep_live)
                pos = self.pos.pop(m); self.sel.pop(m, None); is5 = pos["is5"]
                tr = realized_return(pos["fill"], pos["plan"], vsol, vtok, exit_tip_lam=0, priority_fee_lam=self.prio_lam)
                dur = ts - pos["fill_t"]; same_block = dur < SAME_BLOCK_S   # <1 slot: cross+fill+win same instant = gap-0 illusion, NOT realizable
                self._bump(self.t, is5, "closed"); self._bump(self.t, is5, "win" if e["y"] == 1 else "loss")
                self._bump(self.t, is5, "net_lam", tr.net_pnl_lam)
                self._bump(self.t, is5, "sum_curve", tr.return_on_curve); self._bump(self.t, is5, "sum_outlay", tr.return_on_outlay)
                if not same_block:                                         # REALIZABLE subset (excludes the same-block mirage)
                    self._bump(self.t_rz, is5, "closed"); self._bump(self.t_rz, is5, "win" if e["y"] == 1 else "loss")
                    self._bump(self.t_rz, is5, "net_lam", tr.net_pnl_lam)
                    self._bump(self.t_rz, is5, "sum_curve", tr.return_on_curve); self._bump(self.t_rz, is5, "sum_outlay", tr.return_on_outlay)
                self.rep.update(pos["buyers"], pos["creator"], e["y"])     # keep the reputation map current (causal)
                cl = {"mint": m, "p": round(pos["p"], 4), "is5": is5, "y": e["y"], "bf": False,
                      "ret_curve": round(tr.return_on_curve, 4), "ret_outlay": round(tr.return_on_outlay, 4),
                      "net_sol": round(tr.net_pnl_lam / LAMPORTS_PER_SOL, 5), "dur_s": round(dur, 1),
                      "same_block": same_block, "buy_rep": round(pos["buy_rep"], 3)}
                self.recent_closes.appendleft(cl); self.emit({"kind": "outcome", "t": ts, **cl})
                if self.live:
                    try: await self.broker.sell_all(m, vsol, vtok, slot=slot)
                    except Exception: pass

    def write_status(self):
        """Same schema as the base bot (so the dashboard works) + reputation extras: a model tag,
        buy_rep per open position, live signer count, events, ring status."""
        now = time.time()
        opens = [{"mint": m, "p": (round(v["p"], 3) if v.get("p") is not None else None),
                  "is5": v["is5"], "age_s": round(now - v["fill_t"], 0),
                  "exec_slip": round(v.get("exec_slip", 0.0), 3), "buy_rep": round(v.get("buy_rep", 0.5), 3),
                  "spec": bool(v.get("spec"))}
                 for m, v in list(self.pos.items())[:30]]
        status = {"ts": now, "mode": "LIVE" if self.live else "DRY-RUN", "model": "LEAN",
                  "uptime_s": round(now - self.t0), "rep_signers": len(self.rep.sig),
                  "events": self.stats.get("events", 0), "intent_ring": self.shred_window is not None,
                  "bet_sol": self.bet, "tip_sol": self.tip, "prio_sol": round(self.prio_lam / LAMPORTS_PER_SOL, 5),
                  "fixed_rt_sol": round(self.fixed_rt, 5), "cap_bps": self.args.cap_bps, "bf_n": self.bf_n,
                  "bf_test_frac": round(self.bf_test_frac, 3), "crosses": self.crosses, "open": len(self.pos),
                  "recovered": self.recovered,
                  "realized_net_sol": (round(self.broker.realized_net_sol(), 6) if (self.live and self.broker) else 0.0),
                  "halted": self.halted, "max_loss_sol": self.max_loss_sol, "max_concurrent": self.max_concurrent, "band_only": self.band_only,
                  "cut5": round(self._cut("5", NARROW), 4), "cut10": round(self._cut("10", self.WIDE), 4),
                  "bf5": self._block(self.bf, "5"), "bf10": self._block(self.bf, "10"),
                  "live5": self._block(self.t, "5"), "live10": self._block(self.t, "10"),
                  "live5_rz": self._block(self.t_rz, "5"), "live10_rz": self._block(self.t_rz, "10"),
                  "same_block_s": SAME_BLOCK_S,
                  "open_positions": opens, "recent_closes": list(self.recent_closes)}
        tmp = STATUS + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(status, fh)
        os.replace(tmp, STATUS)

    async def _status_loop(self):
        while True:
            await asyncio.sleep(5)
            try: await self._sweep_live()        # zombie guard: time-stop stalled positions + drop unfilled buys
            except Exception: pass
            try: self.write_status()
            except Exception: pass

    async def run(self):
        await self.setup()
        if self.broker is not None and hasattr(self.broker, "set_fill_callback"):
            self.broker.set_fill_callback(self._on_buy_fill)
        if self.broker is not None and hasattr(self.broker, "set_failure_callback"):
            self.broker.set_failure_callback(self._on_buy_fail)
        try:
            await self._recover_positions()    # rebuild open positions from chain+journal BEFORE trades flow (anti-orphan)
        except Exception as ex:
            print(f"[rep-bot] position recovery error: {ex}", flush=True)
        try:
            from shred_window import ShredWindow
            self.shred_window = ShredWindow(); self.shred_window.start()
            print("[rep-bot] intent ring attached (pumpfun_intents)", flush=True)
        except Exception as ex:
            print(f"[rep-bot] intent ring unavailable: {ex}", flush=True); self.shred_window = None
        from .graduation_listener import grad_grpc_listener_for_harness
        asyncio.create_task(self._status_loop()); self.write_status()
        print(f"[grad-bot] PumpSwap AMM gRPC listener starting (live={self.live}, RICH-only graduation model, "
              f"shred_ring={'on' if self.shred_window else 'off'})", flush=True)
        await grad_grpc_listener_for_harness(self)


if __name__ == "__main__":
    asyncio.run(RepLive(parse_args()).run())
