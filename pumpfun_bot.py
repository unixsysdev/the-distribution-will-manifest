"""pumpfun_bot.py — production bot wrapper around ShadowHarness.

Adds:
  - Position persistence via PositionStore (atomic JSONL); restart recovery force-closes
    open positions using snapshots collected so far so we never silently double-enter.
  - Paper/live mode flag. Live execution is DOUBLE-GATED behind both the `--live` flag
    AND the PUMPFUN_LIVE_OK=1 env var, so accidental execution requires two explicit acts.
  - JitoBroker wiring (live mode) via the existing jito_exec module; PaperBroker (paper
    mode) is a no-op stub that just logs intents.
  - Optional periodic snapshot of stats + open-position summary to a status.json so
    systemd / monitoring can poll without grepping the log.

Restart pattern (paper mode, v1):
  On startup, replay positions.jsonl. Any position with open events but no matching
  close gets force-closed using snapshots collected so far, then logged with reason
  "restart". The mint is added to closed_mints so we don't re-enter on the same trade
  stream. Fresh trades drive a clean accumulator from there.

Restart pattern (live mode, TODO):
  Same JSONL replay, BUT before force-closing we should query the chain to see what
  the wallet actually owns and reconcile. For v1 we just close everything in the book
  and the broker is responsible for emitting a market sell of any on-chain remnants.
  This is conservative and safe — the bot loses any partial profit a stuck position
  might have recovered, but never silently double-enters.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))

from shadow_harness import ShadowHarness
from position_store import PositionStore
from jito_broker import PaperBroker, JitoBroker


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="bot_artifacts_K7V",
                    help="model artifact dir")
    ap.add_argument("--data-dir", default="bot_data",
                    help="dir for positions.jsonl, status.json, etc.")
    ap.add_argument("--log-path", default=None,
                    help="event log path (default: <data-dir>/shadow_run.jsonl)")
    ap.add_argument("--bet-sol", type=float, default=None,
                    help="bet size in SOL")
    ap.add_argument("--live", action="store_true",
                    help="enable live execution (also requires PUMPFUN_LIVE_OK=1 env AND "
                         "JITO_DRY_RUN=0 env to actually POST bundles)")
    ap.add_argument("--source", choices=["ws", "grpc"], default="ws",
                    help="listener source. ws (default) = solana websocket logsSubscribe "
                         "(validated, no slot in events). grpc = Geyser gRPC (lower latency, "
                         "carries slot for same-block aim) — listener_grpc_bot.py.")
    ap.add_argument("--status-interval", type=int, default=30,
                    help="seconds between status.json snapshots")
    ap.add_argument("--entry-threshold", type=float, default=None,
                    help="override the model's loaded entry threshold. Use sparingly; "
                         "intended for DRY_RUN end-to-end validation when calibration is "
                         "still being investigated.")
    return ap.parse_args()


async def status_writer(h: ShadowHarness, path: Path, interval: int):
    while True:
        await asyncio.sleep(interval)
        try:
            rets = h.book.returns()
            n = len(rets)
            doc = {"t": time.time(), "uptime_s": time.time() - h.start_ts,
                   "stats": dict(h.stats),
                   "n_closed": int(n),
                   "n_open_paper": len(h.open_paper),
                   "n_closed_mints": len(h.closed_mints),
                   "n_states_tracked": len(h.states),
                   "mean_net": float(rets.mean()) if n else None,
                   "win_pct": float(100*(rets>0).mean()) if n else None,
                   "model": getattr(h, "model_info", {}),
                   "era": {
                       "start_t": h.start_ts,
                       "ready": h.stats.get("both_ready", 0),
                       "fires": h.stats.get("entry_fire", 0),
                       "closed": len(getattr(h, "era_book_nets", []) or []),
                       "mean_book_net": (sum(h.era_book_nets) / len(h.era_book_nets))
                                        if getattr(h, "era_book_nets", None) else None,
                       "mean_policy_net": (sum(h.era_policy_nets) / len(h.era_policy_nets))
                                          if getattr(h, "era_policy_nets", None) else None,
                       "win_pct_policy": (100.0 * sum(1 for x in h.era_policy_nets if x > 0)
                                          / len(h.era_policy_nets))
                                         if getattr(h, "era_policy_nets", None) else None,
                   },
                   "broker_kind": type(h.broker).__name__ if h.broker is not None else None,
                   "dry_run": bool(getattr(h.broker, "dry_run", True)),
                   "listener_source": args.source,
                   "entry_threshold": float(h.srv.entry_threshold)}
            # Recon stats from JitoBroker (if available)
            if h.broker is not None and hasattr(h.broker, "recon_summary"):
                try:
                    doc["recon"] = h.broker.recon_summary()
                    doc["pending_bundles"] = len(getattr(h.broker, "pending_bundles", {}))
                except Exception:
                    pass
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, default=str))
            tmp.replace(path)
        except Exception as e:
            print(f"[bot] status_writer error: {e}", flush=True)


async def amain(args):
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else (data_dir / "shadow_run.jsonl")
    pos_path = data_dir / "positions.jsonl"
    status_path = data_dir / "status.json"

    # Broker selection — double-gated for live, triple-gated for actual submission
    if args.live:
        if os.getenv("PUMPFUN_LIVE_OK") != "1":
            print("[bot] --live passed but PUMPFUN_LIVE_OK!=1 — refusing to start "
                  "in live mode. Re-run with PUMPFUN_LIVE_OK=1 to override.",
                  flush=True)
            sys.exit(2)
        broker = await JitoBroker.create(bet_sol=args.bet_sol)
        dry = os.getenv("JITO_DRY_RUN", "1") == "1"
        print(f"[bot] mode=LIVE (JitoBroker active, JITO_DRY_RUN={dry})", flush=True)
        if dry:
            print("[bot] DRY_RUN=1 — bundles will be ASSEMBLED + LOGGED but NOT POSTED. "
                  "Set JITO_DRY_RUN=0 to actually submit.", flush=True)
    else:
        broker = PaperBroker(log_path=str(data_dir / "broker_paper.jsonl"))
        print("[bot] mode=PAPER (no execution)", flush=True)

    # Position store + replay-recover
    pstore = PositionStore(pos_path, fsync_every_event=args.live)
    h = ShadowHarness(artifact_dir=args.artifact_dir, log_path=str(log_path),
                      position_store=pstore, broker=broker)
    spec = getattr(h.srv, "spec", {}) or {}
    h.model_info = {
        "artifact": os.path.realpath(args.artifact_dir),
        "artifact_kind": spec.get("artifact_kind"),
        "spec_created_at": spec.get("created_at_utc") or spec.get("created_at"),
        "spec_exit_policy": spec.get("exit_policy"),
        "exit_policy": getattr(h.exit_policy, "NAME", None),
        "threshold": float(h.srv.entry_threshold),
        "n_features": len(h.srv.entry_features),
        "rich_entry": bool(getattr(h.srv, "rich_entry", False)),
    }
    h.log_event("model_loaded", **h.model_info)
    if args.entry_threshold is not None:
        orig = h.srv.entry_threshold
        h.srv.entry_threshold = float(args.entry_threshold)
        print(f"[bot] entry threshold OVERRIDDEN: {orig:.4f} -> "
              f"{h.srv.entry_threshold:.4f} (--entry-threshold)",
              flush=True)
    # Listener source — swappable. WS is the validated default; gRPC available for
    # latency uplift + same-block aim (carries tx.slot per event).
    if args.source == "grpc":
        from listener_grpc_bot import grpc_listener_for_harness
        h.listener = lambda: grpc_listener_for_harness(h)
        from listener_grpc_bot import GRPC_ENDPOINT as _GE; print(f"[bot] listener=gRPC ({_GE}, same-block aim active)", flush=True)
    else:
        print("[bot] listener=WS (slot absent from events; same-block aim degrades to "
              "next-slot at best)", flush=True)
    # Replay positions BEFORE starting the listener. KEEP open positions OPEN
    # so the running exit policy decides when to sell, not the restart event.
    # The harness's listener will resume processing trades for these mints as
    # they arrive on the live gRPC stream; exit policy fires naturally when
    # conditions are met (level_tp_100 at ret>=1.0, c_hybrid_t30 at ret>0
    # de-risk, stale_watchdog at 5min no-activity, etc).
    open_at_restart = pstore.replay(h.book, force_close_on_restart=False,
                                    restart_reason="restart")
    # Seed the stale watchdog for restored-open mints: in a fresh process
    # last_trade_ts is empty and the watchdog's get(m, now) default would age
    # them from "now" forever on a dead token (zombie position). Seeding with
    # the restore time makes them stale out after STALE_SEC like any other.
    _now = time.time()
    for m, pos in h.book.positions.items():
        if not pos.closed:
            h.last_trade_ts.setdefault(m, _now)
    # Mark CLOSED-historic mints as "do not re-enter". Open positions stay
    # ABSENT from closed_mints so the harness recognises them as still-open
    # (and so the entry path skips them anyway — they are already in book).
    for m, pos in h.book.positions.items():
        if pos.closed:
            h.closed_mints.add(m)
    # Restore broker.holdings for the still-open mints so subsequent sells
    # (from the exit policy) actually fire on-chain (LIVE) / log a sell
    # event (DRY_RUN / paper). Without this, broker.sell_all() would see
    # "no_holdings" because broker.holdings dict is reset on each startup.
    n_holdings_restored = 0
    if open_at_restart:
        from pump_fun_ix import tokens_out_for_sol
        bet_lam = int(getattr(broker, "bet_sol", args.bet_sol) * 1e9) if broker else 0
        for mint in open_at_restart:
            pos = h.book.positions.get(mint)
            if pos is None: continue
            # CRITICAL: the harness routes forward-phase trades and runs the
            # stale_watchdog through self.open_paper. Positions only present in
            # book.positions but NOT in open_paper are INVISIBLE to both —
            # they'd sit forever. Restore membership so exit policy actually
            # runs on restored positions.
            h.open_paper.add(mint)
            # Refresh last_trade_ts so the stale_watchdog gives the recovered
            # position a fresh 5-min window (rather than auto-closing it
            # immediately on the assumption the position has been silent since
            # the bot was down).
            h.last_trade_ts[mint] = time.time()
            # Restore broker.holdings (broker dict is empty on each startup,
            # otherwise broker.sell_all() would short-circuit at no_holdings).
            if broker is not None:
                try:
                    tok = tokens_out_for_sol(int(pos.vsK), int(pos.vtK), bet_lam)
                except Exception:
                    tok = 0
                if tok > 0 and hasattr(broker, "holdings"):
                    broker.holdings[mint] = broker.holdings.get(mint, 0) + tok
                    n_holdings_restored += 1
    h.log_event("recovery", positions_open_at_restart=list(open_at_restart),
                n_open_at_restart=len(open_at_restart),
                n_holdings_restored=n_holdings_restored,
                n_book_positions=len(h.book.positions),
                n_closed_mints=len(h.closed_mints))
    print(f"[bot] recovery: {len(open_at_restart)} position(s) restored open "
          f"(holdings re-injected for {n_holdings_restored}); "
          f"book has {len(h.book.positions)} historic position(s); "
          f"{len(h.closed_mints)} closed mint(s) will not be re-entered. "
          f"Open positions remain managed by the active exit policy.",
          flush=True)

    # Signal handling
    stop = asyncio.Event()
    def shutdown(*_): stop.set()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, shutdown)
        except NotImplementedError: pass

    runner = asyncio.create_task(h.run())
    writer = asyncio.create_task(status_writer(h, status_path, args.status_interval))
    closer = asyncio.create_task(stop.wait())
    done, pending = await asyncio.wait([runner, writer, closer],
                                        return_when=asyncio.FIRST_COMPLETED)
    for t in pending: t.cancel()

    # Final close: any positions opened during this run that are still open
    for m in list(h.open_paper):
        pos = h.book.positions.get(m)
        if pos is not None and not pos.closed:
            h.book._close_one(pos)
            h.log_event("position_close", mint=m, exit_kind=pos.kind,
                        net=pos.net_return, reason="shutdown")
            pstore.record_close(m, pos.net_return, pos.kind, "shutdown")

    rets = h.book.returns()
    h.log_event("shutdown", stats=h.stats, n_closed=len(rets),
                mean_net=float(rets.mean()) if len(rets) else None,
                win_pct=float(100*(rets>0).mean()) if len(rets) else None)
    print(f"[bot] shutdown — closed={len(rets)} "
          f"mean={rets.mean() if len(rets) else 0:+.3f}", flush=True)
    pstore.close()
    try: broker.close()
    except Exception: pass
    h.log.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(amain(args))
