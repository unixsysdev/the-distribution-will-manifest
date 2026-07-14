"""Empirical latency probe — gRPC vs WS for the same pump.fun TradeEvent stream.

Subscribes to BOTH grpc-fra1-1.erpc.global:80 and the wss logsSubscribe stream
in parallel. For every parsed TradeEvent records receive_time, signature, and
(for gRPC) the tx.slot from the envelope. Also polls getSlot every 200ms so we
can compute "how many slots behind the chain tip were we at receive time".

After RUN_SEC seconds, prints:
  - p50/p90/p99 of slot_lag at receive for gRPC (lower = fresher)
  - p50/p90/p99 of (receive_grpc - receive_ws) for signatures seen on both
    (positive = gRPC arrived first; what we hope to see)
  - sample count per source

Doesn't write anything to disk. Doesn't interfere with the running bot or capture
(separate subscriptions; erpc allows multiple).
"""
from __future__ import annotations
import asyncio
import base64
import statistics
import time

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent if "tools" in __file__ else _P(__file__).resolve().parent) + "/grpc_stubs")
import grpc
import base58
import geyser_pb2
import geyser_pb2_grpc

import config
from pumpfun_parse import parse_trade_event, TRADE_EVENT_DISC

from solders.pubkey import Pubkey
from solana.rpc.commitment import Confirmed
from solana.rpc.websocket_api import RpcTransactionLogsFilterMentions, connect
from solana.rpc.async_api import AsyncClient

GRPC_ENDPOINT = "grpc-fra1-1.erpc.global:80"
RUN_SEC = 30


async def slot_poller(state):
    """Maintain a fresh latest_slot via getSlot polling."""
    async with AsyncClient(config.rpc_http_url()) as cli:
        while not state["stop"]:
            try:
                resp = await cli.get_slot(commitment=Confirmed)
                state["latest_slot"] = int(resp.value)
                state["latest_slot_t"] = time.time()
            except Exception:
                pass
            await asyncio.sleep(0.2)


async def grpc_sub(state):
    metadata = (("x-token", config.GRPC_TOKEN),) if config.GRPC_TOKEN else None
    async with grpc.aio.insecure_channel(GRPC_ENDPOINT) as ch:
        stub = geyser_pb2_grpc.GeyserStub(ch)
        req = geyser_pb2.SubscribeRequest()
        req.transactions["f"].account_include.append(config.PUMP_FUN_PROGRAM)
        req.transactions["f"].failed = False
        async def itr():
            yield req
            while not state["stop"]:
                await asyncio.sleep(1)
        async for resp in stub.Subscribe(itr(), metadata=metadata):
            if state["stop"]: break
            if not resp.HasField("transaction"): continue
            t = time.time()
            tx = resp.transaction
            slot = int(tx.slot)
            sig = base58.b58encode(tx.transaction.signature).decode()
            meta = tx.transaction.meta
            if meta.err.err: continue
            for ln in meta.log_messages:
                if "Program data:" not in ln: continue
                b64 = ln.split("Program data:", 1)[1].strip()
                try: data = base64.b64decode(b64)
                except Exception: continue
                if len(data) < 8 or data[:8] != TRADE_EVENT_DISC: continue
                ev = parse_trade_event(data)
                if ev is None: continue
                # slot lag at receive
                lag_slots = (state["latest_slot"] - slot) if state["latest_slot"] else None
                state["grpc"].append({"t": t, "sig": sig, "slot": slot,
                                       "lag_slots": lag_slots, "mint": ev.mint,
                                       "ev_ts": ev.timestamp})


async def ws_sub(state):
    program = Pubkey.from_string(config.PUMP_FUN_PROGRAM)
    async with connect(config.rpc_ws_url()) as ws:
        await ws.logs_subscribe(RpcTransactionLogsFilterMentions(program),
                                commitment=Confirmed)
        await ws.recv()
        while not state["stop"]:
            msg = await ws.recv()
            t = time.time()
            if not isinstance(msg, list) or not msg: continue
            item = msg[0]
            val = getattr(item.result, "value", None)
            if val is None or val.err is not None: continue
            sig = str(val.signature)
            for ln in val.logs:
                if "Program data:" not in ln: continue
                b64 = ln.split("Program data:", 1)[1].strip()
                try: data = base64.b64decode(b64)
                except Exception: continue
                if len(data) < 8 or data[:8] != TRADE_EVENT_DISC: continue
                ev = parse_trade_event(data)
                if ev is None: continue
                state["ws"].append({"t": t, "sig": sig, "mint": ev.mint})


async def main():
    state = {"grpc": [], "ws": [], "stop": False,
             "latest_slot": 0, "latest_slot_t": 0.0}
    poller = asyncio.create_task(slot_poller(state))
    # warm-up
    await asyncio.sleep(2)
    print(f"running {RUN_SEC}s probe (gRPC + WS in parallel) ...")
    t0 = time.time()
    g = asyncio.create_task(grpc_sub(state))
    w = asyncio.create_task(ws_sub(state))
    await asyncio.sleep(RUN_SEC)
    state["stop"] = True
    for t in (g, w, poller): t.cancel()
    try: await asyncio.gather(g, w, poller, return_exceptions=True)
    except Exception: pass
    grpc_evs = state["grpc"]; ws_evs = state["ws"]
    print(f"\n=== {len(grpc_evs)} gRPC events  /  {len(ws_evs)} WS events  in {time.time()-t0:.1f}s ===")

    # receive-time vs event_timestamp (gRPC; second precision but converges at scale)
    ts_lags_ms = [int((e["t"] - e["ev_ts"]) * 1000) for e in grpc_evs if e.get("ev_ts")]
    if ts_lags_ms:
        ts_lags_ms.sort()
        def p(q): return ts_lags_ms[int(len(ts_lags_ms)*q/100)]
        print(f"\ngRPC receive_t - event_timestamp (wall-clock lag from on-chain ts):")
        print(f"  n={len(ts_lags_ms)}  min={min(ts_lags_ms)}ms  p50={p(50)}ms  "
              f"p90={p(90)}ms  p99={p(99)}ms  max={max(ts_lags_ms)}ms")
        print(f"  (event_timestamp is set by the validator with ~1s precision, so sub-second "
              f"lags can show as 0 or even slightly negative)")

    # slot lag at receive (gRPC only — WS doesn't carry slot)
    lags = [e["lag_slots"] for e in grpc_evs if e["lag_slots"] is not None]
    if lags:
        lags.sort()
        def pct(p): return lags[int(len(lags)*p/100)]
        print(f"\ngRPC slot lag at receive (chain_tip_slot - tx_slot):")
        print(f"  n={len(lags)}  min={min(lags)}  p50={pct(50)}  p90={pct(90)}  p99={pct(99)}  max={max(lags)}")
        print(f"  in ms (assuming 400ms/slot):  p50={pct(50)*400:.0f}  p90={pct(90)*400:.0f}  p99={pct(99)*400:.0f}")

    # per-signature dedup, compare gRPC vs WS receive times
    by_sig_g = {}; by_sig_w = {}
    for e in grpc_evs: by_sig_g.setdefault(e["sig"], e["t"])
    for e in ws_evs: by_sig_w.setdefault(e["sig"], e["t"])
    common = set(by_sig_g) & set(by_sig_w)
    if common:
        deltas_ms = [(by_sig_w[s] - by_sig_g[s]) * 1000.0 for s in common]
        deltas_ms.sort()
        def pct2(p): return deltas_ms[int(len(deltas_ms)*p/100)]
        print(f"\nWS_receive_t - gRPC_receive_t for {len(common)} signatures seen on both")
        print(f"  (positive => gRPC arrived first)")
        print(f"  mean={statistics.mean(deltas_ms):+.0f}ms  median={statistics.median(deltas_ms):+.0f}ms")
        print(f"  p10={pct2(10):+.0f}ms  p50={pct2(50):+.0f}ms  p90={pct2(90):+.0f}ms  p99={pct2(99):+.0f}ms")
        n_grpc_first = sum(1 for d in deltas_ms if d > 0)
        print(f"  gRPC arrived first in {n_grpc_first}/{len(common)} = {100*n_grpc_first/len(common):.0f}% of cases")
    else:
        print("\n(no signatures seen on both within window — increase RUN_SEC)")


if __name__ == "__main__":
    asyncio.run(main())
