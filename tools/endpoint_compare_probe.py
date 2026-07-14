"""Sequential gRPC-only comparison of two Yellowstone endpoints.

Endpoints are IP-allowed (no token needed), so we just open a stream to each
one in turn and measure how fresh / how fast it is. NO http RPC, NO websocket.

For each endpoint:
  (a) GetLatestBlockhash RTT — N samples, p50/p90/p99 ms
  (b) Subscribe stream for RUN_SEC seconds, count TradeEvents
  (c) per-event "freshness" — how stale the chain-state was when the event hit
      our socket. Two measures, both pure gRPC:
        - slot_lag    = (max_slot_seen_so_far - this_tx.slot)
          where max_slot_seen_so_far comes from the SAME stream's recent
          high-water mark. Lower = closer to the tip you saw.
        - wall_lag_s  = (recv_wall_time - tx.ev_ts)
          ev_ts is the validator-stamped TradeEvent.timestamp; second precision
          so it converges only at scale. Lower = stream pushed faster.

Run sequentially so we don't fight ERPC concurrent-stream limits.

Usage:
  GRPC_TOKEN=... ./venv/bin/python tools/endpoint_compare_probe.py
  RUN_SEC=45 BH_SAMPLES=20 ./venv/bin/python tools/endpoint_compare_probe.py
"""
from __future__ import annotations
import asyncio, base64, statistics, time, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grpc_stubs"))

import grpc
import base58
import geyser_pb2
import geyser_pb2_grpc

import config
from pumpfun_parse import parse_trade_event, TRADE_EVENT_DISC

EP_MAIN  = "grpc-fra1-1.erpc.global:80"
EP_BURST = "grpc-fra1-burst.erpc.global:80"

RUN_SEC    = int(os.getenv("RUN_SEC", "45"))
BH_SAMPLES = int(os.getenv("BH_SAMPLES", "20"))


def _meta():
    # IP-allowlisted endpoints don't need a token, but ERPC accepts it either
    # way — match the production bot.
    return (("x-token", config.GRPC_TOKEN),) if config.GRPC_TOKEN else None


def _q(xs, p):
    xs = sorted(xs)
    if not xs: return float("nan")
    k = min(int(len(xs) * p / 100), len(xs) - 1)
    return xs[k]


async def bh_rtt(endpoint: str, n: int) -> list[float]:
    out = []
    async with grpc.aio.insecure_channel(endpoint) as ch:
        stub = geyser_pb2_grpc.GeyserStub(ch)
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                _ = await stub.GetLatestBlockhash(geyser_pb2.GetLatestBlockhashRequest(),
                                                  metadata=_meta(), timeout=5)
                out.append((time.perf_counter() - t0) * 1000.0)
            except Exception as e:
                print(f"    bh err: {e}")
            await asyncio.sleep(0.3)
    return out


async def subscribe_window(endpoint: str, run_sec: int) -> dict:
    """Subscribe and return collected per-event stats for run_sec seconds."""
    events = []   # each: {"recv_t": ..., "tx_slot": ..., "ev_ts": ..., "sig": ...}
    max_slot = 0  # high-water mark of slots seen on this stream
    t_end = time.time() + run_sec

    async with grpc.aio.insecure_channel(endpoint) as ch:
        stub = geyser_pb2_grpc.GeyserStub(ch)
        req = geyser_pb2.SubscribeRequest()
        req.transactions["f"].account_include.append(config.PUMP_FUN_PROGRAM)
        req.transactions["f"].failed = False
        try:
            req.commitment = geyser_pb2.CommitmentLevel.PROCESSED
        except Exception:
            pass

        cancelled = asyncio.Event()
        async def itr():
            yield req
            await cancelled.wait()

        try:
            stream = stub.Subscribe(itr(), metadata=_meta())
            async for resp in stream:
                if time.time() >= t_end:
                    cancelled.set(); break
                if not resp.HasField("transaction"): continue
                recv_t = time.time()
                tx = resp.transaction
                slot = int(tx.slot)
                if slot > max_slot: max_slot = slot
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
                    events.append({
                        "recv_t": recv_t,
                        "tx_slot": slot,
                        "ev_ts": float(ev.timestamp) if ev.timestamp else 0.0,
                        "max_slot_at_recv": max_slot,
                        "sig": base58.b58encode(tx.transaction.signature).decode(),
                    })
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"    subscribe err: {e}")

    return {"events": events, "max_slot": max_slot}


def report(tag: str, res: dict, run_sec: int):
    evs = res["events"]
    print(f"\n  [{tag}] events={len(evs)}  rate={len(evs)/run_sec:.1f}/s   "
          f"max_slot_seen={res['max_slot']}")
    if not evs: return

    # slot_lag: how many slots behind THIS stream's own tip was each tx
    slot_lags = [e["max_slot_at_recv"] - e["tx_slot"] for e in evs]
    print(f"  [{tag}] slot_lag vs same-stream tip (slots):  "
          f"p50={_q(slot_lags,50)}  p90={_q(slot_lags,90)}  "
          f"p99={_q(slot_lags,99)}  max={max(slot_lags)}")

    # wall_lag: recv_t - on-chain ts (1s precision)
    wall_lags = [(e["recv_t"] - e["ev_ts"]) for e in evs if e["ev_ts"]]
    if wall_lags:
        wall_lags.sort()
        print(f"  [{tag}] wall_lag = recv_t - ev_ts (s):       "
              f"p50={_q(wall_lags,50):.3f}  p90={_q(wall_lags,90):.3f}  "
              f"p99={_q(wall_lags,99):.3f}  max={max(wall_lags):.3f}")

    # inter-arrival
    iats = []
    for i in range(1, len(evs)):
        iats.append((evs[i]["recv_t"] - evs[i-1]["recv_t"]) * 1000.0)
    if iats:
        print(f"  [{tag}] inter-arrival (ms):                  "
              f"p50={_q(iats,50):.1f}  p90={_q(iats,90):.1f}  "
              f"p99={_q(iats,99):.1f}  max={max(iats):.1f}")


async def main():
    print(f"=== endpoint_compare_probe @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"  MAIN  = {EP_MAIN}")
    print(f"  BURST = {EP_BURST}")
    print(f"  RUN_SEC={RUN_SEC}  BH_SAMPLES={BH_SAMPLES}   (sequential, gRPC-only)")

    # (a) GetLatestBlockhash RTT — sequential, each endpoint
    print(f"\n--- (a) GetLatestBlockhash RTT  ({BH_SAMPLES} samples, 0.3s gap) ---")
    for tag, ep in (("MAIN ", EP_MAIN), ("BURST", EP_BURST)):
        rtt = await bh_rtt(ep, BH_SAMPLES)
        if rtt:
            print(f"  [{tag}] n={len(rtt)}  min={min(rtt):.1f}  "
                  f"p50={_q(rtt,50):.1f}  p90={_q(rtt,90):.1f}  p99={_q(rtt,99):.1f}  "
                  f"max={max(rtt):.1f}  mean={statistics.mean(rtt):.1f} ms")

    # (b)+(c) Subscribe windows — back-to-back
    print(f"\n--- (b)+(c) Subscribe stream  ({RUN_SEC}s each, sequential) ---")
    for tag, ep in (("MAIN ", EP_MAIN), ("BURST", EP_BURST)):
        print(f"  [{tag}] subscribing {ep} for {RUN_SEC}s ...")
        res = await subscribe_window(ep, RUN_SEC)
        report(tag, res, RUN_SEC)

    print("\n  done.")


if __name__ == "__main__":
    asyncio.run(main())
