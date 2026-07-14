"""Probe ERPC Direct Shreds endpoint.

Subscribes to ShredstreamProxy.SubscribeEntries for RUN_SEC seconds and
characterizes what comes through. We don't try to decode the Solana entries
bincode yet — that's Week 1 work. This is Week 0 recon.

Reports:
  - whether the endpoint speaks Jito's Shredstream protocol
  - rate of Entry messages
  - distribution of slot numbers vs current chain tip (how fresh are these
    shreds vs our existing Yellowstone executed-tx stream?)
  - byte-size distribution of the entries payload
  - first few raw bytes (so we can see what format the bincode-encoded
    Vec<Entry> looks like)
"""
from __future__ import annotations
import asyncio, sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shred_bot/stubs"))
sys.path.insert(0, str(ROOT / "grpc_stubs"))

import grpc
import shredstream_pb2
import shredstream_pb2_grpc

# For comparison: we already know what slot we're at via the Yellowstone gRPC
import geyser_pb2, geyser_pb2_grpc
import config

ENDPOINT = "shreds-fra6-1.erpc.global:80"
GEYSER_ENDPOINT = "grpc-fra1-1.erpc.global:80"
RUN_SEC = int(os.getenv("RUN_SEC", "10"))


async def get_current_slot():
    """One-shot getLatestBlockhash via Yellowstone to know the current chain tip."""
    async with grpc.aio.insecure_channel(GEYSER_ENDPOINT) as ch:
        stub = geyser_pb2_grpc.GeyserStub(ch)
        meta = (("x-token", config.GRPC_TOKEN),) if config.GRPC_TOKEN else None
        r = await stub.GetLatestBlockhash(geyser_pb2.GetLatestBlockhashRequest(),
                                          metadata=meta, timeout=5)
        return int(r.slot)


async def probe():
    print(f"=== ERPC Direct Shreds probe @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"  endpoint: {ENDPOINT}")
    print(f"  run_sec : {RUN_SEC}")

    # Tip slot (for freshness comparison)
    try:
        tip_at_start = await get_current_slot()
        print(f"  chain tip at start (Yellowstone): {tip_at_start}")
    except Exception as e:
        tip_at_start = None
        print(f"  chain tip lookup failed: {e}")

    print(f"\n--- opening ShredstreamProxy.SubscribeEntries stream ---")
    n = 0
    slots = []
    entry_sizes = []
    first_payloads = []
    t0 = time.time()
    try:
        async with grpc.aio.insecure_channel(ENDPOINT) as ch:
            stub = shredstream_pb2_grpc.ShredstreamProxyStub(ch)
            req = shredstream_pb2.SubscribeEntriesRequest()
            # Try without metadata first (IP-allowed); fall back to token if needed
            try:
                async for entry in stub.SubscribeEntries(req, timeout=RUN_SEC + 5):
                    n += 1
                    slots.append(int(entry.slot))
                    entry_sizes.append(len(entry.entries))
                    if len(first_payloads) < 3:
                        # save first 64 bytes hex for inspection
                        first_payloads.append(entry.entries[:64].hex())
                    if n % 100 == 0:
                        rate = n / max(0.001, time.time() - t0)
                        print(f"    {n} entries  ({rate:.0f}/s)  latest_slot={slots[-1]}",
                              flush=True)
                    if time.time() - t0 >= RUN_SEC: break
            except grpc.aio.AioRpcError as e:
                print(f"  RPC error: {e.code()} {e.details()}")
    except Exception as e:
        print(f"  connection error: {e}")

    dur = time.time() - t0
    print(f"\n--- summary ---")
    print(f"  duration:    {dur:.1f}s")
    print(f"  entries:     {n}  ({n/max(0.001,dur):.0f}/s)")
    if slots:
        print(f"  slot range:  {min(slots)} .. {max(slots)}  (span {max(slots)-min(slots)} slots)")
        if tip_at_start is not None:
            # tip moved during run — get a current one
            try:
                tip_now = await get_current_slot()
            except Exception:
                tip_now = tip_at_start
            print(f"  chain tip now: {tip_now}")
            # How fresh is the latest entry vs chain tip?
            gap = tip_now - max(slots)
            print(f"  freshness: latest entry slot {max(slots)} vs tip {tip_now}  "
                  f"-> gap {gap} slots ({gap*0.4:.1f}s)")
        # Entries are typically bunched per slot — count distinct slots
        from collections import Counter
        slot_counts = Counter(slots)
        print(f"  distinct slots seen: {len(slot_counts)}")
        print(f"  entries per slot: min={min(slot_counts.values())} "
              f"max={max(slot_counts.values())} "
              f"mean={sum(slot_counts.values())/len(slot_counts):.1f}")
    if entry_sizes:
        es = sorted(entry_sizes)
        print(f"  entries-payload bytes: min={min(es)} p50={es[len(es)//2]} "
              f"p90={es[int(len(es)*0.9)]} max={max(es)}")
    if first_payloads:
        print(f"\n  first 64 bytes of first 3 payloads (bincode-encoded Vec<solana_entry::Entry>):")
        for i, hx in enumerate(first_payloads):
            print(f"    [{i}] {hx}")


if __name__ == "__main__":
    asyncio.run(probe())
