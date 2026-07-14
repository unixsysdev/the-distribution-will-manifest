"""graduation_listener.py — gRPC feed for the GRADUATION (PumpSwap AMM) bot.

Clone of listener_grpc_bot.grpc_listener_for_harness, but subscribes to the PumpSwap AMM
program (pAMMBay...) and decodes PumpSwap Buy/SellEvent (offsets locked vs post_tb to 0.4%:
quote_res@48, base_res@56, user@152) instead of bonding-curve TradeEvent. Routes a lightweight
AmmEvent to h.on_trade_amm(ev). Read-only feed; original listener untouched.

Run from the repository root (`python -m research.graduation.graduation_listener`) for a
15-second live self-test that decodes and prints.
"""
from __future__ import annotations
import asyncio, base64, struct, time
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
_sys.path.insert(0, str(_P(__file__).resolve().parent) + "/grpc_stubs")
import grpc
import geyser_pb2
import geyser_pb2_grpc
import config

ENDPOINT = "grpc-fra1-1.erpc.global:80"      # same endpoint as listener_grpc_bot
PUMPSWAP_PROG = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
WSOL = "So11111111111111111111111111111111111111112"
DISC_BUY = bytes.fromhex("67f4521f2cf57777")
DISC_SELL = bytes.fromhex("3e2f370aa503dc2a")


class AmmEvent:
    __slots__ = ("mint", "is_buy", "user", "quote_res", "base_res", "slot", "t")
    def __init__(self, mint, is_buy, user, quote_res, base_res, slot, t):
        self.mint = mint; self.is_buy = is_buy; self.user = user
        self.quote_res = quote_res; self.base_res = base_res; self.slot = slot; self.t = t


def _u64(b, off): return struct.unpack_from("<Q", b, off)[0]


def _decode_amm(data):
    """PumpSwap Buy/Sell -> (is_buy, quote_res, base_res, user_hex) | None."""
    disc = bytes(data[:8])
    if disc == DISC_BUY or disc == DISC_SELL:
        if len(data) < 184:
            return None
        return (disc == DISC_BUY, _u64(data, 48), _u64(data, 56), bytes(data[152:184]).hex())
    return None


def _base_mint_from_meta(meta):
    """The graduated coin = the non-WSOL mint in the tx's post token balances."""
    try:
        for tb in meta.post_token_balances:
            if tb.mint and tb.mint != WSOL:
                return tb.mint
    except Exception:
        pass
    return None


async def grad_grpc_listener_for_harness(h) -> None:
    metadata = (("x-token", config.GRPC_TOKEN),) if config.GRPC_TOKEN else None
    print(f"[grad] connecting gRPC {ENDPOINT} (PumpSwap AMM)", flush=True)
    while True:
        try:
            async with grpc.aio.insecure_channel(ENDPOINT) as channel:
                stub = geyser_pb2_grpc.GeyserStub(channel)
                req = geyser_pb2.SubscribeRequest()
                req.transactions["pumpswap"].account_include.append(PUMPSWAP_PROG)
                req.transactions["pumpswap"].failed = False
                req.commitment = geyser_pb2.CommitmentLevel.PROCESSED

                async def req_iter():
                    yield req
                    while True:
                        await asyncio.sleep(1)

                print("[grad] subscribed (PumpSwap AMM, commitment=processed)", flush=True)
                async for resp in stub.Subscribe(req_iter(), metadata=metadata):
                    if not resp.HasField("transaction"):
                        continue
                    h.stats["events"] += 1
                    tx = resp.transaction
                    slot = int(tx.slot)
                    meta = tx.transaction.meta
                    if meta.err.err:
                        continue
                    mint = None
                    for ln in meta.log_messages:
                        if "Program data:" not in ln:
                            continue
                        try:
                            data = base64.b64decode(ln.split("Program data:", 1)[1].strip())
                        except Exception:
                            continue
                        if len(data) < 8:
                            continue
                        d = _decode_amm(data)
                        if d is None:
                            continue
                        is_buy, qres, bres, user = d
                        if qres <= 0 or bres <= 0:
                            continue
                        if mint is None:
                            mint = _base_mint_from_meta(meta)
                        if not mint:
                            continue
                        await h.on_trade_amm(AmmEvent(mint, is_buy, user, qres, bres, slot, time.time()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[grad] gRPC error: {exc}; reconnect in 3s", flush=True)
            await asyncio.sleep(3)


if __name__ == "__main__":
    import collections

    class _H:
        def __init__(self): self.stats = collections.Counter(); self.n = 0

        async def on_trade_amm(self, ev):
            self.n += 1
            if self.n <= 15:
                print(f"  AMM {'BUY ' if ev.is_buy else 'SELL'} mint={ev.mint[:10]} "
                      f"mid={ev.quote_res / ev.base_res:.3e} user={ev.user[:8]} slot={ev.slot}", flush=True)

    async def _t():
        h = _H()
        task = asyncio.create_task(grad_grpc_listener_for_harness(h))
        await asyncio.sleep(15)
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        print(f"[grad selftest] envelope_events={h.stats['events']}  decoded_amm_trades={h.n}")

    asyncio.run(_t())
