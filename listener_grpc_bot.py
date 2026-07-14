"""gRPC listener for pumpfun_bot — alternative to the websocket-based ShadowHarness.listener.

Subscribes to the pump.fun program via Geyser at grpc-fra1-1.erpc.global:80 (the same
endpoint pumpfun-grpc-capture uses, but as its own independent subscription so the
two processes don't share state). Parses TradeEvent payloads from log_messages,
attaches the slot from the gRPC envelope (tx.slot), and routes to ShadowHarness.on_trade.

WHY a separate subscription instead of reading the capture's JSONL? Latency. The
capture's job is durable archive; this listener's job is in-memory hot-path feed.
A few hundred KB/s extra to the bot process is irrelevant. Same model can later be
trained against the capture's archive for new features (gRPC fields the WS path
cannot expose — slot index, compute units, inner instructions, etc.).

FUTURE-FEATURES NOTE (per user 2026-06-07): if/when we add features that the gRPC
envelope provides (e.g. tx index in slot, pre/post balances, compute units), those
features will be in a DIFFERENT model than the V+K7 one we're currently training.
This listener attaches `slot` to the event so same-block aim works today; richer
fields stay in the gRPC envelope and can be added to a future event extension when
we train a model that consumes them.
"""
from __future__ import annotations
import asyncio
import base64
import time

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent if "tools" in __file__ else _P(__file__).resolve().parent) + "/grpc_stubs")
import grpc
import base58
import geyser_pb2
import geyser_pb2_grpc
import config
from pumpfun_parse import parse_trade_event, TRADE_EVENT_DISC
from pumpfun_create_parse import (CREATE_EVENT_DISC, parse_create_event,
                                  token_program_from_keys)

try:
    from bot_config import cfg as _C
    GRPC_ENDPOINT = _C.listener.grpc_endpoint
    GRPC_INSECURE = _C.listener.grpc_insecure
    GRPC_COMMITMENT = _C.listener.commitment  # "processed" | "confirmed" | "finalized"
except Exception:
    GRPC_ENDPOINT = "grpc-fra1-1.erpc.global:80"
    GRPC_INSECURE = True
    GRPC_COMMITMENT = "processed"


# Same Jito tip + routing constants as grpc_capture.py — keep in sync if added
# to or moved later. We DO NOT consume these as model features yet; we just
# annotate each entry_decision with K-window aggregate sophistication so we
# can later check whether these fields correlate with per-fire P&L.
JITO_TIP_ACCOUNTS = frozenset({
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDe9B",
    "ADuUkR4vqLUMWXxW9gh6D6L8pivKeVBBWhxX59nFXTNb",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
})
KNOWN_ROUTERS = {
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4":  "jupiter_v6",
    "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB":  "jupiter_v4",
    "routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS":  "raydium_router",
    "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K":  "moonshot",
}
COMPUTE_BUDGET = "ComputeBudget111111111111111111111111111111"


def _extract_grpc_extras(tx, meta) -> dict:
    """Pull gRPC-exclusive per-tx fields. Mirrors grpc_capture.py extraction.
    Always returns a dict (possibly empty on parse failure) so callers don't
    have to guard. Cheap — touches well-known fields, no decoding heavy lifting.
    """
    out = {}
    try:
        if meta.fee: out["fee_lam"] = int(meta.fee)
        if meta.compute_units_consumed: out["cu"] = int(meta.compute_units_consumed)
        if meta.inner_instructions:
            out["n_inner_ix"] = sum(len(g.instructions) for g in meta.inner_instructions)
        ak_bytes = (tx.transaction.transaction.message.account_keys
                    if hasattr(tx.transaction, "transaction") else [])
        if ak_bytes:
            ak_b58 = [base58.b58encode(bytes(k)).decode() for k in ak_bytes]
            out["n_keys"] = len(ak_b58)
            message = tx.transaction.transaction.message
            cu_limit = 0
            priority_fee_micro = 0
            for ix in message.instructions:
                if ix.program_id_index >= len(ak_b58):
                    continue
                if ak_b58[ix.program_id_index] != COMPUTE_BUDGET:
                    continue
                d = bytes(ix.data)
                if len(d) >= 5 and d[0] == 0x02:
                    cu_limit = int.from_bytes(d[1:5], "little")
                elif len(d) >= 9 and d[0] == 0x03:
                    priority_fee_micro = int.from_bytes(d[1:9], "little")
            out["cu_limit"] = cu_limit
            out["priority_fee_micro"] = priority_fee_micro
            for k in ak_b58:
                if k in KNOWN_ROUTERS:
                    out["route"] = KNOWN_ROUTERS[k]; break
            for i, k in enumerate(ak_b58):
                if k in JITO_TIP_ACCOUNTS:
                    out["jito_tip_idx"] = i
                    try:
                        pre  = int(meta.pre_balances[i])
                        post = int(meta.post_balances[i])
                        out["jito_tip_lam"] = post - pre
                    except Exception: pass
                    break
    except Exception:
        # Stay quiet — extras are advisory; never break the hot path.
        pass
    return out


def _commitment_to_enum(name: str) -> int:
    """Map config string → CommitmentLevel proto enum (int). Defaults to PROCESSED
    on unknown so we never accidentally slow down to finalized."""
    m = {"processed": geyser_pb2.CommitmentLevel.PROCESSED,
         "confirmed": geyser_pb2.CommitmentLevel.CONFIRMED,
         "finalized": geyser_pb2.CommitmentLevel.FINALIZED}
    return m.get((name or "processed").lower(), geyser_pb2.CommitmentLevel.PROCESSED)


async def grpc_listener_for_harness(h) -> None:
    """Drop-in replacement for ShadowHarness.listener(). Uses gRPC instead of WS."""
    metadata = (("x-token", config.GRPC_TOKEN),) if config.GRPC_TOKEN else None
    print(f"[shadow] connecting gRPC {GRPC_ENDPOINT}", flush=True)
    while True:
        try:
            async with grpc.aio.insecure_channel(GRPC_ENDPOINT) as channel:
                stub = geyser_pb2_grpc.GeyserStub(channel)
                req = geyser_pb2.SubscribeRequest()
                req.transactions["pumpfun"].account_include.append(config.PUMP_FUN_PROGRAM)
                req.transactions["pumpfun"].failed = False
                # gRPC-native wallet reconcile (2026-06-12): fold OUR WALLET into
                # the SAME stream as a second filter — no new gRPC connection
                # (the endpoint is at its max stream count). failed=True so we
                # ALSO catch our own reverted txs (the pumpfun filter drops
                # them). Each of our txs returns with full meta (pre/post SOL +
                # token balances, slot, err) = confirmation + landed-slot +
                # actual fill without any HTTP. Replaces the eRPC
                # getTransaction/getSignatureStatuses path (degraded). Only
                # added when a live JitoBroker with a wallet is present.
                wallet_recon = False
                _bk = getattr(h, "broker", None)
                if (_bk is not None and hasattr(_bk, "reconcile_grpc_tx")
                        and getattr(_bk, "user_pk", None) is not None):
                    req.transactions["wallet"].account_include.append(str(_bk.user_pk))
                    req.transactions["wallet"].failed = True
                    wallet_recon = True
                    print(f"[shadow] gRPC wallet-recon ON for {_bk.user_pk} "
                          f"(same stream, failed=True)", flush=True)
                # Pin commitment explicitly so we are not at the upstream's mercy
                # if its default ever changes. PROCESSED is intentional — we want
                # leader-block-included events ASAP, accepting <1% reorg risk
                # that the holdings-reconciler ground-truths against.
                req.commitment = _commitment_to_enum(GRPC_COMMITMENT)
                async def req_iter():
                    yield req
                    while True:
                        await asyncio.sleep(1)
                print(f"[shadow] subscribed (gRPC, commitment={GRPC_COMMITMENT})",
                      flush=True)
                async for resp in stub.Subscribe(req_iter(), metadata=metadata):
                    if not resp.HasField("transaction"):
                        continue
                    h.stats["events"] += 1
                    tx = resp.transaction
                    slot = int(tx.slot)
                    meta = tx.transaction.meta
                    # gRPC-native reconcile for OUR OWN txs (incl reverts) BEFORE
                    # the err-skip below. Only fires when the "wallet" filter
                    # matched, so it costs a dict lookup for everyone else.
                    if wallet_recon and ("wallet" in resp.filters):
                        try:
                            sig_b58 = base58.b58encode(bytes(tx.transaction.signature)).decode()
                            h.broker.reconcile_grpc_tx(sig_b58, slot, meta)
                            h.stats["grpc_wallet_recon"] = h.stats.get("grpc_wallet_recon", 0) + 1
                        except Exception:
                            pass
                    if meta.err.err:
                        continue
                    for ln in meta.log_messages:
                        if "Program data:" not in ln:
                            continue
                        b64 = ln.split("Program data:", 1)[1].strip()
                        try:
                            data = base64.b64decode(b64)
                        except Exception:
                            continue
                        if len(data) >= 8 and data[:8] == CREATE_EVENT_DISC:
                            # Token birth: seed the broker's (token_program,
                            # creator) cache from the feed so fire-time
                            # assembly needs ZERO RPC for mint meta (was a
                            # 470-600ms cold fetch per fire). ~20-25 creates
                            # per minute; key decode only happens here.
                            b = getattr(h, "broker", None)
                            if b is not None and hasattr(b, "seed_mint_meta"):
                                try:
                                    ce = parse_create_event(data)
                                    if ce is not None:
                                        msg_keys = tx.transaction.transaction.message.account_keys
                                        keys = [base58.b58encode(bytes(k)).decode() for k in msg_keys]
                                        keys += [base58.b58encode(bytes(k)).decode()
                                                 for k in meta.loaded_writable_addresses]
                                        keys += [base58.b58encode(bytes(k)).decode()
                                                 for k in meta.loaded_readonly_addresses]
                                        tp = token_program_from_keys(keys)
                                        if tp is not None:
                                            b.seed_mint_meta(ce["mint"], tp, ce["creator"])
                                            h.stats["creates_seeded"] = \
                                                h.stats.get("creates_seeded", 0) + 1
                                except Exception:
                                    pass
                            continue
                        if len(data) < 8 or data[:8] != TRADE_EVENT_DISC:
                            continue
                        ev = parse_trade_event(data)
                        if ev is None:
                            continue
                        # Attach slot for same-block aim. TradeEvent now has a `slot`
                        # field (default None on WS path). on_trade reads getattr(ev,
                        # "slot", None) for broker calls.
                        ev.slot = slot
                        # Side-car: attach gRPC-exclusive per-tx fields the model
                        # itself doesn't consume yet, but on_trade aggregates into
                        # the per-mint sophistication window so entry_decision logs
                        # carry the K-window signature. Empty dict if extraction
                        # failed — harness checks `is not None` then falls through.
                        ev.grpc_extras = _extract_grpc_extras(tx, meta)
                        await h.on_trade(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[shadow] gRPC error: {exc}; reconnect in 3s", flush=True)
            await asyncio.sleep(3)
