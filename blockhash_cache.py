"""Background blockhash prefetch — keep hot path off RPC.

PRIMARY path: Yellowstone gRPC GetLatestBlockhash on the SAME endpoint we
already pay for via subscription (the trade firehose). Zero per-call credit
cost. Authenticated via x-token metadata header (env GRPC_TOKEN).

FALLBACK path: public mainnet-beta HTTPS — rate-limited but free. Only
used if gRPC primary fails repeatedly.

History: previously this module polled ERPC HTTP /getLatestBlockhash. That
endpoint charges ~43 credits/call from the 10M monthly pool. At 200ms cadence
that's 432K calls/day — 12-15 days to burn the entire monthly budget on
blockhash polling alone. Switching to gRPC eliminates this drain.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# gRPC stubs live in grpc_stubs/ — add to path so import works regardless of cwd
_HERE = Path(__file__).resolve().parent
_STUBS = _HERE / "grpc_stubs"
if str(_STUBS) not in sys.path:
    sys.path.insert(0, str(_STUBS))

import grpc
import geyser_pb2
import geyser_pb2_grpc

# HTTPS fallback only — kept for resilience if gRPC primary errors out
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Processed

import config

# Public mainnet-beta is free but rate-limited. Used only as a fallback for
# the rare case both gRPC primary and our own retry both fail.
FALLBACK_RPC = os.getenv(
    "BLOCKHASH_RPC_FALLBACK",
    "https://api.mainnet-beta.solana.com",
)

# gRPC endpoint — same as the trade firehose. Auth via GRPC_TOKEN. Prefer the
# listener config (cfg.listener.grpc_endpoint) because that's the one we know
# resolves and is paid for. Env BLOCKHASH_GRPC_ENDPOINT overrides explicitly.
def _resolve_grpc_endpoint() -> tuple[str, bool]:
    override = os.getenv("BLOCKHASH_GRPC_ENDPOINT")
    if override:
        return override, os.getenv("BLOCKHASH_GRPC_INSECURE", "1") in ("1","true","True")
    try:
        from bot_config import cfg as _C
        return _C.listener.grpc_endpoint, bool(_C.listener.grpc_insecure)
    except Exception:
        return "grpc-fra1-1.erpc.global:80", True

GRPC_ENDPOINT, GRPC_INSECURE = _resolve_grpc_endpoint()


@dataclass
class BlockhashEntry:
    blockhash: str
    last_valid_block_height: int
    fetched_at: float
    slot: int = 0   # slot number this blockhash was produced at (gRPC only)


_cache: BlockhashEntry | None = None
_lock = asyncio.Lock()
_fail_streak = 0
_last_fail_log = 0.0
_grpc_channel = None
_grpc_stub = None


def _grpc_metadata():
    tok = config.GRPC_TOKEN
    return (("x-token", tok),) if tok else None


async def _ensure_grpc_stub():
    """Lazy-init the gRPC channel + stub. Reused across calls for the lifetime
    of the process — no per-call TCP/TLS handshake."""
    global _grpc_channel, _grpc_stub
    if _grpc_stub is not None:
        return _grpc_stub
    if GRPC_INSECURE:
        _grpc_channel = grpc.aio.insecure_channel(GRPC_ENDPOINT)
    else:
        _grpc_channel = grpc.aio.secure_channel(GRPC_ENDPOINT,
                                                  grpc.ssl_channel_credentials())
    _grpc_stub = geyser_pb2_grpc.GeyserStub(_grpc_channel)
    return _grpc_stub


async def _fetch_grpc() -> BlockhashEntry:
    """Pull latest blockhash via gRPC. Zero credit cost — billed under the
    firehose subscription, not the HTTP credit pool."""
    stub = await _ensure_grpc_stub()
    req = geyser_pb2.GetLatestBlockhashRequest(
        commitment=geyser_pb2.CommitmentLevel.PROCESSED)
    resp = await stub.GetLatestBlockhash(req, metadata=_grpc_metadata())
    return BlockhashEntry(
        blockhash=str(resp.blockhash),
        last_valid_block_height=int(resp.last_valid_block_height),
        fetched_at=time.time(),
        slot=int(resp.slot),
    )


async def _fetch_http(client: AsyncClient) -> BlockhashEntry:
    """Legacy HTTPS fallback path. Only used if gRPC fails repeatedly."""
    resp = await client.get_latest_blockhash(commitment=Processed)
    value = resp.value
    return BlockhashEntry(
        blockhash=str(value.blockhash),
        last_valid_block_height=int(value.last_valid_block_height),
        fetched_at=time.time(),
    )


async def _stream_blocks_meta() -> None:
    """PUSH path (2026-06-12): subscribe to blocks_meta and cache each new
    block's hash at production time. Strictly better than polling: the
    'latest blockhash' only CHANGES once per block (~400ms), so a 200ms poll
    just re-reads the same hash with a fresher fetched_at, while the stream
    hands us each hash at birth (maximum remaining validity window, real
    slot attached for same-block-aim logging). NOTE: bh_age_ms in broker
    logs now measures true HASH age (0-400ms typical between blocks), not
    poll recency; the 500ms freshness gate stays meaningful (a >500ms-old
    hash means skipped slots or a stalled stream -> on-demand refresh).
    Runs until the stream errors; the supervisor falls back to polling."""
    global _cache, _fail_streak
    stub = await _ensure_grpc_stub()
    req = geyser_pb2.SubscribeRequest()
    req.blocks_meta["bh"].CopyFrom(geyser_pb2.SubscribeRequestFilterBlocksMeta())
    req.commitment = geyser_pb2.CommitmentLevel.PROCESSED

    async def _req_iter():
        yield req
        while True:
            await asyncio.sleep(3600)

    print(f"[blockhash] streaming blocks_meta from {GRPC_ENDPOINT} "
          f"(push per block; unary poll is outage-fallback)", flush=True)
    async for resp in stub.Subscribe(_req_iter(), metadata=_grpc_metadata()):
        if not resp.HasField("block_meta"):
            continue
        bm = resp.block_meta
        entry = BlockhashEntry(
            blockhash=str(bm.blockhash),
            last_valid_block_height=int(bm.block_height.block_height) + 150,
            fetched_at=time.time(),
            slot=int(bm.slot),
        )
        async with _lock:
            _cache = entry
        _fail_streak = 0


async def _poll_window(duration_s: float, poll_s: float) -> None:
    """Legacy unary-poll loop, time-bounded; used only while the blocks_meta
    stream is down. gRPC unary first, HTTPS fallback second."""
    global _cache, _fail_streak, _last_fail_log
    end = time.time() + duration_s
    async with AsyncClient(FALLBACK_RPC) as fallback_client:
        while time.time() < end:
            delay = poll_s
            ok = False
            try:
                entry = await _fetch_grpc()
                async with _lock:
                    _cache = entry
                _fail_streak = 0
                ok = True
            except Exception:
                try:
                    entry = await _fetch_http(fallback_client)
                    async with _lock:
                        _cache = entry
                    _fail_streak = 0
                    ok = True
                except Exception:
                    pass
            if not ok:
                _fail_streak += 1
                now = time.time()
                if _fail_streak == 1 or now - _last_fail_log > 30:
                    print(f"[!] blockhash prefetch failed ({_fail_streak}x) — "
                          f"both gRPC and HTTPS fallback returned errors", flush=True)
                    _last_fail_log = now
                delay = min(poll_s * (2 ** min(_fail_streak, 4)), 8.0)
            await asyncio.sleep(delay)


async def blockhash_shadow_loop(poll_s: float = None) -> None:
    """Keep the cache fresh. STREAM-FIRST: blocks_meta push (one update per
    block). On stream failure: rebuild the channel, poll for 10s, retry the
    stream. HTTPS only if the gRPC unary also fails (inside _poll_window)."""
    if poll_s is None:
        try:
            from bot_config import cfg as _C
            poll_s = float(getattr(_C.broker, "blockhash_poll_s", 0.2))
        except Exception:
            poll_s = 0.2
    global _grpc_channel, _grpc_stub
    while True:
        try:
            await _stream_blocks_meta()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[blockhash] stream error: {e}; 10s poll-fallback then "
                  f"re-stream", flush=True)
            try:
                if _grpc_channel is not None:
                    await _grpc_channel.close()
            except Exception:
                pass
            _grpc_channel = None
            _grpc_stub = None
        await _poll_window(10.0, poll_s)


async def get_cached_blockhash() -> BlockhashEntry | None:
    async with _lock:
        return _cache


# HTTPS fallback client cache for the freshness-on-demand path. Only used when
# gRPC on-demand also fails — rare; keeps the system resilient.
_fallback_http: AsyncClient | None = None


async def _get_fallback_http() -> AsyncClient:
    global _fallback_http
    if _fallback_http is None:
        _fallback_http = AsyncClient(FALLBACK_RPC)
    return _fallback_http


async def get_fresh_blockhash(max_age_ms: float = 500.0) -> BlockhashEntry | None:
    """Return the cached blockhash if fresh; otherwise block on a refresh.
    Try gRPC first (free under subscription), then HTTPS fallback. Returns
    stale cache rather than None on total failure."""
    global _cache
    async with _lock:
        cur = _cache
    now = time.time()
    if cur is not None and (now - cur.fetched_at) * 1000.0 <= max_age_ms:
        return cur
    # On-demand: gRPC first
    try:
        entry = await _fetch_grpc()
        async with _lock:
            _cache = entry
        return entry
    except Exception:
        pass
    # HTTPS fallback (RARE — only when gRPC errors)
    try:
        client = await _get_fallback_http()
        entry = await _fetch_http(client)
        async with _lock:
            _cache = entry
        return entry
    except Exception:
        pass
    return cur
