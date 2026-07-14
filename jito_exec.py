"""Jito bundle submission — Frankfurt block engine."""
from __future__ import annotations

import asyncio
import base58
import base64
import json as _json
import time as _time
import urllib.request as _urlreq
import random
from typing import Sequence

from jito_py_rpc import JitoJsonRpcSDK
from solders.hash import Hash
from solders.instruction import Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

JITO_FRANKFURT_URL = "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1"

_jito: JitoJsonRpcSDK | None = None


def jito_client() -> JitoJsonRpcSDK:
    global _jito
    if _jito is None:
        _jito = JitoJsonRpcSDK(url=JITO_FRANKFURT_URL)
    return _jito


# Static set of Jito mainnet tip accounts (fetched once, cached). Fallback list
# is the live getTipAccounts result verified 2026-06-11 — used if the API is
# rate-limited so we NEVER pass None to Pubkey.from_string.
_TIP_FALLBACK = [
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
]
_tip_accounts = None


def get_cached_tip_account() -> str:
    """A Jito tip account from a process-cached list. The set is static, so fetch
    getTipAccounts ONCE and cache it; fall back to the known set on any error /
    rate-limit. Replaces per-bundle get_random_tip_account() which 429'd."""
    global _tip_accounts
    if not _tip_accounts:
        accts = None
        try:
            r = jito_client().get_tip_accounts()
            if isinstance(r, dict):
                data = r.get("data")
                if isinstance(data, dict):
                    accts = data.get("result")
        except Exception:
            accts = None
        if accts and isinstance(accts, list) and all(isinstance(a, str) for a in accts):
            _tip_accounts = accts
        else:
            _tip_accounts = list(_TIP_FALLBACK)
    return random.choice(_tip_accounts)


def send_transaction_b64(b64_tx: str, base_url: str = JITO_FRANKFURT_URL) -> dict:
    """Submit ONE signed tx via Jito's sendTransaction proxy (forwards directly to the
    validator with MEV protection). Lands single txs reliably where a 1-tx sendBundle
    does NOT on the public endpoint (verified 2026-06-11). Returns the JSON-RPC dict
    ({"result": <signature>} on success).

    SYNC + new connection per call: fine for tools/scripts. The BOT's hot path
    must use send_transaction_b64_async below: this one blocks the event loop
    for the whole POST (10s worst case) and pays a TCP+TLS handshake per call."""
    body = _json.dumps({"jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                        "params": [b64_tx, {"encoding": "base64"}]}).encode()
    req = _urlreq.Request(base_url.rstrip("/") + "/transactions", data=body,
                          headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=10) as r:
        return _json.loads(r.read())


# ---- async hot-path sender (2026-06-11 latency changeset) ----
# One persistent httpx client: connection pooling kills the per-call TCP+TLS
# handshake to the block engine, and awaiting the POST keeps the event loop
# (listener, exit dispatch, watchdog) running during submission. The sync
# urllib path above measured as a full event-loop stall per live POST.
_async_http = None

# Region preference list (2026-06-12, corrected after verifying the proxy
# forwards to the current leader regardless of region — Frankfurt-only canary
# always landed). The proxy uses the network-wide leader schedule, so ANY
# region reaches every leader; region choice is a LATENCY tweak, not a
# reachability gate. We therefore PREFER the nearest region (lowest local hop)
# and only fail over to the next on error / 1-rps-exhaustion. Order = measured
# warm round-trip from sol (in Frankfurt): frankfurt 6.9ms, amsterdam 12.4,
# london 18.1, dublin 29.5, ny 82.4. (slc 124 / singapore 156 / tokyo 226 are
# omitted: too far to help inside the ~400ms slot, and only reached if every
# nearer region errored.) The default limit is 1 req/s PER IP PER REGION, so a
# same-second buy+sell naturally split across frankfurt -> amsterdam.
JITO_REGIONS = [
    "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1",
    "https://amsterdam.mainnet.block-engine.jito.wtf/api/v1",
    "https://london.mainnet.block-engine.jito.wtf/api/v1",
    "https://dublin.mainnet.block-engine.jito.wtf/api/v1",
    "https://ny.mainnet.block-engine.jito.wtf/api/v1",
]
try:  # optional config override (cfg.broker.jito_endpoints: list of base urls)
    from bot_config import cfg as _C_jx
    _eps = getattr(getattr(_C_jx, "broker", None), "jito_endpoints", None)
    if _eps:
        JITO_REGIONS = list(_eps)
except Exception:
    pass

_region_last_send: dict[str, float] = {}
_REGION_MIN_INTERVAL = 1.05   # 1 rps limit + 5% safety margin


def _region_name(base_url: str) -> str:
    try:
        return base_url.split("//", 1)[1].split(".", 1)[0]
    except Exception:
        return base_url


def _get_async_http():
    global _async_http
    if _async_http is None:
        import httpx
        # keepalive_expiry must outlive the keepalive ping interval (60s in
        # jito_keepalive_loop) or the pool closes idle connections anyway.
        _async_http = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            limits=httpx.Limits(max_keepalive_connections=5,
                                keepalive_expiry=90.0))
    return _async_http


async def send_transaction_b64_async(b64_tx: str, base_url: str | None = None) -> dict:
    """Submit ONE signed tx via the sendTransaction proxy. base_url=None (the
    bot's hot path) does NEAREST-REGION-FIRST with FAILOVER:

      Why not broadcast / why not pin one region (verified 2026-06-12): the
      proxy "forwards your transaction directly to the validator" using the
      network-wide leader schedule, so ANY region reaches the CURRENT leader
      (this is why our Frankfurt-only canary always landed). Region choice is
      therefore a LATENCY tweak, not a reachability gate, and broadcasting the
      same tx everywhere barely helps when we are geographically fixed (the
      distance to a far leader is paid on either the us->engine or the
      engine->leader leg regardless) while it DID create buy/sell rate
      contention. So: send to the nearest region with 1-rps budget; on send
      error OR rate-exhaustion, fail over to the next region. This reaches the
      leader, keeps the lowest local hop (sol is in Frankfurt, 6.9ms), gives
      redundancy against a transient single-path drop, and lets a same-second
      buy+sell use Frankfurt then Amsterdam with no artificial wait.

    base_url set = force that one region (tools/tests). Returns the JSON-RPC
    dict annotated with _region; raises only if every region errors / is
    exhausted."""
    cli = _get_async_http()
    body = {"jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
            "params": [b64_tx, {"encoding": "base64"}]}
    if base_url is not None:
        _region_last_send[base_url] = _time.time()
        r = await cli.post(base_url.rstrip("/") + "/transactions", json=body)
        r.raise_for_status()
        out = r.json(); out["_region"] = _region_name(base_url)
        return out
    errors = []
    attempted = False
    for url in JITO_REGIONS:
        now = _time.time()
        if _region_last_send.get(url, 0.0) + _REGION_MIN_INTERVAL - now > 0.0:
            continue    # region rate-exhausted this second -> try the next one
        _region_last_send[url] = now    # reserve before awaiting
        attempted = True
        try:
            r = await cli.post(url.rstrip("/") + "/transactions", json=body)
            r.raise_for_status()
            out = r.json(); out["_region"] = _region_name(url)
            return out
        except Exception as e:
            errors.append((_region_name(url), str(e)[:50]))
            continue
    if not attempted:
        # Every region rate-exhausted in this second (burst > len(regions)):
        # wait for the soonest to free, then send there.
        now = _time.time()
        forced = min(JITO_REGIONS, key=lambda u: _region_last_send.get(u, 0.0))
        wait = max(0.0, _region_last_send.get(forced, 0.0) + _REGION_MIN_INTERVAL - now)
        await asyncio.sleep(min(wait, 1.2))
        _region_last_send[forced] = _time.time()
        r = await cli.post(forced.rstrip("/") + "/transactions", json=body)
        r.raise_for_status()
        out = r.json(); out["_region"] = _region_name(forced)
        return out
    raise RuntimeError(f"sendTransaction failed on all regions: {errors[:4]}")


async def warm_jito_connection(base_url: str = JITO_FRANKFURT_URL) -> None:
    """Open the pooled TLS connection ahead of the first real submission so the
    first live POST doesn't pay the handshake. Any HTTP response (even 4xx)
    means the connection is up. Best-effort, never raises."""
    try:
        cli = _get_async_http()
        await cli.get(base_url.rstrip("/") + "/transactions")
    except Exception:
        pass


async def jito_keepalive_loop(interval_s: float = 60.0) -> None:
    """Keep the pooled TLS connections to ALL configured regions warm BETWEEN
    fires. Fires are 20-30min apart while client keep-alive and LB idle
    timeouts are far shorter, so without this every fire's submit paid a fresh
    handshake (the same cold-pool effect that motivated CreateEvent meta
    seeding). One cheap GET per region per minute (405, no server work, far
    under the 1-rps regional limit); no credit pool involved."""
    while True:
        for url in JITO_REGIONS:
            await warm_jito_connection(url)
        await asyncio.sleep(interval_s)


def build_versioned_tx(
    wallet: Keypair,
    instructions: Sequence[Instruction],
    recent_blockhash: str | Hash,
) -> VersionedTransaction:
    bh = recent_blockhash if isinstance(recent_blockhash, Hash) else Hash.from_string(recent_blockhash)
    msg = MessageV0.try_compile(
        payer=wallet.pubkey(),
        instructions=list(instructions),
        address_lookup_table_accounts=[],
        recent_blockhash=bh,
    )
    return VersionedTransaction(msg, [wallet])


async def execute_jito_bundle(
    wallet: Keypair,
    swap_ix: Instruction,
    recent_blockhash: str,
    tip_lamports: int = 100_000,
) -> dict | None:
    """Swap + tip in one tx, submit as Jito bundle to Frankfurt."""
    tip_account = Pubkey.from_string(get_cached_tip_account())
    tip_ix = transfer(
        TransferParams(
            from_pubkey=wallet.pubkey(),
            to_pubkey=tip_account,
            lamports=tip_lamports,
        )
    )
    tx = build_versioned_tx(wallet, [swap_ix, tip_ix], recent_blockhash)
    b64_tx = base64.b64encode(bytes(tx)).decode("utf-8")
    try:
        print("[JITO] Dropping bundle to Frankfurt...")
        return jito_client().send_bundle(params=[b64_tx])
    except Exception as exc:
        print(f"[!] Jito bundle failed: {exc}")
        return None
