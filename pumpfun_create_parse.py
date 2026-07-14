"""Parse pump.fun CreateEvent from gRPC program-data log lines.

Purpose: seed JitoBroker's (token_program, creator) cache at TOKEN BIRTH from
the feed we already consume, so fire-time bundle assembly needs ZERO RPC for
mint meta. Replaces a per-fire 2-RPC fetch that measured 470-600ms whenever
the connection pool had gone cold between fires (httpx keep-alive is seconds,
fires are ~20-30 min apart), and costs nothing against the eRPC credit pool.

Layout VERIFIED LIVE 2026-06-11 on 3/3 sampled creates (two independent
proofs: the bonding_curve field equals the PDA derivation, and the creator
field equals the curve account's data at offset 49 fetched via RPC), plus a
12/12 token-program coverage check across static + ALUT-loaded keys:

  disc 8B = 1b72a94ddeeb6376
  borsh: name str, symbol str, uri str
  mint 32B, bonding_curve 32B, user 32B, creator 32B
  (+114 trailing bytes: timestamp/reserves, unused here)

This module is intentionally SEPARATE from pumpfun_parse.py: that file is in
the frozen collector import closure (tests/collector_frozen_manifest.txt) and
must not change; this one is imported only by the bot-side gRPC listener.
"""
from __future__ import annotations

import base58

CREATE_EVENT_DISC = bytes.fromhex("1b72a94ddeeb6376")
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
TOKEN_LEGACY = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def _skip_borsh_str(buf: bytes, off: int) -> int:
    n = int.from_bytes(buf[off:off + 4], "little")
    off += 4
    if n > 4096 or off + n > len(buf):
        raise ValueError("bad borsh string")
    return off + n


def parse_create_event(data: bytes) -> dict | None:
    """data = full program-data payload including the 8-byte discriminator.
    Returns {"mint": b58, "creator": b58} or None if not a (well-formed)
    CreateEvent. Never raises."""
    if len(data) < 8 or data[:8] != CREATE_EVENT_DISC:
        return None
    try:
        buf = data[8:]
        off = _skip_borsh_str(buf, 0)      # name
        off = _skip_borsh_str(buf, off)    # symbol
        off = _skip_borsh_str(buf, off)    # uri
        if off + 128 > len(buf):
            return None
        mint = base58.b58encode(buf[off:off + 32]).decode()
        off += 32                           # past mint -> bonding_curve
        off += 32                           # past bonding_curve -> user
        off += 32                           # past user -> creator
        creator = base58.b58encode(buf[off:off + 32]).decode()
        return {"mint": mint, "creator": creator}
    except Exception:
        return None


def token_program_from_keys(keys_b58) -> str | None:
    """keys_b58: iterable of base58 account keys, static AND ALUT-loaded
    (loaded_writable/readonly_addresses) — creates routed through lookup
    tables carry the token program only in the loaded set. Returns the token
    program id string, or None when absent (caller should NOT seed; the
    fire-time RPC fallback handles it).

    PREFER Token-2022 when BOTH appear. The earlier first-match loop returned
    whichever program came first in iteration order; for a Token-2022 mint whose
    key set also contained the legacy program (ATA/WSOL refs in the static keys,
    ahead of the Token-2022 program in the loaded set) it seeded LEGACY -> the
    fire-time ATA-create reverted IncorrectProgramId and the buy was lost
    (observed 2026-06-16, mint 2L6qDGdV...). A Token-2022 mint's create always
    references the Token-2022 program, and a legacy mint never references
    Token-2022, so preferring Token-2022 is correct for both."""
    ks = set(keys_b58)
    if TOKEN_2022 in ks:
        return TOKEN_2022
    if TOKEN_LEGACY in ks:
        return TOKEN_LEGACY
    return None
