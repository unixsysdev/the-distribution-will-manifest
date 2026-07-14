"""Shred-stream pump.fun Buy intent extractor.

Subscribes to ERPC's Direct Shreds (Jito ShredstreamProxy.SubscribeEntries),
decodes the bincode wrapper + wire-format VersionedTransactions, finds pump.fun
Buy instructions, and prints per-buy intent in real time:

    (slot, signer, mint, token_amount, max_sol_cost_lam, priority_fee, jito_tip)

This is the FIRE-HOSE side of the front-run augmentation. Output will eventually
go to a shared-memory ring buffer the policy bot consumes; for now we just
print + log so we can characterize what's actually in the stream.

Format notes:
  - The Entry message has `bytes entries` which is bincode-encoded
    Vec<solana_entry::Entry> where each Entry = {num_hashes:u64, hash:[u8;32],
    transactions:Vec<VersionedTransaction>}
  - The OUTER Vec<Entry> + Entry struct uses bincode (8-byte LE lengths)
  - Each VersionedTransaction inside is in SOLANA WIRE FORMAT (with ShortVec
    aka compact-u16 for lengths) — this is the same on-wire serialization
    Solana uses for transactions
  - Pump.fun Buy: first 8 bytes data = 0x66063d1201daebea, then u64 amount,
    then u64 max_sol_cost. Account[2] = mint, account[6] = user (signer).

Usage:
    RUN_SEC=60 ./venv/bin/python shred_bot/intent_extractor.py
"""
from __future__ import annotations
import asyncio, base58, os, struct, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc
import shredstream_pb2
import shredstream_pb2_grpc

ENDPOINT       = os.getenv("SHREDS_ENDPOINT", "shreds-fra6-1.erpc.global:80")
RUN_SEC        = int(os.getenv("RUN_SEC", "30"))
PUMP_FUN_PROG  = base58.b58decode("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
COMPUTE_BUDGET = base58.b58decode("ComputeBudget111111111111111111111111111111")

# Anchor instruction discriminators: sha256("global:<name>")[:8].
# We compute all known pump.fun instruction names so we can label every
# pump.fun-program ix in the shred stream (buy/sell/create/migrate/etc).
import hashlib
def _anchor_disc(name: str) -> bytes:
    return hashlib.sha256(f"global:{name}".encode()).digest()[:8]

PUMPFUN_IX_NAMES: dict[bytes, str] = {
    _anchor_disc("buy"):        "buy",         # 66063d1201daebea
    _anchor_disc("sell"):       "sell",        # 33e685a4017f83ad
    _anchor_disc("create"):     "create",      # token birth
    _anchor_disc("initialize"): "initialize",  # global state init (rare)
    _anchor_disc("withdraw"):   "withdraw",    # creator withdraws (rare)
    _anchor_disc("migrate"):    "migrate",     # bonding-curve -> Raydium graduation
    _anchor_disc("set_params"): "set_params",  # admin (rare)
    # Newer pump.fun ABI (with-creator variants). Identified empirically
    # 2026-06-09 by brute-forcing sha256(global:<name>)[:8] against the
    # most-frequent unknown discriminators in the captured stream.
    # Same data layout as legacy buy/sell (disc + token_amount_u64 + sol_limit_u64).
    # Account layout adds creator_vault, but mint stays at accs[2] and user at
    # accs[6] in the new ABI too (creator_vault inserted between existing
    # accounts and event_authority, near the end of the list).
    _anchor_disc("buy_v2"):     "buy",         # b817ee6167c5d33d  -> treat as buy (same data layout)
    _anchor_disc("sell_v2"):    "sell",        # 5df6823ce7e940b2  -> treat as sell (same data layout)
    _anchor_disc("create_v2"):  "create",      # d6904cec5f8b31b4  -> treat as create
    # USDC-quote router (peer-identified, brute-forced). Same shape (24B)
    # but DIFFERENT semantics: data = disc + quote_amount_in(u64) +
    # min_base_out(u64). Quote = SOL or USDC depending on pool. Do NOT
    # alias to "buy" — the existing buy parser would put quote_amount in
    # the token_amount slot and corrupt analysis. Give it its own type;
    # the parser below will save raw fields under different names.
    _anchor_disc("buy_exact_quote_in_v2"): "buy_quote",  # c2ab1c46684d5b2f (1775 occ)
    # Legacy SOL-as-quote router (data = disc + sol_in + min_tokens_out).
    # Account layout differs from regular buy (mint position may shift).
    # JSONL-only; not aliased to "buy" to keep ring data clean.
    _anchor_disc("buy_exact_sol_in"):     "buy_sol_in",  # 38fc74089edfcd5f (982 occ)
    # Non-trade events (JSONL-only, ring will skip). Named so they show up
    # as known types instead of pumpfun_other in analysis.
    _anchor_disc("collect_creator_fee_v2"):        "collect_creator_fee",   # cf118af204221338
    _anchor_disc("close_user_volume_accumulator"): "close_volume_accum",    # f945a4da9667548a
    _anchor_disc("extend_account"):                "extend_account",         # ea66c2cb96483ee5
    # User cashback claims (identified 2026-06-09 against pump-fun/pump-public-docs
    # idl/pump.json and idl/pump_amm.json — discriminator bytes match the IDL's
    # literal `discriminator` array exactly). Pure user-side fee-rebate flow, no
    # trade impact: 0 args, 5-10 accounts, just creates/closes the user
    # cashback PDA. Same discriminator `253a237ebe35e4c5` appears in BOTH
    # the bonding curve program (pump.json) AND the PumpSwap AMM (pump_amm.json);
    # we treat them identically because both mean "user collected a rebate".
    _anchor_disc("claim_cashback"):    "claim_cashback",   # 253a237ebe35e4c5 (86-219 occ)
    _anchor_disc("claim_cashback_v2"): "claim_cashback",   # 7af3cc415e741d37 (436-726 occ)
    # Rest of the long tail (same 2026-06-09 IDL audit, all non-trade plumbing).
    # All these were appearing as `pumpfun_other`; together with the two
    # cashback rows above they cover 100% of pump.fun-program ix observed in
    # the shred stream so far. Trade-shaped instructions (buy/sell/create
    # variants) are mapped above; ring-buffer logic is unaffected.
    _anchor_disc("collect_creator_fee"):           "collect_creator_fee",   # 1416567bc61cdb84 (95)  legacy v1
    _anchor_disc("init_user_volume_accumulator"):  "init_volume_accum",     # 5e06ca73ff60e8b7 (61)  user-PDA init
    _anchor_disc("sync_user_volume_accumulator"):  "sync_volume_accum",     # 561fc057a3574fee (1)   keep-alive
    _anchor_disc("distribute_creator_fees"):       "distribute_creator",    # a572670079cef751 (21)  legacy distribution
    _anchor_disc("distribute_creator_fees_v2"):    "distribute_creator",    # ffcb134ff444089f (24)  v2 distribution
    _anchor_disc("migrate_v2"):                    "migrate",               # bbcb121fceedfe29 (3)   new graduation flow
    # NOTE on program-id ambiguity (flagged for follow-up, not fixed here):
    # The same disc `66063d1201daebea` (buy) and `33e685a4017f83ad` (sell)
    # are used by BOTH pump.json (bonding curve) and pump_amm.json (PumpSwap).
    # Right now we label them identically because the bot only fires
    # pre-graduation, and PumpSwap activity is post-graduation noise we don't
    # gate on. If we ever want to separate them in analysis we need to also
    # capture `ix_program_id` (currently emitted as None in JSONL) so a
    # downstream consumer can disambiguate.
}
# Back-compat constants used elsewhere
BUY_DISC  = _anchor_disc("buy")
SELL_DISC = _anchor_disc("sell")
# v2 discriminators broken out for any consumer that wants to distinguish
BUY_V2_DISC  = _anchor_disc("buy_v2")
SELL_V2_DISC = _anchor_disc("sell_v2")
# Canonical Jito mainnet tip accounts (verified 2026-06-09 against
# docs.jito.wtf/lowlatencytxnsend `getTipAccounts`). One previously-wrong
# address (ADuUkR4..ZQs) fixed to the canonical (ADuUkR4..cEt).
JITO_TIP_ACCOUNTS = {base58.b58decode(s) for s in [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",   # was ..ZQs (wrong)
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]}


# ---------- ShortVec (Solana's compact-u16 encoding) ----------
def shortvec(buf: memoryview, pos: int) -> tuple[int, int]:
    """Decode Solana's compact-u16. Returns (value, new_pos)."""
    val = 0; shift = 0; n = 0
    while True:
        b = buf[pos + n]; n += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0: break
        shift += 7
        if n > 3:
            raise ValueError("compact-u16 too long")
    return val, pos + n


# ---------- VersionedTransaction wire-format decoder ----------
def parse_vt(buf: memoryview, pos: int) -> tuple[dict, int]:
    """Parse one VersionedTransaction in Solana wire format. Returns (tx, new_pos)."""
    # signatures
    n_sigs, pos = shortvec(buf, pos)
    pos += n_sigs * 64   # skip raw signatures (we don't need them for intent)
    sig_start = pos - n_sigs * 64
    first_sig = bytes(buf[sig_start:sig_start + 64]) if n_sigs > 0 else b""

    # message: first byte may be a version marker (high bit set for V0+)
    first = buf[pos]
    is_versioned = (first & 0x80) != 0
    version = (first & 0x7F) if is_versioned else None
    if is_versioned:
        pos += 1
    # header: 3 bytes
    hdr = (buf[pos], buf[pos+1], buf[pos+2]); pos += 3
    # account_keys: ShortVec<Pubkey>
    n_keys, pos = shortvec(buf, pos)
    keys = [bytes(buf[pos + i*32: pos + (i+1)*32]) for i in range(n_keys)]
    pos += n_keys * 32
    # recent_blockhash: 32 bytes
    pos += 32
    # instructions: ShortVec<CompiledInstruction>
    n_ix, pos = shortvec(buf, pos)
    ixs = []
    for _ in range(n_ix):
        prog_idx = buf[pos]; pos += 1
        n_acc, pos = shortvec(buf, pos)
        accs = bytes(buf[pos: pos + n_acc]); pos += n_acc
        n_dat, pos = shortvec(buf, pos)
        data = bytes(buf[pos: pos + n_dat]); pos += n_dat
        ixs.append((prog_idx, accs, data))
    # V0 only: address_table_lookups
    if is_versioned:
        n_alt, pos = shortvec(buf, pos)
        for _ in range(n_alt):
            pos += 32                                   # account_key
            n_wr, pos = shortvec(buf, pos); pos += n_wr  # writable_indexes
            n_rd, pos = shortvec(buf, pos); pos += n_rd  # readonly_indexes
    return {"first_sig": first_sig, "n_signers": n_sigs,
            "version": version, "keys": keys, "ixs": ixs,
            # message header — (num_required_sigs, num_readonly_signed,
            # num_readonly_unsigned). Together with len(keys) this exactly
            # determines which accounts are writable. Critical for spoof
            # detection: a fake pump.fun intent can omit the write-lock on
            # the bonding curve and still pass surface-level checks.
            "msg_header": [int(hdr[0]), int(hdr[1]), int(hdr[2])]}, pos


# ---------- Outer bincode Vec<Entry> decoder ----------
def parse_entry_batch(buf: bytes) -> list[dict]:
    mv = memoryview(buf)
    pos = 0
    n_entries = struct.unpack_from("<Q", mv, pos)[0]; pos += 8
    out = []
    for _ in range(n_entries):
        pos += 8     # num_hashes (u64) — we don't care
        pos += 32    # hash (32 bytes) — we don't care for intent
        n_tx = struct.unpack_from("<Q", mv, pos)[0]; pos += 8
        for _ in range(n_tx):
            tx, pos = parse_vt(mv, pos)
            out.append(tx)
    return out


SYSTEM_PROG = base58.b58decode("11111111111111111111111111111111")


def _is_writable(account_idx: int, n_keys: int, msg_header: list[int]) -> bool:
    """Solana wire-format writability: account_keys are ordered
    [writable_signers, readonly_signers, writable_unsigned, readonly_unsigned].
    msg_header = (num_req_sigs, num_readonly_signed, num_readonly_unsigned)."""
    n_req, n_ro_signed, n_ro_unsigned = msg_header
    if account_idx < n_req:                      # signer region
        return account_idx < (n_req - n_ro_signed)
    return account_idx < (n_keys - n_ro_unsigned)


# ---------- Pump.fun Buy + Sell intent extraction ----------
def extract_intents(txs: list[dict], slot: int) -> list[dict]:
    """For each tx, find pump.fun Buy AND Sell instructions. Returns a list of intent dicts.

    Records EVERYTHING cheaply extractable per tx — we'd rather save too much
    now and decide later than have to re-process raw shreds. JSON-friendly
    fields only (no raw bytes / no signatures beyond first_sig).

    Captured per intent:
        is_buy           : True for Buy, False for Sell
        slot             : Solana slot
        signer           : fee payer (account_keys[0]) base58
        user             : Buy/Sell instruction's user account (accs[6]) base58
        mint             : pump.fun token mint (accs[2]) base58
        token_amount     : raw u64 token amount (buy: tokens out; sell: tokens in)
        sol_limit_lam    : unified slippage cap u64 (buy: max_sol_cost; sell: min_sol_output)
        sol_limit_sol    : derived from sol_limit_lam / 1e9
        max_sol_cost     : (buy only) same as sol_limit_lam for backward compat
        min_sol_output   : (sell only) same as sol_limit_lam
        priority_fee_micro: ComputeBudget set_compute_unit_price arg (microlamports/CU)
        cu_limit         : ComputeBudget set_compute_unit_limit arg
        jito_tip_lam     : lamports transferred to a known Jito tip account
                           CAVEAT: Solana atomicity means this tip is FREE to spoof
                           (revertable). For real signal, prefer priority_fee_micro
                           which costs the spoofer per CU even on revert.
        first_sig        : base58 of fee-payer signature (lets us cross-ref
                           against grpc_capture's executed-tx stream to measure
                           land-rate and detect spoofs)
        n_ix             : total instructions in tx (proxy for tx complexity)
        n_accounts       : total accounts (proxy for tx complexity)
        n_signers        : signature count
        programs_touched : list of distinct program-id base58 strings in the tx.
                           Reveals if this tx ALSO touches Jupiter, Raydium,
                           Token2022, other pump.fun-ish programs — useful for
                           classifying MEV bundles and discriminating retail
                           vs sophisticated actors.
    """
    intents = []
    for tx in txs:
        keys = tx["keys"]
        if not keys: continue
        # Quick filter: tx must reference pump.fun program in its account keys
        if PUMP_FUN_PROG not in keys:
            continue
        # Pre-pass: count instructions per program, extract compute-budget + tip
        priority_fee_micro = 0
        cu_limit = 0
        jito_tip_lam = 0
        programs_touched = set()
        for prog_idx, accs, data in tx["ixs"]:
            if prog_idx >= len(keys): continue
            prog = keys[prog_idx]
            programs_touched.add(prog)
            if prog == COMPUTE_BUDGET and len(data) >= 5:
                disc = data[0]
                if disc == 2 and len(data) >= 5:
                    cu_limit = struct.unpack_from("<I", data, 1)[0]
                elif disc == 3 and len(data) >= 9:
                    priority_fee_micro = struct.unpack_from("<Q", data, 1)[0]
            elif prog == SYSTEM_PROG and len(data) >= 12:
                if data[:4] == b"\x02\x00\x00\x00":   # SystemTransfer
                    lam = struct.unpack_from("<Q", data, 4)[0]
                    if len(accs) >= 2:
                        dest_idx = accs[1]
                        if dest_idx < len(keys) and keys[dest_idx] in JITO_TIP_ACCOUNTS:
                            jito_tip_lam = lam
        progs_b58 = [base58.b58encode(p).decode() for p in programs_touched]
        # Find EVERY pump.fun instruction (Buy, Sell, Create, Migrate, etc).
        # One record per pump.fun ix.
        for prog_idx, accs, data in tx["ixs"]:
            if prog_idx >= len(keys): continue
            if keys[prog_idx] != PUMP_FUN_PROG: continue
            if len(data) < 8: continue
            disc = bytes(data[:8])
            ix_name = PUMPFUN_IX_NAMES.get(disc, "pumpfun_other")
            # Non-buy/sell records: emit a minimal "pump.fun event" record
            # with the ix discriminator, raw data preview, and accounts list.
            # No mint extraction by default — different ix types put the mint
            # at different account positions; downstream analysis can decode.
            if ix_name not in ("buy", "sell"):
                accs_b58 = [base58.b58encode(keys[a]).decode()
                            for a in accs if a < len(keys)]
                intents.append({
                    "type":                ix_name,
                    "slot":                slot,
                    "signer":              base58.b58encode(keys[0]).decode(),
                    "first_sig":           base58.b58encode(tx["first_sig"]).decode() if tx["first_sig"] else "",
                    "ix_disc_hex":         disc.hex(),
                    "ix_data_hex":         bytes(data[:128]).hex(),  # cap at 128 bytes
                    "ix_data_len":         len(data),
                    "ix_accounts":         accs_b58,
                    "priority_fee_micro":  priority_fee_micro,
                    "cu_limit":            cu_limit,
                    "jito_tip_lam":        jito_tip_lam,
                    "n_ix":                len(tx["ixs"]),
                    "n_accounts":          len(keys),
                    "n_signers":           tx.get("n_signers", 1),
                    "programs_touched":    progs_b58,
                })
                continue
            # Buy/Sell path (existing): need >=24 data bytes for amount+limit
            if len(data) < 24: continue
            is_buy = (ix_name == "buy")
            # Both Buy and Sell have same data layout:
            #   disc(8) + token_amount(u64) + sol_limit(u64)
            # (Buy: sol_limit = max_sol_cost; Sell: sol_limit = min_sol_output)
            # Both have same account layout: accs[2]=mint, accs[6]=user.
            token_amount = struct.unpack_from("<Q", data, 8)[0]
            sol_limit    = struct.unpack_from("<Q", data, 16)[0]
            # Version-aware account layout (2026-06-09 fix: v2 was being parsed
            # with v1 positions -> 26% of buy/sell records had quote_mint
            # written to the `mint` field instead of base_mint).
            #
            # Per official IDL (pump-fun/pump-public-docs idl/pump.json):
            #   v1 buy/sell (16/14 accs):
            #     accs[2]=mint  accs[3]=bonding_curve  accs[6]=user(signer)
            #   v2 buy/sell (27/26 accs):
            #     accs[1]=base_mint  accs[2]=quote_mint(!)  accs[10]=bonding_curve
            #     accs[13]=user(signer)
            #
            # Data layout (offsets 8 and 16) is identical between v1 and v2 -
            # only the account positions shifted. token_amount and sol_limit
            # parsing above stays correct for both versions.
            is_v2 = (disc == BUY_V2_DISC) or (disc == SELL_V2_DISC)
            version = 2 if is_v2 else 1
            if is_v2:
                if len(accs) < 14: continue
                mint_idx          = accs[1]    # base_mint (the actual token)
                user_idx          = accs[13]   # writable+signer
                bonding_curve_idx = accs[10]
                quote_mint_idx    = accs[2]    # SOL or USDC - saved separately
            else:
                if len(accs) < 7: continue
                mint_idx          = accs[2]
                user_idx          = accs[6]
                bonding_curve_idx = accs[3] if len(accs) >= 4 else -1
                quote_mint_idx    = -1         # legacy is SOL-only, implicit

            if mint_idx >= len(keys) or user_idx >= len(keys): continue
            if quote_mint_idx >= len(keys): quote_mint_idx = -1  # bad spoof, drop the field
            # Spoof-resistance: bonding_curve must be writable. Real pump.fun
            # buy/sell tx structurally cannot mutate AMM state otherwise ->
            # guaranteed revert -> ~free spoof.
            hdr = tx.get("msg_header", [0, 0, 0])
            bonding_curve_writable = (
                bonding_curve_idx >= 0 and
                bonding_curve_idx < len(keys) and
                _is_writable(bonding_curve_idx, len(keys), hdr)
            )
            # Save raw disc + per-ix accounts so future schema changes can be
            # repaired retroactively from saved data (the bug we just fixed
            # could have been caught earlier if these had been saved before).
            accs_b58 = [base58.b58encode(keys[a]).decode()
                        for a in accs if a < len(keys)]
            rec = {
                "type":               "buy" if is_buy else "sell",
                "version":            version,    # 1 = legacy, 2 = with-creator ABI
                "is_buy":             is_buy,
                "slot":               slot,
                "signer":             base58.b58encode(keys[0]).decode(),
                "user":               base58.b58encode(keys[user_idx]).decode(),
                "mint":               base58.b58encode(keys[mint_idx]).decode(),
                "quote_mint":         (base58.b58encode(keys[quote_mint_idx]).decode()
                                       if quote_mint_idx >= 0 else None),
                "token_amount":       token_amount,
                "sol_limit_lam":      sol_limit,
                "sol_limit_sol":      sol_limit / 1e9,
                "priority_fee_micro": priority_fee_micro,
                "cu_limit":           cu_limit,
                "jito_tip_lam":       jito_tip_lam,
                "first_sig":          base58.b58encode(tx["first_sig"]).decode() if tx["first_sig"] else "",
                "n_ix":               len(tx["ixs"]),
                "n_accounts":         len(keys),
                "n_signers":          tx.get("n_signers", 1),
                "programs_touched":   progs_b58,
                # Raw evidence for retroactive repair on any future ABI shift:
                "ix_disc_hex":        disc.hex(),
                "ix_accounts":        accs_b58,
                # Spoof-resistance signals (peer review 2026-06-08):
                "msg_header":         hdr,
                "bonding_curve_writable": bonding_curve_writable,
                # cu_limit_too_low: real pump.fun txs need ~30-50k CU. Anything
                # below 30k structurally cannot complete the AMM math + SPL
                # transfer -> guaranteed ComputationalBudgetExceeded revert.
                # Set the threshold conservatively at 25k to leave headroom
                # for ABI changes; analysis can re-bucket later.
                "cu_limit_too_low":   bool(cu_limit > 0 and cu_limit < 25_000),
                # PROBABLE_SPOOF flag = both signals say "guaranteed revert".
                # Use in analysis to drop noise before any reputation work.
                "probable_spoof":     (not bonding_curve_writable) or
                                       (cu_limit > 0 and cu_limit < 25_000),
            }
            # Backward-compat alias used by intent_recorder + ring writer
            if is_buy:
                rec["max_sol_cost"]     = sol_limit
                rec["max_sol_cost_sol"] = sol_limit / 1e9
            else:
                rec["min_sol_output"]     = sol_limit
                rec["min_sol_output_sol"] = sol_limit / 1e9
                # Ring writer reads max_sol_cost; mirror it so sells still get
                # the sol_limit captured in the shm ring.
                rec["max_sol_cost"] = sol_limit
            intents.append(rec)
    return intents


# ---------- Main ----------
async def main():
    print(f"=== shred intent extractor @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"  endpoint: {ENDPOINT}")
    print(f"  run_sec:  {RUN_SEC}")

    n_entries = 0
    n_txs = 0
    n_pump = 0
    n_buys = 0
    parse_errors = 0
    distinct_mints = set()
    distinct_users = set()
    tip_dist = []
    cost_dist = []
    sample_intents = []

    t0 = time.time()
    try:
        async with grpc.aio.insecure_channel(ENDPOINT) as ch:
            stub = shredstream_pb2_grpc.ShredstreamProxyStub(ch)
            req = shredstream_pb2.SubscribeEntriesRequest()
            async for msg in stub.SubscribeEntries(req, timeout=RUN_SEC + 5):
                if time.time() - t0 >= RUN_SEC: break
                n_entries += 1
                try:
                    txs = parse_entry_batch(msg.entries)
                except Exception:
                    parse_errors += 1
                    continue
                n_txs += len(txs)
                # Pump.fun count (any tx that includes the program)
                for tx in txs:
                    if PUMP_FUN_PROG in tx["keys"]:
                        n_pump += 1
                intents = extract_intents(txs, int(msg.slot))
                n_buys += len(intents)
                for it in intents:
                    distinct_mints.add(it["mint"])
                    distinct_users.add(it["user"])
                    if it["jito_tip_lam"] > 0: tip_dist.append(it["jito_tip_lam"])
                    cost_dist.append(it["max_sol_cost"])
                    if len(sample_intents) < 12:
                        sample_intents.append(it)
                if n_entries % 100 == 0:
                    print(f"    .. {n_entries} entries  {n_txs} tx  {n_buys} pump.fun buys "
                          f"(@ slot {int(msg.slot)})", flush=True)
    except Exception as e:
        print(f"  stream error: {e}")

    dur = time.time() - t0
    print(f"\n=== summary ({dur:.1f}s) ===")
    print(f"  entries:        {n_entries}    ({n_entries/max(0.001,dur):.0f}/s)")
    print(f"  transactions:   {n_txs}        ({n_txs/max(0.001,dur):.0f}/s)")
    print(f"  parse errors:   {parse_errors}")
    print(f"  pump.fun txs:   {n_pump}       ({n_pump/max(0.001,dur):.0f}/s)")
    print(f"  pump.fun BUYS:  {n_buys}       ({n_buys/max(0.001,dur):.1f}/s)")
    print(f"  distinct mints: {len(distinct_mints)}")
    print(f"  distinct users: {len(distinct_users)}")
    if cost_dist:
        import statistics as st
        cs = sorted(cost_dist)
        print(f"  max_sol_cost SOL distribution:  "
              f"p50={cs[len(cs)//2]/1e9:.3f}  p90={cs[int(len(cs)*0.9)]/1e9:.3f}  "
              f"max={max(cs)/1e9:.3f}")
    if tip_dist:
        ts = sorted(tip_dist)
        print(f"  jito_tip lamports:              "
              f"n={len(ts)}  p50={ts[len(ts)//2]}  p90={ts[int(len(ts)*0.9)]}  "
              f"max={max(ts)}    ({len(tip_dist)}/{n_buys} buys = "
              f"{100*len(tip_dist)/max(1,n_buys):.0f}% tipped)")
    print(f"\nfirst {len(sample_intents)} intents:")
    for it in sample_intents:
        print(f"  slot={it['slot']}  mint={it['mint'][:14]}  user={it['user'][:14]}  "
              f"buy_max_sol={it['max_sol_cost_sol']:.4f}  tip={it['jito_tip_lam']}  "
              f"prio_µ={it['priority_fee_micro']}")


if __name__ == "__main__":
    asyncio.run(main())
