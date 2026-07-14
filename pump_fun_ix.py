"""pump.fun bonding-curve Anchor instruction builders (CURRENT ABI, 2026-06).

Buy/sell for pump.fun 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P + AMM math.

REWRITTEN 2026-06-11 (the tiny_live_roundtrip canary proved the old legacy-SPL
12-account layout fails on current tokens). Verified against the canonical IDL,
real on-chain buys, the chainstacklabs working bot, AND simulateTransaction
(a full 0.002 buy simulates clean: Buy -> fee GetFees -> Token-2022 TransferChecked
-> success). Current pump.fun mints are Token-2022; ATAs + token_program use the
mint's actual owner program. Buy carries creator_vault, global/user volume
accumulators, fee_config, fee_program, bonding_curve_v2, breaking_fee_recipient,
and args end with track_volume=Some(true) ([1,1]). SELL differs: creator_vault
before token_program, no global_volume_accumulator, breaking_fee_recipient last.
Callers pass the mint's token_program (mint owner) and the bonding-curve creator
(account data offset 49, 32 bytes).

DRY-RUN-SAFE: assembles instructions only; never signs or POSTs.
"""
from __future__ import annotations
import random
from solders.instruction import Instruction, AccountMeta
from solders.pubkey import Pubkey

# Program addresses
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
FEE_PROGRAM = Pubkey.from_string("pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
TOKEN_2022_PROGRAM = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
ATA_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Anchor instruction discriminators (sighash("global:name")[:8])
BUY_DISC = bytes.fromhex("66063d1201daebea")
SELL_DISC = bytes.fromhex("33e685a4017f83ad")

# PDA seeds
GLOBAL_SEED = b"global"
BONDING_CURVE_SEED = b"bonding-curve"
BONDING_CURVE_V2_SEED = b"bonding-curve-v2"
EVENT_AUTHORITY_SEED = b"__event_authority"
CREATOR_VAULT_SEED = b"creator-vault"
GLOBAL_VOL_SEED = b"global_volume_accumulator"
USER_VOL_SEED = b"user_volume_accumulator"
FEE_CONFIG_SEED = b"fee_config"

# Static PDAs
GLOBAL_PDA, _ = Pubkey.find_program_address([GLOBAL_SEED], PUMP_FUN_PROGRAM)
EVENT_AUTHORITY_PDA, _ = Pubkey.find_program_address([EVENT_AUTHORITY_SEED], PUMP_FUN_PROGRAM)
GLOBAL_VOLUME_ACCUMULATOR, _ = Pubkey.find_program_address([GLOBAL_VOL_SEED], PUMP_FUN_PROGRAM)
FEE_CONFIG_PDA, _ = Pubkey.find_program_address([FEE_CONFIG_SEED, bytes(PUMP_FUN_PROGRAM)], FEE_PROGRAM)

# breaking_fee_recipient: NOT a PDA — one of 8 fixed fee-program accounts (rotate).
BREAKING_FEE_RECIPIENTS = [
    Pubkey.from_string("5YxQFdt3Tr9zJLvkFccqXVUwhdTWJQc1fFg2YPbxvxeD"),
    Pubkey.from_string("9M4giFFMxmFGXtc3feFzRai56WbBqehoSeRE5GK7gf7"),
    Pubkey.from_string("GXPFM2caqTtQYC2cJ5yJRi9VDkpsYZXzYdwYpGnLmtDL"),
    Pubkey.from_string("3BpXnfJaUTiwXnJNe7Ej1rcbzqTTQUvLShZaWazebsVR"),
    Pubkey.from_string("5cjcW9wExnJJiqgLjq7DEG75Pm6JBgE1hNv4B2vHXUW6"),
    Pubkey.from_string("EHAAiTxcdDwQ3U4bU6YcMsQGaekdzLS3B5SmYo46kJtL"),
    Pubkey.from_string("5eHhjP8JaYkz83CWwvGU2uMUXefd3AazWGx4gpcuEEYD"),
    Pubkey.from_string("A7hAgCzFw14fejgCp387JUJRMNyz4j89JKnhtKU8piqW"),
]
def pick_breaking_fee_recipient() -> Pubkey:
    return random.choice(BREAKING_FEE_RECIPIENTS)


def derive_bonding_curve(mint: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([BONDING_CURVE_SEED, bytes(mint)], PUMP_FUN_PROGRAM)[0]

def derive_bonding_curve_v2(mint: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([BONDING_CURVE_V2_SEED, bytes(mint)], PUMP_FUN_PROGRAM)[0]

def derive_ata(owner: Pubkey, mint: Pubkey, token_program: Pubkey = TOKEN_PROGRAM) -> Pubkey:
    """ATA; token_program MUST be the mint's owner program (Token-2022 for current mints)."""
    return Pubkey.find_program_address([bytes(owner), bytes(token_program), bytes(mint)], ATA_PROGRAM)[0]

def derive_creator_vault(creator: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([CREATOR_VAULT_SEED, bytes(creator)], PUMP_FUN_PROGRAM)[0]

def derive_user_volume_accumulator(user: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([USER_VOL_SEED, bytes(user)], PUMP_FUN_PROGRAM)[0]


# ---------- AMM math ----------
def tokens_out_for_sol(vsol_lam: int, vtok: int, sol_in_lam: int) -> int:
    if vsol_lam <= 0 or vtok <= 0 or sol_in_lam <= 0: return 0
    return int(vtok - (vsol_lam * vtok) // (vsol_lam + sol_in_lam))

def sol_out_for_tokens(vsol_lam: int, vtok: int, tok_in: int) -> int:
    if vsol_lam <= 0 or vtok <= 0 or tok_in <= 0: return 0
    return int(vsol_lam - (vsol_lam * vtok) // (vtok + tok_in))

def slippage_max_sol_cost(sol_in_lam: int, slippage_bps: int = 1500) -> int:
    return int(sol_in_lam * (10000 + slippage_bps) // 10000)

def slippage_min_sol_output(sol_out_lam: int, slippage_bps: int = 1500) -> int:
    return int(sol_out_lam * (10000 - slippage_bps) // 10000)


# ---------- Instruction builders ----------
def build_ata_create_idempotent_ix(payer: Pubkey, owner: Pubkey, mint: Pubkey,
                                    token_program: Pubkey = TOKEN_PROGRAM) -> Instruction:
    ata = derive_ata(owner, mint, token_program)
    accs = [AccountMeta(payer, True, True), AccountMeta(ata, False, True),
            AccountMeta(owner, False, False), AccountMeta(mint, False, False),
            AccountMeta(SYSTEM_PROGRAM, False, False), AccountMeta(token_program, False, False)]
    return Instruction(ATA_PROGRAM, bytes([1]), accs)


def build_buy_ix(mint: Pubkey, user: Pubkey, fee_recipient: Pubkey,
                 token_amount: int, max_sol_cost_lam: int, *,
                 token_program: Pubkey, creator: Pubkey,
                 breaking_fee_recipient: Pubkey | None = None) -> Instruction:
    """pump.fun BUY (current ABI, 18 accounts; sim-verified)."""
    bc = derive_bonding_curve(mint)
    brk = breaking_fee_recipient or pick_breaking_fee_recipient()
    accs = [
        AccountMeta(GLOBAL_PDA, False, False),                           # 0 global
        AccountMeta(fee_recipient, False, True),                         # 1 fee_recipient
        AccountMeta(mint, False, False),                                 # 2 mint
        AccountMeta(bc, False, True),                                    # 3 bonding_curve
        AccountMeta(derive_ata(bc, mint, token_program), False, True),   # 4 assoc_bonding_curve
        AccountMeta(derive_ata(user, mint, token_program), False, True), # 5 assoc_user
        AccountMeta(user, True, True),                                   # 6 user
        AccountMeta(SYSTEM_PROGRAM, False, False),                       # 7 system
        AccountMeta(token_program, False, False),                        # 8 token_program
        AccountMeta(derive_creator_vault(creator), False, True),         # 9 creator_vault
        AccountMeta(EVENT_AUTHORITY_PDA, False, False),                  # 10 event_authority
        AccountMeta(PUMP_FUN_PROGRAM, False, False),                     # 11 program
        AccountMeta(GLOBAL_VOLUME_ACCUMULATOR, False, False),            # 12 global_vol_accum
        AccountMeta(derive_user_volume_accumulator(user), False, True),  # 13 user_vol_accum
        AccountMeta(FEE_CONFIG_PDA, False, False),                       # 14 fee_config
        AccountMeta(FEE_PROGRAM, False, False),                          # 15 fee_program
        AccountMeta(derive_bonding_curve_v2(mint), False, True),         # 16 bonding_curve_v2
        AccountMeta(brk, False, True),                                   # 17 breaking_fee_recipient
    ]
    data = BUY_DISC + token_amount.to_bytes(8, "little") + max_sol_cost_lam.to_bytes(8, "little") + bytes([1, 1])
    return Instruction(PUMP_FUN_PROGRAM, data, accs)


def build_sell_ix(mint: Pubkey, user: Pubkey, fee_recipient: Pubkey,
                  token_amount: int, min_sol_output_lam: int, *,
                  token_program: Pubkey, creator: Pubkey,
                  breaking_fee_recipient: Pubkey | None = None) -> Instruction:
    """pump.fun SELL (current ABI, 16 accounts). creator_vault before token_program;
    no global_volume_accumulator; breaking_fee_recipient last."""
    bc = derive_bonding_curve(mint)
    brk = breaking_fee_recipient or pick_breaking_fee_recipient()
    accs = [
        AccountMeta(GLOBAL_PDA, False, False),                           # 0 global
        AccountMeta(fee_recipient, False, True),                         # 1 fee_recipient
        AccountMeta(mint, False, False),                                 # 2 mint
        AccountMeta(bc, False, True),                                    # 3 bonding_curve
        AccountMeta(derive_ata(bc, mint, token_program), False, True),   # 4 assoc_bonding_curve
        AccountMeta(derive_ata(user, mint, token_program), False, True), # 5 assoc_user
        AccountMeta(user, True, True),                                   # 6 user
        AccountMeta(SYSTEM_PROGRAM, False, False),                       # 7 system
        AccountMeta(derive_creator_vault(creator), False, True),         # 8 creator_vault
        AccountMeta(token_program, False, False),                        # 9 token_program
        AccountMeta(EVENT_AUTHORITY_PDA, False, False),                  # 10 event_authority
        AccountMeta(PUMP_FUN_PROGRAM, False, False),                     # 11 program
        AccountMeta(FEE_CONFIG_PDA, False, False),                       # 12 fee_config
        AccountMeta(FEE_PROGRAM, False, False),                          # 13 fee_program
        AccountMeta(derive_bonding_curve_v2(mint), False, True),         # 14 bonding_curve_v2
        AccountMeta(brk, False, True),                                   # 15 breaking_fee_recipient
    ]
    data = SELL_DISC + token_amount.to_bytes(8, "little") + min_sol_output_lam.to_bytes(8, "little")
    return Instruction(PUMP_FUN_PROGRAM, data, accs)


def build_close_account_ix(account, owner, token_program, dest=None):
    """Close an empty SPL / Token-2022 token account, reclaiming its rent to `dest`
    (default: owner). CloseAccount = instruction index 9. Fails on-chain if the
    account is non-empty, so callers MUST ensure balance == 0 first."""
    dest = dest or owner
    return Instruction(token_program, bytes([9]),
                       [AccountMeta(account, False, True),
                        AccountMeta(dest, False, True),
                        AccountMeta(owner, True, False)])
