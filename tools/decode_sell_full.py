import asyncio, sys, base58
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
import config
from pump_fun_ix import (PUMP_FUN_PROGRAM, FEE_PROGRAM, SELL_DISC, GLOBAL_PDA, EVENT_AUTHORITY_PDA,
                         GLOBAL_VOLUME_ACCUMULATOR, FEE_CONFIG_PDA, derive_bonding_curve,
                         derive_bonding_curve_v2, derive_creator_vault, derive_user_volume_accumulator, derive_ata)
TOKEN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"; T22 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ATA = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"; SYS = "11111111111111111111111111111111"

async def main():
    async with AsyncClient(config.rpc_http_url()) as c:
        sigs = (await c.get_signatures_for_address(PUMP_FUN_PROGRAM, limit=120)).value
        for si in sigs:
            if si.err is not None: continue
            r = await c.get_transaction(si.signature, encoding="jsonParsed", max_supported_transaction_version=0)
            if r.value is None: continue
            groups = [r.value.transaction.transaction.message.instructions]
            for ii in (getattr(getattr(r.value, "meta", None), "inner_instructions", None) or []): groups.append(ii.instructions)
            for g in groups:
                for ix in g:
                    pid = getattr(ix, "program_id", None); accs = getattr(ix, "accounts", None); data = getattr(ix, "data", None)
                    if pid is None or accs is None or data is None or str(pid) != str(PUMP_FUN_PROGRAM): continue
                    try: db = base58.b58decode(str(data))
                    except Exception: continue
                    if db[:8] != SELL_DISC: continue
                    a = [str(x) for x in accs]; aset = set(a)
                    mint = next((x for x in a if str(derive_bonding_curve(Pubkey.from_string(x))) in aset), None)
                    if not mint: continue
                    mpk = Pubkey.from_string(mint); tp = str((await c.get_account_info(mpk)).value.owner)
                    bc = str(derive_bonding_curve(mpk))
                    user = next((x for x in a if x != bc and str(derive_ata(Pubkey.from_string(x), mpk, Pubkey.from_string(tp))) in aset), None)
                    cr = Pubkey.from_bytes(bytes((await c.get_account_info(Pubkey.from_string(bc), encoding="base64")).value.data)[49:81])
                    L = {str(GLOBAL_PDA): "global", str(EVENT_AUTHORITY_PDA): "event_authority", str(PUMP_FUN_PROGRAM): "program",
                         SYS: "system", TOKEN: "token(legacy)", T22: "token(2022)", ATA: "ata_program", str(FEE_PROGRAM): "fee_program",
                         str(FEE_CONFIG_PDA): "fee_config", str(GLOBAL_VOLUME_ACCUMULATOR): "global_vol_accum",
                         mint: "mint", bc: "bonding_curve", str(derive_ata(Pubkey.from_string(bc), mpk, Pubkey.from_string(tp))): "assoc_bonding_curve",
                         str(derive_creator_vault(cr)): "creator_vault", str(derive_bonding_curve_v2(mpk)): "bonding_curve_v2"}
                    if user:
                        L[user] = "user"
                        L[str(derive_ata(Pubkey.from_string(user), mpk, Pubkey.from_string(tp)))] = "assoc_user"
                        L[str(derive_user_volume_accumulator(Pubkey.from_string(user)))] = "user_vol_accum"
                    print(f"real SELL {str(si.signature)[:14]} n={len(a)} mint={mint[:8]}")
                    for j, x in enumerate(a):
                        print(f"   [{j:2d}] {x}  {L.get(x, '?? (fee_recipient/breaking?)')}")
                    return
asyncio.run(main())
