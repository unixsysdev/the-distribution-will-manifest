import asyncio, sys, struct
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
import config
from pump_fun_ix import (build_sell_ix, sol_out_for_tokens, slippage_min_sol_output,
                         derive_bonding_curve, derive_ata, BREAKING_FEE_RECIPIENTS)
w = Keypair.from_base58_string(open("/root/wallet.key").read().strip()); user = w.pubkey()
mint = Pubkey.from_string("35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump")
fee = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")

async def m():
    async with AsyncClient(config.rpc_http_url()) as c:
        tp = (await c.get_account_info(mint)).value.owner
        ata = derive_ata(user, mint, tp)
        tok = int((await c.get_token_account_balance(ata, commitment=Confirmed)).value.amount)
        bc = derive_bonding_curve(mint); d = bytes((await c.get_account_info(bc, encoding="base64")).value.data)
        vtok, vsol = struct.unpack_from("<QQ", d, 8); creator = Pubkey.from_bytes(d[49:81])
        mn = slippage_min_sol_output(sol_out_for_tokens(vsol, vtok, tok), 1500)
        bh = (await c.get_latest_blockhash()).value.blockhash
        for br in BREAKING_FEE_RECIPIENTS:
            sx = build_sell_ix(mint, user, fee, tok, mn, token_program=tp, creator=creator, breaking_fee_recipient=br)
            tx = VersionedTransaction(MessageV0.try_compile(payer=user, instructions=[sx], address_lookup_table_accounts=[], recent_blockhash=bh), [w])
            sim = await c.simulate_transaction(tx)
            tag = "AUTHORIZED" if sim.value.err is None else str(sim.value.err)
            print(f"  {str(br)[:14]}  {tag}")
asyncio.run(m())
