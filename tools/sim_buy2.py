"""Simulate the buy with the rewritten token-2022-aware builders. Fetches the
mint's token program (owner) and the bonding-curve creator (offset 49)."""
import asyncio, sys, struct
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
import config
from pump_fun_ix import (build_buy_ix, build_ata_create_idempotent_ix, tokens_out_for_sol,
                         slippage_max_sol_cost, derive_bonding_curve)

MINT = sys.argv[1] if len(sys.argv) > 1 else "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"
wallet = Keypair.from_base58_string(open("/root/wallet.key").read().strip())
user = wallet.pubkey()
mint_pk = Pubkey.from_string(MINT)
fee_recipient = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")

async def main():
    async with AsyncClient(config.rpc_http_url()) as cli:
        tp = (await cli.get_account_info(mint_pk)).value.owner
        bc = derive_bonding_curve(mint_pk)
        data = bytes((await cli.get_account_info(bc, encoding="base64")).value.data)
        vtok, vsol = struct.unpack_from("<QQ", data, 8)
        creator = Pubkey.from_bytes(data[49:81])
        sol_in = int(0.002 * 1e9)
        et = tokens_out_for_sol(vsol, vtok, sol_in)
        mc = slippage_max_sol_cost(sol_in, 1500)
        ata = build_ata_create_idempotent_ix(user, user, mint_pk, tp)
        buy = build_buy_ix(mint_pk, user, fee_recipient, et, mc, token_program=tp, creator=creator)
        bh = (await cli.get_latest_blockhash()).value.blockhash
        msg = MessageV0.try_compile(payer=user, instructions=[ata, buy],
                                    address_lookup_table_accounts=[], recent_blockhash=bh)
        tx = VersionedTransaction(msg, [wallet])
        sim = await cli.simulate_transaction(tx)
        print(f"mint={MINT[:10]} token_program={tp} creator={creator}")
        print(f"buy_ix accounts={len(buy.accounts)}")
        print("SIM ERR:", sim.value.err)
        for l in (sim.value.logs or []):
            print("  ", l)
asyncio.run(main())
