"""Simulate the buy tx exactly as _do_buy builds it, to see WHY Jito drops it
(account-layout / fee-recipient / creator-vault staleness). No submission."""
import os, asyncio, sys, struct
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
import config
from pump_fun_ix import (build_buy_ix, build_ata_create_idempotent_ix,
                         tokens_out_for_sol, slippage_max_sol_cost, derive_bonding_curve)

MINT = "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"
wallet = Keypair.from_base58_string(open("/root/wallet.key").read().strip())
user = wallet.pubkey()
mint_pk = Pubkey.from_string(MINT)
fee_recipient = Pubkey.from_string(os.getenv("PUMPFUN_FEE_RECIPIENT",
                                   "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM"))

async def main():
    async with AsyncClient(config.rpc_http_url()) as cli:
        acc = await cli.get_account_info(derive_bonding_curve(mint_pk), encoding="base64")
        data = bytes(acc.value.data)
        vtok, vsol = struct.unpack_from("<QQ", data, 8)
        print(f"curve owner: {acc.value.owner}  data_len: {len(data)}")
        sol_in = int(0.002 * 1e9)
        et = tokens_out_for_sol(vsol, vtok, sol_in)
        mc = slippage_max_sol_cost(sol_in, 1500)
        ata = build_ata_create_idempotent_ix(user, user, mint_pk)
        buy = build_buy_ix(mint_pk, user, fee_recipient, et, mc)
        print(f"buy_ix accounts: {len(buy.accounts)}  data_len: {len(buy.data)}")
        bh = (await cli.get_latest_blockhash()).value.blockhash
        msg = MessageV0.try_compile(payer=user, instructions=[ata, buy],
                                    address_lookup_table_accounts=[], recent_blockhash=bh)
        tx = VersionedTransaction(msg, [wallet])
        sim = await cli.simulate_transaction(tx)
        print("SIM ERR:", sim.value.err)
        print("SIM LOGS:")
        for l in (sim.value.logs or []):
            print("   ", l)

asyncio.run(main())
