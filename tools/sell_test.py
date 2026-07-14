"""Sell the 35Fwy tokens we just bought, via the Jito sendTransaction proxy.
Validates build_sell_ix live + recovers the SOL + completes the round-trip."""
import asyncio, sys, struct, base64, json, urllib.request
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.system_program import TransferParams, transfer
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
import config
from pump_fun_ix import (build_sell_ix, sol_out_for_tokens, slippage_min_sol_output,
                         derive_bonding_curve, derive_ata)
from jito_exec import get_cached_tip_account

MINT = "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"
URL = "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/transactions"
wallet = Keypair.from_base58_string(open("/root/wallet.key").read().strip()); user = wallet.pubkey()
mint = Pubkey.from_string(MINT); fee_recip = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")

async def main():
    async with AsyncClient(config.rpc_http_url()) as cli:
        bal0 = (await cli.get_balance(user, commitment=Confirmed)).value
        tp = (await cli.get_account_info(mint)).value.owner
        ata = derive_ata(user, mint, tp)
        tok = int((await cli.get_token_account_balance(ata, commitment=Confirmed)).value.amount)
        print(f"selling {tok} tokens")
        bc = derive_bonding_curve(mint)
        d = bytes((await cli.get_account_info(bc, encoding="base64")).value.data)
        vtok, vsol = struct.unpack_from("<QQ", d, 8); creator = Pubkey.from_bytes(d[49:81])
        min_sol = slippage_min_sol_output(sol_out_for_tokens(vsol, vtok, tok), 1500)
        ixs = [set_compute_unit_limit(200_000), set_compute_unit_price(2_000_000),
               build_sell_ix(mint, user, fee_recip, tok, min_sol, token_program=tp, creator=creator),
               transfer(TransferParams(from_pubkey=user, to_pubkey=Pubkey.from_string(get_cached_tip_account()), lamports=100_000))]
        bh = (await cli.get_latest_blockhash()).value.blockhash
        tx = VersionedTransaction(MessageV0.try_compile(payer=user, instructions=ixs, address_lookup_table_accounts=[], recent_blockhash=bh), [wallet])
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                           "params": [base64.b64encode(bytes(tx)).decode(), {"encoding": "base64"}]}).encode()
        resp = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=10))
        sig = resp.get("result"); print("sell sendTransaction:", resp)
        if not sig: return
        for i in range(20):
            await asyncio.sleep(2)
            st = (await cli.get_signature_statuses([Signature.from_string(sig)])).value[0]
            print(f"  t={i*2+2:>3}s {st.confirmation_status if st else None} err={st.err if st else None}")
            if st and st.confirmation_status is not None: break
        bal1 = (await cli.get_balance(user, commitment=Confirmed)).value
        tok1 = int((await cli.get_token_account_balance(ata, commitment=Confirmed)).value.amount)
        print(f"SOL {bal0/1e9:.6f} -> {bal1/1e9:.6f} (delta {(bal1-bal0)/1e9:+.6f})  tokens left: {tok1}  sig={sig}")
asyncio.run(main())
