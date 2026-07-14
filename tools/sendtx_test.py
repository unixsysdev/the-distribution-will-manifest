"""Test the documented single-tx path: Jito sendTransaction proxy (/api/v1/transactions,
base64, MEV protection, skip_preflight) instead of sendBundle. Build our buy tx
(compute-budget priority fee + tip in-tx, ~70/30), POST sendTransaction, then poll
getSignatureStatuses for landing. Buy-only (landing test)."""
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
import config
from pump_fun_ix import (build_buy_ix, build_ata_create_idempotent_ix, tokens_out_for_sol,
                         slippage_max_sol_cost, derive_bonding_curve)
from jito_exec import get_cached_tip_account

MINT = "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"
URL = "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/transactions"
wallet = Keypair.from_base58_string(open("/root/wallet.key").read().strip()); user = wallet.pubkey()
mint = Pubkey.from_string(MINT); fee_recip = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
TIP = 100_000  # ~30% vs priority-fee ~0.0004 SOL (70/30 per docs)

async def main():
    async with AsyncClient(config.rpc_http_url()) as cli:
        bal0 = (await cli.get_balance(user)).value
        tp = (await cli.get_account_info(mint)).value.owner
        bc = derive_bonding_curve(mint)
        d = bytes((await cli.get_account_info(bc, encoding="base64")).value.data)
        vtok, vsol = struct.unpack_from("<QQ", d, 8); creator = Pubkey.from_bytes(d[49:81])
        sol_in = int(0.002 * 1e9); et = tokens_out_for_sol(vsol, vtok, sol_in); mc = slippage_max_sol_cost(sol_in, 1500)
        ixs = [set_compute_unit_limit(200_000), set_compute_unit_price(2_000_000),
               build_ata_create_idempotent_ix(user, user, mint, tp),
               build_buy_ix(mint, user, fee_recip, et, mc, token_program=tp, creator=creator),
               transfer(TransferParams(from_pubkey=user, to_pubkey=Pubkey.from_string(get_cached_tip_account()), lamports=TIP))]
        bh = (await cli.get_latest_blockhash()).value.blockhash
        tx = VersionedTransaction(MessageV0.try_compile(payer=user, instructions=ixs, address_lookup_table_accounts=[], recent_blockhash=bh), [wallet])
        b64 = base64.b64encode(bytes(tx)).decode()
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                           "params": [b64, {"encoding": "base64"}]}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=10))
        print("sendTransaction resp:", resp)
        sig = resp.get("result")
        if not sig:
            print("no signature returned"); return
        for i in range(20):
            await asyncio.sleep(2)
            st = (await cli.get_signature_statuses([Signature.from_string(sig)])).value[0]
            print(f"  t={i*2+2:>3}s  {st.confirmation_status if st else None}  err={st.err if st else None}")
            if st and st.confirmation_status is not None:
                break
        bal1 = (await cli.get_balance(user)).value
        print(f"balance {bal0/1e9:.6f} -> {bal1/1e9:.6f}  delta {(bal1-bal0)/1e9:+.6f} SOL  sig={sig}")
asyncio.run(main())
