"""Test bundle LANDING the documented Jito way: pull the live tip_floor, submit a
buy bundle, and poll getInflightBundleStatuses by bundle_id immediately (Pending ->
Landed/Failed/Invalid) instead of the Solana sig check. Tip set to the live landed p95."""
import asyncio, sys, struct, base64, json, urllib.request
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.system_program import TransferParams, transfer
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solana.rpc.async_api import AsyncClient
import config
from pump_fun_ix import (build_buy_ix, build_ata_create_idempotent_ix, tokens_out_for_sol,
                         slippage_max_sol_cost, derive_bonding_curve)
from jito_exec import jito_client, get_cached_tip_account

MINT = "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"
wallet = Keypair.from_base58_string(open("/root/wallet.key").read().strip()); user = wallet.pubkey()
mint = Pubkey.from_string(MINT); fee_recip = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")

tf = json.load(urllib.request.urlopen("https://bundles.jito.wtf/api/v1/bundles/tip_floor", timeout=8))[0]
p50, p75, p95, p99 = (tf["landed_tips_50th_percentile"], tf["landed_tips_75th_percentile"],
                      tf["landed_tips_95th_percentile"], tf["landed_tips_99th_percentile"])
print(f"LIVE tip_floor (SOL): p50={p50:.6f} p75={p75:.6f} p95={p95:.6f} p99={p99:.6f}")
TIP = int(float(sys.argv[1]) if len(sys.argv) > 1 else p95 * 1e9)
print(f"using tip = {TIP} lam ({TIP/1e9:.6f} SOL, = live landed p95)")

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
        resp = jito_client().send_bundle(params=[b64])
        bid = (resp.get("data") or {}).get("result") if isinstance(resp, dict) else None
        print("send ok:", resp.get("success"), " bundle_id:", bid)
        for i in range(30):
            await asyncio.sleep(2)
            st = jito_client().get_inflight_bundle_statuses([bid])
            v = ((((st or {}).get("data") or {}).get("result") or {}).get("value") or [{}])[0]
            print(f"  t={i*2+2:>3}s  status={v.get('status')}  landed_slot={v.get('landed_slot')}")
            if v.get("status") in ("Landed", "Failed"): break
        bal1 = (await cli.get_balance(user)).value
        print(f"balance {bal0/1e9:.6f} -> {bal1/1e9:.6f}  delta {(bal1-bal0)/1e9:+.6f} SOL")
asyncio.run(main())
