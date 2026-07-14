"""Close EMPTY token accounts (balance 0) to reclaim their ~0.002 SOL rent, in
batches via sendTransaction. Standalone now (reclaim what we already locked) and
the same logic will run as a periodic broker loop. Decoupled from the sell path,
so it can never jeopardize an exit."""
import asyncio, sys, struct, base64
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TokenAccountOpts
import config
from jito_exec import send_transaction_b64

TOKEN = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
T22 = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
w = Keypair.from_base58_string(open("/root/wallet.key").read().strip()); user = w.pubkey()

def close_ix(acct, tp):  # SPL/Token-2022 CloseAccount = ix 9
    return Instruction(tp, bytes([9]), [AccountMeta(acct, False, True),
                                        AccountMeta(user, False, True), AccountMeta(user, True, False)])

async def main():
    async with AsyncClient(config.rpc_http_url()) as c:
        empties = []
        for tp in (T22, TOKEN):
            resp = await c.get_token_accounts_by_owner(user, TokenAccountOpts(program_id=tp, encoding="base64"))
            for ka in resp.value:
                d = bytes(ka.account.data)
                amount = struct.unpack_from("<Q", d, 64)[0] if len(d) >= 72 else 0
                if amount == 0:
                    empties.append((ka.pubkey, tp))
        print(f"empty token accounts found: {len(empties)}")
        if not empties:
            print("nothing to reclaim"); return
        bal0 = (await c.get_balance(user, commitment=Confirmed)).value
        for i in range(0, len(empties), 18):
            batch = empties[i:i+18]
            bh = (await c.get_latest_blockhash()).value.blockhash
            tx = VersionedTransaction(MessageV0.try_compile(payer=user, instructions=[close_ix(a, tp) for a, tp in batch],
                                      address_lookup_table_accounts=[], recent_blockhash=bh), [w])
            resp = send_transaction_b64(base64.b64encode(bytes(tx)).decode())
            print(f"  batch {i//18}: closing {len(batch)} accts -> {resp.get('result')}")
            await asyncio.sleep(9)
        await asyncio.sleep(3)
        bal1 = (await c.get_balance(user, commitment=Confirmed)).value
        print(f"SOL {bal0/1e9:.6f} -> {bal1/1e9:.6f}  reclaimed {(bal1-bal0)/1e9:+.6f}")
asyncio.run(main())
