import asyncio, sys, struct
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
import config

PUMP = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
FEE = Pubkey.from_string("pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ")
SYS = Pubkey.from_string("11111111111111111111111111111111")
ATA_PROG = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
BUY_DISC = bytes.fromhex("66063d1201daebea")
BREAKING = "GXPFM2caqTtQYC2cJ5yJRi9VDkpsYZXzYdwYpGnLmtDL"  # one of the 8 observed in real buys
def pda(seeds, prog=PUMP): return Pubkey.find_program_address(seeds, prog)[0]
def ata(o, m, tp): return Pubkey.find_program_address([bytes(o), bytes(tp), bytes(m)], ATA_PROG)[0]
GLOBAL = pda([b"global"]); EVAUTH = pda([b"__event_authority"]); GVA = pda([b"global_volume_accumulator"])
FEE_CONFIG = Pubkey.find_program_address([b"fee_config", bytes(PUMP)], FEE)[0]

MINT = sys.argv[1] if len(sys.argv) > 1 else "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"
wallet = Keypair.from_base58_string(open("/root/wallet.key").read().strip()); user = wallet.pubkey()
mint = Pubkey.from_string(MINT); fee_recip = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")

async def main():
    async with AsyncClient(config.rpc_http_url()) as c:
        tp = (await c.get_account_info(mint)).value.owner
        bc = pda([b"bonding-curve", bytes(mint)])
        d = bytes((await c.get_account_info(bc, encoding="base64")).value.data)
        vtok, vsol = struct.unpack_from("<QQ", d, 8); creator = Pubkey.from_bytes(d[49:81])
        sol_in = int(0.002 * 1e9); et = int(vtok - (vsol * vtok) // (vsol + sol_in)); mc = int(sol_in * 11500 // 10000)
        bc_v2 = pda([b"bonding-curve-v2", bytes(mint)]); uva = pda([b"user_volume_accumulator", bytes(user)])
        cv = pda([b"creator-vault", bytes(creator)]); ata_user = ata(user, mint, tp); ata_bc = ata(bc, mint, tp)
        ata_ix = Instruction(ATA_PROG, bytes([1]), [AccountMeta(user, True, True), AccountMeta(ata_user, False, True),
                 AccountMeta(user, False, False), AccountMeta(mint, False, False), AccountMeta(SYS, False, False), AccountMeta(tp, False, False)])
        accs = [AccountMeta(GLOBAL, False, False), AccountMeta(fee_recip, False, True), AccountMeta(mint, False, False),
                AccountMeta(bc, False, True), AccountMeta(ata_bc, False, True), AccountMeta(ata_user, False, True),
                AccountMeta(user, True, True), AccountMeta(SYS, False, False), AccountMeta(tp, False, False),
                AccountMeta(cv, False, True), AccountMeta(EVAUTH, False, False), AccountMeta(PUMP, False, False),
                AccountMeta(GVA, False, False), AccountMeta(uva, False, True), AccountMeta(FEE_CONFIG, False, False),
                AccountMeta(FEE, False, False), AccountMeta(bc_v2, False, True), AccountMeta(Pubkey.from_string(BREAKING), False, True)]
        data = BUY_DISC + struct.pack("<Q", et) + struct.pack("<Q", mc) + bytes([1, 1])
        buy = Instruction(PUMP, data, accs)
        bh = (await c.get_latest_blockhash()).value.blockhash
        msg = MessageV0.try_compile(payer=user, instructions=[ata_ix, buy], address_lookup_table_accounts=[], recent_blockhash=bh)
        sim = await c.simulate_transaction(VersionedTransaction(msg, [wallet]))
        print(f"mint={MINT[:10]} accts={len(accs)} creator={creator}")
        print("SIM ERR:", sim.value.err)
        for l in (sim.value.logs or []):
            print("  ", l)
asyncio.run(main())
